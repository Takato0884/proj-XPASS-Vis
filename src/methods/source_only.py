import os

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import copy
from torch.utils.data import DataLoader

from ..train_common import earth_mover_distance, build_piaa_model, num_bins
from ..data import collate_fn


def setup(model, args, device):
    return {}


def _train_one_epoch(model, dataloader, optimizer, scaler, device, args, epoch: int = None):
    model.train()
    running_emd_loss = 0.0
    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Train"
    progress_bar = tqdm(dataloader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")
    for sample in progress_bar:
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['Aesthetic'].to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            aesthetic_logits = model(images)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_emd_loss += loss.item()
        progress_bar.set_postfix({'Train EMD': loss.item()})

    return running_emd_loss / len(dataloader)


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    train_dataloader, _, _ = src_dataloaders

    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        train_emd = _train_one_epoch(model, train_dataloader, optimizer, scaler, device, args, epoch=epoch)
        if args.is_log:
            wandb.log({"epoch": epoch, f"{args.genre}/Train EMD GIAA": train_emd}, commit=True)

    os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
    torch.save(model.state_dict(), best_modelname)

    model.load_state_dict(torch.load(best_modelname))


def _train_one_epoch_piaa(model, dataloader, optimizer, scaler, device, args, genre, epoch=None):
    model.train()
    running_loss = 0.0
    running_interaction = 0.0
    running_direct = 0.0
    total_batches = 0

    from torch.amp import autocast
    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Train"
    progress_bar = tqdm(total=len(dataloader), leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample in dataloader:
        optimizer.zero_grad()
        images = sample['image'].to(device)
        sample_pt = sample['traits'].float().to(device)
        sample_attr = sample['QIP'].float().to(device)
        aesthetic_scores = sample['Aesthetic'].to(device).view(-1, 1)

        with autocast('cuda'):
            score_pred = model(images, sample_pt, sample_attr, genre)
            loss = F.mse_loss(score_pred, aesthetic_scores)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_interaction += getattr(model, '_last_interaction_mean', 0.0)
        running_direct += getattr(model, '_last_direct_mean', 0.0)
        total_batches += 1
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': loss.item()})

    progress_bar.close()
    n = total_batches if total_batches > 0 else 1
    return running_loss / n, running_interaction / n, running_direct / n


def trainer_pretrain(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                     num_attr, num_pt, tgt_val_loader=None, tgt_genre=None):
    import wandb
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    train_loader = DataLoader(datasets_dict[genre]['train'], batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers, timeout=300 if args.num_workers else 0, collate_fn=collate_fn)

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)

    pretrained_path = pretrained_model_dict[genre]
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Error: Pretrained NIMA model file not found: {pretrained_path}")
    try:
        state = torch.load(pretrained_path)
        model.nima_dict[genre].load_state_dict(state)
        print(f"Loaded NIMA weights for {genre} from {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"Error: Failed to load NIMA weights for {genre} from {pretrained_path}: {e}")

    model.freeze_backbone()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_{experiment_name}_pretrain.pth')
    best_state_dict = None

    scaler = GradScaler('cuda')
    for epoch in range(args.num_epochs):
        train_loss, _, _ = _train_one_epoch_piaa(model, train_loader, optimizer, scaler, device, args, genre, epoch=epoch)
        if args.is_log:
            wandb.log({"epoch": epoch, f"{genre}/Train Loss": train_loss}, commit=True)

    if args.no_save_model:
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        torch.save(model.state_dict(), best_model_path)

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                     num_attr, num_pt, tgt_val_piaa_dataset=None, tgt_genre=None):
    import wandb
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"Training for user {uid}...")

        user_train_ds = copy.copy(datasets_dict[genre]['train'])
        user_train_ds.data = datasets_dict[genre]['train'].data[datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)

        total_train_samples = len(user_train_ds)
        print(f"User {uid}: train {total_train_samples} samples")
        if total_train_samples == 0:
            print(f"Skipping user {uid}: insufficient data")
            continue

        train_loader = DataLoader(user_train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers, timeout=300 if args.num_workers else 0, collate_fn=collate_fn)

        model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
        pretrained_path = pretrained_model_dict[genre]
        if pretrained_path is not None and os.path.exists(pretrained_path):
            try:
                state = torch.load(pretrained_path)
                model_user.load_state_dict(state)
                print(f"Loaded PIAA weights from {pretrained_path}")
            except Exception as e:
                raise RuntimeError(f"Error: Failed to load model weights from {pretrained_path}: {e}")
        else:
            raise FileNotFoundError(f"Error: Pretrained model file not found: {pretrained_path}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer_user = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)

        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')

        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            _train_one_epoch_piaa(model_user, train_loader, optimizer_user, scaler, device, args, genre, epoch=epoch)

        torch.save(model_user.state_dict(), best_model_path)

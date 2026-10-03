import os
import copy

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..train_common import build_piaa_model, load_weights, num_bins, save_weights
from ..data import collate_fn


def compute_rsd_bmp(feature_source, feature_target, eps=1e-8):
    F_s = feature_source.float().t()
    F_t = feature_target.float().t()

    U_s, _, _ = torch.linalg.svd(F_s, full_matrices=False)
    U_t, _, _ = torch.linalg.svd(F_t, full_matrices=False)

    M = U_s.t() @ U_t
    P_s, cos_theta, P_t_h = torch.linalg.svd(M, full_matrices=False)
    P_t = P_t_h.t()

    cos_sq = cos_theta.pow(2).clamp(max=1.0)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_sq, min=eps))
    L_RSD = sin_theta.sum()

    L_BMP = (P_s.abs() - P_t.abs()).pow(2).sum()

    return L_RSD, L_BMP


def _train_one_epoch_piaa(model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                           epoch=None, desc_suffix=""):
    model.train()
    beta = getattr(args, 'rsd_beta', 0.01)
    gamma = getattr(args, 'rsd_gamma', 1e-5)
    eps = getattr(args, 'rsd_eps', 1e-8)

    running_L_y = running_L_rsd = running_L_bmp = 0.0
    running_norm_src = running_norm_tgt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [RSD{desc_suffix}]" if epoch is not None else f"Train RSD{desc_suffix}"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120,
                        colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src = sample_src['image'].to(device)
        aesthetic_src = sample_src['Aesthetic'].to(device).view(-1, 1)
        pt_src = sample_src['traits'].float().to(device)
        attr_src = sample_src['QIP'].float().to(device)

        images_tgt = sample_tgt['image'].to(device)
        pt_tgt = sample_tgt['traits'].float().to(device)
        attr_tgt = sample_tgt['QIP'].float().to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            score_src, I_ij_src = model(images_src, pt_src, attr_src, genre, return_feat=True)
            _, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)

            L_y = F.mse_loss(score_src, aesthetic_src)

        L_RSD, L_BMP = compute_rsd_bmp(I_ij_src, I_ij_tgt, eps=eps)

        norm_src = I_ij_src.float().norm(dim=1).mean()
        norm_tgt = I_ij_tgt.float().norm(dim=1).mean()

        loss = L_y + beta * L_RSD + gamma * L_BMP

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_L_rsd = beta * L_RSD
        weighted_L_bmp = gamma * L_BMP

        running_L_y += L_y.item()
        running_L_rsd += weighted_L_rsd.item()
        running_L_bmp += weighted_L_bmp.item()
        running_norm_src += norm_src.item()
        running_norm_tgt += norm_tgt.item()
        total_batches += 1

        progress_bar.set_postfix({
            'L_y':   f'{L_y.item():.4f}',
            'L_RSD': f'{weighted_L_rsd.item():.4f}',
            'L_BMP': f'{weighted_L_bmp.item():.4f}',
            'norm_s': f'{norm_src.item():.2f}',
            'norm_t': f'{norm_tgt.item():.2f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_rsd / n, running_L_bmp / n, running_norm_src / n, running_norm_tgt / n


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = domain_tag if domain_tag else genre

    src_loader = DataLoader(datasets_dict[genre]['train'], batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300 if args.num_workers else 0, collate_fn=collate_fn)
    tgt_loader = DataLoader(tgt_train_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300 if args.num_workers else 0, collate_fn=collate_fn)

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)

    pretrained_path = pretrained_model_dict[genre]
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained NIMA model not found: {pretrained_path}")
    try:
        load_weights(model.nima_dict[genre], pretrained_path)
        print(f"Loaded NIMA weights for {genre} from {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load NIMA weights for {genre}: {e}")

    model.freeze_backbone()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    _rsd_run = experiment_name.removeprefix('RSD_')
    best_model_path = os.path.join(dirname, f'{genre_str}_RSD_{args.model_type}_{_rsd_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_rsd, L_bmp, norm_src, norm_tgt = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
            epoch=epoch, desc_suffix=" pretrain")

        if args.is_log:
            total_da = L_y + L_rsd + L_bmp
            ratio_y_da = L_y / total_da if total_da > 0 else 0.0
            ratio_rsd_bmp = L_rsd / (L_rsd + L_bmp) if (L_rsd + L_bmp) > 0 else 0.0
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss":  L_y,
                f"{genre}/Train L_RSD": L_rsd,
                f"{genre}/Train L_BMP": L_bmp,
                f"{genre}/Train ratio L_y/(L_y+L_RSD+L_BMP)": ratio_y_da,
                f"{genre}/Train ratio L_RSD/(L_RSD+L_BMP)":   ratio_rsd_bmp,
                f"{genre}/Train feat_norm_src": norm_src,
                f"{genre}/Train feat_norm_tgt": norm_tgt,
            }, commit=True)

    if args.no_save_model:
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        save_weights(model, best_model_path)

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, rsd_target_genre=None):
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"RSD finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{rsd_target_genre}' train_piaa_dataset.")
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        total_train_src = len(user_train_src)
        total_train_tgt = len(user_train_tgt)
        print(f"User {uid}: train src={total_train_src}, train tgt={total_train_tgt}")
        if total_train_src < batch_size or total_train_tgt < batch_size:
            print(f"Skipping user {uid}: need >={batch_size} per split")
            continue

        src_loader = DataLoader(user_train_src, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300 if args.num_workers else 0, collate_fn=collate_fn)
        tgt_loader = DataLoader(user_train_tgt, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300 if args.num_workers else 0, collate_fn=collate_fn)

        model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
        pretrained_path = pretrained_model_dict[genre]
        if pretrained_path is None or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"RSD pretrained model not found: {pretrained_path}")
        try:
            load_weights(model_user, pretrained_path)
            print(f"Loaded RSD pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)

        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_y, L_rsd, L_bmp, norm_src, norm_tgt = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                epoch=epoch, desc_suffix=" finetune")

            if args.is_log:
                total_da = L_y + L_rsd + L_bmp
                ratio_y_da = L_y / total_da if total_da > 0 else 0.0
                ratio_rsd_bmp = L_rsd / (L_rsd + L_bmp) if (L_rsd + L_bmp) > 0 else 0.0
                wandb.log({
                    "epoch": epoch,
                    f"{genre}/Train Loss user_{uid}":  L_y,
                    f"{genre}/Train L_RSD user_{uid}": L_rsd,
                    f"{genre}/Train L_BMP user_{uid}": L_bmp,
                    f"{genre}/Train ratio L_y/(L_y+L_RSD+L_BMP) user_{uid}": ratio_y_da,
                    f"{genre}/Train ratio L_RSD/(L_RSD+L_BMP) user_{uid}":   ratio_rsd_bmp,
                    f"{genre}/Train feat_norm_src user_{uid}": norm_src,
                    f"{genre}/Train feat_norm_tgt user_{uid}": norm_tgt,
                }, commit=True)

        save_weights(model_user, best_model_path)

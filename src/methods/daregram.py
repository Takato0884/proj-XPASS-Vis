import os
import copy

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..train_common import build_piaa_model, num_bins
from ..data import collate_fn


def _daregram_losses(Z_s, Z_t, T=0.95):
    Zs = Z_s.float()
    Zt = Z_t.float()

    G_s = Zs.t() @ Zs
    G_t = Zt.t() @ Zt

    U_s, S_s, Vh_s = torch.linalg.svd(G_s, full_matrices=False)
    U_t, S_t, Vh_t = torch.linalg.svd(G_t, full_matrices=False)

    def _select_k(S):
        total = S.sum().clamp_min(1e-12)
        cum = torch.cumsum(S, dim=0) / total
        k = int((cum < T).sum().item()) + 1
        return max(1, min(k, S.numel()))

    k_s = _select_k(S_s)
    k_t = _select_k(S_t)
    k = max(k_s, k_t)

    eps = 1e-8
    inv_s = torch.zeros_like(S_s)
    inv_s[:k] = 1.0 / (S_s[:k] + eps)
    G_s_pinv = (U_s[:, :k] * inv_s[:k].unsqueeze(0)) @ Vh_s[:k]

    inv_t = torch.zeros_like(S_t)
    inv_t[:k] = 1.0 / (S_t[:k] + eps)
    G_t_pinv = (U_t[:, :k] * inv_t[:k].unsqueeze(0)) @ Vh_t[:k]

    cos_sim = F.cosine_similarity(G_s_pinv, G_t_pinv, dim=0)
    L_cos = (1.0 - cos_sim).abs().sum()

    L_scale = torch.norm(S_s[:k] - S_t[:k], p=2)

    return L_cos, L_scale, k


def _train_one_epoch_piaa(model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                           epoch=None, desc_suffix=""):
    model.train()
    alpha_cos = getattr(args, 'daregram_alpha_cos', 0.1)
    gamma_scale = getattr(args, 'daregram_gamma_scale', 0.1)
    T = getattr(args, 'daregram_T', 0.95)

    running_L_y = running_L_cos = running_L_scale = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [DAREGRAM{desc_suffix}]" if epoch is not None else f"Train DAREGRAM{desc_suffix}"
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

        L_cos, L_scale, _ = _daregram_losses(I_ij_src, I_ij_tgt, T=T)

        loss = L_y + alpha_cos * L_cos + gamma_scale * L_scale

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_L_cos = alpha_cos * L_cos
        weighted_L_scale = gamma_scale * L_scale

        running_L_y += L_y.item()
        running_L_cos += weighted_L_cos.item()
        running_L_scale += weighted_L_scale.item()
        total_batches += 1

        progress_bar.set_postfix({
            'L_y':     f'{L_y.item():.4f}',
            'L_cos':   f'{weighted_L_cos.item():.4f}',
            'L_scale': f'{weighted_L_scale.item():.4f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_cos / n, running_L_scale / n


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = domain_tag if domain_tag else genre

    src_loader = DataLoader(datasets_dict[genre]['train'], batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    tgt_loader = DataLoader(tgt_train_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)

    pretrained_path = pretrained_model_dict[genre]
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained NIMA model not found: {pretrained_path}")
    try:
        state = torch.load(pretrained_path)
        model.nima_dict[genre].load_state_dict(state)
        print(f"Loaded NIMA weights for {genre} from {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load NIMA weights for {genre}: {e}")

    model.freeze_backbone()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    _dg_run = experiment_name.removeprefix('DAREGRAM_')
    best_model_path = os.path.join(dirname, f'{genre_str}_DAREGRAM_{args.model_type}_{_dg_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_cos, L_scale = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
            epoch=epoch, desc_suffix=" pretrain")

        if args.is_log:
            ratio_y_da = L_y / (L_y + L_scale + L_cos) if (L_y + L_scale + L_cos) > 0 else 0.0
            ratio_cos_scale = L_cos / (L_scale + L_cos) if (L_scale + L_cos) > 0 else 0.0
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss":    L_y,
                f"{genre}/Train L_cos":   L_cos,
                f"{genre}/Train L_scale": L_scale,
                f"{genre}/Train ratio L_y/(L_y+L_scale+L_cos)": ratio_y_da,
                f"{genre}/Train ratio L_cos/(L_scale+L_cos)":   ratio_cos_scale,
            }, commit=True)

    if args.no_save_model:
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(model.state_dict(), best_model_path)

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, daregram_target_genre=None):
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"DAREGRAM finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{daregram_target_genre}' train_piaa_dataset.")
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        total_train_src = len(user_train_src)
        total_train_tgt = len(user_train_tgt)
        print(f"User {uid}: train src={total_train_src}, train tgt={total_train_tgt}")
        if total_train_src < batch_size or total_train_tgt < batch_size:
            print(f"Skipping user {uid}: need >={batch_size} per split")
            continue

        src_loader = DataLoader(user_train_src, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        tgt_loader = DataLoader(user_train_tgt, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)

        model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
        pretrained_path = pretrained_model_dict[genre]
        if pretrained_path is None or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"DAREGRAM pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            model_user.load_state_dict(state)
            print(f"Loaded DAREGRAM pretrain weights from {pretrained_path}")
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
            L_y, L_cos, L_scale = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                epoch=epoch, desc_suffix=" finetune")

            if args.is_log:
                ratio_y_da = L_y / (L_y + L_scale + L_cos) if (L_y + L_scale + L_cos) > 0 else 0.0
                ratio_cos_scale = L_cos / (L_scale + L_cos) if (L_scale + L_cos) > 0 else 0.0
                wandb.log({
                    "epoch": epoch,
                    f"{genre}/Train Loss user_{uid}":    L_y,
                    f"{genre}/Train L_cos user_{uid}":   L_cos,
                    f"{genre}/Train L_scale user_{uid}": L_scale,
                    f"{genre}/Train ratio L_y/(L_y+L_scale+L_cos) user_{uid}": ratio_y_da,
                    f"{genre}/Train ratio L_cos/(L_scale+L_cos) user_{uid}":   ratio_cos_scale,
                }, commit=True)

        torch.save(model_user.state_dict(), best_model_path)

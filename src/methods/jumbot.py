import os

import numpy as np
import ot
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


def _emd_matrix(y_s, y_t):
    cdf_s = torch.cumsum(y_s, dim=-1).unsqueeze(1)
    cdf_t = torch.cumsum(y_t, dim=-1).unsqueeze(0)
    return torch.norm(cdf_s - cdf_t, p=2, dim=-1)


def _solve_uot(C_detached, epsilon, tau):
    C = C_detached.cpu().numpy().astype(np.float64)
    cmax = float(C.max())
    if cmax > 0:
        C = C / cmax
    n_s, n_t = C.shape
    a = np.ones(n_s, dtype=np.float64) / n_s
    b = np.ones(n_t, dtype=np.float64) / n_t
    pi = ot.sinkhorn_unbalanced(
        a, b, C, reg=epsilon, reg_m=tau,
        method='sinkhorn_stabilized', numItermax=1000)
    if not np.isfinite(pi).all():
        pi = np.zeros((n_s, n_t), dtype=np.float64)
    return pi


def _ot_marginal_stats(pi):
    n_s, n_t = pi.shape
    mass = pi.sum()
    row = pi.sum(dim=1)
    col = pi.sum(dim=0)
    a = 1.0 / n_s
    b = 1.0 / n_t
    marg_dev = 0.5 * ((row - a).abs().sum() + (col - b).abs().sum())
    return mass.item(), marg_dev.item()


def _train_one_epoch(model, src_loader, tgt_loader, optimizer, scaler, device, args,
                     epoch=None, global_step=0):
    model.train()
    eta1 = getattr(args, 'jumbot_eta1', 0.1)
    eta2 = getattr(args, 'jumbot_eta2', 0.1)
    eta3 = getattr(args, 'jumbot_eta3', 1.0)
    tau = getattr(args, 'jumbot_tau', 0.5)
    epsilon = getattr(args, 'jumbot_epsilon', 0.1)

    running_L_s = running_L_feat = running_L_label = 0.0
    running_mass = running_marg_dev = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [JUMBOT]" if epoch is not None else "Train JUMBOT"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120,
                        colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src = sample_src['image'].to(device)
        hist_src   = sample_src['Aesthetic'].to(device)
        images_tgt = sample_tgt['image'].to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            logit_src, z_s, _ = model(images_src, return_feat=True)

            logit_tgt, z_t, _ = model(images_tgt, return_feat=True)
            pred_t = F.softmax(logit_tgt, dim=1)

            prob_src = F.softmax(logit_src, dim=1)
            L_s = earth_mover_distance(prob_src, hist_src).mean()

            with torch.no_grad():
                feat_dist_d  = torch.cdist(z_s.float(), z_t.float(), p=2).pow(2)
                label_cost_d = _emd_matrix(hist_src.float(), pred_t.float())
                C_d = eta1 * feat_dist_d + eta2 * label_cost_d
                pi = torch.from_numpy(
                    _solve_uot(C_d, epsilon, tau)
                ).to(dtype=z_s.dtype, device=device)
                ot_mass, ot_marg_dev = _ot_marginal_stats(pi)

            feat_dist  = torch.cdist(z_s, z_t, p=2).pow(2)
            label_cost = _emd_matrix(hist_src, pred_t)

            L_feat  = (pi * feat_dist).sum()
            L_label = (pi * label_cost).sum()

            loss = L_s + eta3 * (eta1 * L_feat + eta2 * L_label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_L_s     += L_s.item()
        running_L_feat  += L_feat.item()
        running_L_label += L_label.item()
        running_mass     += ot_mass
        running_marg_dev += ot_marg_dev
        total_batches   += 1
        global_step     += 1

        progress_bar.set_postfix({
            'L_s':     f'{L_s.item():.4f}',
            'L_feat':  f'{L_feat.item():.4f}',
            'L_label': f'{L_label.item():.4f}',
            'mass':    f'{ot_mass:.3f}',
        })

    n = max(total_batches, 1)
    return {
        'train_emd':   running_L_s     / n,
        'feat_loss':   running_L_feat  / n,
        'label_loss':  running_L_label / n,
        'ot_mass':     running_mass     / n,
        'ot_marg_dev': running_marg_dev / n,
        'global_step': global_step,
    }


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    src_train_loader, val_loader, _ = src_dataloaders

    eta1 = getattr(args, 'jumbot_eta1', 0.1)
    eta2 = getattr(args, 'jumbot_eta2', 0.1)
    eta3 = getattr(args, 'jumbot_eta3', 1.0)

    global_step  = 0
    scaler       = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            model, src_train_loader, tgt_loader, optimizer, scaler, device, args,
            epoch=epoch, global_step=global_step)
        global_step = metrics['global_step']

        align_total = eta3 * (eta1 * metrics['feat_loss'] + eta2 * metrics['label_loss'])
        total_loss = metrics['train_emd'] + align_total
        L_s_ratio = metrics['train_emd'] / total_loss if total_loss > 0 else 0.0
        L_feat_ratio = (eta3 * eta1 * metrics['feat_loss']) / align_total if align_total > 0 else 0.0

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA":        metrics['train_emd'],
                f"{args.genre}/Train Feature Loss":    metrics['feat_loss'],
                f"{args.genre}/Train Label Loss":      metrics['label_loss'],
                f"{args.genre}/Train L_s Ratio":       L_s_ratio,
                f"{args.genre}/Train L_feat Ratio":    L_feat_ratio,
                f"{args.genre}/Train OT Mass":         metrics['ot_mass'],
                f"{args.genre}/Train OT Marginal Dev": metrics['ot_marg_dev'],
            }, commit=True)

    os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
    torch.save(model.state_dict(), best_modelname)

    model.load_state_dict(torch.load(best_modelname))


def _train_one_epoch_piaa(model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                           epoch=None, global_step=0, desc_suffix=""):
    from torch.amp import autocast
    model.train()
    eta1 = getattr(args, 'jumbot_eta1', 0.1)
    eta2 = getattr(args, 'jumbot_eta2', 0.1)
    eta3 = getattr(args, 'jumbot_eta3', 1.0)
    tau = getattr(args, 'jumbot_tau', 0.5)
    epsilon = getattr(args, 'jumbot_epsilon', 0.1)

    running_L_y = running_L_feat = running_L_label = 0.0
    running_mass = running_marg_dev = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [JUMBOT{desc_suffix}]" if epoch is not None else f"Train JUMBOT{desc_suffix}"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

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
            L_y = F.mse_loss(score_src, aesthetic_src)

            score_tgt, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)

            with torch.no_grad():
                feat_dist_d  = torch.cdist(I_ij_src.float(), I_ij_tgt.float(), p=2).pow(2)
                label_cost_d = (aesthetic_src.float() - score_tgt.float().T).pow(2)
                C_d = eta1 * feat_dist_d + eta2 * label_cost_d
                pi = torch.from_numpy(
                    _solve_uot(C_d, epsilon, tau)
                ).to(dtype=I_ij_src.dtype, device=device)
                ot_mass, ot_marg_dev = _ot_marginal_stats(pi)

            feat_dist  = torch.cdist(I_ij_src, I_ij_tgt, p=2).pow(2)
            label_cost = (aesthetic_src - score_tgt.T).pow(2)

            L_feat  = (pi * feat_dist).sum()
            L_label = (pi * label_cost).sum()

            loss = L_y + eta3 * (eta1 * L_feat + eta2 * L_label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_L_y += L_y.item()
        running_L_feat += L_feat.item()
        running_L_label += L_label.item()
        running_mass += ot_mass
        running_marg_dev += ot_marg_dev
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}',
            'L_feat': f'{L_feat.item():.4f}',
            'L_label': f'{L_label.item():.4f}',
            'mass': f'{ot_mass:.3f}',
        })

    n = max(total_batches, 1)
    return (running_L_y / n, running_L_feat / n, running_L_label / n,
            running_mass / n, running_marg_dev / n, global_step)


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    import wandb

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

    global_step = 0
    _jumbot_run = experiment_name.removeprefix('JUMBOT_')
    best_model_path = os.path.join(dirname, f'{genre_str}_JUMBOT_{args.model_type}_{_jumbot_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    eta1 = getattr(args, 'jumbot_eta1', 0.1)
    eta2 = getattr(args, 'jumbot_eta2', 0.1)
    eta3 = getattr(args, 'jumbot_eta3', 1.0)

    for epoch in range(args.num_epochs):
        L_y, L_feat, L_label, ot_mass, ot_marg_dev, global_step = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
            epoch=epoch, global_step=global_step, desc_suffix=" pretrain")

        align_total = eta3 * (eta1 * L_feat + eta2 * L_label)
        total_loss = L_y + align_total
        L_y_ratio = L_y / total_loss if total_loss > 0 else 0.0
        L_feat_ratio = (eta3 * eta1 * L_feat) / align_total if align_total > 0 else 0.0

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss": L_y,
                f"{genre}/Train Feature Loss": L_feat,
                f"{genre}/Train Label Loss": L_label,
                f"{genre}/Train L_y Ratio": L_y_ratio,
                f"{genre}/Train L_feat Ratio": L_feat_ratio,
                f"{genre}/Train OT Mass": ot_mass,
                f"{genre}/Train OT Marginal Dev": ot_marg_dev,
            }, commit=True)

    if args.no_save_model:
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        torch.save(model.state_dict(), best_model_path)

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, jumbot_target_genre=None):
    import wandb

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"JUMBOT finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{jumbot_target_genre}' train_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        total_train_src = len(user_train_src)
        total_train_tgt = len(user_train_tgt)
        print(f"User {uid}: train src={total_train_src}, train tgt={total_train_tgt}")
        if total_train_src < batch_size or total_train_tgt < batch_size:
            print(f"Skipping user {uid}: need >={batch_size} samples per domain")
            continue

        src_loader = DataLoader(user_train_src, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        tgt_loader = DataLoader(user_train_tgt, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)

        model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
        pretrained_path = pretrained_model_dict[genre]
        if pretrained_path is None or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"JUMBOT pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            model_user.load_state_dict(state)
            print(f"Loaded JUMBOT pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)

        eta1 = getattr(args, 'jumbot_eta1', 0.01)
        eta2 = getattr(args, 'jumbot_eta2', 0.5)
        eta3 = getattr(args, 'jumbot_eta3', 0.1)

        global_step = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_y, L_feat, L_label, ot_mass, ot_marg_dev, global_step = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                epoch=epoch, global_step=global_step, desc_suffix=" finetune")

            align_total = eta3 * (eta1 * L_feat + eta2 * L_label)
            total_loss = L_y + align_total
            L_y_ratio = L_y / total_loss if total_loss > 0 else 0.0
            L_feat_ratio = (eta3 * eta1 * L_feat) / align_total if align_total > 0 else 0.0

            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{genre}/Train Loss user_{uid}": L_y,
                    f"{genre}/Train Feature Loss user_{uid}": L_feat,
                    f"{genre}/Train Label Loss user_{uid}": L_label,
                    f"{genre}/Train L_y Ratio user_{uid}": L_y_ratio,
                    f"{genre}/Train L_feat Ratio user_{uid}": L_feat_ratio,
                    f"{genre}/Train OT Mass user_{uid}": ot_mass,
                    f"{genre}/Train OT Marginal Dev user_{uid}": ot_marg_dev,
                }, commit=True)

        torch.save(model_user.state_dict(), best_model_path)


import os
import copy

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..train_common import (
    earth_mover_distance, GradientReversalLayer, DomainDiscriminator, get_da_lambda,
    build_piaa_model, num_bins, da_weight, load_weights, save_weights)
from ..data import collate_fn


class MultilinearMap(nn.Module):
    def __init__(self, d_f: int, d_g: int):
        super().__init__()
        self.d_f = d_f
        self.d_g = d_g
        self.out_dim = d_f * d_g

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return (f.unsqueeze(2) * g.unsqueeze(1)).flatten(start_dim=1)


def gaussian_soft_label(y_hat: torch.Tensor, sigma: float, n_bins: int = num_bins) -> torch.Tensor:
    c = torch.arange(1, n_bins + 1, device=y_hat.device, dtype=y_hat.dtype)
    logits = -(c - y_hat.view(-1, 1)) ** 2 / (2.0 * sigma * sigma)
    return torch.softmax(logits, dim=-1)


def _piaa_score_to_bin(score: torch.Tensor) -> torch.Tensor:
    """Map a PIAA score on the [0, 1] label scale to the 1..num_bins bin scale."""
    return score * (num_bins - 1) + 1.0


def setup(model, args, device):
    d_f = model.feat_dim
    d_g = num_bins
    multilinear = MultilinearMap(d_f, d_g)
    discriminator = DomainDiscriminator(multilinear.out_dim).to(device)
    grl = GradientReversalLayer()
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
    return {
        'multilinear': multilinear,
        'discriminator': discriminator,
        'grl': grl,
        'optimizer_disc': optimizer_disc,
    }


def _train_one_epoch(model, src_loader, tgt_loader, optimizer, scaler, device, args,
                     multilinear, discriminator, grl, optimizer_disc,
                     epoch=None, global_step=0, cdan_total_steps=50):
    model.train()
    discriminator.train()
    running_L_y = running_L_d = running_L_d_tgt = running_disc_acc_tgt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    lambda_ = get_da_lambda(global_step, cdan_total_steps, getattr(args, 'da_gamma', 10.0))
    desc = f"Epoch {epoch} [CDAN λ={lambda_:.3f}]" if epoch is not None else "Train CDAN"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        lambda_ = get_da_lambda(global_step, cdan_total_steps, getattr(args, 'da_gamma', 10.0))

        images_src = sample_src['image'].to(device)
        hist_src   = sample_src['Aesthetic'].to(device)
        images_tgt = sample_tgt['image'].to(device)

        optimizer.zero_grad()
        optimizer_disc.zero_grad()
        with autocast('cuda'):
            logit_src, domain_feat_src, _ = model(images_src, return_feat=True)
            prob_src = F.softmax(logit_src, dim=1)
            L_y = earth_mover_distance(prob_src, hist_src).mean()

            logit_tgt, domain_feat_tgt, _ = model(images_tgt, return_feat=True)
            prob_tgt = F.softmax(logit_tgt, dim=1)

            feat_all = torch.cat([domain_feat_src, domain_feat_tgt], dim=0)
            # g is detached, as in the official CDAN code: no adversarial gradient through the prediction.
            prob_all = torch.cat([prob_src, prob_tgt], dim=0).detach()
            h_all = multilinear(feat_all, prob_all)

            domain_labels = torch.cat([
                torch.zeros(domain_feat_src.size(0), 1),
                torch.ones(domain_feat_tgt.size(0), 1),
            ], dim=0).to(device)
            domain_logit = discriminator(grl(h_all, lambda_))
            L_d = F.binary_cross_entropy_with_logits(domain_logit, domain_labels)
            loss = L_y + da_weight(args, 1.0) * L_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(optimizer_disc)
        scaler.update()

        n_src = domain_feat_src.size(0)
        logit_tgt_d = domain_logit[n_src:]
        label_tgt = domain_labels[n_src:]
        with torch.no_grad():
            L_d_tgt = F.binary_cross_entropy_with_logits(logit_tgt_d, label_tgt).item()
            disc_acc_tgt = ((torch.sigmoid(logit_tgt_d) > 0.5).float() == label_tgt).float().mean().item()

        running_L_y += L_y.item()
        running_L_d += L_d.item()
        running_L_d_tgt += L_d_tgt
        running_disc_acc_tgt += disc_acc_tgt
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}', 'L_d': f'{L_d.item():.4f}',
            'L_d_tgt': f'{L_d_tgt:.4f}', 'acc_tgt': f'{disc_acc_tgt:.3f}',
            'λ': f'{lambda_:.3f}',
        })

    n = max(total_batches, 1)
    return {
        'train_emd': running_L_y / n,
        'domain_loss': running_L_d / n,
        'domain_loss_tgt': running_L_d_tgt / n,
        'disc_acc_tgt': running_disc_acc_tgt / n,
        'global_step': global_step,
    }


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    src_train_loader, _, _ = src_dataloaders
    multilinear = components['multilinear']
    discriminator = components['discriminator']
    grl = components['grl']
    optimizer_disc = components['optimizer_disc']

    if tgt_loader is None:
        raise ValueError("CDAN GIAA requires a target loader (use --da_method CDAN-<target>).")

    steps_per_epoch = len(src_train_loader)
    cdan_total_steps = getattr(args, 'da_schedule_epochs', 50) * steps_per_epoch

    global_step = 0
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            model, src_train_loader, tgt_loader, optimizer, scaler, device, args,
            multilinear=multilinear, discriminator=discriminator, grl=grl,
            optimizer_disc=optimizer_disc,
            epoch=epoch, global_step=global_step, cdan_total_steps=cdan_total_steps)
        global_step = metrics['global_step']
        lambda_ = get_da_lambda(global_step, cdan_total_steps, getattr(args, 'da_gamma', 10.0))

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA": metrics['train_emd'],
                f"{args.genre}/Train Domain Loss": metrics['domain_loss'],
                f"{args.genre}/Train Domain Loss (tgt)": metrics['domain_loss_tgt'],
                f"{args.genre}/Train Disc Acc (tgt)": metrics['disc_acc_tgt'],
                f"{args.genre}/CDAN lambda": lambda_,
            }, commit=True)

    os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
    save_weights(model, best_modelname)

    load_weights(model, best_modelname)


def _train_one_epoch_piaa(model, src_loader, tgt_loader, multilinear, discriminator, grl,
                          optimizer, optimizer_disc, scaler, device, args, genre,
                          sigma, epoch=None, global_step=0, cdan_total_steps=50,
                          desc_suffix=""):
    model.train()
    discriminator.train()
    running_L_y = running_L_d = running_L_d_tgt = running_disc_acc_tgt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    lambda_ = get_da_lambda(global_step, cdan_total_steps, getattr(args, 'da_gamma', 10.0))
    desc = f"Epoch {epoch} [CDAN{desc_suffix} λ={lambda_:.3f}]" if epoch is not None else f"Train CDAN{desc_suffix}"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        lambda_ = get_da_lambda(global_step, cdan_total_steps, getattr(args, 'da_gamma', 10.0))

        images_src = sample_src['image'].to(device)
        aesthetic_src = sample_src['Aesthetic'].to(device).view(-1, 1)
        pt_src = sample_src['traits'].float().to(device)
        attr_src = sample_src['QIP'].float().to(device)

        images_tgt = sample_tgt['image'].to(device)
        pt_tgt = sample_tgt['traits'].float().to(device)
        attr_tgt = sample_tgt['QIP'].float().to(device)

        optimizer.zero_grad()
        optimizer_disc.zero_grad()

        with autocast('cuda'):
            score_src, I_ij_src = model(images_src, pt_src, attr_src, genre, return_feat=True)
            L_y = F.mse_loss(score_src, aesthetic_src)

            score_tgt, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)

            # g: soft label of the predicted score on the bin scale, detached as in the official CDAN code.
            g_src = gaussian_soft_label(_piaa_score_to_bin(score_src.detach().view(-1)), sigma)
            g_tgt = gaussian_soft_label(_piaa_score_to_bin(score_tgt.detach().view(-1)), sigma)

            feat_all = torch.cat([I_ij_src, I_ij_tgt], dim=0)
            g_all = torch.cat([g_src, g_tgt], dim=0)
            h_all = multilinear(feat_all, g_all)

            domain_labels = torch.cat([
                torch.zeros(I_ij_src.size(0), 1),
                torch.ones(I_ij_tgt.size(0), 1),
            ], dim=0).to(device)
            domain_logit = discriminator(grl(h_all, lambda_))
            L_d = F.binary_cross_entropy_with_logits(domain_logit, domain_labels)
            loss = L_y + da_weight(args, 0.1) * L_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(optimizer_disc)
        scaler.update()

        n_src = I_ij_src.size(0)
        logit_tgt_d = domain_logit[n_src:]
        label_tgt = domain_labels[n_src:]
        with torch.no_grad():
            L_d_tgt = F.binary_cross_entropy_with_logits(logit_tgt_d, label_tgt).item()
            disc_acc_tgt = ((torch.sigmoid(logit_tgt_d) > 0.5).float() == label_tgt).float().mean().item()

        running_L_y += L_y.item()
        running_L_d += L_d.item()
        running_L_d_tgt += L_d_tgt
        running_disc_acc_tgt += disc_acc_tgt
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}', 'L_d': f'{L_d.item():.4f}',
            'L_d_tgt': f'{L_d_tgt:.4f}', 'acc_tgt': f'{disc_acc_tgt:.3f}',
            'λ': f'{lambda_:.3f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_d / n, running_L_d_tgt / n, running_disc_acc_tgt / n, global_step


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = domain_tag if domain_tag else genre
    sigma = float(getattr(args, 'cdan_sigma', 1.0))

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

    d_f = model.input_dim
    multilinear = MultilinearMap(d_f, num_bins)
    discriminator = DomainDiscriminator(multilinear.out_dim).to(device)
    grl = GradientReversalLayer()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)

    steps_per_epoch = len(src_loader)
    cdan_total_steps = getattr(args, 'da_schedule_epochs', 50) * steps_per_epoch

    global_step = 0
    _cdan_run = experiment_name.removeprefix('CDAN_')
    best_model_path = os.path.join(dirname, f'{genre_str}_CDAN_{args.model_type}_{_cdan_run}_pretrain.pth')
    best_state_dict = None

    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_d, L_d_tgt, disc_acc_tgt, global_step = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, multilinear, discriminator, grl,
            optimizer, optimizer_disc, scaler, device, args, genre, sigma,
            epoch=epoch, global_step=global_step, cdan_total_steps=cdan_total_steps,
            desc_suffix=" pretrain")
        lambda_ = get_da_lambda(global_step, cdan_total_steps, getattr(args, 'da_gamma', 10.0))

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss": L_y,
                f"{genre}/Train Domain Loss": L_d,
                f"{genre}/Train Domain Loss (tgt)": L_d_tgt,
                f"{genre}/Train Disc Acc (tgt)": disc_acc_tgt,
                f"{genre}/CDAN lambda": lambda_,
            }, commit=True)

    if args.no_save_model:
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        save_weights(model, best_model_path)

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, cdan_target_genre=None):
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre
    sigma = float(getattr(args, 'cdan_sigma', 1.0))

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"CDAN finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{cdan_target_genre}' train_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
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
            raise FileNotFoundError(f"CDAN pretrained model not found: {pretrained_path}")
        try:
            load_weights(model_user, pretrained_path)
            print(f"Loaded CDAN pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        d_f = model_user.input_dim
        multilinear = MultilinearMap(d_f, num_bins)
        discriminator = DomainDiscriminator(multilinear.out_dim).to(device)
        grl = GradientReversalLayer()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)
        optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)

        steps_per_epoch = len(src_loader)
        cdan_total_steps = getattr(args, 'da_schedule_epochs', 50) * steps_per_epoch

        global_step = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_y, L_d, L_d_tgt, disc_acc_tgt, global_step = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, multilinear, discriminator, grl,
                optimizer, optimizer_disc, scaler, device, args, genre, sigma,
                epoch=epoch, global_step=global_step, cdan_total_steps=cdan_total_steps,
                desc_suffix=" finetune")
            lambda_ = get_da_lambda(global_step, cdan_total_steps, getattr(args, 'da_gamma', 10.0))

            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{genre}/Train Loss user_{uid}": L_y,
                    f"{genre}/Train Domain Loss user_{uid}": L_d,
                    f"{genre}/Train Domain Loss (tgt) user_{uid}": L_d_tgt,
                    f"{genre}/Train Disc Acc (tgt) user_{uid}": disc_acc_tgt,
                    f"{genre}/CDAN lambda user_{uid}": lambda_,
                }, commit=True)

        save_weights(model_user, best_model_path)

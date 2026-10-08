"""Interaction features (I_ij) of fine-tuned PIAA models, for the feature-space case study (Additional Analysis).

Each test user of a fold is fine-tuned again exactly as in the src.sweep test run (the selected pre
checkpoint and fine configuration, seed 42; see src.swap), and the 64-dim interaction representation
I_ij (the input to attr_corr) is extracted for the user's own fine-tuning samples ('train' role) of the
source and the target domain, with the user's own attributes.

Outputs (under --out_dir, default output/r16):
    feature_map/fold{k}/{MT}/{method}_{src}2{tgt}/u{uid}.npz
        feat (n x 64), genre, sample_id, sample_file, true, pred

Usage:
    python -m src.feature_map extract --fold 4 --source art --target scenery      # Source-Only and DeepJDOT, ICI
    python -m src.feature_map figure --fold 4 --source art --target scenery --users 1 108 \
        --highlight camille-pissarro_the-louvre-winter-sunshine-morning-1900.jpg joshua-reynolds_portrait-of-a-lady.jpg
"""
import argparse
import os

import numpy as np
import torch
from torch.amp import autocast

from .data import GENRES
from .swap import Swap
from .sweep import DA_METHODS, _subset


def parse_cli():
    parser = argparse.ArgumentParser(description='Interaction features for the t-SNE case study')
    sub = parser.add_subparsers(dest='command', required=True)
    ex = sub.add_parser('extract')
    ex.add_argument('--fold', type=int, required=True)
    ex.add_argument('--source', type=str, required=True, choices=GENRES)
    ex.add_argument('--target', type=str, required=True, choices=GENRES)
    ex.add_argument('--methods', type=str, nargs='+', default=['SourceOnly', 'DJDOT'],
                    choices=['SourceOnly'] + DA_METHODS)
    ex.add_argument('--users', type=int, nargs='*', default=None, help='Default: every test user of the fold')
    ex.add_argument('--model_type', type=str, default='ICI', choices=['ICI', 'MIR'])
    ex.add_argument('--criterion', type=str, default='oracle')
    ex.add_argument('--no_feature_cache', action='store_true')
    ex.add_argument('--split_dir', type=str, default='asset/split')
    ex.add_argument('--maked_dir', type=str, default='asset/maked')
    ex.add_argument('--root_dir', type=str, default='data')
    ex.add_argument('--backbone', type=str, default='clip_vit_b16')
    ex.add_argument('--num_workers', type=int, default=0)
    ex.add_argument('--out_dir', type=str, default='output/r16')
    ex.add_argument('--models_dir', type=str, default='models_pth/r16')
    ex.add_argument('--swap_dir', type=str, default=None)
    fig = sub.add_parser('figure', help='t-SNE of I_ij: rows = users, columns = methods')
    fig.add_argument('--fold', type=int, required=True)
    fig.add_argument('--source', type=str, required=True, choices=GENRES)
    fig.add_argument('--target', type=str, required=True, choices=GENRES)
    fig.add_argument('--users', type=int, nargs='+', required=True)
    fig.add_argument('--methods', type=str, nargs='+', default=['SourceOnly', 'DJDOT'])
    fig.add_argument('--highlight', type=str, nargs='*', default=[], help='Source-domain sample_file names')
    fig.add_argument('--model_type', type=str, default='ICI', choices=['ICI', 'MIR'])
    fig.add_argument('--samples_dir', type=str, default='data/samples')
    fig.add_argument('--out_dir', type=str, default='output/r16')
    fig.add_argument('--out', type=str, default='manuscript/images/feat_map.pdf')
    return parser.parse_args()


def _features(sweep, model, dataset, head, batch_size):
    loader = sweep.loader(dataset, batch_size)
    model.eval()
    feats, preds, trues = [], [], []
    with torch.no_grad():
        for sample in loader:
            with autocast('cuda'):
                out, feat = model(sample['image'].to(sweep.device), sample['traits'].float().to(sweep.device),
                                  sample['QIP'].float().to(sweep.device), head, return_feat=True)
            feats.append(feat.float().cpu().numpy())
            preds.append(out.view(-1).float().cpu().numpy())
            trues.append(sample['Aesthetic'].view(-1).numpy())
    return np.concatenate(feats), np.concatenate(preds), np.concatenate(trues)


def extract(cli):
    cli.stage, cli.pre_metric = 'fine', 'scc'
    src, tgt = cli.source, cli.target
    for method in cli.methods:
        sw = Swap(cli)
        key, pre, hp, _, _ = sw.selected(method, src, tgt, cli.criterion)
        args = sw.args(src, method, key[2], hp)
        users = cli.users or sw.users
        name = f'{method}_{src}2{tgt}'
        out_dir = os.path.join(cli.out_dir, 'feature_map', f'fold{cli.fold}', cli.model_type, name)
        tmp_dir = os.path.join(sw.model_dir, cli.model_type, 'tmp_feature_map', name)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        train_all = sw.fine_set(src, sw.users, 'train')
        unlabeled_all = sw.fine_set(tgt, sw.users, 'unlabeled') if method != 'SourceOnly' else None
        for uid in users:
            path = os.path.join(out_dir, f'u{uid}.npz')
            if os.path.exists(path):
                continue
            print(f'[feature_map] {name} fold {cli.fold}: user {uid}')
            model = sw.train_user(key, pre, args, uid, train_all, unlabeled_all, tmp_dir)
            parts = {k: [] for k in ['feat', 'pred', 'true', 'genre', 'sample_id', 'sample_file']}
            for g in [src, tgt]:
                data = _subset(sw.fine_set(g, sw.users, 'train'), uid)
                feat, pred, true = _features(sw, model, data, src, args.batch_size)
                parts['feat'].append(feat)
                parts['pred'].append(pred)
                parts['true'].append(true)
                parts['genre'] += [g] * len(feat)
                parts['sample_id'] += data.data['sample_id'].astype(int).tolist()
                parts['sample_file'] += data.data['sample_file'].tolist()
            np.savez(path, feat=np.concatenate(parts['feat']), pred=np.concatenate(parts['pred']),
                     true=np.concatenate(parts['true']), genre=np.array(parts['genre']),
                     sample_id=np.array(parts['sample_id']), sample_file=np.array(parts['sample_file']))
            del model
            sw._release()
        os.rmdir(tmp_dir)


METHOD_LABEL = {'SourceOnly': 'Source-Only', 'DJDOT': 'DeepJDOT', 'DEEPCORAL': 'DeepCORAL'}
DOMAIN_STYLE = {'art': ('#D55E00', 'o'), 'fashion': ('#009E73', '^'), 'scenery': ('#0072B2', 's')}  # Okabe-Ito
HIGHLIGHT_COLOUR = '#E69F00'
CORNERS = [(0.14, 0.83), (0.86, 0.83), (0.14, 0.17), (0.86, 0.17)]


def _thumb(path, height=70):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    return np.asarray(im.resize((max(1, int(im.width * height / im.height)), height)))


def figure(cli):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from sklearn.manifold import TSNE

    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
                         'axes.linewidth': 0.6, 'pdf.fonttype': 42})
    src, tgt = cli.source, cli.target
    rows, cols = len(cli.users), len(cli.methods)
    fig, axes = plt.subplots(rows, cols, figsize=(1.75 * cols, 1.65 * rows), squeeze=False)  # IEEE single column
    for i, uid in enumerate(cli.users):
        for j, method in enumerate(cli.methods):
            ax = axes[i, j]
            d = np.load(os.path.join(cli.out_dir, 'feature_map', f'fold{cli.fold}', cli.model_type,
                                     f'{method}_{src}2{tgt}', f'u{uid}.npz'))
            xy = TSNE(n_components=2, random_state=42, perplexity=30, init='pca').fit_transform(d['feat'])
            for g in [src, tgt]:
                colour, marker = DOMAIN_STYLE[g]
                m = d['genre'] == g
                ax.scatter(xy[m, 0], xy[m, 1], s=3, c=colour, marker=marker, alpha=0.6, linewidths=0,
                           label=f'{g} (n={m.sum()})', zorder=2)
            for g in [src, tgt]:
                colour, _ = DOMAIN_STYLE[g]
                c = xy[d['genre'] == g].mean(0)
                ax.scatter(*c, s=60, marker='*', c=colour, edgecolors='black', linewidths=0.4, zorder=4)
            used = set()
            for name in cli.highlight:
                hit = np.where(d['sample_file'] == name)[0]
                if not len(hit):
                    continue
                p = xy[hit[0]]
                ax.scatter(*p, s=22, facecolors='none', edgecolors=HIGHLIGHT_COLOUR, linewidths=1.0, zorder=5)
                # Thumbnail in the free corner farthest from the point's cloud of neighbours.
                lim = (xy.min(0), xy.max(0))
                frac = (p - lim[0]) / (lim[1] - lim[0])
                corner = min((k for k in range(len(CORNERS)) if k not in used),
                             key=lambda k: np.hypot(*(np.array(CORNERS[k]) - frac)))
                used.add(corner)
                img = OffsetImage(_thumb(os.path.join(cli.samples_dir, src, name), height=120), zoom=0.16)
                ab = AnnotationBbox(img, p, xybox=CORNERS[corner], boxcoords='axes fraction', pad=0.08,
                                    bboxprops={'edgecolor': HIGHLIGHT_COLOUR, 'linewidth': 0.7},
                                    arrowprops={'arrowstyle': '-', 'color': '#55554f', 'linewidth': 0.4}, zorder=6)
                ax.add_artist(ab)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            pad = 0.08 * (xy.max(0) - xy.min(0))
            ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
            ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
            if i == 0:
                ax.set_title(METHOD_LABEL.get(method, method), fontsize=8.5, fontweight='bold', pad=3)
            if j == 0:
                ax.set_ylabel(f'User {uid}', fontsize=8.5, fontweight='bold', labelpad=2)
    # Proxy artists, so the legend can be opaque without touching the plotted points.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle='none', marker=DOMAIN_STYLE[g][1], color=DOMAIN_STYLE[g][0], markersize=2.2)
               for g in [src, tgt]]
    labels = [g.capitalize() for g in [src, tgt]]
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               markerscale=3.2, fontsize=9, handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.3, w_pad=0.3)
    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    fig.savefig(cli.out, bbox_inches='tight')
    fig.savefig(os.path.splitext(cli.out)[0] + '.png', dpi=400, bbox_inches='tight')
    print(f'-> {cli.out}')


if __name__ == '__main__':
    cli = parse_cli()
    figure(cli) if cli.command == 'figure' else extract(cli)

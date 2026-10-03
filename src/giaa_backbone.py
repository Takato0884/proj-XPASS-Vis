"""GIAA backbone comparison on the group split (appendix table tab:comp_backbone).

For each backbone, fold and domain, NIMA is tuned exactly as the GIAA stage of src.sweep
(Source-Only, frozen backbone with cached features, fixed epochs, random search, selected by
EMD on the val users' images of that domain) and the selected model is scored on the test
users' images: EMD between the predicted distribution and the test users' score histogram,
and SCC between the predicted and the test users' mean score over the images.
The trials are src.sweep's own records, so clip_vit_b16 reuses the main sweep's GIAA stage;
the other backbones run under <out_dir>_backbone/<backbone>.

The zero-shot LLMs (reports/exp/{model}/{genre}_giaa_results.json, one predicted distribution
per image) are scored on the same test images and histograms.

Usage:
    python -m src.giaa_backbone run --backbone resnet50 --fold 0      # trials + test, one fold
    python -m src.giaa_backbone table                                 # aggregate -> JSON
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .data import GENRES
from .train_common import NIMA, earth_mover_distance, load_weights, num_bins

BACKBONES = ['resnet50', 'vit_b_16', 'clip_rn50', 'clip_vit_b16']
MAIN_BACKBONE = 'clip_vit_b16'  # the main sweep's backbone; its GIAA trials are reused
LLMS = {'gpt': 'GPT 5.4', 'claude': 'Claude Opus 4.6', 'gemini': 'Gemini 3.0 Flash'}


def dirs(backbone, out_dir='output/r16', models_dir='models_pth/r16'):
    if backbone == MAIN_BACKBONE:
        return out_dir, models_dir
    return f'{out_dir}_backbone/{backbone}', f'{models_dir}_backbone/{backbone}'


def _scores(pred_dist, true_dist):
    """EMD (mean over images, as in training) and SCC of the mean scores."""
    scale = torch.arange(num_bins, dtype=torch.float32)
    emd = float(earth_mover_distance(pred_dist, true_dist).mean())
    scc = float(spearmanr((pred_dist * scale).sum(1).numpy(), (true_dist * scale).sum(1).numpy())[0])
    return emd, scc


def _test_set(sweep, genre):
    """The test users' images of `genre`: sample_file, histogram, cached feature."""
    dataset = sweep.data(genre).giaa('test')
    items = [dataset[i] for i in range(len(dataset))]
    return ([it['sample_file'] for it in items], torch.stack([it['Aesthetic'] for it in items]),
            torch.stack([it['image'] for it in items]))


def run(cli):
    from . import sweep as sw

    out_dir, models_dir = dirs(cli.backbone)
    argv = ['--fold', str(cli.fold), '--method', 'SourceOnly', '--source', GENRES[0], '--n_trials', str(cli.n_trials),
            '--backbone', cli.backbone, '--out_dir', out_dir, '--models_dir', models_dir]
    import sys
    sys.argv = ['sweep'] + argv
    sweep = sw.Sweep(sw.parse_cli())
    result_dir = os.path.join(out_dir, 'results', f'fold{cli.fold}', 'giaa_test')
    for genre in GENRES:
        path = os.path.join(result_dir, f'SourceOnly_{genre}.json')
        if os.path.exists(path):
            continue
        records = sweep.giaa_trials(('SourceOnly', genre, None))
        best = sw._best(records, 'val_loss', genre, minimize=True)
        files, hist, feats = _test_set(sweep, genre)
        model = NIMA(num_bins, backbone=cli.backbone, dropout=0.0)
        load_weights(model, best['ckpt'], map_location='cpu')
        model.eval()
        with torch.no_grad():
            pred = F.softmax(model(feats).float(), dim=1)
        emd, scc = _scores(pred, hist)
        sw._write_json(path, {'fold': cli.fold, 'backbone': cli.backbone, 'genre': genre,
                              'selected': {'trial': best['trial'], 'hparams': best['hparams'],
                                           'val_emd': best['val_loss'][genre]},
                              'n_images': len(files), 'test_emd': emd, 'test_scc': scc,
                              'device': sw._device()})
        print(f'{cli.backbone} fold{cli.fold} {genre}: t{best["trial"]:03d} test EMD {emd:.4f} SCC {scc:.4f}')


def _llm_scores(fold, genre, model_name, cli):
    """LLM test EMD/SCC on fold's test images, using the histograms written by `run` (any backbone)."""
    from . import sweep as sw
    import sys
    sys.argv = ['sweep', '--fold', str(fold), '--method', 'SourceOnly', '--source', genre]
    sweep = sw.Sweep(sw.parse_cli())
    files, hist, _ = _test_set(sweep, genre)
    with open(os.path.join(cli.llm_dir, model_name, f'{genre}_giaa_results.json')) as f:
        pred = {e['sample_file']: e['pred_dist'] for e in json.load(f)['per_sample']}
    key = (lambda s: s.replace('.mp4', '.jpg')) if genre == 'scenery' else (lambda s: s)
    missing = [s for s in files if key(s) not in pred]
    if missing:
        raise ValueError(f'{model_name}/{genre} fold{fold}: {len(missing)} test images have no prediction')
    dist = torch.tensor([pred[key(s)] for s in files], dtype=torch.float32)
    dist = dist / dist.sum(1, keepdim=True)
    return _scores(dist, hist)


def table(cli):
    rows = {}
    for name, label in LLMS.items():
        rows[label] = {g: [_llm_scores(f, g, name, cli) for f in range(5)] for g in GENRES}
    for backbone in BACKBONES:
        out_dir, _ = dirs(backbone)
        per = {}
        for g in GENRES:
            vals = []
            for f in range(5):
                path = os.path.join(out_dir, 'results', f'fold{f}', 'giaa_test', f'SourceOnly_{g}.json')
                if os.path.exists(path):
                    with open(path) as fh:
                        r = json.load(fh)
                    vals.append((r['test_emd'], r['test_scc']))
            per[g] = vals
        rows[backbone] = per
    out = {}
    for label, per in rows.items():
        out[label] = {}
        for g, vals in per.items():
            if len(vals) < 5:
                out[label][g] = None
                print(f'{label:16s} {g:8s} {len(vals)}/5 folds')
                continue
            a = np.array(vals)
            out[label][g] = {'emd_mean': a[:, 0].mean(), 'emd_std': a[:, 0].std(),
                             'scc_mean': a[:, 1].mean(), 'scc_std': a[:, 1].std()}
            print(f'{label:16s} {g:8s} EMD {a[:, 0].mean():.3f}±{a[:, 0].std():.3f} '
                  f'SCC {a[:, 1].mean():.3f}±{a[:, 1].std():.3f}')
    with open(cli.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'-> {cli.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('--backbone', required=True, choices=BACKBONES)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--n_trials', type=int, default=20)
    p = sub.add_parser('table')
    p.add_argument('--llm_dir', default='reports/exp')
    p.add_argument('--out', default='output/r16/giaa_backbone.json')
    cli = parser.parse_args()
    run(cli) if cli.cmd == 'run' else table(cli)


if __name__ == '__main__':
    main()

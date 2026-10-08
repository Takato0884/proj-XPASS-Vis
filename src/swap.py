"""R2-1: user-identity swap control for the cross-domain PIAA results.

Does the gain come from transferring each person's preference, or from features
that any user's model would share? For one fold, method, direction and
criterion, every test user is fine-tuned again exactly as in the src.sweep test
run (the selected pre checkpoint and fine configuration, seed 42 before each
user). Each user's model, fed its own user's traits, then predicts the eval
samples of every test user of the fold, in the source and the target domain:

    native  user B's model on B's eval samples        (diagonal; reproduces the sweep's per-user SCC)
    swap    user A's model (A's traits) on B's eval samples, scored against B's ratings

No test image reaches training in any stage (adaptation uses train-group images
only), so the swap needs no further training. The full owner x rated matrix is
saved, so the swap partners can be chosen afterwards:

    set   test users of B's rating session (same stimuli pool; session effects held fixed)
    fold  all test users of B's fold (the two test sessions)

Outputs (under --swap_dir, default <out_dir>/swap = output/r16/swap):
    fold{k}/{MT}/{method}_{src}2{tgt}_{criterion}.json
        users, set of each user, per-sample predictions of every owner's model,
        scc / ccc [genre][owner][rated], and the diagonal check against the sweep
    summary_{MT}_{criterion}.json   (summary command)

The run command writes a .partial file after each owner and resumes from it.

Usage:
    python -m src.swap run --fold 0 --method DJDOT --source art --target fashion --model_type ICI
    python -m src.swap summary --model_type ICI                       # DeepJDOT
    python -m src.swap run ... --stage pre                            # swap traits on the pre model (no fine-tuning)
    python -m src.swap figure [--partner fold] [--all_directions]     # after both summaries
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .data import GENRES
from .evaluate import evaluate_piaa
from .sweep import (CRITERIA, DA_METHODS, SEED, Sweep, _Tee, _device, _key_name, _method_module, _pre_id,
                    _read_json, _seed_everything, _subset, _write_json)
from .train_common import build_piaa_model, load_weights, num_bins

PARTNERS = ['set', 'fold']
STAGES = ['fine', 'pre']


def parse_cli():
    parser = argparse.ArgumentParser(description='R2-1 user-identity swap control')
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='Re-fine-tune the test users of one fold and score every model on every user')
    run.add_argument('--fold', type=int, required=True)
    run.add_argument('--method', type=str, required=True, choices=['SourceOnly'] + DA_METHODS)
    run.add_argument('--source', type=str, required=True, choices=GENRES)
    run.add_argument('--target', type=str, required=True, choices=GENRES)
    run.add_argument('--model_type', type=str, default='ICI', choices=['ICI', 'MIR'])
    run.add_argument('--criterion', type=str, default='oracle', choices=CRITERIA)
    run.add_argument('--stage', type=str, default='fine', choices=STAGES,
                     help='fine: per-user fine-tuned models; pre: the selected pre model, only the traits swapped')
    run.add_argument('--no_feature_cache', action='store_true')
    run.add_argument('--split_dir', type=str, default='asset/split')
    run.add_argument('--maked_dir', type=str, default='asset/maked')
    run.add_argument('--root_dir', type=str, default='data')
    run.add_argument('--backbone', type=str, default='clip_vit_b16')
    run.add_argument('--num_workers', type=int, default=0)
    run.add_argument('--out_dir', type=str, default='output/r16')
    run.add_argument('--models_dir', type=str, default='models_pth/r16')
    run.add_argument('--swap_dir', type=str, default=None, help='Swap outputs and logs (default: <out_dir>/swap)')

    summary = sub.add_parser('summary', help='Native vs swap per direction, both partner sets')
    summary.add_argument('--model_type', type=str, default='ICI', choices=['ICI', 'MIR'])
    summary.add_argument('--methods', type=str, nargs='+', default=['DJDOT'])
    summary.add_argument('--criterion', type=str, default='oracle', choices=CRITERIA)
    summary.add_argument('--stage', type=str, default='fine', choices=STAGES)
    summary.add_argument('--baseline', type=str, default='SourceOnly',
                         help='Difference-in-differences against this method, when it is among --methods')
    summary.add_argument('--out_dir', type=str, default='output/r16')
    summary.add_argument('--swap_dir', type=str, default=None)

    figure = sub.add_parser('figure', help='Paired bars of own vs swapped model per model and metric (summary JSONs)')
    figure.add_argument('--partner', type=str, default='set', choices=PARTNERS)
    figure.add_argument('--method', type=str, default='DJDOT')
    figure.add_argument('--criterion', type=str, default='oracle', choices=CRITERIA)
    figure.add_argument('--stage', type=str, default='fine', choices=STAGES)
    figure.add_argument('--alpha', type=float, default=0.01, help='Mark pairs with p below this (Holm; Avg. raw)')
    figure.add_argument('--all_directions', action='store_true',
                        help='2x2 panels with the six directions and Avg. (default: Avg. only, one panel per metric)')
    figure.add_argument('--out', type=str, default=None,
                        help='Default manuscript/images/swap_control.pdf (swap_control_pre.pdf for --stage pre)')
    figure.add_argument('--out_dir', type=str, default='output/r16')
    figure.add_argument('--swap_dir', type=str, default=None)

    cli = parser.parse_args()
    if cli.command == 'figure' and cli.out is None:
        cli.out = 'manuscript/images/swap_control' + ('' if cli.stage == 'fine' else f'_{cli.stage}') + '.pdf'
    if cli.command == 'run' and cli.source == cli.target:
        parser.error('--source and --target must differ')
    return cli


def _name(method, src, tgt, criterion, stage='fine'):
    return f'{method}_{src}2{tgt}_{criterion}' + ('' if stage == 'fine' else f'_{stage}')


def _swap_dir(cli):
    return cli.swap_dir or os.path.join(cli.out_dir, 'swap')


def _swap_path(cli, fold, model_type, name):
    return os.path.join(_swap_dir(cli), f'fold{fold}', model_type, f'{name}.json')


def _per_user(pred, true, user_ids, users):
    """Per-user SCC and CCC, computed as in evaluate_piaa; an undefined SCC is None."""
    user_ids = np.asarray(user_ids)
    scc, ccc = {}, {}
    for uid in users:
        mask = user_ids == uid
        p, t = pred[mask], true[mask]
        s = spearmanr(p, t)[0]
        scc[str(uid)] = None if np.isnan(s) else float(s)
        cov = ((p - p.mean()) * (t - t.mean())).mean()
        ccc[str(uid)] = float(2 * cov / (p.var() + t.var() + (p.mean() - t.mean()) ** 2 + 1e-8))
    return scc, ccc


class Swap(Sweep):
    def __init__(self, cli):
        super().__init__(cli)
        self.users = self.split['test_users']

    def selected(self, method, src, tgt, criterion):
        """The sweep's selected pre record, fine hyper-parameters and final record for this chain."""
        final_path = os.path.join(self.report_dir, self.cli.model_type, 'final',
                                  f'{_name(method, src, tgt, criterion)}.json')
        final = _read_json(final_path)
        if final is None:
            raise FileNotFoundError(f'{final_path}: run src.sweep for this chain first')
        sel = final['selected']
        key = ('SourceOnly', src, None) if method == 'SourceOnly' else (method, src, tgt)
        giaa_id = f"{sel['giaa']['key']}-t{sel['giaa']['trial']:03d}"
        pre_path = os.path.join(self.report_dir, self.cli.model_type, 'pre', _key_name(key), giaa_id,
                                f"t{sel['pre']['trial']:03d}.json")
        pre = _read_json(pre_path)
        if pre is None or not os.path.exists(pre['ckpt']):
            raise FileNotFoundError(f'selected pre record or checkpoint missing: {pre_path}')
        test_path = os.path.join(self.report_dir, self.cli.model_type, 'test', _key_name(key), _pre_id(pre),
                                 f"fine-t{sel['fine']['trial']:03d}.json")
        return key, pre, sel['fine']['hparams'], final, _read_json(test_path)

    def owner_eval_set(self, genre, owner):
        """Every test user's eval samples of `genre`, with the user columns (traits) replaced by `owner`'s.

        Labels and user_id stay those of the rated user. piaa() gives the new dataset its own memo, so
        items cached under (user_id, sample_file) with the rated user's traits are never reused.
        """
        data = self.data(genre)
        rows = self.fine_set(genre, self.users, 'eval').data.copy()
        person = data._rows([owner]).iloc[0]
        for col in data.user_columns:
            if col != 'user_id':
                rows[col] = person[col]
        return data.piaa(rows, is_train=False)

    def train_user(self, key, pre, args, uid, train_all, unlabeled_all, tmp_dir):
        """User `uid`'s fine-tuned model, trained as in Sweep._fine_users."""
        method, src, tgt = key
        num_attr, num_pt = self.dims(src)
        backbone_dict = {src: args.backbone}
        mod = _method_module(method)
        _seed_everything(SEED)
        train = _subset(train_all, uid)
        datasets_dict = {src: {'train': train, 'val': train, 'test': None}}
        exp = f'u{uid}'
        if method == 'SourceOnly':
            mod.trainer_finetune(datasets_dict, args, self.device, tmp_dir, exp, backbone_dict,
                                 {src: pre['ckpt']}, num_attr, num_pt)
        else:
            unlabeled = _subset(unlabeled_all, uid)
            mod.trainer_finetune(datasets_dict, unlabeled, unlabeled, args, self.device, tmp_dir, exp,
                                 backbone_dict, {src: pre['ckpt']}, num_attr, num_pt,
                                 **{f'{method.lower()}_target_genre': tgt})
        ckpt = os.path.join(tmp_dir, f'{src}_{args.model_type}_user_{uid}_{exp}_finetune.pth')
        model = build_piaa_model(num_bins, num_attr, num_pt, [src], backbone_dict, args).to(self.device)
        load_weights(model, ckpt)
        os.remove(ckpt)
        return model

    def run(self):
        cli = self.cli
        name = _name(cli.method, cli.source, cli.target, cli.criterion, cli.stage)
        path = _swap_path(cli, cli.fold, cli.model_type, name)
        if os.path.exists(path):
            print(f'{path} exists')
            return
        key, pre, hp, final, test = self.selected(cli.method, cli.source, cli.target, cli.criterion)
        method, src, tgt = key[0], cli.source, cli.target
        genres = [src, tgt]
        args = self.args(src, method, key[2], hp)
        sets = pd.read_csv(os.path.join(cli.maked_dir, 'ratings.csv'), usecols=['user_id', 'set']) \
            .drop_duplicates().set_index('user_id')['set']

        partial_path = path + '.partial'
        result = _read_json(partial_path) or {
            'fold': cli.fold, 'model_type': cli.model_type, 'method': method, 'source': src, 'target': tgt,
            'criterion': cli.criterion, 'stage': cli.stage, 'seed': SEED, 'pre_ckpt': pre['ckpt'], 'fine_hparams': hp,
            'users': self.users, 'set': {str(u): int(sets[u]) for u in self.users},
            'rows': {}, 'pred': {g: {} for g in genres}, 'device': _device(),
            'sweep_test_device': (test or {}).get('device'),
        }
        for g in genres:
            if g not in result['rows']:
                rows = [_subset(self.fine_set(g, self.users, 'eval'), u).data for u in self.users]
                result['rows'][g] = {'user_id': [int(u) for r in rows for u in r['user_id']],
                                     'sample_id': [int(i) for r in rows for i in r['sample_id']]}

        if cli.stage == 'pre':
            # One model for everyone: the selected pre checkpoint, as scored in the sweep's pre stage.
            num_attr, num_pt = self.dims(src)
            pre_model = build_piaa_model(num_bins, num_attr, num_pt, [src], {src: args.backbone}, args).to(self.device)
            load_weights(pre_model, pre['ckpt'])
        else:
            tmp_dir = os.path.join(self.model_dir, cli.model_type, 'tmp_swap', name)
            os.makedirs(tmp_dir, exist_ok=True)
            train_all = self.fine_set(src, self.users, 'train')
            unlabeled_all = self.fine_set(tgt, self.users, 'unlabeled') if method != 'SourceOnly' else None
        for i, owner in enumerate(self.users):
            if str(owner) in result['pred'][tgt]:
                continue
            print(f'\n[swap] {name} fold {cli.fold}: owner {owner} ({i + 1}/{len(self.users)})')
            model = pre_model if cli.stage == 'pre' else \
                self.train_user(key, pre, args, owner, train_all, unlabeled_all, tmp_dir)
            for g in genres:
                # One loader per rated user, as in the sweep's test run: under autocast the predictions
                # depend slightly on the batch composition, and this keeps the diagonal identical.
                eval_all = self.owner_eval_set(g, owner)
                user_ids, pred, true = [], [], []
                for rated in self.users:
                    loader = self.loader(_subset(eval_all, rated), args.batch_size)
                    evaluate_piaa(model, {src: loader}, self.device, phase_name=f'swap u{owner} on u{rated} [{g}]')
                    out = model._eval_predictions[src]
                    user_ids += [int(u) for u in out['user_id']]
                    pred += [float(p) for p in out['pred']]
                    true += [float(t) for t in out['true']]
                if user_ids != result['rows'][g]['user_id']:
                    raise RuntimeError('prediction order differs from the eval rows')
                result['pred'][g][str(owner)] = pred
                result['rows'][g].setdefault('true', true)
            if cli.stage == 'fine':
                del model
                self._release()
            _write_json(partial_path, result)
        if cli.stage == 'fine':
            os.rmdir(tmp_dir)

        result['scc'], result['ccc'] = {}, {}
        for g in genres:
            true = np.asarray(result['rows'][g]['true'])
            result['scc'][g], result['ccc'][g] = {}, {}
            for owner in self.users:
                pred = np.asarray(result['pred'][g][str(owner)])
                result['scc'][g][str(owner)], result['ccc'][g][str(owner)] = \
                    _per_user(pred, true, result['rows'][g]['user_id'], self.users)
        # The sweep records no pre-stage scores of test users, so only fine runs can be checked.
        result['diag_check'] = _diag_check(result, final, genres) if cli.stage == 'fine' else None
        _write_json(path, result)
        os.remove(partial_path)
        check = (f"diagonal vs sweep max |dSCC| = {result['diag_check']['max_abs_diff_scc']:.2e}"
                 if result['diag_check'] else 'pre stage (no sweep record to check)')
        print(f"{name} fold {cli.fold}: {check} -> {path}")


def _diag_check(result, final, genres):
    """How closely the native (diagonal) per-user SCC and CCC reproduce the sweep's test run."""
    d_scc, d_ccc = [], []
    for g in genres:
        for u in result['users']:
            new, old = result['scc'][g][str(u)][str(u)], final['per_user'][str(u)][g]
            d_scc.append(abs((new or 0.0) - (old or 0.0)))
            d_ccc.append(abs(result['ccc'][g][str(u)][str(u)] - final['per_user_ccc'][str(u)][g]))
    return {'max_abs_diff_scc': float(max(d_scc)), 'max_abs_diff_ccc': float(max(d_ccc)),
            'n_diff_scc_over_1e-4': int(sum(d > 1e-4 for d in d_scc))}


def _start_log(cli):
    name = _name(cli.method, cli.source, cli.target, cli.criterion, cli.stage)
    path = os.path.join(_swap_dir(cli), 'logs', f'fold{cli.fold}',
                        f"{name}_{cli.model_type}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = open(path, 'a', buffering=1)
    log.write(f"$ python -m src.swap {' '.join(sys.argv[1:])}\n")
    sys.stdout = _Tee(sys.stdout, log, skip_redraws=False)
    sys.stderr = _Tee(sys.stderr, log, skip_redraws=True)
    device = _device()
    print(f"Log: {path}\nDevice: {device['host']} / {device['gpu']}")


# ---------- summary ----------

def _holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def _wilcoxon(x):
    x = np.asarray(x)
    return float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0


def _gaps(record, metric, genre, partner):
    """Per rated user: native score, mean swap score over the partners, and native - swap.

    An undefined SCC counts as 0, as in the sweep's means.
    """
    m = record[metric][genre]
    users, sets = record['users'], record['set']
    val = lambda owner, rated: m[str(owner)][str(rated)] or 0.0
    out = {}
    for b in users:
        partners = [a for a in users if a != b and (partner == 'fold' or sets[str(a)] == sets[str(b)])]
        native, swap = val(b, b), float(np.mean([val(a, b) for a in partners]))
        out[b] = (native, swap, native - swap)
    return out


def summary(cli):
    directions = [(s, t) for s in GENRES for t in GENRES if s != t]
    gaps, missing = {}, []
    for method in cli.methods:
        for src, tgt in directions:
            name = _name(method, src, tgt, cli.criterion, cli.stage)
            records = [_read_json(_swap_path(cli, f, cli.model_type, name)) for f in range(5)]
            if any(r is None for r in records):
                missing.append(f'{name} (folds {[f for f, r in enumerate(records) if r is None]})')
                continue
            for metric in ['scc', 'ccc']:
                for partner in PARTNERS:
                    merged = {}
                    for r in records:
                        merged.update(_gaps(r, metric, tgt, partner))
                    gaps[method, f'{src}2{tgt}', metric, partner] = merged

    rows = []
    for method in cli.methods:
        for metric in ['scc', 'ccc']:
            for partner in PARTNERS:
                cols = [f'{s}2{t}' for s, t in directions if (method, f'{s}2{t}', metric, partner) in gaps]
                if not cols:
                    continue
                per_col = {c: gaps[method, c, metric, partner] for c in cols}
                if len(cols) == len(directions):  # Avg. = per-user mean over the six directions
                    users = sorted(per_col[cols[0]])
                    per_col['avg'] = {u: tuple(np.mean([per_col[c][u][k] for c in cols]) for k in range(3))
                                      for u in users}
                base = {c: gaps.get((cli.baseline, c, metric, partner)) for c in per_col}
                if cli.baseline in cli.methods and 'avg' in per_col and all(base[c] for c in cols):
                    base['avg'] = {u: tuple(np.mean([base[c][u][k] for c in cols]) for k in range(3))
                                   for u in per_col['avg']}
                entries = []
                for c, g in per_col.items():
                    users = sorted(g)
                    diff = [g[u][2] for u in users]
                    entry = {'method': method, 'direction': c, 'metric': metric, 'partner': partner,
                             'n_users': len(users),
                             'native': float(np.mean([g[u][0] for u in users])),
                             'swap': float(np.mean([g[u][1] for u in users])),
                             'gap': float(np.mean(diff)), 'p': _wilcoxon(diff)}
                    if method != cli.baseline and base.get(c):
                        did = [g[u][2] - base[c][u][2] for u in users]
                        entry.update(did=float(np.mean(did)), p_did=_wilcoxon(did))
                    entries.append(entry)
                # Holm within (method, metric, partner) over the six directions; Avg. is its own family.
                dirs = [e for e in entries if e['direction'] != 'avg']
                for field in ['p', 'p_did']:
                    if all(field in e for e in dirs):
                        for e, p in zip(dirs, _holm([e[field] for e in dirs])):
                            e[f'{field}_holm'] = p
                rows += entries

    suffix = '' if cli.stage == 'fine' else f'_{cli.stage}'
    out_path = os.path.join(_swap_dir(cli), f'summary_{cli.model_type}_{cli.criterion}{suffix}.json')
    _write_json(out_path, {'model_type': cli.model_type, 'criterion': cli.criterion, 'stage': cli.stage,
                           'baseline': cli.baseline,
                           'missing': missing, 'rows': rows})
    print(f"{'method':<11}{'dir':<16}{'metric':<7}{'partner':<8}{'n':>4}{'native':>8}{'swap':>8}{'gap':>8}"
          f"{'p_holm':>9}{'DiD':>8}{'p_holm':>9}")
    # Avg. is its own family, so its raw p is shown in the Holm columns.
    fmt = lambda e, k: f"{e.get(k, e.get(k.removesuffix('_holm'))):.3g}" if k in e or k.removesuffix('_holm') in e else '-'
    for e in rows:
        did = f"{e['did']:+.3f}" if 'did' in e else '-'
        print(f"{e['method']:<11}{e['direction']:<16}{e['metric']:<7}{e['partner']:<8}{e['n_users']:>4}"
              f"{e['native']:>8.3f}{e['swap']:>8.3f}{e['gap']:>+8.3f}{fmt(e, 'p_holm'):>9}{did:>8}"
              f"{fmt(e, 'p_did_holm'):>9}")
    if missing:
        print('missing: ' + '; '.join(missing))
    print(f'-> {out_path}')


# ---------- figure ----------

DIR_LABEL = {'art': 'A', 'fashion': 'F', 'scenery': 'S'}
OWN_COLOUR, SWAP_COLOUR = '#0072B2', '#E69F00'  # Okabe-Ito blue and orange
STAR_COLOUR = '#D00000'


def figure(cli):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'pdf.fonttype': 42})
    directions = [f'{s}2{t}' for s in GENRES for t in GENRES if s != t] + ['avg']
    labels = [f"{DIR_LABEL[d.split('2')[0]]}$\\rightarrow${DIR_LABEL[d.split('2')[1]]}" if d != 'avg' else 'Avg.'
              for d in directions]
    model_types = ['ICI', 'MIR']
    metrics = [('scc', 'SCC'), ('ccc', 'CCC')]
    rows = {}
    for mt in model_types:
        suffix = '' if cli.stage == 'fine' else f'_{cli.stage}'
        summary = _read_json(os.path.join(_swap_dir(cli), f'summary_{mt}_{cli.criterion}{suffix}.json'))
        if summary is None:
            raise FileNotFoundError(f'run `python -m src.swap summary --model_type {mt} --stage {cli.stage}` first')
        for e in summary['rows']:
            if e['method'] == cli.method and e['partner'] == cli.partner:
                rows[mt, e['metric'], e['direction']] = e

    if not cli.all_directions:
        return _figure_avg(cli, plt, rows, model_types, metrics)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.7), sharex=True)
    width = 0.38
    x = np.arange(len(directions), dtype=float)
    x[-1] += 0.5  # set Avg. apart
    for j, (metric, metric_label) in enumerate(metrics):
        top = max(max(rows[mt, metric, d]['native'], rows[mt, metric, d]['swap'])
                  for mt in model_types for d in directions)
        for i, mt in enumerate(model_types):
            ax = axes[i, j]
            own = [rows[mt, metric, d]['native'] for d in directions]
            swap = [rows[mt, metric, d]['swap'] for d in directions]
            ax.bar(x - width / 2, own, width, color=OWN_COLOUR, edgecolor='white', linewidth=0.8,
                   label='Personalized model', zorder=2)
            ax.bar(x + width / 2, swap, width, color=SWAP_COLOUR, edgecolor='white', linewidth=0.8,
                   label='Swapped model', zorder=2)
            for k, d in enumerate(directions):
                e = rows[mt, metric, d]
                p = e['p'] if d == 'avg' else e['p_holm']
                y = max(own[k], swap[k]) + top * 0.04
                leg = top * 0.025
                ax.plot([x[k] - width / 2, x[k] - width / 2, x[k] + width / 2, x[k] + width / 2],
                        [y - leg, y, y, y - leg], color='#55554f', linewidth=0.6, zorder=3)
                if p < cli.alpha:
                    ax.text(x[k], y + top * 0.005, '*', ha='center', va='bottom', fontsize=10, color=STAR_COLOUR)
            ax.axvline((x[-2] + x[-1]) / 2, color='#c8c7c0', linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
            ax.set_ylim(0, top * 1.22)
            ax.yaxis.grid(True, color='#e4e3dd', linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
            ax.set_title(f'{mt} — {metric_label}', fontsize=9, pad=3)
            ax.set_ylabel(metric_label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.tick_params(axis='x', length=0)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=1.5)
    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    fig.savefig(cli.out, bbox_inches='tight')
    fig.savefig(os.path.splitext(cli.out)[0] + '.png', dpi=200, bbox_inches='tight')
    print(f'-> {cli.out}')


def _bracket(ax, x0, x1, y, leg, star, size):
    ax.plot([x0, x0, x1, x1], [y - leg, y, y, y - leg], color='#55554f', linewidth=0.6, zorder=3)
    if star:
        ax.text((x0 + x1) / 2, y, '*', ha='center', va='bottom', fontsize=size, color=STAR_COLOUR)


def _figure_avg(cli, plt, rows, model_types, metrics):
    """Avg. over the six directions: one panel per metric, own vs swapped model per PIAA model."""
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    width = 0.36
    x = np.arange(len(model_types), dtype=float)
    top = max(max(rows[mt, m, 'avg']['native'], rows[mt, m, 'avg']['swap']) for mt in model_types for m, _ in metrics)
    for ax, (metric, metric_label) in zip(axes, metrics):
        own = [rows[mt, metric, 'avg']['native'] for mt in model_types]
        swap = [rows[mt, metric, 'avg']['swap'] for mt in model_types]
        ax.bar(x - width / 2, own, width, color=OWN_COLOUR, edgecolor='white', linewidth=0.8,
               label='Personalized model', zorder=2)
        ax.bar(x + width / 2, swap, width, color=SWAP_COLOUR, edgecolor='white', linewidth=0.8,
               label='Swapped model', zorder=2)
        for k, mt in enumerate(model_types):
            _bracket(ax, x[k] - width / 2, x[k] + width / 2, max(own[k], swap[k]) + top * 0.05, top * 0.03,
                     rows[mt, metric, 'avg']['p'] < cli.alpha, 10)
        ax.set_ylim(0, top * 1.25)
        ax.yaxis.grid(True, color='#e4e3dd', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.set_title(metric_label, fontsize=9, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(model_types)
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(-0.6, len(model_types) - 0.4)
    axes[0].set_ylabel('Score (avg. of 6 directions)')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.0)
    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    fig.savefig(cli.out, bbox_inches='tight')
    fig.savefig(os.path.splitext(cli.out)[0] + '.png', dpi=300, bbox_inches='tight')
    print(f'-> {cli.out}')


if __name__ == '__main__':
    cli = parse_cli()
    if cli.command == 'summary':
        summary(cli)
    elif cli.command == 'figure':
        figure(cli)
    else:
        _start_log(cli)
        Swap(cli).run()

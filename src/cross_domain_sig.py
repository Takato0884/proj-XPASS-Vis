"""R1-4: significance of each UDA method against Source-Only and Target-Only (Table tab:cross_domain).

Per test user (129, 5 folds pooled) and direction s->t, the method's target-domain SCC / CCC from the
sweep's final records (oracle by default) is compared with that of Source-Only (same direction) and of
Target-Only (within-domain model of t), by a two-sided Wilcoxon signed-rank test. For Avg., each user's
mean over the six directions is tested. Holm correction within each model and column (six directions
and Avg.) over the 16 comparisons of the 8 methods in SCC and CCC, separately per reference.
An undefined SCC (constant prediction) counts as 0, as in the table.

    python -m src.cross_domain_sig [--criterion oracle]
Output: output/r16/cross_domain_sig_{criterion}.json
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'output/r16/results')
G = ('art', 'fashion', 'scenery')
DIRS = [(s, t) for s in G for t in G if s != t]
METHODS = ['DANN', 'CDAN', 'ALDA', 'DJDOT', 'JUMBOT', 'DEEPCORAL', 'RSD', 'DAREGRAM']
REFS = ['SourceOnly', 'TargetOnly']
METRICS = ['scc', 'ccc']
ALPHA = .01


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


def _load(model_type, name, t):
    """user -> {'scc', 'ccc'} on target t over the 5 folds."""
    out = {}
    for k in range(5):
        rec = json.load(open(os.path.join(RESULTS, f'fold{k}', model_type, 'final', f'{name}.json')))
        for u in rec['per_user']:
            scc = rec['per_user'][u][t]
            out[u] = {'scc': 0.0 if scc is None else float(scc), 'ccc': float(rec['per_user_ccc'][u][t])}
    return out


def run(model_type, criterion):
    recs = {m: {(s, t): _load(model_type, f'{m}_{s}2{t}_{criterion}', t) for s, t in DIRS}
            for m in METHODS + ['SourceOnly']}
    recs['TargetOnly'] = {(s, t): _load(model_type, f'TargetOnly_{t}', t) for s, t in DIRS}
    users = sorted(recs['SourceOnly'][DIRS[0]], key=int)
    assert len(users) == 129, len(users)
    for name in recs:
        for d in DIRS:
            assert set(recs[name][d]) == set(users), (name, d)

    def values(name, col, metric):
        if col == 'avg':
            return np.mean([values(name, d, metric) for d in DIRS], axis=0)
        return np.array([recs[name][col][u][metric] for u in users])

    out = {}
    for ref in REFS:
        out[ref] = {}
        for col in DIRS + ['avg']:
            entries = []
            for metric in METRICS:
                for m in METHODS:
                    x, y = values(m, col, metric), values(ref, col, metric)
                    entries.append({'method': m, 'metric': metric, 'mean': float(x.mean()), 'ref': float(y.mean()),
                                    'diff': float((x - y).mean()), 'p': _wilcoxon(x - y)})
            for e, p in zip(entries, _holm([e['p'] for e in entries])):
                e['p_holm'] = p
                e['sig'] = (1 if e['diff'] > 0 else -1) if p < ALPHA else 0
            out[ref]['avg' if col == 'avg' else f'{col[0]}2{col[1]}'] = entries
    return out


def main():
    parser = argparse.ArgumentParser(description='R1-4: UDA methods vs Source-Only / Target-Only')
    parser.add_argument('--criterion', type=str, default='oracle')
    cli = parser.parse_args()
    result = {mt: run(mt, cli.criterion) for mt in ['MIR', 'ICI']}
    for mt, by_ref in result.items():
        for ref, by_col in by_ref.items():
            cols = list(by_col)
            print(f'\n=== {mt} vs {ref} (cell: SCC | CCC; +/- p_holm<{ALPHA}, else p_holm) ===')
            print(f"{'method':10s}" + ''.join(c.ljust(18) for c in cols))
            for m in METHODS:
                row = f'{m:10s}'
                for c in cols:
                    cell = []
                    for metric in METRICS:
                        e = next(e for e in by_col[c] if e['method'] == m and e['metric'] == metric)
                        cell.append({1: '+', -1: '-'}[e['sig']] if e['sig'] else f"{e['p_holm']:.2f}")
                    row += ' | '.join(cell).ljust(18)
                print(row)
    path = os.path.join(ROOT, f'output/r16/cross_domain_sig_{cli.criterion}.json')
    json.dump(result, open(path, 'w'), indent=1)
    print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()

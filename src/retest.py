"""R1-7: test-retest agreement of the overall aesthetic score (Tables tab:dataset_stats, tab:within_domain,
tab:cross_domain).

Within each domain, 12 stimuli were shown twice to every annotator. Per user and domain, the first and
second ratings of these stimuli are compared by SCC, CCC, Pearson r and MAE, and each statistic is
averaged over the 129 users (95% CI by a bootstrap over users). The first rating is the one used for
training and evaluation (src/make_split.py keeps it).

    python -m src.retest
Output: output/r16/retest.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKED = os.path.join(ROOT, 'asset/maked')
G = ('art', 'fashion', 'scenery')
STATS = ['scc', 'ccc', 'pearson', 'mae']


def _ccc(a, b):
    return 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


def per_user():
    r = pd.read_csv(os.path.join(MAKED, 'ratings.csv'))
    r['k'] = r.groupby(['user_id', 'sample_file']).cumcount()
    pairs = r.pivot_table(index=['user_id', 'genre', 'sample_file'], columns='k', values='Aesthetic').dropna()
    rows = []
    for (u, g), d in pairs.reset_index().groupby(['user_id', 'genre']):
        a, b = d[0].to_numpy(float), d[1].to_numpy(float)
        assert a.std() > 0 and b.std() > 0, (u, g)  # no constant raters among the repeated stimuli
        rows.append({'user_id': u, 'genre': g, 'n': len(d), 'scc': spearmanr(a, b).correlation, 'ccc': _ccc(a, b),
                     'pearson': pearsonr(a, b)[0], 'mae': float(np.abs(a - b).mean())})
    df = pd.DataFrame(rows)
    assert df.user_id.nunique() == 129
    return df


def main():
    df = per_user()
    rng = np.random.default_rng(0)
    out = {}
    for g in G:
        d = df[df.genre == g]
        out[g] = {'n_users': len(d), 'n_pairs': [int(d.n.min()), float(d.n.mean()), int(d.n.max())]}
        for s in STATS:
            x = d[s].to_numpy()
            boot = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
            out[g][s] = {'mean': float(x.mean()), 'std': float(x.std(ddof=1)), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist()}
        print(g.ljust(8) + '  '.join(f"{s} {out[g][s]['mean']:.3f}±{out[g][s]['std']:.3f}" for s in STATS))
    out['avg'] = {s: float(np.mean([out[g][s]['mean'] for g in G])) for s in STATS}
    print('avg     ' + '  '.join(f"{s} {out['avg'][s]:.3f}" for s in STATS))
    path = os.path.join(ROOT, 'output/r16/retest.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()

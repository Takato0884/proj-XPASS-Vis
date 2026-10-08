"""User-level analysis of domain adaptation (Sec. sec:who_gain, Table tab:da_factors_top5).

Per test user and direction s->t, Delta_u = SCC_u(DA, t) - SCC_u(Source-Only, t), from the sweep's
final records (oracle, 5 folds, 129 users). Delta_u is regressed (OLS) on z-standardized user features;
p-values are BH-FDR corrected per direction and the standardized betas are averaged across the six
directions. Port of proj-xpass-DA `analysis.py analyze_da_factors --keep-shift-only` (same features).

    python -m src.da_factors run [--model_type ICI] [--method DJDOT] [--criterion oracle]
    python -m src.da_factors run --check_old DIR   # refit the old per_user_features.csv files in DIR
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'output/r16/results')
MAKED = os.path.join(ROOT, 'asset/maked')
G = ('art', 'fashion', 'scenery')
DIRS = [(s, t) for s in G for t in G if s != t]
LEARN = {'art': 'art', 'fashion': 'fashion', 'scenery': 'photoVideo'}  # users.csv prefix per domain
EDU = {'high_school': 1, 'vocational': 2, 'junior_college': 3, 'technical_college': 4,
       'university': 5, 'graduate': 6, '博士': 6}


def _z(v):
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def per_user(model_type, method, criterion, s, t):
    """Per-user target/source SCC of Source-Only and of `method` over the 5 folds (None/NaN -> 0)."""
    rows = []
    for k in range(5):
        d = os.path.join(RESULTS, f'fold{k}', model_type, 'final')
        no = json.load(open(os.path.join(d, f'SourceOnly_{s}2{t}_{criterion}.json')))['per_user']
        da = json.load(open(os.path.join(d, f'{method}_{s}2{t}_{criterion}.json')))['per_user']
        assert set(no) == set(da)
        for u in no:
            rows.append({'user_id': int(u), 'fold': k,
                         'baseline_scc_target': _z(no[u][t]), 'baseline_scc_source': _z(no[u][s]),
                         'delta_target': _z(da[u][t]) - _z(no[u][t])})
    assert len(rows) == 129, len(rows)
    return pd.DataFrame(rows)


def user_features(s, t, score_col='Aesthetic'):
    users = pd.read_csv(os.path.join(MAKED, 'users.csv')).set_index('user_id')
    ratings = pd.read_csv(os.path.join(MAKED, 'ratings.csv'))
    # TIPI: Q1..Q10 are 0-based (0..6); each factor averages an item and its reversed partner
    q = {i: users[f'Q{i}'].astype(float) + 1.0 for i in range(1, 11)}
    feat = users[['age'] + [f'{LEARN[d]}_interest' for d in G]].copy()
    feat = feat.join(pd.DataFrame({
        'big5_E': (q[1] + (8 - q[6])) / 2, 'big5_A': ((8 - q[2]) + q[7]) / 2,
        'big5_C': (q[3] + (8 - q[8])) / 2, 'big5_ES': ((8 - q[4]) + q[9]) / 2,
        'big5_O': (q[5] + (8 - q[10])) / 2}))
    feat['edu_level'] = users['edu'].map(EDU)
    feat = feat.join(pd.get_dummies(users[['gender', 'nationality']], prefix=['gender', 'nationality'],
                                    drop_first=True, dtype=int))

    def style(df):
        return df.groupby('user_id')[score_col].agg(
            mean='mean', std='std', skew=lambda x: float(pd.Series(x).skew()),
            kurt=lambda x: float(pd.Series(x).kurt()))

    def retest_mae(df):
        out = {}
        for (uid, _), x in df.groupby(['user_id', 'sample_file'])[score_col]:
            if len(x) >= 2:
                out.setdefault(uid, []).append(abs(x.iloc[0] - x.iloc[1]))
        return pd.Series({u: float(np.mean(v)) for u, v in out.items()})

    def generality(df):
        # Pearson r between a user's ratings and the leave-one-out mean of the other raters
        tot = df.groupby('sample_file')[score_col].sum()
        cnt = df.groupby('sample_file')[score_col].count()
        out = {}
        for uid, sub in df.groupby('user_id'):
            if len(sub) < 5:
                continue
            x = sub[score_col].values.astype(float)
            others = (tot.loc[sub['sample_file']].values - x) / np.maximum(cnt.loc[sub['sample_file']].values - 1, 1)
            if np.std(x) and np.std(others):
                out[uid] = float(np.corrcoef(x, others)[0, 1])
        return pd.Series(out)

    rs, rt = ratings[ratings.genre == s], ratings[ratings.genre == t]
    ss, st_ = style(rs), style(rt)
    # keep-shift-only: only |target - source| of each per-domain statistic enters the regression
    for stat in ('mean', 'std', 'skew', 'kurt'):
        feat[f'shift_{stat}'] = (st_[stat] - ss[stat]).abs()
    feat['shift_retest_mae'] = (retest_mae(rt) - retest_mae(rs)).abs()
    feat['shift_generality'] = (generality(rt) - generality(rs)).abs()
    feat['shift_interest'] = (users[f'{LEARN[t]}_interest'] - users[f'{LEARN[s]}_interest']).abs()
    # the on-domain interest levels are replaced by shift_interest; the off-domain one stays (as in the port)
    return feat.drop(columns=[f'{LEARN[s]}_interest', f'{LEARN[t]}_interest'])


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(adj, 0, 1)
    return out


def ols(X, y):
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])
    inv = np.linalg.pinv(Xc.T @ Xc)
    beta = inv @ Xc.T @ y
    resid = y - Xc @ beta
    dof = n - p - 1
    se = np.sqrt(np.maximum(np.diag(resid @ resid / dof * inv), 0))
    tval = beta / se
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return beta[1:], se[1:], 2 * stats.t.sf(np.abs(tval[1:]), dof), r2, 1 - (1 - r2) * (n - 1) / dof


def vif(X):
    out = []
    for j in range(X.shape[1]):
        Xc = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        resid = X[:, j] - Xc @ (np.linalg.pinv(Xc) @ X[:, j])
        out.append(((X[:, j] - X[:, j].mean()) ** 2).sum() / (resid @ resid))
    return np.array(out)


def regress(df):
    """df: one row per user with delta_target and the features. Returns (table, summary)."""
    cols = ['baseline_scc_target', 'baseline_scc_source'] + [
        c for c in df.columns if c not in ('user_id', 'fold', 'delta_target', 'baseline_scc_target',
                                           'baseline_scc_source')]
    Xy = df[cols + ['delta_target']].apply(pd.to_numeric, errors='coerce')
    cols = [c for c in cols if Xy[c].notna().sum() >= 10 and Xy[c].dropna().nunique() >= 2]
    Xy = Xy[cols + ['delta_target']].dropna()
    X = Xy[cols].values.astype(float)
    Xz = (X - X.mean(0)) / X.std(0)  # population SD, as StandardScaler
    beta, se, p, r2, adj = ols(Xz, Xy['delta_target'].values.astype(float))
    tab = pd.DataFrame({'feature': cols, 'beta': beta, 'se': se, 'p': p, 'p_fdr': bh(p), 'vif': vif(Xz)})
    tab = tab.reindex(tab.beta.abs().sort_values(ascending=False).index).reset_index(drop=True)
    return tab, {'n': len(Xy), 'p': len(cols), 'r2': float(r2), 'adj_r2': float(adj),
                 'delta_mean': float(Xy['delta_target'].mean()),
                 'p_positive': float((Xy['delta_target'] > 0).mean())}


def canonical(f, s, t):
    """Pair-specific names -> names shared by every direction; None for features dropped in aggregation."""
    f = f.replace('srocc', 'scc').replace(f'_{s}_to_{t}', '')
    if f.startswith(('baseline_', 'shift_', 'big5_', 'gender_', 'edu_', 'nationality_')) or f == 'age':
        return f
    if f == 'photoVideo_interest':
        return 'scenery_interest'
    return None  # off-domain art/fashion interest: a different column per direction


def run(cli):
    pairs = {}
    for s, t in DIRS:
        if cli.check_old:
            old = pd.read_csv(os.path.join(cli.check_old, f'{s}2{t}_ICI_DAREGRAM_shiftonly', 'per_user_features.csv'))
            df = per_user_old(old, s, t)
        else:
            df = per_user(cli.model_type, cli.method, cli.criterion, s, t).merge(
                user_features(s, t), left_on='user_id', right_index=True, how='left')
        tab, summ = regress(df)
        tab['feature'] = [canonical(f, s, t) or f for f in tab.feature]
        pairs[f'{s}2{t}'] = {'summary': summ, 'table': tab.to_dict('records')}
        print(f'{s}->{t}: n={summ["n"]} p={summ["p"]} R2={summ["r2"]:.3f} mean Delta={summ["delta_mean"]:+.3f} '
              f'P(Delta>0)={summ["p_positive"]:.2f} maxVIF={tab.vif.max():.2f}')
        for r in tab.head(5).itertuples():
            print(f'   {r.feature:22s} {r.beta:+.3f}  p_fdr={r.p_fdr:.3f}')
    agg = {}
    for (s, t), (key, v) in zip(DIRS, pairs.items()):
        for r in v['table']:
            c = canonical(r['feature'], s, t)
            if c is not None:
                agg.setdefault(c, []).append(r)
    rows = []
    for f, rs in agg.items():
        b = [r['beta'] for r in rs]
        rows.append({'feature': f, 'beta_mean': float(np.mean(b)), 'beta_sd': float(np.std(b, ddof=1)),
                     'abs_beta_mean': float(np.mean(np.abs(b))), 'n_pairs': len(rs),
                     'n_sig01': sum(r['p_fdr'] < .01 for r in rs), 'n_sig05': sum(r['p_fdr'] < .05 for r in rs)})
    rows.sort(key=lambda r: -r['abs_beta_mean'])
    print('\naggregated (by mean |beta|):')
    for r in rows[:12]:
        print(f'   {r["feature"]:22s} {r["beta_mean"]:+.3f} ± {r["beta_sd"]:.3f}  '
              f'sig01 {r["n_sig01"]}/{r["n_pairs"]}  sig05 {r["n_sig05"]}/{r["n_pairs"]}')
    if not cli.check_old:
        out = os.path.join(ROOT, f'output/r16/da_factors_{cli.model_type}_{cli.method}_{cli.criterion}.json')
        json.dump({'pairs': pairs, 'aggregate': rows}, open(out, 'w'), indent=1)
        print('wrote', out)


def per_user_old(old, s, t):
    """Old per_user_features.csv -> the same frame per_user()+user_features() build (validation only)."""
    df = old.rename(columns={'baseline_srocc_target': 'baseline_scc_target',
                             'baseline_srocc_source': 'baseline_scc_source'})
    keep = ['user_id', 'fold', 'baseline_scc_target', 'baseline_scc_source', 'delta_target']
    return df[keep].merge(user_features(s, t), left_on='user_id', right_index=True, how='left')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model_type', default='ICI')
    r.add_argument('--method', default='DJDOT')
    r.add_argument('--criterion', default='oracle')
    r.add_argument('--check_old', default=None)
    cli = ap.parse_args()
    run(cli)

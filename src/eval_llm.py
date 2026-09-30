"""Score the zero-shot LLM baselines on the group split (within-domain table).

The LLMs rated every (user, image) pair once (src.methods.{gpt,claude,gemini},
results in reports/exp/{model}/{genre}_piaa_results.json). Each fold's test
users are scored on their 50 'eval' samples per domain from fine_samples.csv,
the same samples the PIAA models are scored on, so every one of the 129 users
is scored exactly once.

LLM scores are on the 1-7 scale of the prompt and ratings on 0-6, so the
predictions are shifted by -1 before computing CCC (SCC is unaffected). As in
src.sweep, an undefined SCC (constant prediction) counts as 0 in the mean.

Usage:
    python -m src.eval_llm
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import GENRES

MODELS = {'gpt': 'GPT 5.4', 'claude': 'Claude Opus 4.6', 'gemini': 'Gemini 3 Flash'}


def _ccc(pred, true):
    mp, mt = pred.mean(), true.mean()
    cov = ((pred - mp) * (true - mt)).mean()
    return 2 * cov / (pred.var() + true.var() + (mp - mt) ** 2 + 1e-8)


def _test_users(split_dir):
    users = {}
    for fold in range(5):
        with open(os.path.join(split_dir, f'fold{fold}', 'test_users.txt')) as f:
            users.update({int(u): fold for u in f.read().split()})
    return users


def _predictions(path):
    with open(path) as f:
        results = json.load(f)
    pred = {}
    for entry in results['per_sample']:
        for r in entry['ratings']:
            pred.setdefault((int(r['user_id']), entry['sample_file']), float(r['pred_score']))
    return results['model'], pred


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--split_dir', default='asset/split')
    parser.add_argument('--maked_dir', default='asset/maked')
    parser.add_argument('--results_dir', default='reports/exp')
    parser.add_argument('--out', default='output/r16/llm_table5.json')
    cli = parser.parse_args()

    ratings = pd.read_csv(os.path.join(cli.maked_dir, 'ratings.csv'))
    ratings = ratings.drop_duplicates(['user_id', 'genre', 'sample_file'], keep='first')
    fine = pd.read_csv(os.path.join(cli.split_dir, 'fine_samples.csv'))
    test = _test_users(cli.split_dir)
    evals = fine[(fine['role'] == 'eval') & fine['user_id'].isin(test)]

    out = {}
    for name, label in MODELS.items():
        out[name] = {'label': label, 'summary': {}, 'per_user': {}}
        for genre in GENRES:
            model_id, pred = _predictions(os.path.join(cli.results_dir, name, f'{genre}_piaa_results.json'))
            rows = evals[evals['genre'] == genre].merge(
                ratings[ratings['genre'] == genre][['user_id', 'sample_id', 'sample_file', 'Aesthetic']],
                on=['user_id', 'sample_id'], how='left')
            rows['pred'] = [pred.get(k) for k in zip(rows['user_id'], rows['sample_file'])]
            if rows['pred'].isna().any():
                raise ValueError(f'{name}/{genre}: {rows["pred"].isna().sum()} eval samples have no prediction')

            per_user = {}
            for uid, g in rows.groupby('user_id'):
                p = g['pred'].to_numpy(float) - 1
                t = g['Aesthetic'].to_numpy(float)
                scc = spearmanr(p, t)[0]
                per_user[str(uid)] = {'scc': None if np.isnan(scc) else float(scc), 'ccc': float(_ccc(p, t))}
            scc = np.array([v['scc'] if v['scc'] is not None else 0.0 for v in per_user.values()])
            ccc = np.array([v['ccc'] for v in per_user.values()])
            out[name]['model'] = model_id
            out[name]['per_user'][genre] = per_user
            out[name]['summary'][genre] = {'n_users': len(per_user), 'scc_mean': scc.mean(), 'scc_std': scc.std(),
                                           'ccc_mean': ccc.mean(), 'ccc_std': ccc.std(),
                                           'n_undefined_scc': int(sum(v['scc'] is None for v in per_user.values()))}
            print(f"{label:16s} {genre:8s} n={len(per_user)} SCC {scc.mean():.3f}±{scc.std():.3f} "
                  f"CCC {ccc.mean():.3f}±{ccc.std():.3f}")
        s = out[name]['summary']
        s['avg'] = {'scc': float(np.mean([s[g]['scc_mean'] for g in GENRES])),
                    'ccc': float(np.mean([s[g]['ccc_mean'] for g in GENRES]))}
        print(f"{label:16s} avg      SCC {s['avg']['scc']:.3f} CCC {s['avg']['ccc']:.3f}")

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    with open(cli.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'-> {cli.out}')


if __name__ == '__main__':
    main()

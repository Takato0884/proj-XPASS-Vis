"""Build the 10-group / 5-fold user split shared by all methods and models.

Groups are the rating sessions (`set` 0-9). Each session has its own annotators
and its own stimuli, so splitting by group also keeps images disjoint across
train / val / test. Fold k tests on the two groups whose `fold` column is k,
validates on one group drawn at random from the remaining eight, and trains on
the other seven.

Per-user fine-tuning samples are drawn per domain from a seed fixed by
(seed, user_id, domain): `n_fine_eval` evaluation samples first, then
`n_fine_train` training samples from the rest. The draw does not depend on the
fold, so a user gets the same samples whether they are a val or a test user.

Output (under --out_dir):
    meta.json                  global settings and checksums
    fine_samples.csv           user_id, genre, sample_id, role (train / eval)
    fold{k}/train_users.txt    one id per line
    fold{k}/val_users.txt
    fold{k}/test_users.txt
    fold{k}/giaa_train_images.txt  sample_ids rated by train users (all domains)
    fold{k}/meta.json

Usage:
    python -m src.make_split --out_dir asset/split_v2
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

GENRES = ['art', 'fashion', 'scenery']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Build the group-based user split')
    parser.add_argument('--maked_dir', type=str, default='asset/maked', help='Directory containing ratings.csv')
    parser.add_argument('--out_dir', type=str, default='asset/split')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_fine_train', type=int, default=120, help='Per-user, per-domain fine-tuning samples')
    parser.add_argument('--n_fine_eval', type=int, default=50, help='Per-user, per-domain evaluation samples')
    parser.add_argument('--overwrite', action='store_true', help='Allow writing into a non-empty out_dir')
    return parser.parse_args()


def load_ratings(maked_dir):
    path = os.path.join(maked_dir, 'ratings.csv')
    ratings = pd.read_csv(path)
    # Retest rows repeat (user_id, sample_file); keep the first rating as src/data.py does.
    ratings = ratings.drop_duplicates(subset=['user_id', 'sample_file'], keep='first')
    with open(path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return ratings, md5


def build_groups(ratings):
    if ratings.groupby('user_id')['set'].nunique().max() != 1:
        raise ValueError('A user appears in more than one set')
    if ratings.groupby('sample_id')['set'].nunique().max() != 1:
        raise ValueError('A sample appears in more than one set')
    groups = {int(g): sorted(int(u) for u in df['user_id'].unique()) for g, df in ratings.groupby('set')}
    set_to_fold = ratings.groupby('set')['fold'].unique()
    if set_to_fold.map(len).max() != 1:
        raise ValueError('A set spans more than one fold')
    test_pairs = {}
    for g, folds in set_to_fold.items():
        test_pairs.setdefault(int(folds[0]), []).append(int(g))
    return groups, {k: sorted(v) for k, v in sorted(test_pairs.items())}


def build_folds(ratings, groups, test_pairs, rng):
    folds = {}
    for k, test_groups in test_pairs.items():
        candidates = [g for g in groups if g not in test_groups]
        val_group = int(rng.choice(candidates))
        train_groups = [g for g in candidates if g != val_group]

        users = lambda gs: sorted(u for g in gs for u in groups[g])
        train_users = users(train_groups)
        giaa_images = sorted(int(i) for i in ratings[ratings['user_id'].isin(train_users)]['sample_id'].unique())
        folds[k] = {
            'test_groups': test_groups,
            'val_group': val_group,
            'train_groups': train_groups,
            'train_users': train_users,
            'val_users': users([val_group]),
            'test_users': users(test_groups),
            'giaa_train_images': giaa_images,
        }
    return folds


def build_fine_samples(ratings, seed, n_train, n_eval):
    rows = []
    for (user_id, genre), df in ratings.groupby(['user_id', 'genre']):
        samples = np.sort(df['sample_id'].to_numpy())
        if len(samples) < n_train + n_eval:
            raise ValueError(f'user {user_id} has only {len(samples)} {genre} ratings (< {n_train + n_eval})')
        rng = np.random.default_rng([seed, int(user_id), GENRES.index(genre)])
        perm = rng.permutation(samples)
        rows += [(int(user_id), genre, int(s), 'eval') for s in perm[:n_eval]]
        rows += [(int(user_id), genre, int(s), 'train') for s in perm[n_eval:n_eval + n_train]]
    return pd.DataFrame(rows, columns=['user_id', 'genre', 'sample_id', 'role'])


def write_ids(path, ids):
    with open(path, 'w') as f:
        f.write('\n'.join(str(i) for i in ids) + '\n')


def write_split(out_dir, folds, fine, meta):
    os.makedirs(out_dir, exist_ok=True)
    for k, fold in folds.items():
        fold_dir = os.path.join(out_dir, f'fold{k}')
        os.makedirs(fold_dir, exist_ok=True)
        for name in ['train_users', 'val_users', 'test_users', 'giaa_train_images']:
            write_ids(os.path.join(fold_dir, f'{name}.txt'), fold[name])
        fold_meta = {
            'test_groups': fold['test_groups'],
            'val_group': fold['val_group'],
            'train_groups': fold['train_groups'],
            'n_test_users': len(fold['test_users']),
            'n_val_users': len(fold['val_users']),
            'n_train_users': len(fold['train_users']),
            'n_giaa_images': len(fold['giaa_train_images']),
        }
        with open(os.path.join(fold_dir, 'meta.json'), 'w') as f:
            json.dump(fold_meta, f, indent=2)
    fine.to_csv(os.path.join(out_dir, 'fine_samples.csv'), index=False)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def verify_split(out_dir, ratings, n_train, n_eval):
    """Re-read the written files and check the split requirements."""
    read_ids = lambda p: [int(x) for x in open(p).read().split()]
    all_users = set(ratings['user_id'])
    images_of = lambda us: set(ratings[ratings['user_id'].isin(us)]['sample_id'])
    fold_dirs = sorted(d for d in os.listdir(out_dir) if d.startswith('fold'))
    tested = []
    for d in fold_dirs:
        p = os.path.join(out_dir, d)
        meta = json.load(open(os.path.join(p, 'meta.json')))
        tr, va, te, im = (read_ids(os.path.join(p, f'{n}.txt'))
                          for n in ['train_users', 'val_users', 'test_users', 'giaa_train_images'])
        for ids in (tr, va, te, im):
            assert len(ids) == len(set(ids)), f'{d}: duplicate ids'
        tr, va, te, im = set(tr), set(va), set(te), set(im)
        assert len(meta['test_groups']) == 2 and len(meta['train_groups']) == 7, f'{d}: group counts'
        assert not (tr & va or tr & te or va & te), f'{d}: user sets overlap'
        assert tr | va | te == all_users, f'{d}: users missing'
        assert (meta['n_train_users'], meta['n_val_users'], meta['n_test_users'], meta['n_giaa_images']) \
            == (len(tr), len(va), len(te), len(im)), f'{d}: meta counts'
        tr_img, va_img, te_img = images_of(tr), images_of(va), images_of(te)
        assert not (tr_img & va_img or tr_img & te_img or va_img & te_img), f'{d}: images shared across roles'
        assert im == tr_img, f'{d}: giaa_train_images != train-user images'
        assert set(ratings[ratings['sample_id'].isin(im)]['genre']) == set(GENRES), f'{d}: domain missing'
        tested += te
    assert sorted(tested) == sorted(all_users), 'each user must be tested exactly once'

    fine = pd.read_csv(os.path.join(out_dir, 'fine_samples.csv'))
    counts = fine.groupby(['user_id', 'genre', 'role']).size().unstack()
    assert len(counts) == len(all_users) * len(GENRES), 'fine: user x domain missing'
    assert (counts['train'] == n_train).all() and (counts['eval'] == n_eval).all(), 'fine: sample counts'
    assert not fine.duplicated(['user_id', 'genre', 'sample_id']).any(), 'fine: train/eval overlap'
    rated = set(zip(ratings['user_id'], ratings['sample_id'], ratings['genre']))
    assert set(zip(fine['user_id'], fine['sample_id'], fine['genre'])) <= rated, 'fine: unrated samples'
    print(f'Verified {len(fold_dirs)} folds and fine_samples.csv')


def main():
    args = parse_arguments()
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
        raise SystemExit(f'{args.out_dir} is not empty; pass --overwrite to replace it')

    ratings, md5 = load_ratings(args.maked_dir)
    groups, test_pairs = build_groups(ratings)
    rng = np.random.default_rng(args.seed)
    folds = build_folds(ratings, groups, test_pairs, rng)
    fine = build_fine_samples(ratings, args.seed, args.n_fine_train, args.n_fine_eval)

    meta = {
        'seed': args.seed,
        'ratings_md5': md5,
        'dedup': 'drop_duplicates(user_id, sample_file, keep=first)',
        'n_users': ratings['user_id'].nunique(),
        'n_images': ratings['sample_id'].nunique(),
        'groups': 'rating set (session) 0-9',
        'group_sizes': {g: len(u) for g, u in groups.items()},
        'test_groups': 'the two sets whose fold column equals k',
        'val_group': 'drawn uniformly per fold from the 8 non-test groups',
        'n_fine_train': args.n_fine_train,
        'n_fine_eval': args.n_fine_eval,
        'fine_seed': '[seed, user_id, genre index in %s]' % GENRES,
    }
    write_split(args.out_dir, folds, fine, meta)
    verify_split(args.out_dir, ratings, args.n_fine_train, args.n_fine_eval)
    for k, fold in folds.items():
        print(f'fold{k}: test={fold["test_groups"]} val={fold["val_group"]} '
              f'users train/val/test={len(fold["train_users"])}/{len(fold["val_users"])}/{len(fold["test_users"])} '
              f'giaa_images={len(fold["giaa_train_images"])}')


if __name__ == '__main__':
    main()

"""R1-6 protocol: random search with stage-wise model selection on val users.

For one fold, method and direction (source -> target) the stages run in order,
each starting from the configuration selected in the previous stage:

    giaa  NIMA on train users' score histograms   selected by EMD on val users' images
    pre   PIAA on train users' ratings            selected by MSE on val users' ratings
    fine  per-user fine-tuning of the val users   selected by mean per-user SCC on their eval samples

and the test users are then fine-tuned with the selected configuration and
scored on their target-domain eval samples.

Two selection criteria (Gulrajani & Lopez-Paz, DomainBed, 2020):
    train_domain  val users' source-domain data  (main results)
    oracle        val users' target-domain data  (reference)

Every trial trains for a fixed number of epochs and is scored once, at its final
checkpoint, on every domain of interest, so both criteria select from the same
trials and the oracle queries the target once per trial. Trainers receive
training data in place of their validation arguments, so no val or test data
reaches training. Fine-stage adaptation uses train-group target images paired
with the fine-tuned user's traits (never that user's own target samples).

Methods: SourceOnly, TargetOnly and the UDA methods. DAREGRAM and RSD have no
GIAA stage and start from Source-Only's selected NIMA. TargetOnly is Source-Only
trained and selected on the target domain.

Results are cached per trial, so an interrupted run resumes where it stopped and
chains that share a stage (e.g. Source-Only for two targets) reuse its trials.
Only the best checkpoint per domain of interest is kept.

Usage:
    python -m src.sweep --fold 0 --method DANN --source art --target fashion --n_trials 20
    python -m src.sweep --fold 0 --method SourceOnly --source art          # both targets
    python -m src.sweep --fold 0 --method TargetOnly --target fashion
"""
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .argflags import parse_arguments
from .data import GENRES, GroupSplitData, build_global_encoders, collate_fn, load_group_split
from .evaluate import evaluate, evaluate_piaa, evaluate_piaa_mse
from .search_space import sample_hparams
from .train_common import NIMA, build_piaa_model, num_bins

DA_METHODS = ['DANN', 'DJDOT', 'JUMBOT', 'DEEPCORAL', 'CDAN', 'ALDA', 'DAREGRAM', 'RSD']
NO_GIAA = {'DAREGRAM', 'RSD'}
CRITERIA = ['train_domain', 'oracle']


def parse_cli():
    parser = argparse.ArgumentParser(description='R1-6 random search with stage-wise selection')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['SourceOnly', 'TargetOnly'] + DA_METHODS)
    parser.add_argument('--source', type=str, choices=GENRES, help='Not used by TargetOnly')
    parser.add_argument('--target', type=str, choices=GENRES, help='Omit to run every other domain')
    parser.add_argument('--model_type', type=str, default='ICI', choices=['ICI', 'MIR'])
    parser.add_argument('--n_trials', type=int, default=20, help='Configurations per stage, common to all methods')
    parser.add_argument('--search_seed', type=int, default=0)
    parser.add_argument('--criteria', type=str, nargs='+', default=CRITERIA, choices=CRITERIA)
    parser.add_argument('--split_dir', type=str, default='asset/split')
    parser.add_argument('--maked_dir', type=str, default='asset/maked')
    parser.add_argument('--root_dir', type=str, default='data', help='Contains samples/ and the cash/ cache')
    parser.add_argument('--backbone', type=str, default='clip_vit_b16',
                        choices=['resnet50', 'vit_b_16', 'clip_rn50', 'clip_vit_b16'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='reports/r16')
    parser.add_argument('--models_dir', type=str, default='models_pth/r16')
    cli = parser.parse_args()
    if cli.method == 'TargetOnly':
        if cli.target is None:
            parser.error('--method TargetOnly requires --target')
    elif cli.source is None:
        parser.error(f'--method {cli.method} requires --source')
    if cli.source is not None and cli.source == cli.target:
        parser.error('--source and --target must differ')
    return cli


def _stable_seed(*parts):
    return int(hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest(), 16) % (2 ** 31)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _key_name(key):
    method, src, tgt = key
    return f'{method}_{src}' if tgt is None else f'{method}_{src}2{tgt}'


def _trial_id(record):
    return f"{record['key']}-t{record['trial']:03d}"


def _score(record, metric, genre, minimize):
    value = record[metric][genre]
    if value is None or math.isnan(value):
        return math.inf
    return value if minimize else -value


def _best(records, metric, genre, minimize):
    return min(records, key=lambda r: (_score(r, metric, genre, minimize), r['trial']))


def _subset(dataset, uid):
    part = copy.copy(dataset)
    part.data = dataset.data[dataset.data['user_id'] == uid].reset_index(drop=True)
    return part


def _method_module(method):
    name = 'source_only' if method == 'SourceOnly' else method.lower()
    return importlib.import_module(f'.methods.{name}', package=__package__)


class Sweep:
    def __init__(self, cli):
        self.cli = cli
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = load_group_split(cli.split_dir, cli.fold)
        self.encoders = build_global_encoders(cli.root_dir, maked_dir=cli.maked_dir)
        self.report_dir = os.path.join(cli.out_dir, f'fold{cli.fold}')
        self.model_dir = os.path.join(cli.models_dir, f'fold{cli.fold}')
        self._data = {}
        self._fine_sets = {}
        self._dims = {}

    # ---------- data ----------

    def data(self, genre):
        if genre not in self._data:
            self._data[genre] = GroupSplitData(self.args(genre), genre, self.split, *self.encoders)
        return self._data[genre]

    def fine_set(self, genre, users, role):
        key = (genre, tuple(users), role)
        if key not in self._fine_sets:
            if role == 'unlabeled':
                self._fine_sets[key] = self.data(genre).unlabeled_personal(users)
            else:
                self._fine_sets[key] = self.data(genre).fine(users, role)
        return self._fine_sets[key]

    def dims(self, genre):
        if genre not in self._dims:
            sample = self.data(genre).pre('train')[0]
            self._dims[genre] = (len(sample['QIP']), len(sample['traits']))
        return self._dims[genre]

    def loader(self, dataset, batch_size, train=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                          num_workers=self.cli.num_workers, timeout=300, collate_fn=collate_fn)

    def args(self, genre, method=None, target=None, hp=None):
        args = parse_arguments(parse=False).parse_args(['--genre', genre])
        args.root_dir = self.cli.root_dir
        args.maked_dir = self.cli.maked_dir
        args.backbone = self.cli.backbone
        args.num_workers = self.cli.num_workers
        args.model_type = self.cli.model_type
        args.dataset_ver = f'group_fold{self.cli.fold}'
        args.is_log = False
        args.fixed_epochs = True
        args.no_save_model = False
        args.da_method = f'{method}-{target}' if method in DA_METHODS else None
        for name, value in (hp or {}).items():
            setattr(args, name, value)
        return args

    def eval_genres(self, key):
        method, src, tgt = key
        return list(GENRES) if method == 'SourceOnly' else [src, tgt]

    def _prune(self, records, genres, minimize):
        keep = {_best(records, 'val_loss', g, minimize)['ckpt'] for g in genres}
        for r in records:
            if r['ckpt'] not in keep and os.path.exists(r['ckpt']):
                os.remove(r['ckpt'])

    def _release(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- giaa ----------

    def giaa_trials(self, key):
        records = []
        for t in range(self.cli.n_trials):
            path = os.path.join(self.report_dir, 'giaa', _key_name(key), f't{t:03d}.json')
            record = _read_json(path)
            if record is None:
                record = self._run_giaa(key, t)
                _write_json(path, record)
            records.append(record)
            self._prune(records, self.eval_genres(key), minimize=True)
        return records

    def _run_giaa(self, key, t):
        method, src, tgt = key
        hp = sample_hparams(method, 'giaa', t, self.cli.search_seed)
        args = self.args(src, method, tgt, hp)
        seed = _stable_seed(self.cli.fold, 'giaa', _key_name(key), t)
        _seed_everything(seed)
        print(f'\n[giaa] {_key_name(key)} trial {t}: {hp}')

        mod = _method_module(method)
        model = NIMA(num_bins, backbone=args.backbone, dropout=args.dropout).to(self.device)
        model.freeze_backbone()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        components = mod.setup(model, args, self.device)
        train_loader = self.loader(self.data(src).giaa('train'), args.batch_size, train=True)
        tgt_loader = self.loader(self.data(tgt).giaa('train'), args.batch_size, train=True) if tgt else None
        ckpt = os.path.join(self.model_dir, 'giaa', _key_name(key), f't{t:03d}.pth')
        mod.trainer((train_loader, None, None), tgt_loader, model, optimizer, args, self.device, ckpt,
                    components, tgt_val_loader=None, tgt_genre=tgt)

        val_loss = {}
        for g in self.eval_genres(key):
            val_loader = self.loader(self.data(g).giaa('val'), args.batch_size)
            val_loss[g] = float(evaluate(model, val_loader, self.device, phase_name=f'Val EMD [{g}]')[0])
        del model, optimizer, components
        self._release()
        return {'stage': 'giaa', 'key': _key_name(key), 'trial': t, 'seed': seed, 'hparams': hp,
                'metric': 'EMD on val users images', 'val_loss': val_loss, 'ckpt': ckpt}

    # ---------- pre ----------

    def pre_trials(self, key, giaa):
        records = []
        for t in range(self.cli.n_trials):
            path = os.path.join(self.report_dir, self.cli.model_type, 'pre', _key_name(key), _trial_id(giaa),
                                f't{t:03d}.json')
            record = _read_json(path)
            if record is None:
                record = self._run_pre(key, giaa, t)
                _write_json(path, record)
            records.append(record)
            self._prune(records, self.eval_genres(key), minimize=True)
        return records

    def _run_pre(self, key, giaa, t):
        method, src, tgt = key
        hp = sample_hparams(method, 'pre', t, self.cli.search_seed)
        args = self.args(src, method, tgt, hp)
        seed = _stable_seed(self.cli.fold, self.cli.model_type, 'pre', _key_name(key), _trial_id(giaa), t)
        _seed_everything(seed)
        print(f'\n[pre] {_key_name(key)} on {_trial_id(giaa)} trial {t}: {hp}')

        num_attr, num_pt = self.dims(src)
        backbone_dict = {src: args.backbone}
        run_dir = os.path.join(self.model_dir, self.cli.model_type, 'pre', _key_name(key), _trial_id(giaa),
                               f't{t:03d}')
        os.makedirs(run_dir, exist_ok=True)
        train = self.data(src).pre('train')
        datasets_dict = {src: {'train': train, 'val': train, 'test': None}}
        exp = f'{method}_t{t:03d}'
        mod = _method_module(method)
        if method == 'SourceOnly':
            ckpt, _ = mod.trainer_pretrain(datasets_dict, args, self.device, run_dir, exp, backbone_dict,
                                           {src: giaa['ckpt']}, num_attr, num_pt)
        else:
            tgt_train = self.data(tgt).pre('train')
            ckpt, _ = mod.trainer_pretrain(datasets_dict, tgt_train, tgt_train, args, self.device, run_dir, exp,
                                           backbone_dict, {src: giaa['ckpt']}, num_attr, num_pt,
                                           domain_tag=f'{src}2{tgt}')

        model = build_piaa_model(num_bins, num_attr, num_pt, [src], backbone_dict, args).to(self.device)
        model.load_state_dict(torch.load(ckpt))
        val_loss = {g: evaluate_piaa_mse(model, self.loader(self.data(g).pre('val'), args.batch_size),
                                         self.device, head=src)
                    for g in self.eval_genres(key)}
        del model
        self._release()
        return {'stage': 'pre', 'key': _key_name(key), 'base': _trial_id(giaa), 'trial': t, 'seed': seed,
                'hparams': hp, 'metric': 'MSE on val users ratings', 'val_loss': val_loss, 'ckpt': ckpt}

    # ---------- fine / test ----------

    def fine_trials(self, key, pre):
        records = []
        for t in range(self.cli.n_trials):
            path = os.path.join(self.report_dir, self.cli.model_type, 'fine', _key_name(key), _pre_id(pre),
                                f't{t:03d}.json')
            record = _read_json(path)
            if record is None:
                method = key[0]
                hp = sample_hparams(method, 'fine', t, self.cli.search_seed)
                print(f'\n[fine] {_key_name(key)} on {_pre_id(pre)} trial {t}: {hp}')
                seed = _stable_seed(self.cli.fold, self.cli.model_type, 'fine', _key_name(key), _pre_id(pre), t)
                result = self._fine_users(key, pre, hp, self.split['val_users'], seed)
                record = {'stage': 'fine', 'key': _key_name(key), 'base': _pre_id(pre), 'trial': t, 'seed': seed,
                          'hparams': hp, 'metric': 'mean per-user SCC on val users eval samples',
                          'val_scc': result['scc'], 'per_user': result['per_user']}
                _write_json(path, record)
            records.append(record)
        return records

    def test_run(self, key, pre, fine):
        path = os.path.join(self.report_dir, self.cli.model_type, 'test', _key_name(key), _pre_id(pre),
                            f"fine-t{fine['trial']:03d}.json")
        record = _read_json(path)
        if record is None:
            print(f"\n[test] {_key_name(key)} on {_pre_id(pre)} fine trial {fine['trial']}")
            result = self._fine_users(key, pre, fine['hparams'], self.split['test_users'], fine['seed'])
            record = {'stage': 'test', 'key': _key_name(key), 'base': _pre_id(pre), 'fine_trial': fine['trial'],
                      'hparams': fine['hparams'], 'metric': 'mean per-user SCC on test users eval samples',
                      'test_scc': result['scc'], 'per_user': result['per_user']}
            _write_json(path, record)
        return record

    def _fine_users(self, key, pre, hp, users, seed):
        """Fine-tune each user on their source 'train' samples; SCC on their 'eval' samples per domain."""
        method, src, tgt = key
        args = self.args(src, method, tgt, hp)
        num_attr, num_pt = self.dims(src)
        backbone_dict = {src: args.backbone}
        tmp_dir = os.path.join(self.model_dir, self.cli.model_type, 'tmp_fine')
        os.makedirs(tmp_dir, exist_ok=True)
        mod = _method_module(method)
        train_all = self.fine_set(src, users, 'train')
        unlabeled_all = self.fine_set(tgt, users, 'unlabeled') if method != 'SourceOnly' else None

        per_user = {}
        for uid in users:
            _seed_everything(_stable_seed(seed, uid))
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
            model.load_state_dict(torch.load(ckpt))
            os.remove(ckpt)
            per_user[str(uid)] = {}
            for g in self.eval_genres(key):
                loader = self.loader(_subset(self.fine_set(g, users, 'eval'), uid), args.batch_size)
                srocc = evaluate_piaa(model, {src: loader}, self.device, phase_name=f'SCC u{uid} [{g}]')[0][src]['srocc']
                per_user[str(uid)][g] = None if np.isnan(srocc) else float(srocc)
            del model
            self._release()

        # An undefined SCC (constant prediction) counts as 0 in the mean; per_user keeps it as null.
        scc = {g: float(np.mean([v[g] if v[g] is not None else 0.0 for v in per_user.values()]))
               for g in self.eval_genres(key)}
        return {'scc': scc, 'per_user': per_user}

    # ---------- chains ----------

    def run_chain(self, method, src, tgt, criterion):
        """Select giaa -> pre -> fine on `criterion`, then evaluate the test users."""
        sel = src if criterion == 'train_domain' else tgt
        key = ('SourceOnly', src, None) if method == 'SourceOnly' else (method, src, tgt)
        giaa_key = ('SourceOnly', src, None) if method in NO_GIAA else key

        giaa_records = self.giaa_trials(giaa_key)
        giaa = _best(giaa_records, 'val_loss', sel, minimize=True)
        pre_records = self.pre_trials(key, giaa)
        pre = _best(pre_records, 'val_loss', sel, minimize=True)
        fine_records = self.fine_trials(key, pre)
        fine = _best(fine_records, 'val_scc', sel, minimize=False)
        test = self.test_run(key, pre, fine)

        stage = lambda r, m: {'trial': r['trial'], 'hparams': r['hparams'], 'val': r[m][sel]}
        return {
            'fold': self.cli.fold, 'model_type': self.cli.model_type, 'method': method,
            'source': src, 'target': tgt, 'criterion': criterion, 'selection_domain': sel,
            'n_trials': self.cli.n_trials, 'search_seed': self.cli.search_seed,
            'selected': {'giaa': dict(stage(giaa, 'val_loss'), key=giaa['key']),
                         'pre': stage(pre, 'val_loss'), 'fine': stage(fine, 'val_scc')},
            'test_scc': test['test_scc'],
            'per_user': test['per_user'],
        }

    def run(self):
        cli = self.cli
        if cli.method == 'TargetOnly':
            # Source-Only trained and selected on the target domain; one criterion.
            result = self.run_chain('SourceOnly', cli.target, None, 'train_domain')
            result.update(method='TargetOnly', source=cli.target, target=cli.target, criterion='target')
            path = os.path.join(self.report_dir, cli.model_type, 'final', f'TargetOnly_{cli.target}.json')
            _write_json(path, result)
            print(f"TargetOnly {cli.target}: test SCC = {result['test_scc'][cli.target]:.4f} -> {path}")
            return
        targets = [cli.target] if cli.target else [g for g in GENRES if g != cli.source]
        for tgt in targets:
            for criterion in cli.criteria:
                result = self.run_chain(cli.method, cli.source, tgt, criterion)
                name = f'{cli.method}_{cli.source}2{tgt}_{criterion}.json'
                path = os.path.join(self.report_dir, cli.model_type, 'final', name)
                _write_json(path, result)
                print(f"{cli.method} {cli.source}->{tgt} [{criterion}]: "
                      f"test SCC target={result['test_scc'][tgt]:.4f} source={result['test_scc'][cli.source]:.4f}"
                      f" -> {path}")


def _pre_id(record):
    return f"{record['base']}-pre-t{record['trial']:03d}"


if __name__ == '__main__':
    Sweep(parse_cli()).run()

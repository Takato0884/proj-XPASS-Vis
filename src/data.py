import os
import math
import pandas as pd
import numpy as np
import copy
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import cv2
from tqdm import tqdm
import pickle
import hashlib
import json
from torchvision import transforms

def build_global_encoders(root_dir, num_age_bins=5, maked_dir=None):
    maked_dir = maked_dir or os.path.join(root_dir, 'maked')
    users = pd.read_csv(os.path.join(maked_dir, 'users.csv'))

    encoded_trait_columns = ['age', 'gender', 'edu', 'nationality', 'art_learn', 'fashion_learn', 'photoVideo_learn']

    min_age, max_age = users['age'].min(), users['age'].max()
    global_age_bins = np.linspace(min_age, max_age, num=num_age_bins + 1)
    age_labels = [f'{int(global_age_bins[i])}-{int(global_age_bins[i+1])-1}' for i in range(num_age_bins)]

    global_trait_encoders = []
    for attr in encoded_trait_columns:
        if attr == 'age':
            encoder = {label: idx for idx, label in enumerate(age_labels)}
        else:
            unique_vals = sorted(users[attr].dropna().unique(), key=str)
            encoder = {val: idx for idx, val in enumerate(unique_vals)}
        global_trait_encoders.append(encoder)

    return global_trait_encoders, global_age_bins


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, genre=None, backbone=None, max_frames=None, is_train=False,
                 global_trait_encoders=None, global_age_bins=None, maked_dir=None):
        self.genre = genre
        self.backbone = backbone
        self.use_image = (self.backbone != "i3d")
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames
        self.is_train = is_train
        self.maked_dir = maked_dir or os.path.join(root_dir, 'maked')

        self.ratings_data = pd.read_csv(os.path.join(self.maked_dir, 'ratings.csv'))
        self.ratings_data = self.ratings_data[self.ratings_data['genre'] == self.genre]
        if {'user_id', 'sample_file'}.issubset(self.ratings_data.columns):
            self.ratings_data = self.ratings_data.drop_duplicates(subset=['user_id', 'sample_file'], keep='first')
        self.users_data = pd.read_csv(os.path.join(self.maked_dir, 'users.csv'))
        self.users_data = self.users_data.set_index("user_id")
        self.data = pd.merge(self.ratings_data, self.users_data, on="user_id", how="left")

        if self.genre == 'scenery' and self.backbone == 'i3d':
            qip_filename = f'QIP_{self.genre}_video.csv'
        elif self.genre == 'scenery':
            qip_filename = f'QIP_{self.genre}_image.csv'
        else:
            qip_filename = f'QIP_{self.genre}.csv'
        self.qip_df = pd.read_csv(os.path.join(self.maked_dir, qip_filename))
        self.qip_df = self.qip_df.set_index("img_file")
        samples_root = os.path.join(root_dir, 'samples')
        if self.genre == 'scenery' and self.backbone == 'i3d':
            self.samples_dir = os.path.join(samples_root, 'scenery_video')
        elif self.genre == 'scenery':
            self.samples_dir = os.path.join(samples_root, 'scenery_image')
        else:
            self.samples_dir = samples_root

        self.score_fields  = [f'Q{i}' for i in range(1, 11)] + ['art_interest', 'fashion_interest', 'photoVideo_interest']
        self.metadata = ['Aesthetic', 'fold', 'user_id']
        self.encoded_trait_columns = ['age', 'gender', 'edu', 'nationality', 'art_learn', 'fashion_learn', 'photoVideo_learn']

        if global_age_bins is not None:
            interval_edges = global_age_bins
        else:
            min_age, max_age = self.data['age'].min(), self.data['age'].max()
            interval_edges = np.linspace(min_age, max_age, num=6)
        interval_labels = [f'{int(interval_edges[i])}-{int(interval_edges[i+1])-1}' for i in range(len(interval_edges)-1)]
        age_intervals = pd.cut(self.data['age'], bins=interval_edges, labels=interval_labels, include_lowest=True)
        self.data['age'] = age_intervals

        if global_trait_encoders is not None:
            self.trait_encoders = global_trait_encoders
        else:
            self.trait_encoders = [
                {group: idx for idx, group in enumerate(sorted(self.data[attribute].dropna().unique(), key=str))}
                for attribute in self.encoded_trait_columns
            ]

        qip_min = self.qip_df.min()
        qip_max = self.qip_df.max()
        qip_range = qip_max - qip_min
        qip_range[qip_range == 0] = 1
        self.qip_df = (self.qip_df - qip_min) / qip_range
        self.qip_data = self.qip_df.to_dict(orient="index")

    def one_hot_personality(self, trait_value):
        if trait_value is None:
            raise ValueError("one_hot_personality: trait_value is None")
        if isinstance(trait_value, float) and math.isnan(trait_value):
            raise ValueError("one_hot_personality: trait_value is NaN")
        try:
            idx = int(trait_value)
        except Exception as e:
            raise ValueError(f"one_hot_personality: trait_value '{trait_value}' is not convertible to int") from e

        if not (0 <= idx <= 6):
            raise ValueError(f"one_hot_personality: trait_value index out of range [0,6]: {idx}")

        return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=7)

    def __len__(self):
        return len(self.data)

    def _row(self, idx):
        """Row `idx` of self.data as a dict. DataFrame.iloc per item dominated data loading, so the
        rows are converted once per data frame (self.data is reassigned after __init__, hence the check)."""
        if getattr(self, '_rows_of', None) is not self.data:
            self._rows = self.data.to_dict('records')
            self._rows_of = self.data
        return self._rows[idx]

    def __getitem__(self, idx):
        row = self._row(idx)

        sample = {
            attribute: torch.tensor(row[attribute], dtype=torch.int) for attribute in self.metadata
        }

        sample.update({
            f'{attribute}_onehot': self.one_hot_personality(row[attribute]) for attribute in self.score_fields
        })

        sample.update({
            trait: torch.tensor(encoder[row[trait]], dtype=torch.int)
            for trait, encoder in zip(self.encoded_trait_columns, self.trait_encoders)
        })

        for trait, encoder in zip(self.encoded_trait_columns, self.trait_encoders):
            original_val = sample[trait]
            onehot_val = F.one_hot(original_val.long(), num_classes=len(encoder))
            sample[f'{trait}_onehot'] = onehot_val

        for trait in self.encoded_trait_columns:
            del sample[trait]

        sample_file = row['sample_file']
        features = getattr(self, 'features', None)
        feature = features[sample_file] if features is not None else None
        if self.genre == 'scenery':
            if self.use_image and sample_file.endswith('.mp4'):
                sample_file = sample_file.replace('.mp4', '.jpg')
            sample_path = os.path.join(self.samples_dir, sample_file)
        else:
            sample_path = os.path.join(self.samples_dir, self.genre, sample_file)
        sample['sample_path'] = sample_path
        sample['sample_file'] = sample_file

        if feature is not None:
            # Cached frozen-backbone features replace the image (and its augmentation).
            sample['image'] = feature
        elif self.use_image:
            sample['image'] = Image.open(sample_path).convert('RGB')
            if self.transform:
                sample['image'] = self.transform(sample['image'])
        else:
            cap = cv2.VideoCapture(sample_path)
            frames = []

            video_seed = None
            if self.transform:
                video_seed = int(np.random.randint(0, 2**31 - 1))
            frame_idx = 0
            while True:
                ret, frm = cap.read()
                if not ret:
                    break


                try:
                    frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
                img = Image.fromarray(frm)

                if self.transform:
                    random.seed(video_seed)
                    np.random.seed(video_seed)
                    torch.manual_seed(video_seed)
                    img = self.transform(img)

                if isinstance(img, torch.Tensor):
                    frames.append(img)
                else:
                    arr = np.array(img)
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    frames.append(torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32))

                frame_idx += 1
            cap.release()

            if len(frames) == 0:
                sample['image'] = torch.empty(0)
            else:
                video_tensor = torch.stack(frames, dim=0).float()
                video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()

                if (self.max_frames is not None) and (self.max_frames > 0):
                    C, T, H, W = video_tensor.shape
                    mf = int(self.max_frames)
                    if T > mf:
                        if self.is_train:
                            start = random.randint(0, T - mf)
                        else:
                            start = max(0, (T - mf) // 2)
                        video_tensor = video_tensor[:, start:start + mf, :, :]
                    elif T < mf:
                        pad_len = mf - T
                        pad = torch.zeros((C, pad_len, H, W), dtype=video_tensor.dtype)
                        video_tensor = torch.cat([video_tensor, pad], dim=1)

                sample['image'] = video_tensor

        sample['QIP'] = self.qip_data[sample['sample_file']]
        sample['user_id'] = row['user_id']
        return sample

def create_GIAA_split_dataset(dataset, fold_id):
    root_dir = dataset.root_dir
    genre = dataset.genre

    version = str(getattr(dataset, 'dataset_ver', fold_id))
    train_file_path = os.path.join(root_dir, 'split', version, genre, 'train_images_GIAA.txt')
    val_file_path = os.path.join(root_dir, 'split', version, genre, 'val_images_GIAA.txt')
    piaa_train_file_path = os.path.join(root_dir, 'split', version, genre, 'train_PIAA.txt')
    piaa_val_file_path = os.path.join(root_dir, 'split', version, genre, 'val_PIAA.txt')
    piaa_test_file_path = os.path.join(root_dir, 'split', version, genre, 'test_PIAA.txt')

    print('Read Image Set')
    with open(train_file_path, "r") as train_file:
        train_image_names = train_file.read().splitlines()
    with open(val_file_path, "r") as val_file:
        val_image_names = val_file.read().splitlines()

    piaa_user_ids = set()
    def _parse_user_ids(lines):
        ids = []
        for ln in lines:
            ln = ln.strip()
            if ln == '':
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            ids.append(parts[0])
        return ids

    try:
        if os.path.exists(piaa_train_file_path):
            with open(piaa_train_file_path, "r") as f:
                piaa_user_ids.update(_parse_user_ids(f.read().splitlines()))
        if os.path.exists(piaa_val_file_path):
            with open(piaa_val_file_path, "r") as f:
                piaa_user_ids.update(_parse_user_ids(f.read().splitlines()))
        if os.path.exists(piaa_test_file_path):
            with open(piaa_test_file_path, "r") as f:
                piaa_user_ids.update(_parse_user_ids(f.read().splitlines()))
    except Exception:
        piaa_user_ids = set()

    train_dataset_GIAA, val_dataset_GIAA = copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_dataset_GIAA.data = train_dataset_GIAA.data[train_dataset_GIAA.data['sample_file'].isin(train_image_names)]
    val_dataset_GIAA.data = val_dataset_GIAA.data[val_dataset_GIAA.data['sample_file'].isin(val_image_names)]

    if len(piaa_user_ids) > 0:
        train_dataset_GIAA.data = train_dataset_GIAA.data.assign(user_id_str=train_dataset_GIAA.data['user_id'].astype(str))
        val_dataset_GIAA.data   = val_dataset_GIAA.data.assign(user_id_str=val_dataset_GIAA.data['user_id'].astype(str))
        train_dataset_GIAA.data = train_dataset_GIAA.data[~train_dataset_GIAA.data['user_id_str'].isin(piaa_user_ids)].drop(columns=['user_id_str'])
        val_dataset_GIAA.data   = val_dataset_GIAA.data[~val_dataset_GIAA.data['user_id_str'].isin(piaa_user_ids)].drop(columns=['user_id_str'])

    return train_dataset_GIAA, val_dataset_GIAA

def create_GIAA_test_dataset(dataset, fold_id):
    root_dir = dataset.root_dir
    genre = dataset.genre
    version = str(getattr(dataset, 'dataset_ver', fold_id))
    test_file_path = os.path.join(root_dir, 'split', version, genre, 'test_images_GIAA.txt')

    with open(test_file_path, "r") as f:
        test_image_names = f.read().splitlines()

    test_dataset_GIAA = copy.deepcopy(dataset)
    test_dataset_GIAA.data = test_dataset_GIAA.data[test_dataset_GIAA.data['sample_file'].isin(test_image_names)]
    return test_dataset_GIAA

def create_GIAA_user_split_dataset(dataset, fold_id):
    root_dir = dataset.root_dir
    genre = dataset.genre
    version = str(getattr(dataset, 'dataset_ver', fold_id))

    train_user_file = os.path.join(root_dir, 'split', version, genre, 'train_users_GIAA.txt')
    val_user_file = os.path.join(root_dir, 'split', version, genre, 'val_users_GIAA.txt')

    with open(train_user_file, "r") as f:
        train_user_ids = set(line.strip() for line in f if line.strip())
    with open(val_user_file, "r") as f:
        val_user_ids = set(line.strip() for line in f if line.strip())

    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)

    train_dataset.data = train_dataset.data[
        train_dataset.data['user_id'].astype(str).isin(train_user_ids)
    ]
    val_dataset.data = val_dataset.data[
        val_dataset.data['user_id'].astype(str).isin(val_user_ids)
    ]

    return train_dataset, val_dataset

def create_PIAA_split_dataset(dataset, fold_id):
    root_dir = dataset.root_dir
    genre = dataset.genre

    version = str(getattr(dataset, 'dataset_ver', fold_id))
    train_file_path = os.path.join(root_dir, 'split', version, genre, 'train_PIAA.txt')
    val_file_path = os.path.join(root_dir, 'split', version, genre, 'val_PIAA.txt')
    test_file_path = os.path.join(root_dir, 'split', version, genre, 'test_PIAA.txt')

    with open(train_file_path, "r") as train_file:
        train_lines = train_file.read().splitlines()
    with open(val_file_path, "r") as val_file:
        val_lines = val_file.read().splitlines()
    with open(test_file_path, "r") as test_file:
        test_lines = test_file.read().splitlines()

    def _parse_userfile_pairs(lines):
        pairs = []
        for ln in lines:
            ln = ln.strip()
            if ln == '':
                continue
            parts = ln.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed split line (expected 'user_id filename'): '{ln}'")
            uid = str(parts[0])
            fname = parts[-1]
            pairs.append((uid, fname))
        return pairs

    train_pairs = pd.DataFrame(_parse_userfile_pairs(train_lines), columns=['user_id', 'sample_file'])
    val_pairs = pd.DataFrame(_parse_userfile_pairs(val_lines), columns=['user_id', 'sample_file'])
    test_pairs = pd.DataFrame(_parse_userfile_pairs(test_lines), columns=['user_id', 'sample_file'])

    train_dataset_PIAA, val_dataset_PIAA, test_dataset_PIAA = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)

    train_dataset_PIAA.data = train_dataset_PIAA.data.assign(user_id=train_dataset_PIAA.data['user_id'].astype(str))
    train_dataset_PIAA.data = train_dataset_PIAA.data.merge(train_pairs, on=['user_id', 'sample_file'], how='inner')

    val_dataset_PIAA.data = val_dataset_PIAA.data.assign(user_id=val_dataset_PIAA.data['user_id'].astype(str))
    val_dataset_PIAA.data = val_dataset_PIAA.data.merge(val_pairs, on=['user_id', 'sample_file'], how='inner')

    test_dataset_PIAA.data = test_dataset_PIAA.data.assign(user_id=test_dataset_PIAA.data['user_id'].astype(str))
    test_dataset_PIAA.data = test_dataset_PIAA.data.merge(test_pairs, on=['user_id', 'sample_file'], how='inner')

    train_dataset_PIAA.data['user_id'] = pd.to_numeric(train_dataset_PIAA.data['user_id'], errors='coerce').astype('Int64')
    val_dataset_PIAA.data['user_id']   = pd.to_numeric(val_dataset_PIAA.data['user_id'], errors='coerce').astype('Int64')
    test_dataset_PIAA.data['user_id']  = pd.to_numeric(test_dataset_PIAA.data['user_id'], errors='coerce').astype('Int64')

    return train_dataset_PIAA, val_dataset_PIAA, test_dataset_PIAA


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(backbone=None):
    if backbone in ('clip_rn50', 'clip_vit_b16'):
        normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    elif backbone in ('resnet50', 'vit_b_16', 'i3d'):
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        normalize = None

    train_ops = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
    ]
    test_ops = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        train_ops.append(normalize)
        test_ops.append(normalize)

    return transforms.Compose(train_ops), transforms.Compose(test_ops)

def load_data(args, global_trait_encoders=None, global_age_bins=None):
    root_dir = args.root_dir
    genre = args.genre
    backbone = getattr(args, 'backbone', None)
    dataset_ver = getattr(args, 'dataset_ver', 'v1')

    train_transform, test_transform = get_transforms(backbone)

    all_dataset = ImageDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone,
                               global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
    setattr(all_dataset, 'dataset_ver', dataset_ver)
    train_giaa_raw, val_giaa_raw = create_GIAA_split_dataset(all_dataset, dataset_ver)
    train_piaa_dataset, val_piaa_dataset, test_piaa_dataset = create_PIAA_split_dataset(all_dataset, dataset_ver)

    train_giaa_usersplit, val_giaa_usersplit = create_GIAA_user_split_dataset(all_dataset, dataset_ver)
    pretrain_train_data = train_giaa_usersplit.data
    pretrain_val_data = val_giaa_usersplit.data

    """Precompute"""
    version = str(dataset_ver) if dataset_ver is not None else 'v1'
    pkl_dir_base = os.path.join(root_dir, 'cash', version)
    pkl_dir = os.path.join(pkl_dir_base, f'{genre}_dataset_pkl')
    ensure_dir_exists(pkl_dir)
    map_file = os.path.join(pkl_dir, 'trainset_image_dct.pkl')
    enc_kwargs = dict(global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
    _train_max_frames = None if backbone == 'i3d' else 16
    train_giaa_hist_dataset = Image_GIAA_HistogramDataset(root_dir, transform=train_transform,
        genre=genre, backbone=backbone, data=train_giaa_raw.data, map_file=map_file,
    precompute_file=os.path.join(pkl_dir, 'trainset_GIAA_dct.pkl'),
        max_frames=_train_max_frames, is_train=True, **enc_kwargs)

    val_mapfile = os.path.join(pkl_dir, 'valset_image_dct.pkl')
    val_precompute_file = os.path.join(pkl_dir, 'valset_GIAA_dct.pkl')
    val_giaa_hist_dataset = Image_GIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, \
        backbone=backbone, data=val_giaa_raw.data, map_file=val_mapfile, precompute_file=val_precompute_file,
        max_frames=None, is_train=False, **enc_kwargs)

    train_giaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone, data=pretrain_train_data, max_frames=_train_max_frames, is_train=True, **enc_kwargs)
    val_giaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, backbone=backbone, data=pretrain_val_data, max_frames=None, is_train=False, **enc_kwargs)
    train_piaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone, data=train_piaa_dataset.data, max_frames=_train_max_frames, is_train=True, **enc_kwargs)
    val_piaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, backbone=backbone, data=val_piaa_dataset.data, max_frames=None, is_train=False, **enc_kwargs)
    test_piaa_dataset = Image_PIAA_HistogramDataset(root_dir, transform=test_transform, genre=genre, backbone=backbone, data=test_piaa_dataset.data, max_frames=None, is_train=False, **enc_kwargs)

    def _unique_users(ds):
        try:
            df = getattr(ds, 'data', None)
            if df is not None and 'user_id' in df.columns:
                return int(df['user_id'].astype(str).nunique())
        except Exception:
            pass
        return None

    ug_train = _unique_users(train_giaa_hist_dataset)
    ug_val = _unique_users(val_giaa_hist_dataset)
    upre_train = _unique_users(train_giaa_dataset)
    upre_val = _unique_users(val_giaa_dataset)
    ufine_train = _unique_users(train_piaa_dataset)
    ufine_val = _unique_users(val_piaa_dataset)
    ufine_test = _unique_users(test_piaa_dataset)

    def _fmt_users(n):
        return f', unique_users={n}' if n is not None else ''

    print(f'Train size GIAA: {len(train_giaa_hist_dataset)}{_fmt_users(ug_train)}, Val size GIAA: {len(val_giaa_hist_dataset)}{_fmt_users(ug_val)}')
    print(f'Train size PIAA-Pre: {len(train_giaa_dataset)}{_fmt_users(upre_train)}, Val size PIAA-Pre: {len(val_giaa_dataset)}{_fmt_users(upre_val)}')
    print(f'Train size PIAA-Fine: {len(train_piaa_dataset)}{_fmt_users(ufine_train)}, Val size PIAA-Fine: {len(val_piaa_dataset)}{_fmt_users(ufine_val)}, Test size PIAA-Fine: {len(test_piaa_dataset)}{_fmt_users(ufine_test)}')
    return train_giaa_hist_dataset, train_piaa_dataset, train_giaa_dataset, val_giaa_hist_dataset, val_piaa_dataset, val_giaa_dataset, test_piaa_dataset

def load_data_giaa_only(args):
    root_dir = args.root_dir
    genre = args.genre
    backbone = getattr(args, 'backbone', None)
    dataset_ver = getattr(args, 'dataset_ver', 'v1')

    train_transform, test_transform = get_transforms(backbone)

    all_dataset = ImageDataset(root_dir, transform=train_transform, genre=genre, backbone=backbone)
    setattr(all_dataset, 'dataset_ver', dataset_ver)

    train_giaa_raw, val_giaa_raw = create_GIAA_split_dataset(all_dataset, dataset_ver)
    test_giaa_raw = create_GIAA_test_dataset(all_dataset, dataset_ver)

    version = str(dataset_ver) if dataset_ver is not None else 'v1'
    pkl_dir = os.path.join(root_dir, 'cash', version, f'{genre}_dataset_pkl')
    ensure_dir_exists(pkl_dir)

    _train_max_frames = None if backbone == 'i3d' else 16
    train_giaa_dataset = Image_GIAA_HistogramDataset(
        root_dir, transform=train_transform, genre=genre, backbone=backbone,
        data=train_giaa_raw.data,
        map_file=os.path.join(pkl_dir, 'trainset_image_dct.pkl'),
        precompute_file=os.path.join(pkl_dir, 'trainset_GIAA_dct.pkl'),
        max_frames=_train_max_frames, is_train=True,
    )
    val_giaa_dataset = Image_GIAA_HistogramDataset(
        root_dir, transform=test_transform, genre=genre, backbone=backbone,
        data=val_giaa_raw.data,
        map_file=os.path.join(pkl_dir, 'valset_image_dct.pkl'),
        precompute_file=os.path.join(pkl_dir, 'valset_GIAA_dct.pkl'),
        max_frames=None, is_train=False,
    )
    test_giaa_dataset = Image_GIAA_HistogramDataset(
        root_dir, transform=test_transform, genre=genre, backbone=backbone,
        data=test_giaa_raw.data,
        map_file=os.path.join(pkl_dir, 'testset_image_dct.pkl'),
        precompute_file=os.path.join(pkl_dir, 'testset_GIAA_dct.pkl'),
        max_frames=None, is_train=False,
    )

    print(f'[val_backbone] Train GIAA: {len(train_giaa_dataset)}, Val GIAA: {len(val_giaa_dataset)}, Test GIAA: {len(test_giaa_dataset)}')
    return train_giaa_dataset, val_giaa_dataset, test_giaa_dataset

class Image_GIAA_HistogramDataset(ImageDataset):
    def __init__(self, root_dir, transform=None, genre=None, backbone=None, data=None, map_file=None, precompute_file=None, max_frames=None, is_train=False,
                 global_trait_encoders=None, global_age_bins=None, maked_dir=None):
        super().__init__(root_dir, transform, genre, backbone=backbone, max_frames=max_frames, is_train=is_train,
                         global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins, maked_dir=maked_dir)
        if data is not None:
            self.data = data

        if map_file and os.path.exists(map_file):
            print('Loading image to indices map from file...')
            self.image_to_indices_map = self._load_map(map_file)
            self.unique_images = [img for img in self.image_to_indices_map.keys()]
        else:
            self.image_to_indices_map = dict()
            for image in tqdm(self.data['sample_file'].unique(), desc='Processing images'):
                indices_for_image = [i for i, img in enumerate(self.data['sample_file']) if img == image]
                if any(not idx < len(self.data) for idx in indices_for_image):
                    print(indices_for_image)
                    raise Exception('Index out of bounds for the data.')
                if len(indices_for_image) > 0:
                    self.image_to_indices_map[image] = indices_for_image

            self.unique_images = [img for img in self.image_to_indices_map.keys()]

            if map_file:
                print(f"Saving image to indices map to {map_file}")
                self._save_map(map_file)

        if precompute_file and os.path.exists(precompute_file):
            print(f'Loading precomputed data from {precompute_file}...')
            self.load(precompute_file)
        else:
            self.precompute_data()
            if precompute_file:
                print(f"Saving precomputed data to {precompute_file}")
                self.save(precompute_file)

    def precompute_data(self):
        self.precomputed_data = []
        for idx in tqdm(range(len(self)), desc='Precompute images'):
            self.precomputed_data.append(self._compute_item(idx))

    def _compute_item(self, idx):
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]

        max_response_score = 7

        accumulated_response = torch.zeros(max_response_score, dtype=torch.float32)

        for ai in associated_indices:
            row = self.data.iloc[ai]
            round_score = int(row['Aesthetic'])
            response_one_hot = F.one_hot(torch.tensor(round_score), num_classes=max_response_score)
            accumulated_response += response_one_hot

        accumulated_histogram = {'Aesthetic': accumulated_response}

        total_samples = len(associated_indices)
        accumulated_histogram['Aesthetic'] /= total_samples
        accumulated_histogram['n_samples'] = total_samples
        accumulated_histogram['sample_file'] = self.unique_images[idx]

        return accumulated_histogram

    def __getitem__(self, idx):
        # With cached features an item depends only on idx; GroupSplitData fills the memo up front.
        memo = getattr(self, 'memo', None)
        if memo is not None and idx in memo:
            return memo[idx]
        item_data = copy.deepcopy(self.precomputed_data[idx])
        img_sample = super().__getitem__(self.image_to_indices_map[self.unique_images[idx]][0])
        inherit_list = ['image', 'sample_file']
        for item in inherit_list:
            item_data[item] = img_sample[item]
        item_data['traits'] = torch.tensor([i for i in range(10)])
        item_data['QIP'] = torch.tensor([i for i in range(10)])

        if memo is not None:
            memo[idx] = item_data
        return item_data

    def _save_map(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.image_to_indices_map, f)

    def _load_map(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.image_to_indices_map)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.precomputed_data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.precomputed_data = pickle.load(f)

def collate_fn(batch):
    batch_dict = {
        'image': torch.stack([item['image'] for item in batch]),
        'Aesthetic': torch.stack([item['Aesthetic'] for item in batch]),
        'traits': torch.stack([item['traits'] for item in batch]),
        'QIP': torch.stack([item['QIP'] for item in batch]),
    }
    optional_keys = ['user_id']
    for k in optional_keys:
        if k in batch[0]:
            try:
                batch_dict[k] = torch.stack([item[k] for item in batch])
            except Exception:
                batch_dict[k] = [item[k] for item in batch]
    return batch_dict

class Image_PIAA_HistogramDataset(ImageDataset):
    def __init__(self, root_dir, transform=None, data=None, genre=None, backbone=None, max_frames=None, is_train=False,
                 global_trait_encoders=None, global_age_bins=None, maked_dir=None):
        super().__init__(root_dir, transform, genre=genre, backbone=backbone, max_frames=max_frames, is_train=is_train,
                         global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins, maked_dir=maked_dir)
        if data is not None:
            self.data = data

    def __getitem__(self, idx):
        max_response_score = 7

        # With cached features an item depends only on its row, so it is built once per dataset.
        # GroupSplitData fills the memo up front; loader workers inherit it, and copy.copy subsets share it.
        memo = getattr(self, 'memo', None)
        if memo is not None:
            row = self._row(idx)
            key = (row['user_id'], row['sample_file'])
            if key in memo:
                return memo[key]

        sample = super().__getitem__(idx)

        round_score = int(sample['Aesthetic'])
        accumulated_histogram = {
            'Aesthetic': round_score,
        }

        score_vecs = []
        trait_vecs = []
        for k, v in sample.items():
            if k.endswith('_onehot') and k.startswith('Q'):
                score_vecs.append(v.float())
            elif k.endswith('_onehot') and not k.startswith('Q'):
                trait_vecs.append(v.float())

        try:
            combined = torch.cat(score_vecs + trait_vecs) if (len(score_vecs) + len(trait_vecs)) > 0 else torch.tensor([])
        except Exception:
            combined = torch.tensor([], dtype=torch.float32)

        accumulated_histogram['traits'] = combined
        accumulated_histogram['n_samples'] = 1

        scalar = float(round_score)
        accumulated_histogram['Aesthetic'] = torch.tensor([scalar / (max_response_score - 1)], dtype=torch.float32)

        qip = sample.get('QIP', {})
        parts = []
        for k in sorted(qip.keys()):
            v = qip[k]
            if isinstance(v, torch.Tensor):
                parts.append(v.view(-1).float())
            else:
                parts.append(torch.tensor(v, dtype=torch.float32).view(-1))

        qip_tensor = torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)
        accumulated_histogram['QIP'] = qip_tensor

        inherit_list = ['image', 'sample_file', 'user_id']
        for item in inherit_list:
            if item in sample:
                accumulated_histogram[item] = sample[item]

        accumulated_histogram['genre'] = self.genre

        if memo is not None:
            memo[key] = accumulated_histogram
        return accumulated_histogram


GENRES = ['art', 'fashion', 'scenery']
FEATURE_BACKBONES = ('clip_rn50', 'clip_vit_b16')


def load_image_features(root_dir, backbone, genre, maked_dir=None, device=None, batch_size=64):
    """Frozen-backbone features of every stimulus of `genre`, keyed by sample_file.

    Computed once with the test transform (no augmentation) and cached under
    <root_dir>/cash/features/. PIAA models keep the backbone frozen, so feeding these
    features gives the same forward pass as the image without the augmentation.
    """
    path = os.path.join(root_dir, 'cash', 'features', f'{backbone}_{genre}.pt')
    if os.path.exists(path):
        return torch.load(path)
    from torch.amp import autocast
    from .train_common import NIMA, num_bins

    _, test_transform = get_transforms(backbone)
    base = ImageDataset(root_dir, transform=test_transform, genre=genre, backbone=backbone, maked_dir=maked_dir)
    files = sorted(base.data['sample_file'].unique())
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = NIMA(num_bins, backbone=backbone).backbone.to(device).eval()
    features = {}
    for i in tqdm(range(0, len(files), batch_size), desc=f'Features [{backbone} {genre}]', ncols=120):
        chunk = files[i:i + batch_size]
        images = []
        for f in chunk:
            name = f.replace('.mp4', '.jpg') if genre == 'scenery' else f
            sub = '' if genre == 'scenery' else genre
            images.append(test_transform(Image.open(os.path.join(base.samples_dir, sub, name)).convert('RGB')))
        with torch.no_grad(), autocast(device.type):
            out = net(torch.stack(images).to(device)).float().cpu()
        features.update({f: v.clone() for f, v in zip(chunk, out)})
    del net
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(features, path)
    return features


def load_group_split(split_dir, fold):
    """Read one fold of the group split written by src/make_split.py."""
    fold_dir = os.path.join(split_dir, f'fold{fold}')

    def _ids(name):
        with open(os.path.join(fold_dir, f'{name}.txt')) as f:
            return [int(x) for x in f.read().split()]

    split = {name: _ids(name) for name in ['train_users', 'val_users', 'test_users', 'giaa_train_images']}
    split['fine'] = pd.read_csv(os.path.join(split_dir, 'fine_samples.csv'))
    split['fold'] = fold
    # Cache key for precomputed GIAA histograms; changes whenever the split does.
    key = json.dumps([split['train_users'], split['val_users'], split['giaa_train_images']])
    split['tag'] = hashlib.md5(key.encode()).hexdigest()[:8]
    return split


class GroupSplitData:
    """Datasets of one genre for one fold of the group split.

    GIAA: per-image score histograms from train users (train) or val users (val).
    PIAA pre: individual ratings of train users (train) or val users (val).
    PIAA fine: each user's 'train' / 'eval' samples listed in fine_samples.csv.
    With args.feature_cache, every stage feeds cached frozen-backbone features (no augmentation).
    """

    def __init__(self, args, genre, split, global_trait_encoders=None, global_age_bins=None):
        self.args = args
        self.genre = genre
        self.split = split
        self.train_transform, self.test_transform = get_transforms(args.backbone)
        self.enc_kwargs = dict(global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins,
                               maked_dir=args.maked_dir)
        base = ImageDataset(args.root_dir, genre=genre, backbone=args.backbone, **self.enc_kwargs)
        self.data = base.data
        users_header = pd.read_csv(os.path.join(base.maked_dir, 'users.csv'), nrows=0).columns
        self.user_columns = [c for c in self.data.columns
                             if c == 'user_id' or c.removesuffix('_x').removesuffix('_y') in users_header]
        self.train_max_frames = None if args.backbone == 'i3d' else 16
        self.pkl_dir = os.path.join(args.root_dir, 'cash', f"group_{split['tag']}", f"fold{split['fold']}", genre)
        self._giaa = {}
        self._pre = {}
        self.features = None
        if getattr(args, 'feature_cache', False) and args.backbone in FEATURE_BACKBONES:
            self.features = load_image_features(args.root_dir, args.backbone, genre, maked_dir=args.maked_dir)

    def _rows(self, users):
        return self.data[self.data['user_id'].isin(users)]

    def giaa(self, role):
        if role not in self._giaa:
            is_train = role == 'train'
            rows = self._rows(self.split['train_users'] if is_train else self.split['val_users'])
            if is_train:
                rows = rows[rows['sample_id'].isin(self.split['giaa_train_images'])]
            ensure_dir_exists(self.pkl_dir)
            self._giaa[role] = Image_GIAA_HistogramDataset(
                self.args.root_dir, transform=self.train_transform if is_train else self.test_transform,
                genre=self.genre, backbone=self.args.backbone, data=rows.reset_index(drop=True),
                map_file=os.path.join(self.pkl_dir, f'giaa_{role}_map.pkl'),
                precompute_file=os.path.join(self.pkl_dir, f'giaa_{role}_hist.pkl'),
                max_frames=self.train_max_frames if is_train else None, is_train=is_train, **self.enc_kwargs)
            self._fill_features(self._giaa[role])
        return self._giaa[role]

    def _fill_features(self, dataset):
        """Feed cached frozen-backbone features in place of images (no augmentation) and build every item once."""
        if self.features is not None:
            dataset.features = self.features
            dataset.memo = {}
            for i in range(len(dataset)):
                dataset[i]

    def piaa(self, rows, is_train):
        dataset = Image_PIAA_HistogramDataset(
            self.args.root_dir, transform=self.train_transform if is_train else self.test_transform,
            data=rows.reset_index(drop=True), genre=self.genre, backbone=self.args.backbone,
            max_frames=self.train_max_frames if is_train else None, is_train=is_train, **self.enc_kwargs)
        self._fill_features(dataset)
        return dataset

    def pre(self, role):
        if role not in self._pre:
            is_train = role == 'train'
            users = self.split['train_users'] if is_train else self.split['val_users']
            self._pre[role] = self.piaa(self._rows(users), is_train)
        return self._pre[role]

    def fine(self, users, role):
        fine = self.split['fine']
        keys = fine[(fine['genre'] == self.genre) & (fine['role'] == role) & fine['user_id'].isin(users)]
        rows = self.data.merge(keys[['user_id', 'sample_id']], on=['user_id', 'sample_id'], how='inner')
        return self.piaa(rows, is_train=role == 'train')

    def unlabeled_personal(self, users):
        """Train-group images of this genre, each paired with every given user's traits.

        Unlabeled target data for fine-stage adaptation. The images come from the
        train groups only, so no val/test image is used for adaptation (R2-2).
        DA losses never read target labels; Aesthetic is zeroed to make that explicit.
        """
        images = self._rows(self.split['train_users']).drop_duplicates('sample_id')
        images = images.drop(columns=self.user_columns).assign(Aesthetic=0)
        people = self._rows(users).drop_duplicates('user_id')[self.user_columns]
        return self.piaa(images.merge(people, how='cross')[self.data.columns], is_train=True)

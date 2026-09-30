<p align="center">
  <img src="logo.png" alt="proj-xpass-logo" width="650">
</p>

## Overview

XPASS-Vis is the first large-scale dataset for cross-domain Personalized Image Aesthetic Assessment (PIAA). It collects 6,528 samples across three domains—artworks, fashion images, and scenery videos—and includes more than 98,000 user-item interactions from 150 annotators. This repository provides the development of General Image Aesthetic Assessment (GIAA) models and Personalized Image Aesthetic Assessment (PIAA) models for each domain, and proposes a novel approach to cross-domain PIAA.

<p align="center">
  <a href="https://arxiv.org/abs/2606.15629">
    <img src="https://img.shields.io/badge/arXiv-Paper-555.svg?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=b31b1b" alt="Paper">
  </a>
  <a href="https://drive.google.com/drive/folders/1zjPbqPoOegC88-0EhBqgoXuTcAykQkYx?usp=drive_link">
    <img src="https://img.shields.io/badge/Google%20Drive-Dataset-555.svg?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=4285F4" alt="Dataset on Google Drive">
  </a>
</p>

> 📩 **Requesting the dataset:** If you would like to use the dataset, please contact **[hayashi0884@jaist.ac.jp](mailto:hayashi0884@jaist.ac.jp)**.

> 🔒 **About dataset access:** The dataset zip files are password-protected. Obtaining the password requires submitting an application form, but **the application form is currently being prepared (coming soon).** In the meantime, please contact [hayashi0884@jaist.ac.jp](mailto:hayashi0884@jaist.ac.jp).

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Training (GIAA)](#giaa-general-image-aesthetic-assessment)
3. [Domain Adaptation Methods](#domain-adaptation-methods)
4. [Training (PIAA)](#piaa-pretraining--finetuning-ici--mir)
   - [Group-Split Protocol (Random Search & Model Selection)](#group-split-protocol-random-search--model-selection)
5. [Inference (Standalone)](#inference-standalone)
6. [LLM Zero-Shot Inference](#llm-zero-shot-inference)
7. [Analysis Tools](#analysis-tools) ([fold aggregation](#fold-result-aggregation-aggregate))
8. [Citation](#citation)
9. [License](#license)

---

## Environment Setup

### Creating and activating a virtual environment

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

> **Note:** All of the following commands must be run from the root directory of the repository (the folder containing `src/`). Running them from another directory may cause `ModuleNotFoundError: No module named 'src'`. Alternatively, you can set `PYTHONPATH` to the repository root: `export PYTHONPATH=$PWD`

### Data placement

Training and inference assume that preprocessed and split data exists under `data/`. Download the [dataset (Google Drive)](https://drive.google.com/drive/folders/1zjPbqPoOegC88-0EhBqgoXuTcAykQkYx?usp=drive_link) and extract it directly under the repository root with the following structure.

```
data/
├── maked/               # users.csv, ratings.csv, QIP_{genre}.csv
├── split/               # {dataset_ver}_fold*/{genre}/*.txt (train/val/test splits)
└── samples/             # stimulus data (images/videos)
    ├── art/             # art images
    ├── fashion/         # fashion images
    ├── scenery_image/   # scenery images
    └── scenery_video/   # scenery videos
```

> **Dataset requests:** To obtain access, contact [hayashi0884@jaist.ac.jp](mailto:hayashi0884@jaist.ac.jp).

> **Note:** The stimulus data folder is named `samples` (plural) (`src/data.py` references `{--root_dir}/samples/`). If you change `--root_dir`, place the data under it with the same structure.

---

## Training

### GIAA (General Image Aesthetic Assessment)

#### Optional arguments

| Argument | Type | Default | Description |
|------|------|------|------|
| `--genre` | str | (required) | Training genre (e.g., `art`, `fashion`, `scenery`) |
| `--dataset_ver` | str | `v1_all` | Data split version (references `data/split/<version>/`) |
| `--backbone` | str | `clip_vit_b16` | Backbone architecture (`resnet50` / `i3d` / `vit_b_16` / `clip_rn50` / `clip_vit_b16`) |
| `--use_video` | flag | `False` | Use video data for the scenery genre. Only when combined with `--backbone resnet50` does it automatically switch to the I3D backbone. If not specified, image data is used (the backbone follows the `--backbone` setting) |
| `--root_dir` | str | `data` | Root directory of image/video data |
| `--num_epochs` | int | `200` | Maximum number of epochs |
| `--batch_size` | int | `32` | Batch size |
| `--lr` | float | `1e-5` | Learning rate |
| `--lr_decay_factor` | float | `0.5` | ReduceLROnPlateau decay rate (factor) |
| `--lr_patience` | int | `5` | ReduceLROnPlateau patience (number of epochs tolerated without improvement) |
| `--max_patience_epochs` | int | `10` | Early stopping patience epochs |
| `--dropout` | float | `0.1` | Dropout rate (applied to the intermediate layers of `fc_aesthetic`) |
| `--num_workers` | int | `4` | Number of DataLoader workers |
| `--no_log` | flag | `False` | Disable wandb logging |
| `--wandb_project` | str | `XPASS` | wandb project name |
| `--da_method` | str | `None` | Specify a domain adaptation method and target domain. Format: `METHOD-target` (e.g., `DANN-fashion`, `DJDOT-scenery`). Omitting it means no domain adaptation |
| `--eval_target` | str | `None` | Evaluate the target genre during source-only training (e.g., `fashion`). Records the target val EMD without domain adaptation |
| `--da_schedule_epochs` | int | `50` | `[DANN]` λ schedule: number of epochs until λ reaches ~1.0. Internally converted to `total_steps = da_schedule_epochs × (data_size / batch_size)` |
| `--da_gamma` | float | `10.0` | `[DANN]` λ schedule: sharpness of the sigmoid (Ganin et al.) |
| `--djdot_alpha` | float | `0.1` | `[DJDOT]` Weight of the feature alignment term (L2 feature distance) |
| `--djdot_lambda_t` | float | `0.1` | `[DJDOT]` Weight of the label alignment term (EMD label cost) |
| `--jumbot_eta1` | float | `0.1` | `[JUMBOT]` Weight of the feature distance term (L2^2) in the OT cost matrix. Matched to DeepJDOT's `djdot_alpha` |
| `--jumbot_eta2` | float | `0.1` | `[JUMBOT]` Weight of the label cost term (GIAA: EMD) in the OT cost matrix. Matched to DeepJDOT's `djdot_lambda_t` |
| `--jumbot_eta3` | float | `1.0` | `[JUMBOT]` Scale of the transfer loss `<pi, C>` added to the source task loss. Set to 1.0 so that `eta3*eta1` and `eta3*eta2` match DeepJDOT's effective weights |
| `--jumbot_tau` | float | `0.5` | `[JUMBOT]` Marginal KL penalty (`reg_m`) of Unbalanced OT. Smaller values relax the marginal constraint |
| `--jumbot_epsilon` | float | `0.1` | `[JUMBOT]` Entropy regularization (`reg`) of Sinkhorn Unbalanced OT |
| `--coral_lambda` | float | `1.0` | `[DEEPCORAL]` Fixed weight of the CORAL alignment loss (no schedule) |
| `--cdan_sigma` | float | `1.0` | `[CDAN]` Width of the Gaussian Soft Ordinal Distribution used for the conditioning vector g (fixed) |
| `--alda_sigma` | float | `1.0` | `[ALDA]` Width of the Gaussian Soft Ordinal Distribution used for p_t in PIAA L_T (fixed). GIAA uses softmax(logit_tgt) directly and ignores this flag |
| `--alda_threshold` | float | `0.2` | `[ALDA]` Confidence threshold δ for target pseudo-labels (filtering of L_T) |

> **Note:** Cross-domain evaluation (evaluation on all genres other than `--genre`) is always performed.

#### Example commands

```bash
# No domain adaptation
python -m src.train_GIAA --genre art

# With domain adaptation (switch METHOD via --da_method METHOD-target; example is DANN)
python -m src.train_GIAA --genre art --da_method DANN-fashion

# scenery (image: CLIP ViT-B/16)
python -m src.train_GIAA --genre scenery

# scenery (video: I3D)
python -m src.train_GIAA --genre scenery --use_video
```

> For available `METHOD` values, refer to the table in [Domain Adaptation Methods](#domain-adaptation-methods) (`DANN` / `DJDOT` / `JUMBOT` / `DEEPCORAL` / `CDAN` / `ALDA`, etc.).

---

## Domain Adaptation Methods

In GIAA training (`train_GIAA`), domain adaptation can be enabled by specifying `--da_method METHOD-target`. The currently supported methods are as follows.

| Method | `--da_method` example | Description | Supported |
|------|---------------------|------|------|
| **DANN** | `DANN-fashion` | Adversarial training of a domain discriminator using a Gradient Reversal Layer | GIAA / PIAA |
| **DeepJDOT** | `DJDOT-fashion` | Joint distribution alignment via Optimal Transport (OT). Uses feature distance and EMD label cost in the cost matrix | GIAA / PIAA |
| **JUMBOT** | `JUMBOT-fashion` | An Unbalanced minibatch OT version of DeepJDOT. Replaces exact OT (`ot.emd`) with unbalanced entropic OT (`ot.sinkhorn_unbalanced`) | GIAA / PIAA |
| **DeepCORAL** | `DEEPCORAL-fashion` | Aligns distributions by minimizing the difference in second-order statistics (covariance matrices) of source and target features | GIAA / PIAA |
| **CDAN** | `CDAN-fashion` | Adversarial training with a domain discriminator conditioned on the multilinear combination of features and class predictions | GIAA / PIAA |
| **ALDA** | `ALDA-fashion` | Adversarial-Learned Loss that corrects the target pseudo-label loss (L_T) using a confusion matrix learned via adversarial training | GIAA / PIAA |
| **DARE-GRAM** | `DAREGRAM-fashion` | Aligns the source and target feature spaces at the geometric-structure level of linear regression by matching the angle (cosine similarity) and scale (singular values) of Gram matrices | PIAA (ICI only) |
| **RSD** | `RSD-fashion` | Regression-oriented Representation Subspace Distance that aligns the principal angles of the source and target feature subspaces | PIAA (ICI only) |

For the formulation and theory of each method, please refer to the paper and the original publications. Below are practical usage notes specific to this repository.

### Usage notes

- **Specifying a method**: Pass `--da_method METHOD-target` (e.g., `DJDOT-fashion`). The relevant hyperparameters are listed in the argument tables above (per `[METHOD]` tag).
- **Save path**: Models trained with domain adaptation are saved to `models_pth/{dataset_ver}/{source}2{target}/` (e.g., `models_pth/v_giaa/art2fashion/`).
- **DARE-GRAM / RSD (ICI only)**: These have no GIAA mode, so the NIMA backbone used in PIAA pretrain is borrowed from another method via `--nima_da_method` (`source_only` loads from `models_pth/{ver}/{genre}/`; `DANN`, etc. load from `models_pth/{ver}/{src2tgt}/`).

### Examples

```bash
# DeepJDOT: art → fashion (GIAA)
python -m src.train_GIAA --genre art --da_method DJDOT-fashion \
  --dataset_ver v_giaa --djdot_alpha 0.1 --djdot_lambda_t 0.1

# JUMBOT: art → fashion (GIAA)
python -m src.train_GIAA --genre art --da_method JUMBOT-fashion \
  --dataset_ver v_giaa --jumbot_eta1 0.1 --jumbot_eta2 0.1 \
  --jumbot_eta3 1.0 --jumbot_tau 0.5 --jumbot_epsilon 0.1

# DARE-GRAM: art → fashion (PIAA, ICI only)
python -m src.train_PIAA --genre art --dataset_ver v2_all \
  --piaa_mode PIAA_pretrain --da_method DAREGRAM-fashion \
  --nima_da_method source_only \
  --daregram_alpha_cos 0.01 --daregram_gamma_scale 0.01 --daregram_T 0.95
python -m src.train_PIAA --genre art --dataset_ver v2_all \
  --piaa_mode PIAA_finetune --da_method DAREGRAM-fashion \
  --daregram_alpha_cos 0.01 --daregram_gamma_scale 0.01
```

---

### PIAA Pretraining & Finetuning (ICI / MIR)

#### Optional arguments

| Argument | Type | Default | Description |
|------|------|------|------|
| `--genre` | str | (required) | Training genre (e.g., `art`, `fashion`, `scenery`) |
| `--dataset_ver` | str | `v1_all` | Data split version (references `data/split/<version>/`) |
| `--piaa_mode` | str | `PIAA_pretrain` | PIAA mode (`PIAA_pretrain` / `PIAA_finetune`) |
| `--model_type` | str | `ICI` | PIAA model architecture (`ICI`: interaction-based / `MIR`: MLP Interaction Regression) |
| `--backbone` | str | `clip_vit_b16` | Backbone architecture (`resnet50` / `i3d` / `vit_b_16` / `clip_rn50` / `clip_vit_b16`) |
| `--use_video` | flag | `False` | Use video data for the scenery genre. Only when combined with `--backbone resnet50` does it automatically switch to the I3D backbone. If not specified, image data is used (the backbone follows the `--backbone` setting) |
| `--root_dir` | str | `data` | Root directory of image/video data |
| `--num_epochs` | int | `200` | Maximum number of epochs |
| `--batch_size` | int | `32` (pretrain) / `8` (finetune) | Batch size (auto-set according to `--piaa_mode` when `--batch_size` is unspecified) |
| `--lr` | float | `5e-6` (pretrain) / `1e-6` (finetune) | Learning rate (auto-set according to `--piaa_mode` when `--lr` is unspecified) |
| `--lr_decay_factor` | float | `0.5` | ReduceLROnPlateau decay rate (factor) |
| `--lr_patience` | int | `5` | ReduceLROnPlateau patience (number of epochs tolerated without improvement) |
| `--max_patience_epochs` | int | `10` | Early stopping patience epochs |
| `--dropout` | float | `0.1` | Dropout rate (applied to the intermediate layers of all MLPs) |
| `--num_workers` | int | `4` | Number of DataLoader workers |
| `--start_fold` | int | `1` | Fold number to resume from (1-indexed). Used when `--dataset_ver` ends with `_all` |
| `--no_log` | flag | `False` | Disable wandb logging |
| `--wandb_project` | str | `XPASS` | wandb project name |
| `--no_save_model` | flag | `False` | Do not save the model to disk; keep the best model in memory |
| `--da_method` | str | `None` | Specify a domain adaptation method and target domain. Format: `METHOD-target` (e.g., `DANN-fashion`, `DJDOT-scenery`). Omitting it means no domain adaptation |
| `--eval_target` | str | `None` | Evaluate the target genre during source-only training (e.g., `fashion`). Records the target val EMD without domain adaptation |
| `--da_schedule_epochs` | int | `50` | `[DANN]` λ schedule: number of epochs until λ reaches ~1.0. Internally converted to `total_steps = da_schedule_epochs × (data_size / batch_size)` |
| `--da_gamma` | float | `10.0` | `[DANN]` λ schedule: sharpness of the sigmoid (Ganin et al.) |
| `--djdot_alpha` | float | `0.1` | `[DJDOT]` Weight of the feature alignment term (L2 feature distance) |
| `--djdot_lambda_t` | float | `1` | `[DJDOT]` Weight of the label alignment term (EMD label cost) |
| `--jumbot_eta1` | float | `0.1` | `[JUMBOT]` Weight of the feature distance term (L2^2) in the OT cost matrix. Matched to DeepJDOT's `djdot_alpha` (PIAA) |
| `--jumbot_eta2` | float | `1.0` | `[JUMBOT]` Weight of the label cost term (PIAA: squared error) in the OT cost matrix. Matched to DeepJDOT's `djdot_lambda_t` (PIAA) |
| `--jumbot_eta3` | float | `1.0` | `[JUMBOT]` Scale of the transfer loss `<pi, C>` (shared with GIAA) |
| `--jumbot_tau` | float | `0.5` | `[JUMBOT]` Marginal KL penalty (`reg_m`) of Unbalanced OT. Smaller values relax the marginal constraint |
| `--jumbot_epsilon` | float | `0.1` | `[JUMBOT]` Entropy regularization (`reg`) of Sinkhorn Unbalanced OT |
| `--coral_lambda` | float | `1.0` | `[DEEPCORAL]` Fixed weight of the CORAL alignment loss (no schedule) |
| `--cdan_sigma` | float | `1.0` | `[CDAN]` Width of the Gaussian Soft Ordinal Distribution used for the conditioning vector g (fixed) |
| `--alda_sigma` | float | `1.0` | `[ALDA]` Width of the Gaussian Soft Ordinal Distribution used for p_t in PIAA L_T (fixed) |
| `--alda_threshold` | float | `0.2` | `[ALDA]` Confidence threshold δ for target pseudo-labels (filtering of L_T) |
| `--daregram_alpha_cos` | float | `0.01` | `[DAREGRAM]` Weight of the angle alignment loss L_cos |
| `--daregram_gamma_scale` | float | `0.01` | `[DAREGRAM]` Weight of the scale alignment loss L_scale |
| `--daregram_T` | float | `0.95` | `[DAREGRAM]` Cumulative singular value threshold of the truncated pseudo-inverse |
| `--rsd_beta` | float | `0.01` | `[RSD]` Weight β of the RSD loss (sum of sin of principal angles) |
| `--rsd_gamma` | float | `1e-5` | `[RSD]` Weight γ of the BMP loss (basis mismatch penalty) |
| `--rsd_eps` | float | `1e-8` | `[RSD]` Numerical stabilization term ε for `sqrt(1 - cos²θ)` |
| `--nima_da_method` | str | `None` | `[DAREGRAM/RSD]` DA method of the NIMA backbone loaded in PIAA_pretrain (`source_only` / `DANN` / `DJDOT` / `DEEPCORAL` / `CDAN` / `ALDA`). Required for methods that have no GIAA training |

> **Note:** Cross-domain evaluation (evaluation on all genres other than `--genre`) is always performed. The loss function is fixed to MSE.

#### Example commands

```bash
# Pretrain
python -m src.train_PIAA --genre art --dataset_ver v2_all \
  --piaa_mode PIAA_pretrain --batch_size 128

# Finetune
python -m src.train_PIAA --genre art --dataset_ver v2_all \
  --piaa_mode PIAA_finetune --batch_size 16

# Finetune: scenery (video: I3D)
python -m src.train_PIAA --genre scenery --dataset_ver v2_all \
  --use_video --piaa_mode PIAA_finetune --batch_size 16

# To use the MIR model, add --model_type MIR (for both Pretrain / Finetune)
python -m src.train_PIAA --genre art --dataset_ver v2_all \
  --model_type MIR --piaa_mode PIAA_pretrain --batch_size 128
```

---

## Group-Split Protocol (Random Search & Model Selection)

`src/sweep.py` tunes every method (Source-Only, Target-Only and the UDA methods) with the same procedure and the same number of trials, on a 10-group / 5-fold user split.

### Split

`python -m src.make_split --out_dir asset/split` builds the split from `asset/maked/ratings.csv`. It uses 10 groups, one per rating session, and each group has its own annotators and stimuli. Each fold tests on 2 groups, validates on 1 randomly drawn group, and trains on the remaining 7. Users and images never overlap across train, val and test. `fine_samples.csv` fixes 120 fine-tuning and 50 evaluation samples per user and domain.

### Procedure

The stages run in order. Each stage starts from the configuration selected in the previous stage.

| Stage | Trained on | Selected by |
|---|---|---|
| GIAA | train users' score histograms | EMD on val users' images |
| PIAA pre | train users' ratings | MSE on val users' ratings |
| PIAA fine | each val user's 120 source samples | mean per-user SCC on their 50 eval samples |

The test users are then fine-tuned with the selected configuration and scored on their target-domain eval samples.

Two selection criteria are used (DomainBed):
- `train_domain`: selects on the val users' source domain. This criterion gives the main results.
- `oracle`: selects on the val users' target domain. This criterion is for reference only.

Every trial trains for a fixed 20 epochs in every stage and is scored once, at its final checkpoint. There is no early stopping, and no validation during training. Unlabeled target data comes from the train groups only. At the fine stage, the train-group target images are paired with the fine-tuned user's own traits. Search ranges, epochs and the per-stage batch sizes are defined in `src/search_space.py`. Each stage runs 20 trials. Trial 0 uses the values adopted by the original papers. Hyperparameters that are not searched are fixed to the original papers' values.

### Example commands

```bash
# UDA method, one direction (both criteria)
python -m src.sweep --fold 0 --method DANN --source art --target fashion --n_trials 20

# Source-Only for every target of one source; Target-Only for one domain
python -m src.sweep --fold 0 --method SourceOnly --source art --n_trials 20
python -m src.sweep --fold 0 --method TargetOnly --target fashion --n_trials 20
```

- **Outputs:** everything except checkpoints goes under `--out_dir` (default `output/r16`):

  | Folder | Contents |
  |---|---|
  | `results/fold{k}/` | one JSON record per trial; `{ICI\|MIR}/final/` holds the selected configurations and the test SCC |
  | `predictions/fold{k}/{ICI\|MIR}/` | per-sample predictions for the test users (CSV: `user_id, sample_id, sample_file, genre, true, pred`) |
  | `logs/fold{k}/` | the console output of each command; progress bars are left out |

- **Resuming and caching:** results are cached per trial. Rerunning a command resumes where it stopped. Runs that share a stage reuse its trials; for example, DARE-GRAM and RSD reuse Source-Only's GIAA stage.
- **Checkpoints:** only the best checkpoint per domain is kept under `--models_dir` (default `models_pth/r16`).
- **Data location:** `--root_dir` (default `data`) must contain `samples/`.
- **Feature cache:** the backbone is frozen in PIAA, so the PIAA stages feed cached CLIP features instead of images. The features are computed once per domain with the test transform (no augmentation) and saved to `<root_dir>/cash/features/`. Pass `--no_feature_cache` to feed images instead.
- **Partial runs:** `--stop_after {giaa,pre,fine}` stops the chain after that stage, without the test run or the final record. For example, `python -m src.sweep --fold 0 --method TargetOnly --target art --model_type ICI --stop_after pre`.

---

## Inference (Standalone)

Using a trained model, you can run inference in bulk over all folds under `models_pth/`. By simply specifying `genre` and `pattern` (a glob pattern against `.pth` filenames), it automatically detects the corresponding folds, runs inference on the test set, and saves result JSON.

- Filenames containing `NIMA` → inference in **GIAA** mode
- Filenames ending with `_pretrain.pth` → inference in **PIAA_pretrain** mode
- If the result JSON (`reports/exp/{fold}/{genre}/*.json`) already exists, it is skipped (force re-run with `--force`)

```bash
# Inference of GIAA (NIMA) models (all folds)
python -m src.inference --genre art --pattern "*NIMA*"

# Inference of PIAA_pretrain (ICI) models (all folds)
python -m src.inference --genre fashion --pattern "*_pretrain*"

# Force re-run even if result JSON exists (--pattern can also filter by experiment name)
python -m src.inference --genre art --pattern "*NIMA*" --force
```

#### Optional arguments

| Argument | Type | Default | Description |
|------|------|------|------|
| `--genre` | str | (required) | Target genre for inference (`art` / `fashion` / `scenery`) |
| `--pattern` | str | (required) | Glob pattern against `.pth` filenames (e.g., `"*NIMA*"`, `"*_pretrain*"`) |
| `--root_dir` | str | `data` | Root directory of image/video data |
| `--backbone` | str | `clip_vit_b16` | Backbone used for model initialization (must match the one used during training) |
| `--batch_size` | int | `16` | Batch size for inference |
| `--num_workers` | int | `4` | Number of DataLoader workers |
| `--dropout` | float | `0.1` | Dropout rate matching the model structure |
| `--model_type` | str | `None` | Filter pretrain files by model type (`ICI` / `MIR`). If omitted, all matching files are run |
| `--force` | flag | `False` | Re-run even if the result JSON already exists |

> **Note:** Cross-domain evaluation is always performed. Evaluation is automatically run on the test sets of all genres other than `--genre`.

---

## Analysis Tools

### Fold result aggregation (aggregate)

In cross-validation experiments, it aggregates the JSON files of each fold and outputs metrics (PIAA: SROCC / NDCG@10 / MAE / CCC, GIAA: EMD / SROCC / MAE / CCC) as fold averages and all-user averages. If the inference JSON contains **cross-domain evaluation results** for target domains other than `--genre`, those are also automatically aggregated and displayed.

It supports the following four modes (switched via `--pattern` and `--giaa_mode`):

| Mode | How to switch | Input JSON | Output metrics |
|--------|-----------|----------|----------|
| **PIAA / NN** (default) | `--pattern finetune`, etc. | `per_user_metrics` / `cross_domain_metrics` of `reports/exp/{version}_fold*/{genre}/*.json` | SROCC / NDCG@10 / MAE / CCC |
| **GIAA / NN** | `--giaa_mode` | `average_metrics` / `cross_domain_metrics` of the same (only JSON with `mode: "GIAA"`) | EMD / SROCC / MAE / CCC |
| **PIAA / LLM** | `--pattern claude \| gemini \| gpt` | `reports/exp/{model}/{genre}_piaa_results*.json` (per-user `ratings`). If absent, falls back to zero-shot from `pred_dist` of `{genre}_results*.json` | SROCC / NDCG@10 / MAE / CCC |
| **GIAA / LLM** | `--pattern claude\|gemini\|gpt --giaa_mode` | Evaluate `reports/exp/{model}/{genre}_giaa_results*.json` (or `{genre}_results*.json`) against `test_images_GIAA.txt` with per-image average GT | EMD / SROCC / MAE / CCC |

`--genre` can also take the form `art2fashion` (transfer-domain fold) or `art-scenery` (multiple sub-genres).

Below are representative examples for each of the four modes (fold/run ID filtering can be specified via `--folds` / `--ids` / `--min-id` / `--max-id`, and transfer folds via `--genre art2fashion`).

```bash
# PIAA/NN: Aggregate ICI finetune results from all folds of v3
python src/analysis.py aggregate \
  --version v3 --genre art --pattern finetune --method ICI

# GIAA/NN: Aggregate average_metrics of GIAA JSON (cross-domain also auto-output)
python src/analysis.py aggregate \
  --version v_giaa --genre art --pattern NIMA --giaa_mode

# PIAA/LLM: Evaluate Claude's per-user ratings under the same protocol as other PIAA methods
python src/analysis.py aggregate \
  --version v3 --genre art --pattern claude

# GIAA/LLM: Evaluate GPT's GIAA (pred_dist) against test_images_GIAA.txt
python src/analysis.py aggregate \
  --version v_giaa --genre art --pattern gpt --giaa_mode
```

#### Optional arguments

| Argument | Type | Default | Description |
|------|------|------|------|
| `--version` | str | (required) | Dataset version (e.g., `v3`) — searches `v3_fold*` directories |
| `--genre` | str | (required) | Genre to analyze. `art` / `scenery` / `art2fashion` (transfer fold) / `art-scenery` (multiple sub-genres) are also allowed |
| `--pattern` | str | `""` | Glob pattern to filter JSON files (e.g., `pretrain`, `finetune`). Specifying `claude` / `gemini` / `gpt` switches to LLM mode |
| `--method` | str | none | Method name to further filter JSON files (e.g., `ICI`, `MIR`, `NIMA`) |
| `--folds` | list | none | Fold indices to aggregate (e.g., `--folds 0 2 4`). If omitted, all folds |
| `--ids` | list | none | Explicitly specify run IDs to aggregate (e.g., `--ids 61 65 70`). Only files with the specified IDs are targeted |
| `--min-id` | int | none | Lower bound of run IDs to aggregate (e.g., `61` → only `name-61_*.json` and later) |
| `--max-id` | int | none | Upper bound of run IDs to aggregate (e.g., `80` → only `name-80_*.json` and earlier) |
| `--giaa_mode` | flag | `False` | Aggregate `average_metrics` of GIAA JSON (`mode: "GIAA"`). For LLM, evaluate `{genre}_giaa_results.json` against `test_images_GIAA.txt` |
| `--reports_dir` | str | `reports/exp` | Search directory for JSON files |
| `--data_dir` | str | `<project_root>/data` | Data directory containing `split/` and `maked/` (used in LLM mode) |

---

## LLM Zero-Shot Inference

This is a GIAA / PIAA baseline implementation that uses a multimodal LLM as a zero-shot evaluator without using a trained model. It supports three models—Claude, GPT, and Gemini—and switches between GIAA and PIAA via `--mode`. All of them run sequential inference over **all images** (or all pairs) in the genre folder without train/test split, saving a checkpoint every 100 items.

### Per-model configuration

| Model | Module | Model used | Required environment variable | Output directory |
|--------|-----------|-----------|--------------|----------------|
| Claude | `src.methods.claude` | `claude-opus-4-6` | `ANTHROPIC_API_KEY` | `reports/exp/claude/` |
| GPT    | `src.methods.gpt`    | `gpt-5.4`          | `OPENAI_API_KEY`    | `reports/exp/gpt/`    |
| Gemini | `src.methods.gemini` | `gemini-3-flash-preview` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `reports/exp/gemini/` |

### Common prompt design

All models use the same system and user prompts.

**System prompt (GIAA)**
> "You are a researcher specializing in empirical aesthetics, skilled at predicting how general audiences perceive and rate visual content."

**User prompt (GIAA, art example)**
> "Imagine approximately 13 ordinary people with no special training in art or photography are shown the art image below and asked to rate its aesthetic quality. … Predict the distribution of their ratings as a probability distribution over scores 1 through 7. … Respond only with a valid JSON array of exactly 7 floats …"

- The label switches to `art image` / `fashion image` / `landscape image` depending on the genre

**System prompt (PIAA)**
> "You are a researcher specializing in empirical aesthetics, skilled at predicting how a specific individual perceives and rates visual content based on their psychological profile and demographic background."

**User prompt (PIAA)**: Embeds an individual profile including age, gender, education, nationality, domain learning experience, interests, and Big Five personality questions (Q1–Q10), and asks for an answer as an integer score from 1 to 7

---

### GIAA mode

For each image, it predicts a 7-class rating distribution (probability vector) in JSON format. If output parsing fails, it falls back to a uniform distribution (1/7 × 7).

#### Example commands

```bash
# Evaluate all art images (for GPT/Gemini, change to src.methods.{gpt,gemini})
python -m src.methods.claude --mode giaa --genre art

# Resume from the middle (--resume) / try only the first 10 images (--trial)
python -m src.methods.gemini --mode giaa --genre art --resume
python -m src.methods.claude --mode giaa --genre art --trial 10
```

#### Output

`reports/exp/{model}/{genre}_results.json` (for GPT/Gemini, `{genre}_results_sequential.json`)

```json
{
  "genre": "art",
  "model": "claude-opus-4-6",
  "timestamp": "2026-04-16 12:00:00",
  "n_total_images": 2345,
  "per_sample": [
    { "sample_file": "foo.jpg", "pred_dist": [0.02, 0.05, 0.10, 0.20, 0.35, 0.20, 0.08] },
    ...
  ]
}
```

- `pred_dist` is the predicted probability distribution corresponding to scores 1–7 (7 elements, summing to 1.0)

---

### PIAA mode

Based on `ratings.csv`, it predicts a scalar score (integer from 1 to 7) for all **(image, user) pairs** within the genre. If parsing fails, it falls back to 4 (the median). All models support `--resume`.

#### Example commands

```bash
# Evaluate all art pairs (for GPT/Gemini, change to src.methods.{gpt,gemini})
python -m src.methods.claude --mode piaa --genre art

# Resume from the middle (--resume) / try only the first 10 images (--trial)
python -m src.methods.claude --mode piaa --genre art --resume
python -m src.methods.claude --mode piaa --genre art --trial 10
```

#### Output

`reports/exp/{model}/{genre}_piaa_results.json`

```json
{
  "genre": "art",
  "model": "claude-opus-4-6",
  "timestamp": "2026-04-16 12:00:00",
  "n_total_pairs": 30012,
  "per_sample": [
    {
      "sample_file": "foo.jpg",
      "ratings": [
        { "user_id": 3, "pred_score": 5 },
        ...
      ]
    },
    ...
  ]
}
```

---

### Optional arguments

| Argument | Type | Default | Description |
|------|----|-----------|------|
| `--mode` | str | (required) | Evaluation mode (`giaa` / `piaa`) |
| `--genre` | str | (required) | Evaluation genre (`art` / `fashion` / `scenery`) |
| `--trial` | int | `0` | Evaluate only the first N images (`0` = all) |
| `--resume` | flag | `False` | Resume from existing result JSON, skipping already-processed items (for GIAA, only Gemini is supported) |
| `--batch` | flag | `False` | **[Gemini only]** Run in batch API mode (enqueue and process as capacity frees up; avoids 503 and roughly halves cost). On interruption, re-attach with `--batch --resume` |

> **Note:** Running requires setting each model's API key in environment variables (or a `.env` file). `--batch` is only available for Gemini (`src.methods.gemini`).

---

## Citation

If you use this dataset or code, please cite the following paper.

```bibtex
@misc{hayashi2026xpassvisdatasetcrossdomainpersonalized,
      title={XPASS-Vis: A Dataset for Cross-Domain Personalized Image Aesthetic Assessment},
      author={Takato Hayashi and Hiroaki Takahara and Candy Olivia Mawalim and Hiromi Narimatsu and Akisato Kimura and Shiro Kumano and Shogo Okada},
      year={2026},
      eprint={2606.15629},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.15629},
}
```

---

## License

XPASS-Vis may be used **for non-commercial academic research purposes only**. See [LICENSE](LICENSE) for details.

- **Annotations, annotator attributes, metadata, and the source code in this repository:** [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Citation of this paper is required when used.
- **Stimulus images/video clips:** Inherit the license of each original source per domain.
  - Art (LAPIS): CC BY-NC-SA 4.0
  - Fashion (Clothing Co-Parsing): Apache License 2.0
  - Scenery (Sekai-Real-Walking-HQ): The original non-commercial research license

All annotator data was collected under written informed consent and anonymized before release. This research was approved by the research ethics committee of the authors' affiliated institution.

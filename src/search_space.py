"""Random-search space for the R1-6 protocol, fixed before any run.

Every method runs the same number of trials (20) per stage. Each stage (giaa /
pre / fine) draws its values independently. The design and its sources are in
note/hparam_proposal.html.

- Searched: the learning rate, plus the method's loss weights (whose scale no
  longer matches the originals once cross-entropy is replaced by EMD / MSE) and
  the parameters the original papers show to matter (ALDA delta, JUMBOT tau).
- Trial 0 uses the values adopted by the original papers (TRIAL0).
- Everything else is fixed to the original papers' values (FIXED); the soft
  label width sigma of CDAN / ALDA has no original counterpart and stays 1.0.
- The adversarial schedule follows the originals: gamma = 10 over the whole
  training run (da_schedule_epochs = num_epochs).

Spec forms:
    ('log', low, high)      log-uniform float
    ('uniform', low, high)  uniform float
A spec or trial-0 value may also be a {stage: ...} dict when it differs by stage.
"""
import hashlib
import math
import random

STAGES = ['giaa', 'pre', 'fine']

# Fixed per stage (not searched). Batch sizes follow the previous runs.
BATCH_SIZE = {'giaa': 32, 'pre': 128, 'fine': 16}
NUM_EPOCHS = {'giaa': 20, 'pre': 20, 'fine': 20}

COMMON = {'lr': ('log', 1e-7, 1e-2)}
COMMON_TRIAL0 = {'lr': 1e-5}  # the manuscript's AdamW setting; the originals use SGD

_ADV = {'da_weight': ('log', 1e-3, 10.0)}

METHOD = {
    'SourceOnly': {},
    'DANN': dict(_ADV),
    'CDAN': dict(_ADV),
    'ALDA': dict(_ADV, alda_threshold={'giaa': ('uniform', 0.0, 0.9),
                                       'pre': ('uniform', 0.0, 0.6),
                                       'fine': ('uniform', 0.0, 0.6)}),
    'DEEPCORAL': {'coral_lambda': ('log', 1e-1, 1e5)},
    'DJDOT': {'djdot_alpha': ('log', 1e-4, 10.0), 'djdot_lambda_t': ('log', 1e-5, 10.0)},
    'JUMBOT': {'jumbot_eta3': ('log', 1e-1, 1e2), 'jumbot_tau': ('log', 0.05, 5.0)},
    'DAREGRAM': {'daregram_alpha_cos': ('log', 1e-3, 1e2), 'daregram_gamma_scale': ('log', 1e-6, 1e-1)},
    'RSD': {'rsd_beta': ('log', 1e-5, 1e-1)},
}

TRIAL0 = {
    'SourceOnly': {},
    'DANN': {'da_weight': 1.0},                        # GRL coefficient capped at 1
    'CDAN': {'da_weight': 1.0},                        # "we fix lambda = 1"
    'ALDA': {'da_weight': 1.0,
             'alda_threshold': {'giaa': 0.9, 'pre': 0.6, 'fine': 0.6}},  # 0.9 (Office), 0.6 (digits)
    'DEEPCORAL': {'coral_lambda': 1.0},                # no value in the paper; DomainBed default
    'DJDOT': {'djdot_alpha': 1e-3, 'djdot_lambda_t': 1e-4},
    'JUMBOT': {'jumbot_eta3': 1.0, 'jumbot_tau': 0.5},  # Office-Home
    'DAREGRAM': {'daregram_alpha_cos': 0.05, 'daregram_gamma_scale': 1e-3},  # official code defaults
    'RSD': {'rsd_beta': 1e-3},                         # centre of Fig. 8(b)
}

FIXED = {
    'DANN': {'da_gamma': 10.0},
    'CDAN': {'da_gamma': 10.0, 'cdan_sigma': 1.0},
    'ALDA': {'da_gamma': 10.0, 'alda_sigma': 1.0, 'alda_reg_weight': 1.0},
    'JUMBOT': {'jumbot_eta1': 0.01, 'jumbot_eta2': 0.5, 'jumbot_epsilon': 0.01},  # Office-Home
    'DAREGRAM': {'daregram_T': 0.9},
    'RSD': {'rsd_eps': 1e-8},
}
SCHEDULED = {'DANN', 'CDAN', 'ALDA'}
RSD_GAMMA_RATIO = 1e-2  # BMP weight gamma = beta / 100, the ratio of the original Fig. 8(b)


def _for_stage(value, stage):
    return value[stage] if isinstance(value, dict) else value


def _draw(spec, rng):
    kind = spec[0]
    if kind == 'log':
        return float(math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2]))))
    if kind == 'uniform':
        return float(rng.uniform(spec[1], spec[2]))
    raise ValueError(f'Unknown spec {spec}')


def sample_hparams(method, stage, trial, seed=0):
    """Configuration `trial` of `method` at `stage`; identical across folds and directions."""
    space = dict(COMMON, **METHOD[method])
    if trial == 0:
        trial0 = dict(COMMON_TRIAL0, **TRIAL0[method])
        hp = {name: float(_for_stage(trial0[name], stage)) for name in sorted(space)}
    else:
        key = f'{seed}|{method}|{stage}|{trial}'
        rng = random.Random(int(hashlib.md5(key.encode()).hexdigest(), 16))
        hp = {name: _draw(_for_stage(spec, stage), rng) for name, spec in sorted(space.items())}
    hp.update(FIXED.get(method, {}))
    if method in SCHEDULED:
        hp['da_schedule_epochs'] = NUM_EPOCHS[stage]
    if method == 'RSD':
        hp['rsd_gamma'] = hp['rsd_beta'] * RSD_GAMMA_RATIO
    hp['batch_size'] = BATCH_SIZE[stage]
    hp['num_epochs'] = NUM_EPOCHS[stage]
    return hp

"""Random-search space for the R1-6 protocol, fixed before any run.

Every method samples the same number of configurations per stage from these
ranges. The common part applies to all methods; the method part adds that
method's own weights. Each stage (giaa / pre / fine) draws its values
independently.

PROVISIONAL (epochs fixed at 20): ranges are centred on the previous defaults and must be reviewed
before the final runs (the review response discloses them as a table).

Spec forms:
    ('log', low, high)      log-uniform float
    ('uniform', low, high)  uniform float
    ('choice', [values])    one of the values
"""
import hashlib
import math
import random

STAGES = ['giaa', 'pre', 'fine']

# Fixed per stage (not searched). Batch sizes follow the previous runs.
BATCH_SIZE = {'giaa': 32, 'pre': 128, 'fine': 16}
NUM_EPOCHS = {'giaa': 20, 'pre': 20, 'fine': 20}

COMMON = {
    'giaa': {'lr': ('log', 1e-6, 1e-4)},
    'pre':  {'lr': ('log', 1e-6, 1e-4)},
    'fine': {'lr': ('log', 1e-6, 1e-4)},
}

_SCHEDULE = {'da_gamma': ('log', 1.0, 30.0), 'da_schedule_epochs': ('choice', [10, 25, 50])}

METHOD = {
    'SourceOnly': {},
    'DANN': dict(_SCHEDULE),
    'CDAN': dict(_SCHEDULE, cdan_sigma=('log', 0.3, 3.0)),
    'ALDA': dict(_SCHEDULE, alda_sigma=('log', 0.3, 3.0), alda_threshold=('uniform', 0.0, 0.9)),
    'DEEPCORAL': {'coral_lambda': ('log', 0.1, 10.0)},
    'DJDOT': {'djdot_alpha': ('log', 1e-4, 1.0), 'djdot_lambda_t': ('log', 1e-5, 1.0)},
    'JUMBOT': {'jumbot_eta1': ('log', 1e-3, 1.0), 'jumbot_eta2': ('log', 1e-2, 10.0),
               'jumbot_eta3': ('log', 0.1, 10.0), 'jumbot_tau': ('log', 0.05, 5.0),
               'jumbot_epsilon': ('log', 0.01, 1.0)},
    'DAREGRAM': {'daregram_alpha_cos': ('log', 1e-3, 1.0), 'daregram_gamma_scale': ('log', 1e-3, 1.0),
                 'daregram_T': ('uniform', 0.9, 0.99)},
    'RSD': {'rsd_beta': ('log', 1e-3, 1.0), 'rsd_gamma': ('log', 1e-6, 1e-3)},
}


def _draw(spec, rng):
    kind = spec[0]
    if kind == 'log':
        return float(math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2]))))
    if kind == 'uniform':
        return float(rng.uniform(spec[1], spec[2]))
    if kind == 'choice':
        return rng.choice(spec[1])
    raise ValueError(f'Unknown spec {spec}')


def sample_hparams(method, stage, trial, seed=0):
    """Configuration `trial` of `method` at `stage`; identical across folds and directions."""
    key = f'{seed}|{method}|{stage}|{trial}'
    rng = random.Random(int(hashlib.md5(key.encode()).hexdigest(), 16))
    space = dict(COMMON[stage], **METHOD[method])
    hp = {name: _draw(spec, rng) for name, spec in sorted(space.items())}
    hp['batch_size'] = BATCH_SIZE[stage]
    hp['num_epochs'] = NUM_EPOCHS[stage]
    return hp

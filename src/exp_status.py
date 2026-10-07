"""Experiment status page for the R1-6 sweep (and the R2-1 swap control): what is done, running or not started, and on which GPU.

Reads only the files src.sweep writes (results/ JSON records and logs/), so it runs with the
system python and works on any machine that receives output/ (e.g. over Syncthing).

Each cell is one chain (fold, PIAA model, method, direction, criterion) with four stage bars:
G (GIAA), P (pre), F (fine) and T (test). A bar fills with the colour of the machine whose
records it holds; a running stage is striped. A stage is running when a log under logs/ was
written in the last --running_min minutes and its last stage marker names a trial whose record
does not exist yet.

A chain is queued (実行予約) when a launcher has registered it in <out_dir>/queue/ and it is neither done
nor running. Launchers write one file per queue, `queue/<origin>/<name>.txt`, with one line per job
    fold model_type method source target criteria stop
(target `*` = every other domain, `-` = Target-Only on the source domain, criteria comma-separated, stop = --stop_after or `-`), and delete it
on exit. A line `# parallel N` gives the jobs the launcher runs at once (for the ETA). `origin` is `local` for queues of this machine; output/r16/pull_pod.sh mirrors the pod's
queue/local/ into queue/pod/.

Usage:
    python -m src.exp_status                       # writes note/exp_status.html
    python -m src.exp_status --watch 120           # regenerate every 2 min (the page reloads itself)
    python -m src.exp_status --pre_metric mse --html note/exp_status_pre-mse.html   # reference criterion
"""
import argparse
import glob
import html
import json
import math
import os
import re
import socket
import statistics
import time
from collections import Counter, defaultdict

# Mirrors src.sweep / src.data; kept literal so this script needs neither torch nor cv2.
GENRES = ['art', 'fashion', 'scenery']
DA_METHODS = ['DANN', 'DJDOT', 'JUMBOT', 'DEEPCORAL', 'CDAN', 'ALDA', 'DAREGRAM', 'RSD']
GIAA_FROM = {'DAREGRAM': 'DJDOT', 'RSD': 'DJDOT'}  # as in src.sweep
METHODS = ['SourceOnly'] + DA_METHODS
MODEL_TYPES = ['ICI', 'MIR']
FOLDS = range(5)
STAGES = ['giaa', 'pre', 'fine', 'test']
PRE_METRICS = {'mse': ('val_loss', True), 'scc': ('val_scc', False)}  # as src.sweep --pre_metric

# Known machines: hostname -> (label, colour). Unknown hosts get the next palette colour, labelled
# with their hostname and GPU. Records written before devices were recorded count as MAIN_HOST.
MAIN_HOST = 'hayashi0884-Z690-S01'
MACHINES = {MAIN_HOST: ('メイン機', '#2a78d6'), 'a156d1e4b161': ('クラウド3090', '#e07b00'),
            'eaa4b40fa4ec': ('クラウド3090-2', '#1a9a6c'), 'bc231cf2e2c5': ('クラウドL40S', '#b4489a')}
PALETTE = ['#e07b00', '#1a9a6c', '#b4489a', '#7a5af0', '#c0392b']
# Queue origin -> (host its jobs run on, jobs it runs at once). `local` is this machine, one job at a time
# (run_sourceonly_oracle.sh); the pod's run_da.sh / run_giaa_da.sh run 5 jobs through xargs -P 5.
# The two pods share one network volume; run_da_shared.sh's queue gives its total parallelism by `# parallel N`.
ORIGINS = {'pod': ('a156d1e4b161', 5)}
PARALLEL = {}  # (origin, queue name) -> jobs at once, from a queue file's `# parallel N` line (overrides ORIGINS)


def parse_cli():
    parser = argparse.ArgumentParser(description='Experiment status page for the R1-6 sweep')
    parser.add_argument('--out_dir', type=str, default='output/r16', help='The sweep --out_dir')
    parser.add_argument('--html', type=str, default='note/exp_status.html')
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--pre_metric', type=str, default='scc', choices=list(PRE_METRICS),
                        help='Pre-stage selection metric of the chains shown, as in src.sweep')
    parser.add_argument('--running_min', type=float, default=15, help='A log newer than this counts as live')
    parser.add_argument('--eta_hours', type=float, default=12,
                        help='Trial durations for the ETA come from records written in the last N hours')
    parser.add_argument('--watch', type=float, default=0, help='Regenerate every N seconds (0: once)')
    return parser.parse_args()


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _records(directory):
    out = [_read_json(p) for p in sorted(glob.glob(os.path.join(directory, 't[0-9][0-9][0-9].json')))]
    return [r for r in out if r]


def _score(record, metric, genre, minimize):
    value = (record.get(metric) or {}).get(genre)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.inf
    return value if minimize else -value


def _best(records, metric, genre, minimize):
    return min(records, key=lambda r: (_score(r, metric, genre, minimize), r['trial']))


def _host(record):
    return (record.get('device') or {}).get('host', MAIN_HOST)


class Machines:
    """Hostname -> label and colour, assigning palette colours to hosts seen for the first time."""

    def __init__(self):
        self.known = dict(MACHINES)
        self.gpus = {MAIN_HOST: {'RTX 3090'}}
        self.trials = Counter()
        self.seen = set()

    def see(self, uid, record):
        """Note a record; `uid` identifies it, since chains share records (e.g. one GIAA trial)."""
        host = _host(record)
        gpu = (record.get('device') or {}).get('gpu')
        if gpu:
            self.gpus.setdefault(host, set()).add(gpu.replace('NVIDIA ', '').replace('GeForce ', ''))
        if uid not in self.seen:
            self.seen.add(uid)
            self.trials[host] += 1
        self.colour(host)

    def colour(self, host):
        if host not in self.known:
            used = {c for _, c in self.known.values()}
            free = [c for c in PALETTE if c not in used] or PALETTE
            self.known[host] = (host, free[0])
        return self.known[host][1]

    def label(self, host):
        gpus = ', '.join(sorted(self.gpus.get(host, ())))
        name = self.known.get(host, (host, ''))[0]
        return f'{name} ({gpus})' if gpus else name


# ---------- queued chains, from the launchers' queue files ----------

def load_queue(out_dir):
    """Jobs registered by live launchers: list of (fold, model_type, method, src, tgt or '*', criteria, stop, origin, name)."""
    jobs = []
    for path in glob.glob(os.path.join(out_dir, 'queue', '*', '*.txt')):
        origin = os.path.basename(os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, errors='replace') as f:
            for line in f:
                parts = line.split()
                if parts[:2] == ['#', 'parallel'] and len(parts) == 3:
                    PARALLEL[(origin, name)] = int(parts[2])
                    continue
                if len(parts) != 7:
                    continue
                fold, model_type, method, src, tgt, criteria, stop = parts
                tgt = None if tgt == '-' else tgt  # '-': Target-Only (src.sweep --method TargetOnly --target src)
                jobs.append((int(fold), model_type, method, src, tgt, set(criteria.split(',')),
                             None if stop == '-' else stop, origin, name))
    return jobs


def queued_by(queue, fold, model_type, method, src, tgt, criterion, stages, n_trials):
    """The origin of the first queued job that still has work on this chain, or None."""
    for q_fold, q_mt, q_method, q_src, q_tgt, q_criteria, q_stop, origin, _ in queue:
        if (q_fold, q_method, q_src) != (fold, method, src) or criterion not in q_criteria:
            continue
        if q_tgt != '*' and q_tgt != tgt:
            continue
        if q_stop is None and q_mt != model_type:
            continue
        if q_stop is not None and stages[q_stop]['n'] >= n_trials:
            continue  # the job stops at a stage this chain has already finished
        return origin
    return None


# ---------- running stages, from fresh logs ----------

MARKER = re.compile(r'^\[(giaa|pre|fine|test)\] (\S+)(?: on (\S+))?(?: fine)? trial (\d+)')


def live_markers(out_dir, running_min):
    """(fold, model_type, stage, key, base, trial) -> host, for each log written in the last `running_min` min."""
    live = {}
    now = time.time()
    for path in glob.glob(os.path.join(out_dir, 'logs', 'fold*', '*.log')):
        if now - os.path.getmtime(path) > running_min * 60:
            continue
        fold = int(re.search(r'fold(\d+)', path).group(1))
        model_type, host, marker = 'ICI', MAIN_HOST, None
        with open(path, errors='replace') as f:
            for line in f:
                if line.startswith('$ python'):
                    m = re.search(r'--model_type (\S+)', line)
                    model_type = m.group(1) if m else 'ICI'
                elif line.startswith('Device: '):
                    host = line[len('Device: '):].split(' / ')[0].strip()
                else:
                    m = MARKER.match(line)
                    if m:
                        marker = m.groups()
        if marker:
            stage, key, base, trial = marker
            live[(fold, model_type if stage != 'giaa' else None, stage, key, base, int(trial))] = host
    return live


# ---------- chain status ----------

def chain(out_dir, fold, model_type, method, src, tgt, criterion, n_trials, pre_metric, live, machines, queue):
    """Stage-by-stage status of one chain, following src.sweep.Sweep.run_chain's selections."""
    out = _chain(out_dir, fold, model_type, method, src, tgt, criterion, n_trials, pre_metric, live, machines)
    out['queued'] = None if out['test'] else queued_by(queue, fold, model_type, method, src, tgt, criterion,
                                                       out['stages'], n_trials)
    return out


def _chain(out_dir, fold, model_type, method, src, tgt, criterion, n_trials, pre_metric, live, machines):
    sel = src if criterion == 'train_domain' else tgt
    key = f'SourceOnly_{src}' if method == 'SourceOnly' else f'{method}_{src}2{tgt}'
    giaa_key = f'{GIAA_FROM[method]}_{src}2{tgt}' if method in GIAA_FROM else key
    res = os.path.join(out_dir, 'results', f'fold{fold}')
    stages = {s: {'n': 0, 'hosts': Counter(), 'running': None, 'id': None} for s in STAGES}

    def collect(stage, records, live_key):
        info = stages[stage]
        info['n'] = len(records)
        info['id'] = live_key
        for r in records:
            info['hosts'][_host(r)] += 1
            machines.see((fold, live_key, r['trial']), r)
        done = {r['trial'] for r in records}
        for t in range(n_trials):
            if t not in done and (*live_key, t) in live:
                info['running'] = live[(*live_key, t)]
                break
        return len(records) >= n_trials

    out = {'stages': stages, 'selected': {}, 'test': None}
    if not collect('giaa', _records(os.path.join(res, 'giaa', giaa_key)), (fold, None, 'giaa', giaa_key, None)):
        return out
    giaa = _best(_records(os.path.join(res, 'giaa', giaa_key)), 'val_loss', sel, True)
    base = f"{giaa_key}-t{giaa['trial']:03d}"
    out['selected']['giaa'] = giaa['trial']

    pre_records = _records(os.path.join(res, model_type, 'pre', key, base))
    if not collect('pre', pre_records, (fold, model_type, 'pre', key, base)):
        return out
    field, minimize = PRE_METRICS[pre_metric]
    pre = _best(pre_records, field, sel, minimize)
    pre_id = f"{base}-pre-t{pre['trial']:03d}"
    out['selected']['pre'] = pre['trial']

    fine_records = _records(os.path.join(res, model_type, 'fine', key, pre_id))
    if not collect('fine', fine_records, (fold, model_type, 'fine', key, pre_id)):
        return out
    fine = _best(fine_records, 'val_scc', sel, False)
    out['selected']['fine'] = fine['trial']

    test = _read_json(os.path.join(res, model_type, 'test', key, pre_id, f"fine-t{fine['trial']:03d}.json"))
    info = stages['test']
    info['id'] = (fold, model_type, 'test', key, pre_id, fine['trial'])
    if test:
        info['n'] = n_trials  # drawn as a full bar
        info['hosts'][_host(test)] += 1
        machines.see((fold, model_type, 'test', key, pre_id), test)
        target = tgt or src
        out['test'] = {'scc': test['test_scc'].get(target), 'ccc': test['test_ccc'].get(target)}
    elif (fold, model_type, 'test', key, pre_id, fine['trial']) in live:
        info['running'] = live[(fold, model_type, 'test', key, pre_id, fine['trial'])]
    return out


def chain_state(c, n_trials):
    stages = c['stages']
    if c['test']:
        return 'done'
    if any(s['running'] for s in stages.values()):
        return 'running'
    if c['queued']:
        return 'queued'
    return 'todo'


# ---------- ETA of the queued work ----------

def stage_durations(out_dir, hours):
    """(host, model, stage) -> median seconds per trial, from records written in the last `hours` hours.

    `model` is the PIAA model (ICI, MIR), or None for GIAA, since the PIAA models differ greatly in speed.

    A trial's duration is the gap between consecutive records in one directory (one job writes them in
    order, so the gap includes the slowdown from jobs running beside it); a test's duration is the gap
    between its fine stage's last record and the test record.
    """
    since = time.time() - hours * 3600
    gaps = defaultdict(list)
    res = os.path.join(out_dir, 'results')
    by_dir = defaultdict(list)
    for pattern in ['fold*/giaa/*/t[0-9][0-9][0-9].json', 'fold*/*/pre/*/*/t[0-9][0-9][0-9].json',
                    'fold*/*/fine/*/*/t[0-9][0-9][0-9].json']:
        for path in glob.glob(os.path.join(res, pattern)):
            by_dir[os.path.dirname(path)].append(os.path.getmtime(path))
    last_fine = {}
    for directory, mtimes in by_dir.items():
        mtimes.sort()
        if mtimes[-1] < since:
            continue
        parts = directory.split(os.sep)
        stage, model = ('giaa', None) if f'{os.sep}giaa{os.sep}' in directory else (parts[-3], parts[-4])
        if stage == 'fine':
            last_fine[directory] = mtimes[-1]
        record = _read_json(glob.glob(os.path.join(directory, 't[0-9][0-9][0-9].json'))[-1])
        host = _host(record) if record else MAIN_HOST
        gaps[(host, model, stage)] += [b - a for a, b in zip(mtimes, mtimes[1:]) if a >= since]
    for path in glob.glob(os.path.join(res, 'fold*/*/test/*/*/fine-t[0-9][0-9][0-9].json')):
        end = os.path.getmtime(path)
        fine_dir = os.path.dirname(path).replace(f'{os.sep}test{os.sep}', f'{os.sep}fine{os.sep}')
        if end >= since and fine_dir in last_fine and end > last_fine[fine_dir]:
            record = _read_json(path)
            model = path.split(os.sep)[-5]
            gaps[(_host(record) if record else MAIN_HOST, model, 'test')].append(end - last_fine[fine_dir])
    return {k: statistics.median(v) for k, v in gaps.items() if v}


def _duration(durations, host, model, stage):
    """Seconds per trial of `stage` of `model` on `host`, falling back to the median over other hosts; None if unknown."""
    model = None if stage == 'giaa' else model
    if (host, model, stage) in durations:
        return durations[(host, model, stage)]
    others = [v for (h, m, s), v in durations.items() if (m, s) == (model, stage)]
    return statistics.median(others) if others else None


def queue_etas(cli, queue, durations):
    """One entry per launcher queue: its remaining trials per stage and the estimated finishing time."""
    machines = Machines()  # throwaway: chain() registers hosts, which must not affect the page's counts
    local_host = socket.gethostname()
    by_name = defaultdict(list)
    for job in queue:
        by_name[(job[7], job[8])].append(job)
    out = []
    for (origin, name), jobs in sorted(by_name.items()):
        host, parallel = ORIGINS.get(origin, (local_host, 1))
        parallel = PARALLEL.get((origin, name), parallel)
        units, chains = {}, 0
        for fold, model_type, method, src, tgt, criteria, stop, _, _ in jobs:
            targets = [t for t in GENRES if t != src] if tgt == '*' else [tgt]
            stages = STAGES[:STAGES.index(stop) + 1] if stop else STAGES
            for criterion in sorted(criteria):
                for t in targets:
                    c = _chain(cli.out_dir, fold, model_type, method, src, t, criterion, cli.n_trials,
                               cli.pre_metric, {}, machines)
                    if c['test'] or (stop and c['stages'][stop]['n'] >= cli.n_trials):
                        continue
                    chains += 1
                    for stage in stages:
                        s = c['stages'][stage]
                        left = (0 if c['test'] else 1) if stage == 'test' else cli.n_trials - s['n']
                        if left > 0:
                            # Stages not reached yet have no id; they count once per chain.
                            units[s['id'] or (fold, model_type, method, src, t, criterion, stage)] = (model_type, stage, left)
        left, unknown, seconds = Counter(), set(), 0.0
        for model, stage, n in units.values():
            left[stage] += n
            d = _duration(durations, host, model, stage)
            if d is None:
                unknown.add(stage)
            else:
                seconds += n * d
        seconds /= parallel
        out.append({'name': name, 'origin': origin, 'host': host, 'parallel': parallel, 'chains': chains,
                    'left': left, 'seconds': seconds, 'unknown': unknown})
    return out


def _hm(seconds):
    minutes = int(round(seconds / 60))
    return f'{minutes // 60}時間{minutes % 60:02d}分' if minutes >= 60 else f'{minutes}分'


def eta_html(etas, durations, machines, n_trials):
    now = time.time()
    hosts = sorted({h for h, _, _ in durations}, key=lambda h: h != MAIN_HOST)
    speed = ' ・ '.join(
        f"{html.escape(machines.label(h))} {m or 'GIAA'}: " + ' / '.join(
            f"{STAGE_LABEL[s]} {durations[(h, m, s)] / 60:.1f}分" for s in STAGES if (h, m, s) in durations)
        for h in hosts for m in [None] + MODEL_TYPES if any((h, m, s) in durations for s in STAGES))
    if not etas:
        return (f'<section><h2>完走予想</h2><p class="note">実行予約・実行中のキューはありません。'
                f'1 trial の所要時間: {speed or "記録なし"}</p></section>')
    rows = []
    finish = {}
    for e in sorted(etas, key=lambda e: e['seconds']):
        end = now + e['seconds']
        finish[e['host']] = max(finish.get(e['host'], 0), end)
        left = ' '.join(f"{STAGE_LABEL[s]}{e['left'][s]}" for s in STAGES if e['left'][s])
        warn = f" <small>（{'/'.join(STAGE_LABEL[s] for s in STAGES if s in e['unknown'])} の所要時間不明・除外）</small>" \
            if e['unknown'] else ''
        rows.append(f"<tr><td><i class=\"sw\" style=\"background:{machines.colour(e['host'])}\"></i>"
                    f"{html.escape(e['name'])}</td><td>{html.escape(machines.label(e['host']))}</td>"
                    f"<td class=\"num\">{e['parallel']}</td><td class=\"num\">{e['chains']}</td><td>{left or '–'}</td>"
                    f"<td class=\"num\">{_hm(e['seconds'])}</td>"
                    f"<td class=\"num\"><b>{time.strftime('%m/%d %H:%M', time.localtime(end))}</b>{warn}</td></tr>")
    overall = ' ・ '.join(f"{html.escape(machines.label(h))} <b>{time.strftime('%m/%d %H:%M', time.localtime(t))}</b>"
                          for h, t in sorted(finish.items(), key=lambda x: x[1]))
    return (f'<section><h2>完走予想</h2>'
            f'<p class="note">キューの残り trial 数 × マシン・段ごとの 1 trial 所要時間（直近の記録の中央値）÷ 並列数。'
            f'実行中の trial の経過分は差し引かず、選択前の段はチェーンごとに別 trial として数えるため、やや遅めに出ます。'
            f'失敗して終わったジョブも残りとして数えます。</p>'
            f'<div class="summary"><span>完走予想（マシン別）</span> {overall}</div>'
            f'<div class="scroll"><table class="eta"><thead><tr><th>キュー</th><th>マシン</th><th>並列</th>'
            f'<th>残りチェーン</th><th>残り trial（G/P/F/T）</th><th>残り時間</th><th>完走予想</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="note">1 trial の所要時間: {speed}（1 段 {n_trials} trial）</p></section>')


# ---------- R2-1 swap control (src.swap) ----------

SWAP_METHOD, SWAP_CRITERION = 'DJDOT', 'oracle'


def _swap_gaps(rec, genre):
    """Mean over rated users of native SCC - mean swap SCC, for same-set and same-fold partners (None -> 0)."""
    m, users, sets = rec['scc'][genre], rec['users'], rec['set']
    val = lambda a, b: m[str(a)][str(b)] or 0.0
    out = {}
    for partner in ['set', 'fold']:
        gaps = []
        for b in users:
            ps = [a for a in users if a != b and (partner == 'fold' or sets[str(a)] == sets[str(b)])]
            gaps.append(val(b, b) - sum(val(a, b) for a in ps) / len(ps))
        out[partner] = sum(gaps) / len(gaps)
    out['native'] = sum(val(b, b) for b in users) / len(users)
    return out


def _swap_launchers():
    """Model types a running run_swap_local.sh (or a chain that will start one) still has to run."""
    pending = set()
    for proc in glob.glob('/proc/[0-9]*'):
        try:
            with open(os.path.join(proc, 'cmdline'), 'rb') as f:
                cmd = f.read().replace(b'\0', b' ').decode(errors='replace')
            if 'run_swap_local.sh' not in cmd:
                continue
            pending |= set(re.findall(r'MODEL_TYPE=(\w+)', cmd))
            with open(os.path.join(proc, 'environ'), 'rb') as f:
                env = dict(kv.split('=', 1) for kv in f.read().decode(errors='replace').split('\0') if '=' in kv)
            pending.add(env.get('MODEL_TYPE', 'ICI'))
        except OSError:
            continue
    return pending


def swap_cell(cli, fold, model_type, src, tgt, pending, machines):
    base = os.path.join(cli.out_dir, 'swap', f'fold{fold}', model_type, f'{SWAP_METHOD}_{src}2{tgt}_{SWAP_CRITERION}.json')
    rec = _read_json(base)
    c = {'done': rec is not None, 'n': 0, 'total': 0, 'running': False, 'host': None, 'gaps': None, 'path': base}
    if rec:
        c['n'] = c['total'] = len(rec['users'])
        c['host'] = (rec.get('device') or {}).get('host')
        c['gaps'] = _swap_gaps(rec, tgt)
        c['diag'] = (rec.get('diag_check') or {}).get('max_abs_diff_scc')
        c['state'] = 'done'
        return c
    part = _read_json(base + '.partial')
    if part:
        c['n'], c['total'] = len(part['pred'][tgt]), len(part['users'])
        c['host'] = (part.get('device') or {}).get('host')
        c['running'] = time.time() - os.path.getmtime(base + '.partial') < cli.running_min * 60
    c['state'] = 'running' if c['running'] else 'queued' if model_type in pending else 'todo'
    return c


def swap_cell_html(c, machines):
    frac = c['n'] / c['total'] if c['total'] else 0.0
    colour = machines.colour(c['host']) if c['host'] else 'transparent'
    cls = 'bar running' if c['running'] else 'bar'
    bar = (f'<span class="{cls}" style="--c:{colour}"><span class="fill" style="width:{frac * 100:.0f}%"></span>'
           f'<b>{c["n"]}/{c["total"] or "–"}</b></span>')
    tips = [f"{STATE_LABEL[c['state']]}  {c['n']}/{c['total'] or '?'} 人"]
    if c['host']:
        tips.append(machines.label(c['host']))
    score = ''
    if c['gaps']:
        g = c['gaps']
        score = f'<span class="scc">{g["set"]:+.3f} / {g["fold"]:+.3f}</span>'
        tips.append(f"ターゲット SCC 本来 {g['native']:.4f}  本来−入れ替え: set {g['set']:+.4f} / fold {g['fold']:+.4f}")
        if c.get('diag') is not None:
            tips.append(f"対角の再現 max|dSCC| = {c['diag']:.1e}")
    title = html.escape('\n'.join(tips))
    return f'<td class="cell {c["state"]}" title="{title}"><div class="bars">{bar}</div>{score}</td>'


def swap_eta(cli, cells):
    """Remaining users x median seconds per user (finished jobs: log start -> record) / jobs at once."""
    per_user = []
    for c in cells.values():
        if not c['done']:
            continue
        name = os.path.basename(c['path'])[:-len('.json')]
        fold_dir = os.path.basename(os.path.dirname(os.path.dirname(c['path'])))
        logs = sorted(glob.glob(os.path.join(cli.out_dir, 'swap', 'logs', fold_dir, f'{name}_*.log')))
        if logs:
            start = time.mktime(time.strptime(logs[-1].rsplit('_', 1)[1][:-len('.log')], '%Y%m%d-%H%M%S'))
            seconds = os.path.getmtime(c['path']) - start
            if 0 < seconds < 6 * 3600:
                per_user.append(seconds / c['total'])
    left = sum((c['total'] or 26) - c['n'] for c in cells.values() if c['state'] in ('running', 'queued'))
    if not per_user or not left:
        return None, left
    return left * statistics.median(per_user) / 4, left


def swap_section(cli, machines):
    pending = _swap_launchers()
    directions = [(s, t) for s in GENRES for t in GENRES if s != t]
    cells = {(mt, f, s, t): swap_cell(cli, f, mt, s, t, pending, machines)
             for mt in MODEL_TYPES for f in FOLDS for s, t in directions}
    head = ''.join(f'<th colspan="5" class="mt">{mt}</th>' for mt in MODEL_TYPES)
    sub = ''.join(f'<th class="fold">f{f}</th>' for _ in MODEL_TYPES for f in FOLDS)
    body = []
    for i, (s, t) in enumerate(directions):
        first = f'<th class="method" rowspan="{len(directions)}">DeepJDOT</th>' if i == 0 else ''
        tds = ''.join(swap_cell_html(cells[mt, f, s, t], machines) for mt in MODEL_TYPES for f in FOLDS)
        body.append(f'<tr>{first}<th class="dir">{s} → {t}</th>{tds}</tr>')
    counts = Counter(c['state'] for c in cells.values())
    pills = ''.join(f'<span class="pill {st}">{STATE_LABEL[st]} <b>{counts.get(st, 0)}</b></span>'
                    for st in ['done', 'running', 'queued', 'todo'])
    means = []
    for mt in MODEL_TYPES:
        done = [c['gaps'] for (m, *_), c in cells.items() if m == mt and c['gaps']]
        if done:
            avg = lambda k: sum(g[k] for g in done) / len(done)
            means.append(f"{mt}（{len(done)}/30 ジョブ）: 本来 {avg('native'):.3f}、本来−入れ替え "
                         f"set <b>{avg('set'):+.3f}</b> / fold <b>{avg('fold'):+.3f}</b>")
    seconds, left = swap_eta(cli, cells)
    eta = (f"残り {left} 人ぶん ・ 完走予想 <b>{time.strftime('%m/%d %H:%M', time.localtime(time.time() + seconds))}</b>"
           f"（4 並列、終わったジョブの 1 人あたり所要時間の中央値から）") if seconds else ''
    return (f'<section><h2>R2-1 入れ替え統制 ・ DeepJDOT（oracle）</h2>'
            f'<p class="note">テストユーザーを選択済みの設定で fine-tune し直し、各ユーザーのモデル（属性もその人）で同じ fold の'
            f'全テストユーザーの評価画像を予測。バー = fine-tune を終えたユーザー数。数字 = ターゲット SCC の'
            f'「本来 − 入れ替え」を評定者で平均（左: 同じ set の相手、右: 同じ fold の相手）。</p>'
            f'<div class="summary">{pills}<span class="total">全 {len(cells)} ジョブ</span> {eta}</div>'
            + (f'<p class="note">{" ・ ".join(means)}（fold・方向の単純平均。検定は python -m src.swap summary）</p>' if means else '')
            + f'<div class="scroll"><table><thead><tr><th rowspan="2">手法</th><th rowspan="2">方向</th>{head}</tr>'
            f'<tr>{sub}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>')


# ---------- LLM baselines (src.methods.gpt / src.methods.qwen) ----------

# (results dir under reports/exp, label, where it runs, run_qwen.sh log pulled by output/r16/pull_qwen.sh)
LLM_RUNS = [('gpt', 'GPT-5.4', None, None),
            ('qwen3.8-27b-fp8', 'Qwen3.8-27B', 'bc231cf2e2c5', 'output/r16/qwen_l40s.log'),
            ('qwen3.5-9b', 'Qwen3.5-9B', 'bc231cf2e2c5', None)]
LLM_CELLS = [(task, shot, g) for task in ['giaa', 'piaa'] for shot in [0, 3] for g in GENRES]
# Labels of the rows in the scorers' outputs (src.giaa_backbone table, src.eval_llm).
LLM_SCORES = {'giaa': 'output/r16/giaa_backbone.json', 'piaa': 'output/r16/llm_table5.json'}


def _llm_log(path):
    """Runs started by run_qwen.sh: [(start epoch, task, genre, shot)], and whether it finished.
    The pod's clock is UTC."""
    import calendar
    starts, finished = [], False
    for line in open(path, errors='replace'):
        m = re.match(r'=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\S+)(?: (\w+) (\w+) (\d)-shot| all done)', line)
        if not m:
            continue
        t = calendar.timegm(time.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'))
        if m.group(3):
            starts.append((t, m.group(3), m.group(4), int(m.group(5))))
        else:
            finished = True
    return starts, finished


def _llm_scores(task, label):
    rec = _read_json(LLM_SCORES[task]) or {}
    if task == 'giaa':
        return {g: f"EMD {v['emd_mean']:.3f} / SCC {v['scc_mean']:.3f}" for g, v in rec.get(label, {}).items() if v}
    row = next((v for v in rec.values() if v.get('label') == label), {})  # src.eval_llm keys rows by name
    return {g: f"SCC {v['scc_mean']:.3f} / CCC {v['ccc_mean']:.3f}" for g, v in row.get('summary', {}).items()
            if g in GENRES}


def llm_section(cli, machines):
    totals = {}
    for task, shot, g in LLM_CELLS:  # GPT-5.4's query counts, the same for every model
        r = _read_json(os.path.join('reports/exp/gpt', f'{g}_{task}_{shot}shot.json'))
        totals[task, shot, g] = r['n_queries'] if r else None
    rows, etas = [], []
    for model_dir, label, host, log in LLM_RUNS:
        starts, finished = _llm_log(log) if log and os.path.exists(log) else ([], False)
        launched = bool(starts) and not finished
        cells, rate = {}, {}
        for i, (t0, task, g, shot) in enumerate(starts):  # seconds per query of each finished run
            end = starts[i + 1][0] if i + 1 < len(starts) else None
            if end and totals.get((task, shot, g)):
                rate.setdefault((task, shot), []).append((end - t0) / totals[task, shot, g])
        for task, shot, g in LLM_CELLS:
            path = os.path.join('reports/exp', model_dir, f'{g}_{task}_{shot}shot.json')
            r = _read_json(path)
            c = {'n': r['n_done'] if r else 0, 'total': (r or {}).get('n_queries') or totals[task, shot, g],
                 'fail': r['n_parse_failures'] if r else 0}
            done = r is not None and c['n'] == c['total']
            fresh = r is not None and time.time() - os.path.getmtime(path) < cli.running_min * 60
            current = launched and starts[-1][1:] == (task, g, shot)
            c['state'] = ('done' if done else 'running' if fresh or current else 'queued' if launched else 'todo')
            cells[task, shot, g] = c
        if launched:
            left = 0.0
            for (task, shot, g), c in cells.items():
                if c['state'] in ('running', 'queued'):
                    per = (rate.get((task, shot)) or rate.get(('giaa', shot)) or sum(rate.values(), []) or [None])
                    if per[0] is None:
                        left = None
                        break
                    left += (c['total'] - c['n']) * statistics.median(per)
            if left is not None:
                etas.append(f"{label}（{html.escape(machines.label(host))}）完走予想 "
                            f"<b>{time.strftime('%m/%d %H:%M', time.localtime(time.time() + left))}</b>")
        scores = {(task, shot): _llm_scores(task, f'{label} ({shot}-shot)') for task in ['giaa', 'piaa'] for shot in [0, 3]}
        rows.append((label, host, cells, scores))

    head = ''.join(f'<th colspan="3" class="mt">{task.upper()} {shot}-shot</th>' for task in ['giaa', 'piaa'] for shot in [0, 3])
    sub = ''.join(f'<th class="fold">{g}</th>' for _ in range(4) for g in GENRES)
    body, counts = [], Counter()
    for label, host, cells, scores in rows:
        if not any(c['n'] or c['state'] != 'todo' for c in cells.values()):
            continue
        where = html.escape(machines.label(host)) if host else 'OpenAI API'
        tds = []
        for key in LLM_CELLS:
            c, (task, shot, g) = cells[key], key
            counts[c['state']] += 1
            frac = c['n'] / c['total'] if c['total'] else 0.0
            colour = machines.colour(host) if host else '#888'
            cls = 'bar running' if c['state'] == 'running' else 'bar'
            tips = [f"{STATE_LABEL[c['state']]}  {c['n']}/{c['total'] or '?'} 件", where]
            if c['fail']:
                tips.append(f"パース失敗 {c['fail']} 件（一様分布 / 中点 4 として採点）")
            score = scores[task, shot].get(g)
            if score:
                tips.append(score)
            fail = f'<span class="scc">失敗 {c["fail"]}</span>' if c['fail'] else ''
            val = f'<span class="scc">{score.split(" / ")[1]}</span>' if score and c['state'] == 'done' else ''
            tds.append(f'<td class="cell {c["state"]}" title="{html.escape(chr(10).join(tips))}"><div class="bars">'
                       f'<span class="{cls}" style="--c:{colour}"><span class="fill" style="width:{frac * 100:.0f}%"></span>'
                       f'<b>{c["n"]}/{c["total"] or "–"}</b></span></div>{val}{fail}</td>')
        body.append(f'<tr><th class="method">{html.escape(label)}<br><small>{where}</small></th>{"".join(tds)}</tr>')
    pills = ''.join(f'<span class="pill {st}">{STATE_LABEL[st]} <b>{counts.get(st, 0)}</b></span>'
                    for st in ['done', 'running', 'queued', 'todo'])
    eta = ' ・ '.join(etas)
    return (f'<section><h2>LLM ベースライン ・ GPT-5.4 / Qwen（オープンウェイト）</h2>'
            f'<p class="note">同じクエリ（プロンプト・k-shot 例・224px 画像・JSON schema・temperature 0）を GPT-5.4 と、vLLM で'
            f'動かす Qwen（thinking オフ）に送る。バー = 回答済みクエリ数。数字 = 採点済みのスコア（GIAA: SCC、PIAA: CCC。'
            f'詳細はカーソル）。L40S の結果は output/r16/pull_qwen.sh が 5 分ごとに取得。</p>'
            f'<div class="summary">{pills}<span class="total">全 {sum(counts.values())} 本</span> {eta}</div>'
            f'<div class="scroll"><table><thead><tr><th rowspan="2">モデル</th>{head}</tr><tr>{sub}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>'
            f'<p class="note">完走予想: 残りクエリ数 × 終わった実行の 1 件あたり所要時間（同じタスク・shot、なければ同じ shot の GIAA）の中央値。</p>'
            f'</section>')


# ---------- html ----------

STATE_LABEL = {'done': '完了', 'running': '実行中', 'queued': '実行予約', 'todo': '未実行'}
QUEUE_LABEL = {'local': 'このマシン', 'pod': 'クラウド'}
STAGE_LABEL = {'giaa': 'G', 'pre': 'P', 'fine': 'F', 'test': 'T'}


def cell_html(c, n_trials, machines):
    bars, tips = [], []
    for stage in STAGES:
        s = c['stages'][stage]
        frac = min(s['n'] / n_trials, 1.0)
        host = s['running'] or (s['hosts'].most_common(1)[0][0] if s['hosts'] else None)
        colour = machines.colour(host) if host else 'transparent'
        cls = 'bar running' if s['running'] else 'bar'
        bars.append(f'<span class="{cls}" style="--c:{colour}"><span class="fill" style="width:{frac * 100:.0f}%">'
                    f'</span><b>{STAGE_LABEL[stage]}</b></span>')
        count = ('済' if c['test'] else '–') if stage == 'test' else f"{s['n']}/{n_trials}"
        where = ', '.join(machines.label(h) for h in s['hosts']) or '–'
        sel = c['selected'].get(stage)
        tip = f"{stage}: {count}  [{where}]"
        if sel is not None:
            tip += f'  選択 t{sel:03d}'
        if s['running']:
            tip += f"  ← 実行中 @ {machines.label(s['running'])}"
        tips.append(tip)
    state = chain_state(c, n_trials)
    score = f"<span class=\"scc\">{c['test']['scc']:.3f}</span>" if c['test'] and c['test']['scc'] is not None else ''
    if c['queued'] and state != 'running':
        tips.append(f"実行予約（{QUEUE_LABEL.get(c['queued'], c['queued'])}のキュー）")
    if c['test']:
        tips.append(f"test SCC {c['test']['scc']:.4f} / CCC {c['test']['ccc']:.4f}")
    title = html.escape('\n'.join(tips))
    return f'<td class="cell {state}" title="{title}"><div class="bars">{"".join(bars)}</div>{score}</td>'


def table_html(rows, n_trials, machines):
    """rows: list of (group, label, {(model_type, fold): chain})."""
    head = ''.join(f'<th colspan="5" class="mt">{mt}</th>' for mt in MODEL_TYPES)
    sub = ''.join(f'<th class="fold">f{f}</th>' for _ in MODEL_TYPES for f in FOLDS)
    body = []
    prev = None
    for group, label, chains in rows:
        span = sum(1 for g, _, _ in rows if g == group)
        first = f'<th class="method" rowspan="{span}">{html.escape(group)}</th>' if group != prev else ''
        prev = group
        cells = ''.join(cell_html(chains[(mt, f)], n_trials, machines) for mt in MODEL_TYPES for f in FOLDS)
        body.append(f'<tr>{first}<th class="dir">{html.escape(label)}</th>{cells}</tr>')
    return (f'<div class="scroll"><table><thead><tr><th rowspan="2">手法</th><th rowspan="2">方向</th>{head}</tr>'
            f'<tr>{sub}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def summary_html(rows, n_trials):
    counts = Counter(chain_state(c, n_trials) for _, _, chains in rows for c in chains.values())
    total = sum(counts.values())
    items = ''.join(f'<span class="pill {s}">{STATE_LABEL[s]} <b>{counts.get(s, 0)}</b></span>'
                    for s in ['done', 'running', 'queued', 'todo'])
    return f'<div class="summary">{items}<span class="total">全 {total} チェーン</span></div>'


def build(cli):
    machines = Machines()
    live = live_markers(cli.out_dir, cli.running_min)
    queue = load_queue(cli.out_dir)
    args = (cli.n_trials, cli.pre_metric, live, machines, queue)
    sections = []
    for criterion, title, note in [
            ('oracle', 'Cross-domain ・ oracle 基準（本文）', '選択: val ユーザーのターゲットドメイン'),
            ('train_domain', 'Cross-domain ・ train_domain 基準（付録）', '選択: val ユーザーのソースドメイン')]:
        rows = []
        for method in METHODS:
            for src in GENRES:
                for tgt in GENRES:
                    if src == tgt:
                        continue
                    chains = {(mt, f): chain(cli.out_dir, f, mt, method, src, tgt, criterion, *args)
                              for mt in MODEL_TYPES for f in FOLDS}
                    rows.append((method, f'{src} → {tgt}', chains))
        sections.append((title, note, rows))
    rows = [('TargetOnly', d, {(mt, f): chain(cli.out_dir, f, mt, 'SourceOnly', d, None, 'train_domain', *args)
                               for mt in MODEL_TYPES for f in FOLDS}) for d in GENRES]
    sections.append(('Within-domain ・ Target-Only', '学習・選択・評価とも同一ドメイン（基準の区別なし）', rows))

    durations = stage_durations(cli.out_dir, cli.eta_hours)
    body = eta_html(queue_etas(cli, queue, durations), durations, machines, cli.n_trials)
    body += llm_section(cli, machines)
    body += swap_section(cli, machines)
    body += ''.join(f'<section><h2>{html.escape(t)}</h2><p class="note">{html.escape(n)}</p>'
                   f'{summary_html(r, cli.n_trials)}{table_html(r, cli.n_trials, machines)}</section>'
                   for t, n, r in sections)
    legend = ''.join(f'<span class="key"><i style="background:{machines.colour(h)}"></i>{html.escape(machines.label(h))}'
                     f' <small>{machines.trials[h]} records</small></span>' for h in machines.known)
    refresh = f'<meta http-equiv="refresh" content="{int(cli.watch)}">' if cli.watch else ''
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    return TEMPLATE.format(refresh=refresh, stamp=stamp, legend=legend, body=body, n=cli.n_trials,
                           pre_metric=cli.pre_metric.upper(),
                           running_min=f'{cli.running_min:g}')


TEMPLATE = '''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}<title>R1-6 実験管理</title>
<style>
:root {{ --bg:#f7f7f5; --panel:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e2e0; --track:#ececea;
  --done:#e8f4ec; --run:#fff4e0; --queue:#e6effb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#16171a; --panel:#1f2024; --ink:#ececee;
  --muted:#9a9aa2; --line:#33343a; --track:#2c2d33; --done:#1c2e24; --run:#352a16; --queue:#1b2a40; }} }}
:root[data-theme="dark"] {{ --bg:#16171a; --panel:#1f2024; --ink:#ececee; --muted:#9a9aa2; --line:#33343a;
  --track:#2c2d33; --done:#1c2e24; --run:#352a16; --queue:#1b2a40; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--ink);
  font:13px/1.5 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif; }}
header {{ max-width:1400px; margin:0 auto 20px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:0 0 2px; }}
.meta, .note {{ color:var(--muted); margin:0 0 10px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; padding:10px 12px; background:var(--panel);
  border:1px solid var(--line); border-radius:8px; }}
.key {{ display:inline-flex; align-items:center; gap:6px; }} .key i {{ width:14px; height:14px; border-radius:3px; display:inline-block; }}
.key small {{ color:var(--muted); }}
.demo {{ display:inline-flex; gap:2px; vertical-align:middle; }}
section {{ max-width:1400px; margin:0 auto 28px; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; padding:16px; }}
.summary {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }}
.pill {{ padding:2px 10px; border-radius:999px; border:1px solid var(--line); }}
.pill.done {{ background:var(--done); }} .pill.running {{ background:var(--run); }} .pill.queued {{ background:var(--queue); }}
.total {{ color:var(--muted); margin-left:4px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:12px; }}
th, td {{ border:1px solid var(--line); padding:3px 5px; }}
thead th {{ background:var(--bg); position:sticky; top:0; }}
th.mt {{ font-size:13px; }} th.fold {{ font-weight:500; color:var(--muted); }}
th.method {{ text-align:left; vertical-align:top; background:var(--bg); white-space:nowrap; }}
th.dir {{ text-align:left; font-weight:500; white-space:nowrap; }}
td.cell {{ min-width:92px; cursor:default; }}
td.done {{ background:var(--done); }} td.running {{ background:var(--run); }} td.queued {{ background:var(--queue); }}
.bars {{ display:flex; gap:2px; }}
.bar {{ position:relative; flex:1; height:16px; min-width:18px; background:var(--track); border-radius:3px; overflow:hidden; }}
.bar .fill {{ position:absolute; inset:0 auto 0 0; background:var(--c); }}
.bar b {{ position:relative; display:block; text-align:center; font-size:10px; line-height:16px; color:var(--ink);
  mix-blend-mode:normal; text-shadow:0 0 3px var(--panel); }}
.bar.running {{ outline:2px solid var(--c); outline-offset:-2px; }}
.bar.running .fill {{ background:repeating-linear-gradient(45deg,var(--c) 0 5px,transparent 5px 10px);
  background-size:14px 14px; animation:slide 1s linear infinite; }}
@keyframes slide {{ from {{ background-position:0 0; }} to {{ background-position:14px 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .bar.running .fill {{ animation:none; }} }}
table.eta td {{ white-space:nowrap; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }}
.scc {{ display:block; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); font-size:11px; }}
</style></head><body>
<header>
<h1>R1-6 実験管理</h1>
<p class="meta">更新 {stamp} ・ pre の選択指標 {pre_metric} ・ 1 段 {n} trial ・ セルにカーソルを置くと詳細（件数・マシン・選択 trial・test SCC/CCC）</p>
<div class="legend"><b>マシン</b>{legend}
<span class="key"><span class="demo"><span class="bar" style="--c:#999;width:22px"><span class="fill" style="width:50%"></span><b>P</b></span></span>記録のある分だけ塗る</span>
<span class="key"><span class="demo"><span class="bar running" style="--c:#999;width:22px"><span class="fill" style="width:100%"></span><b>P</b></span></span>実行中（ログ更新 {running_min} 分以内）</span>
<span class="key"><i style="background:var(--queue);border:1px solid var(--line)"></i>実行予約（キューに登録済み・未実行）</span>
<span class="key">G=GIAA P=pre F=fine T=test ・ 数字=ターゲット test SCC</span></div>
</header>
{body}
</body></html>
'''


def main():
    cli = parse_cli()
    while True:
        page = build(cli)
        os.makedirs(os.path.dirname(cli.html) or '.', exist_ok=True)
        tmp = cli.html + '.tmp'
        with open(tmp, 'w') as f:
            f.write(page)
        os.replace(tmp, cli.html)
        print(f"{time.strftime('%H:%M:%S')} wrote {cli.html}")
        if not cli.watch:
            break
        time.sleep(cli.watch)


if __name__ == '__main__':
    main()

"""Experiment status page for the R1-6 sweep: what is done, running or not started, and on which GPU.

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
(target `*` = every other domain, criteria comma-separated, stop = --stop_after or `-`), and delete it
on exit. `origin` is `local` for queues of this machine; output/r16/pull_pod.sh mirrors the pod's
queue/local/ into queue/pod/.

Usage:
    python -m src.exp_status                       # writes note/exp_status.html
    python -m src.exp_status --watch 120           # regenerate every 2 min (the page reloads itself)
    python -m src.exp_status --pre_metric scc --html note/exp_status_pre-scc.html
"""
import argparse
import glob
import html
import json
import math
import os
import re
import time
from collections import Counter

# Mirrors src.sweep / src.data; kept literal so this script needs neither torch nor cv2.
GENRES = ['art', 'fashion', 'scenery']
DA_METHODS = ['DANN', 'DJDOT', 'JUMBOT', 'DEEPCORAL', 'CDAN', 'ALDA', 'DAREGRAM', 'RSD']
NO_GIAA = {'DAREGRAM', 'RSD'}
METHODS = ['SourceOnly'] + DA_METHODS
MODEL_TYPES = ['ICI', 'MIR']
FOLDS = range(5)
STAGES = ['giaa', 'pre', 'fine', 'test']
PRE_METRICS = {'mse': ('val_loss', True), 'scc': ('val_scc', False)}  # as src.sweep --pre_metric

# Known machines: hostname -> (label, colour). Unknown hosts get the next palette colour, labelled
# with their hostname and GPU. Records written before devices were recorded count as MAIN_HOST.
MAIN_HOST = 'hayashi0884-Z690-S01'
MACHINES = {MAIN_HOST: ('メイン機', '#2a78d6'), 'a156d1e4b161': ('クラウド3090', '#e07b00')}
PALETTE = ['#e07b00', '#1a9a6c', '#b4489a', '#7a5af0', '#c0392b']


def parse_cli():
    parser = argparse.ArgumentParser(description='Experiment status page for the R1-6 sweep')
    parser.add_argument('--out_dir', type=str, default='output/r16', help='The sweep --out_dir')
    parser.add_argument('--html', type=str, default='note/exp_status.html')
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--pre_metric', type=str, default='mse', choices=list(PRE_METRICS),
                        help='Pre-stage selection metric of the chains shown, as in src.sweep')
    parser.add_argument('--running_min', type=float, default=15, help='A log newer than this counts as live')
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
    """Jobs registered by live launchers: list of (fold, model_type, method, src, tgt or '*', criteria, stop, origin)."""
    jobs = []
    for path in glob.glob(os.path.join(out_dir, 'queue', '*', '*.txt')):
        origin = os.path.basename(os.path.dirname(path))
        with open(path, errors='replace') as f:
            for line in f:
                parts = line.split()
                if len(parts) != 7:
                    continue
                fold, model_type, method, src, tgt, criteria, stop = parts
                jobs.append((int(fold), model_type, method, src, tgt, set(criteria.split(',')),
                             None if stop == '-' else stop, origin))
    return jobs


def queued_by(queue, fold, model_type, method, src, tgt, criterion, stages, n_trials):
    """The origin of the first queued job that still has work on this chain, or None."""
    for q_fold, q_mt, q_method, q_src, q_tgt, q_criteria, q_stop, origin in queue:
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
    giaa_key = f'SourceOnly_{src}' if method in NO_GIAA or method == 'SourceOnly' else key
    res = os.path.join(out_dir, 'results', f'fold{fold}')
    stages = {s: {'n': 0, 'hosts': Counter(), 'running': None} for s in STAGES}

    def collect(stage, records, live_key):
        info = stages[stage]
        info['n'] = len(records)
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
    args = (cli.n_trials, cli.pre_metric, live, machines, load_queue(cli.out_dir))
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

    body = ''.join(f'<section><h2>{html.escape(t)}</h2><p class="note">{html.escape(n)}</p>'
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

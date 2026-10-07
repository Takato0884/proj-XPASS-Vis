import os
import json
import base64
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import numpy as np


_SAVE_DIR = 'reports/exp/gpt'

_SAMPLES_DIR_MAP = {
    'art':     'data/samples/art',
    'fashion': 'data/samples/fashion',
    'scenery': 'data/samples/scenery_image',
}

_GENRE_IMG_EXT = {'scenery': '.jpg'}

_MODEL = "gpt-5.4"


_SYSTEM_PROMPT = (
    "You are a researcher specializing in empirical aesthetics, skilled at predicting "
    "how general audiences perceive and rate visual content."
)

_GENRE_LABEL_EN = {
    'art':     'art image',
    'fashion': 'fashion image',
    'scenery': 'landscape image',
}

def _make_user_prompt(genre: str, n_shot: int = 0) -> str:
    label_en = _GENRE_LABEL_EN.get(genre, genre)
    shots = (
        f"The images before it are {n_shot} other {label_en}s from the same study, each followed by "
        f"the distribution of the ratings that ordinary people gave it.\n\n"
    ) if n_shot else ""
    return (
        f"Imagine approximately 13 ordinary people with no special training in art or photography "
        f"are shown the last {label_en} above and asked to rate its aesthetic quality.\n"
        f"{shots}"
        f"In the study, participants were asked the following question:\n"
        f"\"Overall, how aesthetic do you find this {label_en}?\"\n\n"
        f"Each person rates the {label_en} using the following 7-point scale:\n"
        f"- 1 = Highly unaesthetic\n"
        f"- 2 = Unaesthetic\n"
        f"- 3 = Slightly unaesthetic\n"
        f"- 4 = Neutral\n"
        f"- 5 = Slightly aesthetic\n"
        f"- 6 = Aesthetic\n"
        f"- 7 = Highly aesthetic\n\n"
        f"Predict the distribution of their ratings of the last {label_en} as a probability distribution "
        f"over scores 1 through 7. The 7 probabilities must sum to 1.0.\n\n"
        f"Respond only with a JSON object of the form {{\"distribution\": [p1, ..., p7]}}, where p1 to p7 "
        f"are the predicted proportions of raters giving each score from 1 to 7, in order, "
        f"rounded to 3 decimal places.\n"
        f"Example: {{\"distribution\": [0.020, 0.050, 0.100, 0.200, 0.350, 0.200, 0.080]}}"
    )


# GIAA queries use the same settings as PIAA (see below): no reasoning, temperature 0, images
# downscaled to 224x224, structured output.
_GIAA_MAX_TOKENS = 96
_GIAA_SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'rating_distribution',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {'distribution': {'type': 'array', 'items': {'type': 'number'},
                                            'minItems': 7, 'maxItems': 7}},
            'required': ['distribution'],
            'additionalProperties': False,
        },
    },
}


def _parse_giaa_dist(text: str):
    """The predicted distribution (normalized to sum 1, rounded to 3 decimals), or None when the
    response is not 7 non-negative numbers with a positive sum."""
    try:
        dist = np.array(json.loads(text)['distribution'], dtype=np.float64)
    except (ValueError, KeyError, TypeError):
        return None
    if dist.shape != (7,) or not np.isfinite(dist).all() or dist.min() < 0 or dist.sum() <= 0:
        return None
    return [round(float(x), 3) for x in dist / dist.sum()]


_PIAA_SYSTEM_PROMPT = (
    "You are a researcher specializing in empirical aesthetics, skilled at predicting "
    "how a specific individual perceives and rates visual content based on their "
    "psychological profile and demographic background."
)

_NATIONALITY_MAP = {'JPN': 'Japan', 'KOR': 'Korea', 'CHN': 'China'}

# PIAA queries GPT-5.4 without reasoning (temperature is only accepted with effort "none").
_PIAA_REASONING = 'none'
_PIAA_MAX_TOKENS = 32
_PIAA_IMAGE_SIZE = 224   # images are downscaled to fit 224x224, the input size of the CLIP backbone
_N_SHOT_BINS = 3         # k-shot examples: one per rating tertile of the user's fine-tuning samples
_PIAA_SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'rating',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {'score': {'type': 'number'}},
            'required': ['score'],
            'additionalProperties': False,
        },
    },
}


def _make_piaa_user_prompt(user: dict, n_shot: int = 0) -> str:
    nat = _NATIONALITY_MAP.get(user['nationality'], user['nationality'])
    shots = (
        f"The images before it are {n_shot} other images that this individual rated in the study, "
        f"each followed by the individual's rating.\n\n"
    ) if n_shot else ""
    return (
        f"A specific individual with the following profile is shown the last image above\n"
        f"and asked to rate its aesthetic quality.\n"
        f"{shots}"
        f"In the study, the participant was asked the following question:\n"
        f"\"Overall, how aesthetic do you find this image?\"\n\n"
        f"=== Individual Profile ===\n"
        f"Age            : {user['age']}\n"
        f"Gender         : {user['gender']}\n"
        f"Education      : {user['edu']}\n"
        f"Nationality    : {nat}\n\n"
        f"Domain training (0 = no formal training, 1 = formally trained):\n"
        f"  Art: {user['art_learn']},  Fashion: {user['fashion_learn']},  Photo/Video: {user['photoVideo_learn']}\n\n"
        f"Domain interest (1–7 scale, 1 = not interested at all, 7 = strongly interested):\n"
        f"  Art: {int(user['art_interest']) + 1},  Fashion: {int(user['fashion_interest']) + 1},  Photo/Video: {int(user['photoVideo_interest']) + 1}\n\n"
        f"Psychological questionnaire (1–7 scale, 1 = Disagree strongly, 7 = Agree strongly):\n"
        f"  Q1  (Extraverted, enthusiastic):          {int(user['Q1']) + 1}\n"
        f"  Q2  (Critical, quarrelsome):              {int(user['Q2']) + 1}\n"
        f"  Q3  (Dependable, self-disciplined):       {int(user['Q3']) + 1}\n"
        f"  Q4  (Anxious, easily upset):              {int(user['Q4']) + 1}\n"
        f"  Q5  (Open to new experiences, complex):   {int(user['Q5']) + 1}\n"
        f"  Q6  (Reserved, quiet):                    {int(user['Q6']) + 1}\n"
        f"  Q7  (Sympathetic, warm):                  {int(user['Q7']) + 1}\n"
        f"  Q8  (Disorganized, careless):             {int(user['Q8']) + 1}\n"
        f"  Q9  (Calm, emotionally stable):           {int(user['Q9']) + 1}\n"
        f"  Q10 (Conventional, uncreative):           {int(user['Q10']) + 1}\n\n"
        f"The individual rates the image using the following 7-point scale:\n"
        f"- 1 = Highly unaesthetic\n"
        f"- 2 = Unaesthetic\n"
        f"- 3 = Slightly unaesthetic\n"
        f"- 4 = Neutral\n"
        f"- 5 = Slightly aesthetic\n"
        f"- 6 = Aesthetic\n"
        f"- 7 = Highly aesthetic\n\n"
        f"Predict the individual's rating of the last image as a real number between 1.00 and 7.00 "
        f"with two decimal places (e.g., 4.35); values between two scale points express your "
        f"expected rating. Respond only with a JSON object of the form {{\"score\": <number>}}."
    )


def _parse_piaa_score(text: str):
    """The predicted rating rounded to 2 decimals, or None when the response is not a number in [1, 7]."""
    try:
        val = float(json.loads(text)['score'])
    except (ValueError, KeyError, TypeError):
        return None
    return round(val, 2) if 1.0 <= val <= 7.0 else None


def _encode_small_image(path: str) -> str:
    """JPEG of the image downscaled (aspect kept) to fit _PIAA_IMAGE_SIZE x _PIAA_IMAGE_SIZE, base64."""
    import io
    from PIL import Image

    img = Image.open(path).convert('RGB')
    img.thumbnail((_PIAA_IMAGE_SIZE, _PIAA_IMAGE_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode('utf-8')


def _select_shots(rows: 'pd.DataFrame', rating_col: str, seed: list, n_shot: int) -> list:
    """`n_shot` of `rows`, one drawn at random from each rating bin.

    The rows are ordered by `rating_col` (ties broken at random) and split into n_shot bins of equal
    size, so every bin is non-empty whatever the rating distribution. The examples are shown in
    random order. `seed` fixes the draw (per user and domain for PIAA, per fold and domain for GIAA).
    """
    if n_shot == 0:
        return []
    rng = np.random.default_rng(seed)
    rows = rows.iloc[rng.permutation(len(rows))]
    rows = rows.iloc[np.argsort(rows[rating_col].to_numpy(), kind='stable')]
    picks = [rows.iloc[b[rng.integers(len(b))]] for b in np.array_split(np.arange(len(rows)), n_shot)]
    return [picks[i] for i in rng.permutation(n_shot)]


_GENRES = ['art', 'fashion', 'scenery']


def _image_blocks(genre: str):
    """A function sample_file -> image content block (downscaled JPEG, encoded once per file)."""
    samples_dir = _SAMPLES_DIR_MAP[genre]
    img_ext = _GENRE_IMG_EXT.get(genre)
    images = {}

    def _image_block(fname):
        if fname not in images:
            img_fname = os.path.splitext(fname)[0] + img_ext if img_ext else fname
            images[fname] = _encode_small_image(os.path.join(samples_dir, img_fname))
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images[fname]}"}}
    return _image_block


def _load_ratings(genre: str, maked_dir: str) -> 'pd.DataFrame':
    import pandas as pd

    ratings = pd.read_csv(os.path.join(maked_dir, 'ratings.csv'))
    return ratings[ratings['genre'] == genre].drop_duplicates(['user_id', 'sample_file'], keep='first')


def _fold_ids(split_dir: str, fold: int, name: str) -> list:
    with open(os.path.join(split_dir, f'fold{fold}', f'{name}.txt')) as f:
        return [int(x) for x in f.read().split()]


def _giaa_jobs(genre: str, n_shot: int, trial: int, split_dir: str, maked_dir: str):
    """One job (fold, sample_file, content) per test image of `genre`, the images NIMA is scored on in
    src.giaa_backbone (every image the fold's test users rated; each image is a test image of exactly
    one fold), and the examples shown in each fold ({fold: [{sample_file, distribution}]}).

    The k-shot examples are fold's GIAA training images (rated by its train users, as NIMA is trained
    on), each with the train users' rating histogram; they are drawn by `_select_shots` on the mean
    rating, with seed [fold, genre index], and are the same for every test image of the fold.
    """
    ratings = _load_ratings(genre, maked_dir)
    image_block = _image_blocks(genre)
    prompt = _make_user_prompt(genre, n_shot)
    if trial > 0:
        print(f"[trial] mode ON — using the first {trial} test images of each fold")

    jobs, shots_by_fold, seen = [], {}, set()
    for fold in range(5):
        train = ratings[ratings['user_id'].isin(_fold_ids(split_dir, fold, 'train_users'))
                        & ratings['sample_id'].isin(_fold_ids(split_dir, fold, 'giaa_train_images'))]
        hist = (train.groupby('sample_file')['Aesthetic'].value_counts(normalize=True)
                .unstack(fill_value=0.0).reindex(columns=range(7), fill_value=0.0))
        hist['mean'] = hist[list(range(7))].to_numpy() @ np.arange(1, 8)
        hist['sample_file'] = hist.index
        shots = _select_shots(hist.reset_index(drop=True), 'mean', [fold, _GENRES.index(genre)], n_shot)
        shots_by_fold[fold] = [{'sample_file': s['sample_file'],
                                'distribution': [round(float(s[k]), 3) for k in range(7)]} for s in shots]
        content = []
        for s in shots_by_fold[fold]:
            content += [image_block(s['sample_file']),
                        {"type": "text", "text": "Distribution of the ratings given to this image (scores 1 to 7): "
                                                 + "[" + ", ".join(f"{p:.3f}" for p in s['distribution']) + "]"}]
        test = sorted(set(ratings[ratings['user_id'].isin(_fold_ids(split_dir, fold, 'test_users'))]['sample_file']))
        assert not seen & set(test), f'{genre}: fold{fold} test images overlap another fold'
        assert not set(test) & set(hist.index), f'{genre}: fold{fold} test images are also training images'
        seen |= set(test)
        if trial > 0:
            test = test[:trial]
        for fname in test:
            jobs.append((fold, fname, content + [image_block(fname), {"type": "text", "text": prompt}]))
    print(f"GIAA {genre} {n_shot}-shot: {len(jobs)} queries")
    return jobs, shots_by_fold


def _piaa_jobs(genre: str, n_shot: int, trial: int, split_dir: str, maked_dir: str):
    """One job (user_id, sample_file, content) per test user's 'eval' sample of `genre`, the samples the
    PIAA models are scored on, and the examples shown to each user ({user_id: [{sample_file, rating}]})."""
    import pandas as pd

    users = pd.read_csv(os.path.join(maked_dir, 'users.csv')).set_index('user_id')
    ratings = _load_ratings(genre, maked_dir)
    fine = pd.read_csv(os.path.join(split_dir, 'fine_samples.csv'))
    fine = fine[fine['genre'] == genre].merge(ratings[['user_id', 'sample_id', 'sample_file', 'Aesthetic']],
                                              on=['user_id', 'sample_id'], how='left')
    assert fine['sample_file'].notna().all(), genre
    test_users = sorted(u for k in range(5) for u in _fold_ids(split_dir, k, 'test_users'))
    if trial > 0:
        test_users = test_users[:trial]
        print(f"[trial] mode ON — using {len(test_users)} users")

    image_block = _image_blocks(genre)
    jobs, shots_by_user = [], {}
    for uid in test_users:
        mine = fine[fine['user_id'] == uid]
        shots = _select_shots(mine[mine['role'] == 'train'], 'Aesthetic', [uid, _GENRES.index(genre)], n_shot)
        shots_by_user[uid] = [{'sample_file': s['sample_file'], 'rating': int(s['Aesthetic']) + 1} for s in shots]
        content = []
        for s in shots_by_user[uid]:
            content += [image_block(s['sample_file']),
                        {"type": "text", "text": f"Rating given by this individual: {s['rating']}"}]
        prompt = _make_piaa_user_prompt(users.loc[uid].to_dict(), n_shot)
        evals = mine[mine['role'] == 'eval']
        assert len(evals) == 50, (uid, genre, len(evals))
        for fname in evals['sample_file']:
            jobs.append((uid, fname, content + [image_block(fname), {"type": "text", "text": prompt}]))
    print(f"{genre} {n_shot}-shot: {len(test_users)} users, {len(jobs)} queries")
    return jobs, shots_by_user


def _body(task: str, content: list) -> dict:
    """The Chat Completions request for one query (shared by the synchronous and the Batch API runs)."""
    system, schema, max_tokens = ((_SYSTEM_PROMPT, _GIAA_SCHEMA, _GIAA_MAX_TOKENS) if task == 'giaa' else
                                  (_PIAA_SYSTEM_PROMPT, _PIAA_SCHEMA, _PIAA_MAX_TOKENS))
    return {
        "model": _MODEL,
        "reasoning_effort": _PIAA_REASONING,
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
        "response_format": schema,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
    }


def _record(task: str, unit: int, fname: str, response: dict) -> dict:
    """A result record from a Chat Completions response body (as a dict)."""
    text = response['choices'][0]['message'].get('content') or ""
    pred = ({"fold": unit, "sample_file": fname, "pred_dist": _parse_giaa_dist(text)} if task == 'giaa' else
            {"user_id": unit, "sample_file": fname, "pred_score": _parse_piaa_score(text)})
    return {**pred, "raw": text,
            "usage": {"prompt_tokens": response['usage']['prompt_tokens'],
                      "completion_tokens": response['usage']['completion_tokens']}}


def _unit_key(task: str) -> str:
    return 'fold' if task == 'giaa' else 'user_id'


def _result_path(task: str, genre: str, n_shot: int, trial: int, save_dir: str = _SAVE_DIR) -> str:
    return os.path.join(save_dir, f"{genre}_{task}_{n_shot}shot{'_trial' if trial else ''}.json")


def _load_done(task: str, save_path: str) -> dict:
    if not os.path.exists(save_path):
        return {}
    with open(save_path) as fp:
        return {(r[_unit_key(task)], r['sample_file']): r for r in json.load(fp)['queries']}


def _save_results(task, save_path, genre, n_shot, jobs, shots, done, settings=None):
    """`settings` replaces the GPT-5.4 query settings recorded in the file (used by src.methods.qwen)."""
    records = list(done.values())
    pred = 'pred_dist' if task == 'giaa' else 'pred_score'
    output = {
        "genre": genre, "n_shot": n_shot,
        **(settings or {"model": _MODEL, "reasoning_effort": _PIAA_REASONING, "temperature": 0.0}),
        "image_size": _PIAA_IMAGE_SIZE,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_queries": len(jobs), "n_done": len(records),
        "n_parse_failures": sum(r[pred] is None for r in records),
        "usage": {k: sum(r['usage'][k] for r in records) for k in ('prompt_tokens', 'completion_tokens')},
        "shots": {str(u): s for u, s in shots.items()},
        "queries": records,
    }
    # The layout read by src.giaa_backbone (GIAA: one distribution per image) and src.eval_llm
    # (PIAA: one entry per image, a rating per user).
    if task == 'giaa':
        output["per_sample"] = [{"sample_file": r['sample_file'], "fold": r['fold'], "pred_dist": r['pred_dist']}
                                for r in sorted(records, key=lambda r: (r['fold'], r['sample_file']))]
    else:
        output["per_sample"] = [{"sample_file": f, "ratings": [{"user_id": r['user_id'], "pred_score": r['pred_score']}
                                                              for r in records if r['sample_file'] == f]}
                                for f in sorted({r['sample_file'] for r in records})]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp = save_path + '.tmp'
    with open(tmp, 'w') as fp:
        json.dump(output, fp, indent=1)
    os.replace(tmp, save_path)


def _jobs(task, genre, n_shot, trial, split_dir, maked_dir):
    return (_giaa_jobs if task == 'giaa' else _piaa_jobs)(genre, n_shot, trial, split_dir, maked_dir)


def _client():
    from openai import OpenAI

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in .env")
    return OpenAI(api_key=api_key)


def run(task: str, genre: str, n_shot: int = 0, trial: int = 0, workers: int = 8, dry: bool = False,
        split_dir: str = 'asset/split', maked_dir: str = 'asset/maked',
        client=None, body=_body, save_dir: str = _SAVE_DIR, settings: dict = None):
    """Run every query of `task` for `genre` synchronously (`workers` requests at a time).

    giaa: every test image (one predicted rating distribution each); piaa: every test user's 50
    eval samples (one predicted rating each). Results go to reports/exp/gpt/{genre}_{task}_{n_shot}shot.json;
    an existing file is resumed. With `trial` > 0, only the first `trial` test images of each fold
    (giaa) or test users (piaa) are queried (written to a *_trial.json file).
    With `dry`, the queries are built and the first one is printed, but nothing is sent.
    `client`, `body`, `save_dir` and `settings` let src.methods.qwen send the same queries to another
    OpenAI-compatible server.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs, shots = _jobs(task, genre, n_shot, trial, split_dir, maked_dir)
    if dry:
        unit, fname, content = jobs[0]
        print(f"{_unit_key(task)} {unit}, target {fname}, examples {shots[unit]}")
        print(json.dumps({k: v for k, v in body(task, content).items() if k != 'messages'}))
        print(f"[system] {body(task, content)['messages'][0]['content']}")
        for block in content:
            print(block['text'] if block['type'] == 'text' else f"<image {len(block['image_url']['url'])} chars>")
        return

    client = client or _client()
    save_path = _result_path(task, genre, n_shot, trial, save_dir)
    done = _load_done(task, save_path)
    if done:
        print(f"[resume] Loaded {len(done)} already-processed queries")
    lock = threading.Lock()
    pred = 'pred_dist' if task == 'giaa' else 'pred_score'

    def _save():
        with lock:
            snapshot = dict(done)
        _save_results(task, save_path, genre, n_shot, jobs, shots, snapshot, settings)

    def _query(job):
        unit, fname, content = job
        response = client.chat.completions.create(**body(task, content))
        return _record(task, unit, fname, response.model_dump())

    todo = [j for j in jobs if (j[0], j[1]) not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_query, j) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with lock:
                done[(r[_unit_key(task)], r['sample_file'])] = r
            print(f"  [{len(done)}/{len(jobs)}] {_unit_key(task)} {r[_unit_key(task)]} {r['sample_file']} → {r[pred]}")
            if n % 200 == 0:
                _save()
    _save()
    print(f"\nResults saved → {save_path}")


_BATCH_DIR = os.path.join(_SAVE_DIR, 'batch')
_BATCH_MAX_BYTES = 150 * 1024 ** 2   # per input file; the Batch API limit is 200 MB
_BATCH_MAX_REQUESTS = 50000
_BATCH_ALIVE = ('validating', 'in_progress', 'finalizing', 'completed')


def run_batch(action: str, task: str, genre: str, n_shot: int = 0, trial: int = 0,
              split_dir: str = 'asset/split', maked_dir: str = 'asset/maked'):
    """Run the same queries as `run` through the Batch API (half price, results within 24 h).

    submit:  write the queries that are neither done nor in a live batch as JSONL files of at most
             150 MB, upload them and create one batch per file. The batch ids go to
             reports/exp/gpt/batch/{tag}/state.json, tag = {genre}_{n_shot}shot[_trial] for piaa and
             giaa_{genre}_{n_shot}shot[_trial] for giaa.
    collect: print each batch's status and merge the results of completed, cancelled or expired
             batches into the same results file as `run`. Unfinished requests are not merged;
             `submit` again, or `run`, sends them.
    """
    tag = f"{'giaa_' if task == 'giaa' else ''}{genre}_{n_shot}shot{'_trial' if trial else ''}"
    batch_dir = os.path.join(_BATCH_DIR, tag)
    state_path = os.path.join(batch_dir, 'state.json')
    state = json.load(open(state_path)) if os.path.exists(state_path) else {'batches': []}

    def _write_state():
        os.makedirs(batch_dir, exist_ok=True)
        with open(state_path, 'w') as fp:
            json.dump(state, fp, indent=1)

    jobs, shots = _jobs(task, genre, n_shot, trial, split_dir, maked_dir)
    key = {f"q{i:05d}": (unit, fname) for i, (unit, fname, _) in enumerate(jobs)}
    save_path = _result_path(task, genre, n_shot, trial)
    done = _load_done(task, save_path)
    client = _client()

    if action == 'submit':
        live = set()
        for b in state['batches']:
            b['status'] = client.batches.retrieve(b['id']).status
            if b['status'] in _BATCH_ALIVE and not b.get('collected'):
                live |= set(b['custom_ids'])
        todo = [(f"q{i:05d}", content) for i, (unit, fname, content) in enumerate(jobs)
                if (unit, fname) not in done and f"q{i:05d}" not in live]
        print(f"{len(done)} done, {len(live)} in live batches, {len(todo)} to submit")
        chunks, lines, size = [], [], 0
        for cid, content in todo:
            line = json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                               "body": _body(task, content)}) + "\n"
            if lines and (size + len(line) > _BATCH_MAX_BYTES or len(lines) >= _BATCH_MAX_REQUESTS):
                chunks.append(lines)
                lines, size = [], 0
            lines.append(line)
            size += len(line)
        if lines:
            chunks.append(lines)
        os.makedirs(batch_dir, exist_ok=True)
        for lines in chunks:
            path = os.path.join(batch_dir, f"input_{len(state['batches']):03d}.jsonl")
            with open(path, 'w') as fp:
                fp.writelines(lines)
            with open(path, 'rb') as fp:
                file = client.files.create(file=fp, purpose='batch')
            batch = client.batches.create(input_file_id=file.id, endpoint='/v1/chat/completions',
                                          completion_window='24h',
                                          metadata={'tag': tag, 'part': str(len(state['batches']))})
            state['batches'].append({'id': batch.id, 'input_file': path, 'input_file_id': file.id,
                                     'n_requests': len(lines), 'status': batch.status,
                                     'custom_ids': [json.loads(l)['custom_id'] for l in lines]})
            _write_state()
            os.remove(path)   # the uploaded copy is kept by OpenAI; the local one is several hundred MB
            print(f"  submitted {batch.id}: {len(lines)} requests ({os.path.basename(path)})")
        _write_state()

    elif action == 'collect':
        n_new = 0
        for b in state['batches']:
            batch = client.batches.retrieve(b['id'])
            b['status'] = batch.status
            c = batch.request_counts
            print(f"  {b['id']}: {batch.status} ({c.completed}/{c.total} completed, {c.failed} failed)")
            # A cancelled or expired batch still returns the requests it finished.
            if batch.status not in ('completed', 'cancelled', 'expired') or b.get('collected'):
                continue
            if batch.output_file_id:
                for line in client.files.content(batch.output_file_id).text.splitlines():
                    r = json.loads(line)
                    if r.get('error') or r['response']['status_code'] != 200:
                        continue
                    unit, fname = key[r['custom_id']]
                    done[(unit, fname)] = _record(task, unit, fname, r['response']['body'])
                    n_new += 1
            b['collected'] = True
        _write_state()
        if n_new:
            _save_results(task, save_path, genre, n_shot, jobs, shots, done)
        print(f"{n_new} new results merged; {len(done)}/{len(jobs)} done → {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GPT-5.4 evaluation (GIAA and PIAA, k-shot)')
    parser.add_argument('--mode',  required=True, choices=['giaa', 'piaa'], help='Evaluation mode')
    parser.add_argument('--genre', required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--shots', type=int, default=0, choices=[0, 3], help='in-context examples per query')
    parser.add_argument('--trial', type=int, default=0,
                        help='GIAA: first N test images per fold; PIAA: first N test users (0 = all)')
    parser.add_argument('--workers', type=int, default=8, help='concurrent requests')
    parser.add_argument('--dry', action='store_true', help='build the queries and print one, send nothing')
    parser.add_argument('--batch', choices=['submit', 'collect'],
                        help='run through the Batch API instead (submit, then collect when done)')
    cli = parser.parse_args()

    if cli.batch:
        run_batch(cli.batch, cli.mode, genre=cli.genre, n_shot=cli.shots, trial=cli.trial)
    else:
        run(cli.mode, genre=cli.genre, n_shot=cli.shots, trial=cli.trial, workers=cli.workers, dry=cli.dry)

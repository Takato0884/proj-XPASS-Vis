"""Open-weight LLM baseline: the GPT-5.4 GIAA and PIAA queries of src.methods.gpt, sent to Qwen served by vLLM.

The requests are those of src.methods.gpt (same prompts, k-shot examples, 224x224 images, strict JSON
schema, temperature 0, max tokens), sent to vLLM's OpenAI-compatible server; vLLM enforces the schema
by structured decoding. GPT-5.4's reasoning effort "none" becomes Qwen's thinking switch
(enable_thinking=False in the chat template).

Results go to reports/exp/{model name}/{genre}_{task}_{n_shot}shot.json (e.g. reports/exp/qwen3.8-27b-fp8),
in the layout read by src.eval_llm and src.giaa_backbone; an existing file is resumed.

Usage:
    vllm serve Qwen/Qwen3.8-27B-FP8 --max-model-len 8192 --max-num-seqs 64 \\
        --limit-mm-per-prompt '{"image": 4, "video": 0}' --port 8000
    python -m src.methods.qwen --mode piaa --genre art --shots 0 [--trial 2] [--model Qwen/Qwen3.8-27B-FP8]
"""
import os

from . import gpt

_MODEL = 'Qwen/Qwen3.8-27B-FP8'


def save_dir(model: str) -> str:
    return os.path.join('reports/exp', model.split('/')[-1].lower())


def _body_fn(model: str):
    def body(task, content):
        b = gpt._body(task, content)
        b.pop('reasoning_effort')
        return {**b, 'model': model, 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
    return body


def run(task: str, genre: str, n_shot: int = 0, trial: int = 0, model: str = _MODEL,
        base_url: str = 'http://localhost:8000/v1', workers: int = 32, dry: bool = False):
    from openai import OpenAI

    settings = {'model': model, 'server': 'vllm', 'enable_thinking': False, 'temperature': 0.0}
    gpt.run(task, genre, n_shot, trial, workers=workers, dry=dry,
            client=None if dry else OpenAI(base_url=base_url, api_key='EMPTY', timeout=600),
            body=_body_fn(model), save_dir=save_dir(model), settings=settings)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Open-weight LLM (Qwen via vLLM) evaluation (GIAA and PIAA, k-shot)')
    parser.add_argument('--mode', required=True, choices=['giaa', 'piaa'], help='Evaluation mode')
    parser.add_argument('--genre', required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--shots', type=int, default=0, choices=[0, 3], help='in-context examples per query')
    parser.add_argument('--trial', type=int, default=0,
                        help='GIAA: first N test images per fold; PIAA: first N test users (0 = all)')
    parser.add_argument('--model', default=_MODEL, help='model name served by vLLM (Hugging Face id)')
    parser.add_argument('--base_url', default='http://localhost:8000/v1')
    parser.add_argument('--workers', type=int, default=32, help='concurrent requests')
    parser.add_argument('--dry', action='store_true', help='build the queries and print one, send nothing')
    cli = parser.parse_args()

    run(cli.mode, genre=cli.genre, n_shot=cli.shots, trial=cli.trial, model=cli.model,
        base_url=cli.base_url, workers=cli.workers, dry=cli.dry)

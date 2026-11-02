"""Common paths and local-only runtime. Import this before third-party modules."""
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CACHE = next((p for p in Path(__file__).resolve().parents if (p / '0.emotion_test_final').is_dir()), ROOT) / 'scripts' / 'final_model_runtime' / 'positive_affect'
sys.dont_write_bytecode = True
for name, subdir in {'TEMP': 'tmp', 'TMP': 'tmp', 'HF_HOME': 'huggingface',
                     'TORCH_HOME': 'torch', 'XDG_CACHE_HOME': 'xdg',
                     'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TRITON_CACHE_DIR': 'triton',
                     'CUDA_CACHE_PATH': 'cuda', 'MPLCONFIGDIR': 'matplotlib'}.items():
    location = CACHE / subdir
    location.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(location)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig') as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (BASE / p).resolve()


def load_config(path=None):
    cfg = read_json(path or BASE / 'config.json')
    if cfg['model']['max_length'] < 3 or cfg['model']['max_length'] > 512:
        raise ValueError('max_length must be in [3,512] for this BERT base')
    t = cfg['training']
    for key in ['epochs', 'batch_size', 'eval_batch_size', 'gradient_accumulation_steps', 'early_stopping_patience', 'log_steps']:
        if not isinstance(t[key], int) or t[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= t['warmup_ratio'] < 1 or not 0 <= cfg['model']['dropout'] < 1:
        raise ValueError('Invalid warmup/dropout')
    for key in ['encoder_learning_rate', 'head_learning_rate', 'max_grad_norm']:
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(t['weight_decay']) or t['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if not isinstance(cfg['evaluation']['batch_size'], int) or cfg['evaluation']['batch_size'] <= 0:
        raise ValueError('evaluation.batch_size must be a positive integer')
    return cfg


def setup(seed, device='cuda:0'):
    import numpy as np
    import torch
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Use D:\\Anacoda\\envs\\pytorch\\python.exe and check the GPU driver.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    d = torch.device(device)
    print(f'Python: {sys.executable}\nCUDA: {torch.cuda.get_device_name(d)}', flush=True)
    return d

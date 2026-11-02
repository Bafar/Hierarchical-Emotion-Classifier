"""Final cascade paths, cache placement and strict JSON input/output."""
import sys
sys.dont_write_bytecode = True
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CACHE = ROOT/'scripts'/'final_model_runtime'
SECONDARY_NAMES = ('positive_acknowledgment','positive_affect','conflict_repair',
                   'emotional_support','cognitive_guidance','casual_engagement')
for name, subdir in {'TEMP':'tmp', 'TMP':'tmp', 'HF_HOME':'huggingface',
                     'TORCH_HOME':'torch', 'XDG_CACHE_HOME':'xdg',
                     'TORCHINDUCTOR_CACHE_DIR':'inductor', 'TRITON_CACHE_DIR':'triton',
                     'CUDA_CACHE_PATH':'cuda', 'MPLCONFIGDIR':'matplotlib'}.items():
    location = CACHE/subdir
    location.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(location)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (BASE/path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8')


def load_config(path=None):
    raw = read_json(resolve(path or 'config.json'))
    if not isinstance(raw, dict) or not all(isinstance(raw.get(key), dict)
                                           for key in ('paths', 'routing', 'inference', 'validation')):
        raise ValueError('Config requires paths, routing, inference and validation objects')
    cfg = {**raw, **raw['routing'], **raw['inference']}
    diagnostics=raw.get('diagnostics',{})
    if not isinstance(diagnostics,dict) or not isinstance(diagnostics.get('routing_trace_enabled',False),bool):
        raise ValueError('diagnostics.routing_trace_enabled must be boolean')
    if diagnostics.get('routing_trace_enabled') and not isinstance(diagnostics.get('routing_trace_jsonl'),str):
        raise ValueError('diagnostics.routing_trace_jsonl must name an internal JSONL log')
    cfg['enable_correction'] = raw['routing'].get('enable_correction', False)
    for key in ('primary_device', 'secondary_device'):
        cfg[key] = raw['inference'].get(key, cfg['device'])
        if not isinstance(cfg[key], str) or not cfg[key].startswith('cuda:') or not cfg[key][5:].isdigit():
            raise ValueError(f'{key} must name a CUDA device, e.g. cuda:0')
    if not isinstance(cfg.get('enable_correction'), bool):
        raise ValueError('routing.enable_correction must be boolean')
    value = cfg.get('top2_min_probability')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('top2_min_probability must be a finite number in [0,1]')
    for key in ('compatibility_log_odds_weight', 'minimum_reversal_log_gain'):
        value = cfg.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0):
            raise ValueError(f'{key} must be finite and nonnegative')
    clip = cfg.get('compatibility_log_odds_clip')
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip <= 0:
        raise ValueError('compatibility_log_odds_clip must be finite and positive')
    thresholds = cfg.get('rejection_thresholds')
    if not isinstance(thresholds, dict) or set(thresholds) != set(SECONDARY_NAMES):
        raise ValueError('rejection_thresholds must list all six secondary branches')
    for branch, values in thresholds.items():
        if not isinstance(values, dict) or set(values) != {'probability', 'margin'}:
            raise ValueError(f'Invalid rejection thresholds for {branch}')
        probability, margin = values['probability'], values['margin']
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not math.isfinite(x) or not 0 <= x <= 1 for x in (probability, margin)):
            raise ValueError(f'Invalid rejection probability/margin for {branch}')
    if not isinstance(cfg.get('device'), str) or not cfg['device'].startswith('cuda'):
        raise ValueError('Final model requires a CUDA device')
    temperatures = {'primary_temperature': cfg['primary_temperature']}
    secondary = cfg['secondary_temperatures']
    if not isinstance(secondary, dict) or set(secondary) != set(SECONDARY_NAMES):
        raise ValueError('secondary_temperatures must list all six secondary branches')
    temperatures.update({f'secondary_temperatures.{key}':value for key,value in secondary.items()})
    for key, value in temperatures.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.1 <= value <= 10:
            raise ValueError(f'{key} must be a finite number in [0.1,10]')
    for key, lower, upper in [('max_length',3,512), ('max_history_messages',0,4),
                              ('batch_size',1,128)]:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f'{key} must be an integer in [{lower},{upper}]')
    delay = cfg['gpu_acquire_timeout_seconds']
    if isinstance(delay, bool) or not isinstance(delay, (int,float)) or not math.isfinite(delay) or delay < 0:
        raise ValueError('gpu_acquire_timeout_seconds must be finite and nonnegative')
    if not isinstance(cfg['clear_cuda_cache_after_branch'], bool):
        raise ValueError('clear_cuda_cache_after_branch must be boolean')
    if not isinstance(cfg.get('warmup_on_startup'), bool):
        raise ValueError('inference.warmup_on_startup must be boolean')
    for key in ('output_json','output_jsonl','primary_bert_base','secondary_bert_base'):
        if not isinstance(raw['paths'].get(key), str) or not raw['paths'][key].strip():
            raise ValueError(f'paths.{key} must be a nonempty path')
    check_base = raw['validation'].get('validate_bert_base_architecture')
    if not isinstance(check_base, bool):
        raise ValueError('validation.validate_bert_base_architecture must be boolean')
    if check_base:
        validate_bert_bases(raw['paths'])
    return cfg


def validate_bert_bases(paths):
    """Check reference base architectures; inference still uses bundled weights."""
    comparable = ('model_type','vocab_size','hidden_size','num_hidden_layers',
                  'num_attention_heads','intermediate_size','max_position_embeddings',
                  'type_vocab_size')
    for key, folders in (
        ('primary_bert_base', [BASE/'model'/'primary_emotion_classifier']),
        ('secondary_bert_base', [BASE/'model'/name for name in SECONDARY_NAMES])):
        source_path = resolve(paths[key])/'config.json'
        if not source_path.is_file():
            raise FileNotFoundError(f'Configured BERT base config not found: {source_path}')
        source = read_json(source_path)
        for folder in folders:
            bundled = read_json(folder/'encoder_config'/'config.json')
            mismatched = [field for field in comparable if source.get(field) != bundled.get(field)]
            if mismatched:
                raise ValueError(f'{key} is incompatible with {folder.name}: {mismatched}')


def load_input(path, max_history_messages=4):
    return validate_input(read_json(path), max_history_messages)


def validate_input(row, max_history_messages=4):
    if not isinstance(row, dict):
        raise ValueError('Input must be one JSON object, not a JSONL file or list')
    text = row.get('current_user_message')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('current_user_message must be a nonempty string')
    history = row.get('history_user_messages', [])
    if not isinstance(history, list) or len(history) > 4 or not all(isinstance(h, str) for h in history):
        raise ValueError('history_user_messages must be an array of at most four strings, latest first')
    return {'current_user_message': text, 'history_user_messages': history[:max_history_messages]}


@contextmanager
def gpu_slot(cfg):
    """Limit concurrent CLI processes sharing one GPU using OS file locks."""
    count = cfg.get('gpu_max_concurrent_jobs', 1)
    timeout = cfg['gpu_acquire_timeout_seconds']
    folder = CACHE/'gpu_slots'/cfg['device'].replace(':','_')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    acquired = None
    announced = False
    while acquired is None:
        for index in range(count):
            handle = (folder/f'slot_{index}.lock').open('a+b')
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = (handle, index)
                break
            except OSError:
                handle.close()
        if acquired is None:
            if not announced:
                print(f'Waiting for model resource {cfg["device"]} ...', file=sys.stderr, flush=True)
                announced = True
            if timeout > 0 and time.monotonic()-started >= timeout:
                raise TimeoutError(f'No GPU inference slot available within {timeout} seconds')
            time.sleep(0.1)
    handle, index = acquired
    try:
        yield {'slot':index, 'wait_seconds':time.monotonic()-started}
    finally:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def model_slot(name, cfg):
    """One active model per physical execution device; wait by default."""
    device = cfg.get('primary_device' if name == 'primary' else 'secondary_device', cfg['device'])
    local = {**cfg, 'device': device, 'gpu_max_concurrent_jobs': 1}
    with gpu_slot(local):
        yield local

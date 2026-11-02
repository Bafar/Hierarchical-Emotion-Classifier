"""Shared full-text features, train-only scaling, current-first tokenization."""
import runtime
import numpy as np
import torch
from torch.utils.data import Dataset
from features import FEATURE_NAMES, LABELS, extract_features

FEATURE_DIM = len(FEATURE_NAMES)
SCALED = [i for i, n in enumerate(FEATURE_NAMES) if n.endswith('.count_log')
          or (n.startswith('structure.') and (n.split('.')[1].startswith('log_') or n.endswith('_log')))
          or n == 'history.nearest_length_logratio']


def feature_matrix(rows):
    result = []
    for i, row in enumerate(rows):
        result.append(list(extract_features(row).values()))
        if (i + 1) % 2000 == 0:
            print(f'Features: {i+1}/{len(rows)}', flush=True)
    x = np.asarray(result, dtype=np.float32).reshape(-1, FEATURE_DIM)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite extracted feature')
    return x


def fit_scaler(train_features):
    if not len(train_features):
        raise ValueError('No model-routed training rows')
    x = np.asarray(train_features, dtype=np.float64)
    mean, scale = np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)
    mean[SCALED] = x[:, SCALED].mean(axis=0)
    sd = x[:, SCALED].std(axis=0, ddof=0)
    scale[SCALED] = np.where(sd == 0, 1., sd)
    return {'feature_names': FEATURE_NAMES, 'scaled_indices': SCALED, 'mean': mean.tolist(),
            'scale': scale.tolist(), 'fit_rows': len(x), 'fit_partition': 'train_model_routed_only'}


def transform(x, scaler):
    if scaler.get('feature_names') != FEATURE_NAMES or scaler.get('scaled_indices') != SCALED:
        raise ValueError('Feature scaler incompatible with this code')
    mean, scale = np.asarray(scaler['mean']), np.asarray(scaler['scale'])
    if mean.shape != (FEATURE_DIM,) or scale.shape != (FEATURE_DIM,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid scaler statistics')
    y = ((x - mean) / scale).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError('Non-finite normalized feature')
    return y


def encode_record(tokenizer, row, max_length=512):
    if max_length < 3 or max_length > 512:
        raise ValueError('max_length must be in [3,512]')
    texts = [row['current_user_message'].strip()] + [x.strip() for x in row['history_user_messages'][:4]]
    encoded = [tokenizer.encode(t, add_special_tokens=False, verbose=False) for t in texts]
    ids, types = list(encoded[0]), [0] * len(encoded[0])
    for j, h in enumerate(encoded[1:]):
        ids += [tokenizer.sep_token_id] + h
        types += [0 if j == 0 else 1] + [1] * len(h)
    raw_length = len(ids) + 2
    ids, types = ids[:max_length-2], types[:max_length-2]
    final_type = types[-1] if types else 0
    tokens = {'input_ids': [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id],
              'token_type_ids': [0] + types + [final_type],
              'attention_mask': [1] * (len(ids)+2)}
    info = {'input_tokens': len(ids)+2, 'untruncated_tokens': raw_length,
            'truncation': 'current' if len(encoded[0]) > max_length-2 else
                          ('history_only' if raw_length > max_length else 'none')}
    return tokens, info


class PreparedDataset(Dataset):
    def __init__(self, rows, tokenizer, scaler, max_length=512, raw_features=None):
        self.rows = rows
        raw = feature_matrix(rows) if raw_features is None else raw_features
        self.numeric = transform(raw, scaler)
        self.tokens, self.info = [], []
        for i, row in enumerate(rows):
            tokens, info = encode_record(tokenizer, row, max_length)
            self.tokens.append(tokens)
            self.info.append(info)
            if (i+1) % 2000 == 0:
                print(f'Tokenized: {i+1}/{len(rows)}', flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        label = self.rows[i].get('label', {}).get('primary')
        return {'tokens': self.tokens[i], 'numeric': self.numeric[i], 'index': i,
                'label': LABELS.index(label) if label in LABELS else -1}


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        tokens = self.tokenizer.pad([x['tokens'] for x in items], padding=True, return_tensors='pt')
        return {**dict(tokens), 'numeric': torch.tensor(np.stack([x['numeric'] for x in items])),
                'labels': torch.tensor([x['label'] for x in items], dtype=torch.long),
                'indices': [x['index'] for x in items]}

"""Load only the seven local safetensors bundles in isolated module namespaces."""
import runtime  # Set every temporary/cache directory under ../scripts first.
import importlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import torch

PROJECT_DIRS = {'primary': runtime.BASE/'model'/'primary_emotion_classifier',
                **{name: runtime.BASE/'model'/name for name in
                   ('positive_acknowledgment', 'positive_affect', 'conflict_repair',
                    'emotional_support', 'cognitive_guidance', 'casual_engagement')}}
PROJECT_MODULES = ('runtime', 'features', 'inputs', 'model', 'evaluation')
MODULE_IMPORT_LOCK = threading.RLock()


@contextmanager
def isolated_project(name):
    """Bundled projects use bare imports; keep modules from crossing branches."""
    with MODULE_IMPORT_LOCK:
        folder = PROJECT_DIRS[name]
        prior = {module: sys.modules.pop(module, None) for module in PROJECT_MODULES}
        sys.path.insert(0, str(folder))
        try:
            modules = {module: importlib.import_module(module) for module in PROJECT_MODULES}
            yield modules
        finally:
            sys.path.remove(str(folder))
            for module in PROJECT_MODULES:
                sys.modules.pop(module, None)
                if prior[module] is not None:
                    sys.modules[module] = prior[module]


def checkpoint_for(name, modules, cfg):
    # No fallback to the original training project is allowed.
    return PROJECT_DIRS[name]


def calibrated_prediction(prediction, name, cfg):
    """Scale model softmax probabilities; preserve direct-rule one-hot output."""
    if prediction['decision_source'] == 'rule':
        return prediction
    key = 'primary_probabilities' if name == 'primary' else 'secondary_probabilities'
    temperature = (cfg['primary_temperature'] if name == 'primary'
                   else cfg['secondary_temperatures'][name])
    if temperature == 1:
        return prediction
    values = [float(probability)**(1.0/temperature) for probability in prediction[key]]
    total = sum(values)
    if total <= 0:
        raise ValueError('Temperature scaling produced zero probability mass')
    return {**prediction, key:[value/total for value in values]}


def infer(name, row, cfg):
    with runtime.model_slot(name, cfg) as local:
        return _infer(name, row, local)


def _infer(name, row, cfg):
    """Return (prediction or None, label order, checkpoint or None, issue or None)."""
    with isolated_project(name) as modules:
        feature = modules['features']
        primary = name == 'primary'
        labels = list(feature.LABELS)
        rule_input = row if hasattr(feature, 'CONTEXT_RULE_VERSION') or hasattr(getattr(feature, 'primary', None), 'CONTEXT_RULE_VERSION') else row['current_user_message']
        rule = (feature.primary_rule if primary else feature.secondary_rule)(rule_input)
        if rule is not None:
            key = 'primary_label' if primary else 'secondary_label'
            prob_key = 'primary_probabilities' if primary else 'secondary_probabilities'
            rule[prob_key] = [float(label == rule[key]) for label in labels]
            return rule, labels, None, None
        checkpoint = checkpoint_for(name, modules, cfg)
        if not all((checkpoint/file).is_file() for file in ('metadata.json', 'model.safetensors')):
            issue = f'Bundled safetensors checkpoint incomplete: {checkpoint}'
            if primary:
                raise FileNotFoundError(issue)
            return None, labels, checkpoint, issue
        model = None
        try:
            model, tokenizer, metadata = modules['model'].load_checkpoint(checkpoint)
            model = model.to(torch.device(cfg['device']))
            if cfg['max_length'] > metadata['model']['max_length']:
                raise ValueError(f'max_length exceeds trained length for {name}')
            prepared = modules['evaluation'].prepare_eval(
                [row], tokenizer, metadata['scaler'], cfg['max_length'])
            result = modules['evaluation'].run_inference(model, [row], prepared,
                                                        tokenizer, torch.device(cfg['device']), 1)[0]
            return calibrated_prediction(result['prediction'], name, cfg), labels, checkpoint, None
        finally:
            if model is not None:
                del model
            if cfg['clear_cuda_cache_after_branch']:
                with torch.cuda.device(cfg['device']):
                    torch.cuda.empty_cache()


def infer_many(name, rows, cfg, batch_size=None):
    with runtime.model_slot(name, cfg) as local:
        return _infer_many(name, rows, local, batch_size)


def _infer_many(name, rows, cfg, batch_size=None):
    """Batched local-bundle inference for final test-set evaluation."""
    if not rows:
        return [], [], PROJECT_DIRS[name]
    with isolated_project(name) as modules:
        labels = list(modules['features'].LABELS)
        checkpoint = checkpoint_for(name, modules, cfg)
        rule_predictions, pending = modules['evaluation'].route(rows)
        if not pending:
            return rule_predictions, labels, checkpoint
        model = None
        try:
            model, tokenizer, metadata = modules['model'].load_checkpoint(checkpoint)
            model = model.to(torch.device(cfg['device']))
            if cfg['max_length'] > metadata['model']['max_length']:
                raise ValueError(f'max_length exceeds trained length for {name}')
            prepared = modules['evaluation'].prepare_eval(
                rows, tokenizer, metadata['scaler'], cfg['max_length'])
            result = modules['evaluation'].run_inference(
                model, rows, prepared, tokenizer, torch.device(cfg['device']),
                batch_size or cfg['batch_size'])
            return [calibrated_prediction(item['prediction'], name, cfg) for item in result], labels, checkpoint
        finally:
            if model is not None:
                del model
            if cfg['clear_cuda_cache_after_branch']:
                with torch.cuda.device(cfg['device']):
                    torch.cuda.empty_cache()

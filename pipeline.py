"""Bounded two-stage JSONL pipeline, using spawn on Windows and Linux."""
import runtime
import json
import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from model_adapters import infer_many
from predict import format_output
from rejection_routing import (PRIMARY_LABELS, score_branch, select_candidates,
                               should_review_top2)


def batches(path, cfg):
    batch = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = runtime.validate_input(json.loads(line), cfg['max_history_messages'])
            except (ValueError, TypeError) as error:
                raise ValueError(f'{path}: line {line_number}: {error}') from error
            batch.append(row)
            if len(batch) == cfg['batch_size']:
                yield batch
                batch = []
    if batch:
        yield batch


def primary_batch(rows, cfg):
    predictions, labels, _ = infer_many('primary', rows, cfg)
    if labels != PRIMARY_LABELS:
        raise ValueError('Unexpected primary label order')
    return rows, predictions


def secondary_batch(rows, predictions, cfg):
    choices = [select_candidates(p, cfg) for p in predictions]
    grouped = {}
    scored = [{} for _ in rows]
    for i, choice in enumerate(choices):
        for candidate in [choice['top1']]:
            name = candidate['label']
            if name == 'neutral':
                scored[i][name] = score_branch(candidate, None, [], cfg)
            else:
                grouped.setdefault(name, []).append((i, candidate))
    for name, items in grouped.items():
        results, labels, checkpoint = infer_many(name, [rows[i] for i, _ in items], cfg)
        for (i, candidate), prediction in zip(items, results):
            scored[i][name] = score_branch(candidate, prediction, labels,
                                           cfg, checkpoint)
    # Only rejected Top-1 branches (and uncertain neutral Top-1) may inspect
    # Top-2. This is the single permitted correction hop.
    grouped = {}
    for i, choice in enumerate(choices):
        top1 = scored[i][choice['top1']['label']]
        if should_review_top2(choice, top1, cfg):
            candidate = choice['top2']
            if candidate['label'] == 'neutral':
                scored[i]['neutral'] = score_branch(candidate, None, [], cfg)
            else:
                grouped.setdefault(candidate['label'], []).append((i, candidate))
            choice['route'] = 'dual'
        else:
            choice['route'] = 'single'
    for name, items in grouped.items():
        results, labels, checkpoint = infer_many(name, [rows[i] for i, _ in items], cfg)
        for (i, candidate), prediction in zip(items, results):
            scored[i][name] = score_branch(candidate, prediction, labels, cfg, checkpoint)
    outputs = []
    for i, (prediction, choice) in enumerate(zip(predictions, choices)):
        branches = [scored[i][choice['top1']['label']]]
        if choice['route'] == 'dual':
            branches.append(scored[i][choice['top2']['label']])
        outputs.append(format_output(prediction, choice, branches, cfg))
    return outputs


def pipeline_results(source, cfg):
    iterator = iter(source)
    first = next(iterator, None)
    if first is None:
        return
    context = multiprocessing.get_context('spawn')
    # Each stage has a separate module namespace, avoiding bare-import clashes.
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as primary_pool, \
            ProcessPoolExecutor(max_workers=1, mp_context=context) as secondary_pool:
        primary = primary_pool.submit(primary_batch, first, cfg)
        secondary = None
        while primary is not None:
            rows, predictions = primary.result()
            following = next(iterator, None)
            primary = (primary_pool.submit(primary_batch, following, cfg)
                       if following is not None else None)
            if secondary is not None:
                yield from secondary.result()
            secondary = secondary_pool.submit(secondary_batch, rows, predictions, cfg)
        if secondary is not None:
            yield from secondary.result()


def predict_jsonl(input_path, output_path, cfg):
    # A failed batch never replaces an existing result with an incomplete file.
    folder = runtime.CACHE/'tmp'
    folder.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.jsonl',
                                         dir=folder, delete=False) as stream:
            temporary = Path(stream.name)
            for result in pipeline_results(batches(input_path, cfg), cfg):
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
                count += 1
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary, output_path)
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

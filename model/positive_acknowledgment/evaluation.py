"""Primary-only routing, model inference, and four-class evaluation."""
import runtime
import numpy as np
import torch
from torch.utils.data import DataLoader
from features import LABELS, secondary_rule
from inputs import PreparedDataset, Collator


def route(rows, use_rules=True):
    predictions = [secondary_rule(r)
                   if use_rules and r.get('label', {}).get('secondary') != 'not_this_branch'
                   else None for r in rows]
    # Encode deterministic routing at the output boundary; keep feature code and
    # checkpoint compatibility unchanged. decision_source distinguishes softmax.
    for prediction in predictions:
        if prediction is not None:
            prediction['secondary_probabilities'] = [
                float(label == prediction['secondary_label']) for label in LABELS]
    indices = [i for i, p in enumerate(predictions) if p is None]
    return predictions, indices


def prepare_eval(rows, tokenizer, scaler, max_length=512, use_rules=True):
    predictions, indices = route(rows, use_rules)
    dataset = PreparedDataset([rows[i] for i in indices], tokenizer, scaler, max_length)
    return predictions, indices, dataset


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ('labels', 'indices')}


@torch.inference_mode()
def run_inference(model, rows, prepared, tokenizer, device, batch_size=8):
    initial, indices, dataset = prepared
    predictions = [dict(x) if x is not None else None for x in initial]
    truncations = [None] * len(rows)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer), num_workers=0)
    for batch in loader:
        probabilities = model(**to_device(batch, device)).float().softmax(-1).cpu().tolist()
        for local, p in zip(batch['indices'], probabilities):
            target = indices[local]
            pred_id = int(np.argmax(p))
            predictions[target] = {'secondary_label': LABELS[pred_id], 'decision_source': 'model',
                                   'rule_id': None, 'secondary_probabilities': p}
            truncations[target] = dataset.info[local]
    results = []
    for r, p, t in zip(rows, predictions, truncations):
        gold = r.get('label', {}).get('secondary')
        results.append({**r, 'prediction': p,
                        'correct': p['secondary_label'] == gold if gold in LABELS else None,
                        'bert_input': t})
    return results


def metrics(rows):
    class_count = len(LABELS)
    cm = np.zeros((class_count, class_count), dtype=np.int64)
    for r in rows:
        cm[LABELS.index(r['label']['secondary']), LABELS.index(r['prediction']['secondary_label'])] += 1
    support, predicted = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(class_count), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(class_count), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(class_count), where=precision+recall != 0)
    n = int(cm.sum())
    return {'n': n, 'accuracy': float(tp.sum()/n) if n else None,
            'macro_f1': float(f1.mean()) if n else None,
            'weighted_f1': float((f1*support).sum()/n) if n else None,
            'confusion_matrix': cm.tolist(),
            'classes': {label: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                'f1': float(f1[i]), 'support': int(support[i])} for i, label in enumerate(LABELS)}}


def build_report(results, split):
    labeled = [r for r in results if r.get('label', {}).get('secondary') in LABELS]
    model = [r for r in labeled if r['prediction']['decision_source'] == 'model']
    rules = [r for r in labeled if r['prediction']['decision_source'] == 'rule']
    subsets = {'overall': labeled, 'model_routed': model, 'rule_routed': rules}
    for key in ('none', 'current', 'history_only'):
        subsets['truncation_'+key] = [r for r in model if r['bert_input']['truncation'] == key]
    return {'split': split, 'label_order': LABELS, 'rows': len(results),
            'rule_coverage': len(rules)/len(labeled) if labeled else None,
            'metrics': {k: metrics(v) for k, v in subsets.items()},
            'matrix_axes': 'rows=true, columns=predicted; macro F1 uses every label in label_order, including internal rejection',
            'probabilities': 'Model path: softmax; rule path: one-hot (matched class 1, others 0), in label_order.'}


def print_report(report):
    print('\nLabel indices: ' + ', '.join(f'{i}={x}' for i, x in enumerate(LABELS)))
    for subset in ('overall', 'model_routed', 'rule_routed'):
        m = report['metrics'][subset]
        print(f'\n[{subset}] n={m["n"]} accuracy={m["accuracy"]} macro_f1={m["macro_f1"]}')
        print('Confusion matrix: rows=true, columns=predicted')
        print('true\\pred ' + ''.join(f'{i:>7}' for i in range(len(LABELS))))
        for i, row in enumerate(m['confusion_matrix']):
            print(f'{i:>9} ' + ''.join(f'{v:>7}' for v in row))
    print('\nPer-class metrics (overall):')
    for label, m in report['metrics']['overall']['classes'].items():
        print(f'{label:26} P={m["precision"]:.4f} R={m["recall"]:.4f} F1={m["f1"]:.4f} n={m["support"]}')

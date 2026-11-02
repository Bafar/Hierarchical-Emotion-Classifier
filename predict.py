"""Classify JSON or ordered JSONL samples through the two-stage models."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import logging
import os
import time
import runtime

from model_adapters import infer
from rejection_routing import (PRIMARY_LABELS, pick_winner, score_branch,
                               select_candidates, should_review_top2)


def record_routing(choice, branches, cfg):
    """Internal raw acceptance evidence; never add fields to public output."""
    diagnostics=cfg.get('diagnostics',{})
    if not diagnostics.get('routing_trace_enabled',False):return
    trace={'time':time.time(),'top1':choice['top1'],'top2':choice['top2'],
           'route':choice['route'],'branches':[]}
    for branch in branches:
        leaf=branch.get('best_leaf');q_out=branch.get('rejection_probability')
        trace['branches'].append({'primary':branch['primary_label'],'rejected':branch.get('rejected'),
            'raw_rejection_probability':q_out,'leaf':leaf['secondary_label'] if leaf else None,
            'raw_leaf_probability':leaf['secondary_probability']*(1-q_out) if leaf and q_out is not None else None,
            'public_conditional_probability':leaf['secondary_probability'] if leaf else None})
    try:
        path=runtime.resolve(diagnostics['routing_trace_jsonl']);path.parent.mkdir(parents=True,exist_ok=True)
        descriptor=os.open(str(path),os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
        try:os.write(descriptor,(json.dumps(trace,ensure_ascii=False)+'\n').encode('utf-8'))
        finally:os.close(descriptor)
    except OSError as exc:
        logging.getLogger(__name__).warning('Unable to write routing diagnostics: %s',exc)


def classify(row, cfg):
    primary_prediction, labels, _, _ = infer('primary', row, cfg)
    if labels != PRIMARY_LABELS:
        raise ValueError('Primary checkpoint label order differs from final-model contract')
    choice = select_candidates(primary_prediction, cfg)
    branches = []
    for candidate in [choice['top1']]:
        name = candidate['label']
        if name == 'neutral':
            branch = score_branch(candidate, None, [], cfg)
        else:
            prediction, secondary_labels, checkpoint, issue = infer(name, row, cfg)
            branch = score_branch(candidate, prediction, secondary_labels,
                                  cfg, checkpoint, issue)
        branches.append(branch)
    if should_review_top2(choice, branches[0], cfg):
        candidate = choice['top2']
        name = candidate['label']
        if name == 'neutral':
            branch = score_branch(candidate, None, [], cfg)
        else:
            prediction, secondary_labels, checkpoint, issue = infer(name, row, cfg)
            branch = score_branch(candidate, prediction, secondary_labels, cfg,
                                  checkpoint, issue)
        branches.append(branch)
        choice['route'] = 'dual'
    else:
        choice['route'] = 'single'
    return format_output(primary_prediction, choice, branches, cfg)


def format_output(primary_prediction, choice, branches, cfg):
    record_routing(choice,branches,cfg)
    winner, status = pick_winner(branches, cfg['minimum_reversal_log_gain'])
    probabilities = primary_prediction['primary_probabilities']
    selected_path = ('path_2_multi_path' if choice['route'] == 'dual'
                     else 'path_1_fast_path')
    output = {
        # Until a winner is available, expose the original Top-1.  On a
        # successful decision this is replaced below by the winning branch so
        # primary and secondary can never describe different taxonomy paths.
        'primary': {'label': choice['top1']['label'],
                    'score': choice['top1']['probability'],
                    'scores': {label: float(probabilities[i])
                               for i, label in enumerate(PRIMARY_LABELS)}},
        'secondary': None,
        'routing_meta': {
            'selected_path': selected_path,
        },
        'final_decision': None,
    }
    if status != 'complete':
        output['status'] = status
        return output
    selected_branch = next(branch for branch in branches if branch['best_leaf'] is winner)
    if (winner['primary_label'] != selected_branch['primary_label'] or
            winner['secondary_label'] not in selected_branch['secondary_label_order']):
        raise RuntimeError('Final primary/secondary taxonomy path is inconsistent')
    output['primary']['label'] = winner['primary_label']
    output['primary']['score'] = winner['primary_probability']
    output['secondary'] = {
        'label': winner['secondary_label'],
        'score': winner['secondary_probability'],
        'scores': selected_branch['secondary_probabilities'],
    }
    output['final_decision'] = {
        'primary_label': winner['primary_label'],
        'secondary_label': winner['secondary_label'],
        'primary_score': winner['primary_probability'],
        'secondary_score': winner['secondary_probability'],
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSON object or JSONL file')
    parser.add_argument('--output', help='Override paths.output_json or paths.output_jsonl for the input format')
    parser.add_argument('--config', help='Config JSON; default: final-model/config.json')
    args = parser.parse_args()
    cfg = runtime.load_config(args.config)
    input_path = runtime.resolve(args.input)
    if input_path.suffix.lower() not in ('.json', '.jsonl'):
        raise ValueError('Input must have a .json or .jsonl extension')
    output_key = 'output_jsonl' if input_path.suffix.lower() == '.jsonl' else 'output_json'
    output = runtime.resolve(args.output or cfg['paths'][output_key])
    if output.suffix.lower() != input_path.suffix.lower():
        raise ValueError('Input and output must use the same JSON/JSONL format')
    if input_path.resolve() == output.resolve():
        raise ValueError('Input and output paths must differ')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the saved BERT models')
    for device in set((cfg['primary_device'], cfg['secondary_device'])):
        print(f'{device}: {torch.cuda.get_device_name(torch.device(device))}', flush=True)
    if input_path.suffix.lower() == '.jsonl':
        from pipeline import predict_jsonl
        count = predict_jsonl(input_path, output, cfg)
        print(f'Completed {count} samples; JSONL: {output}', flush=True)
        return
    row = runtime.load_input(input_path, cfg['max_history_messages'])
    result = classify(row, cfg)
    runtime.write_json(output, result)
    print(f'Result: {result["final_decision"]}; JSON: {output}', flush=True)


if __name__ == '__main__':
    main()

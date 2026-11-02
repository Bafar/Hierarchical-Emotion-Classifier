"""Two-stage routing with an internal secondary rejection class."""
import math

PRIMARY_LABELS = ['neutral', 'positive_acknowledgment', 'positive_affect',
                  'conflict_repair', 'emotional_support', 'cognitive_guidance',
                  'casual_engagement']
REJECTION_LABEL = 'not_this_branch'


def validate_probabilities(values, expected):
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise ValueError(f'Expected {expected} probabilities')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v < 0 or v > 1 for v in values):
        raise ValueError('Probabilities must be finite numbers in [0,1]')
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError('Probabilities must sum to one')


def select_candidates(prediction, cfg):
    probabilities = prediction['primary_probabilities']
    validate_probabilities(probabilities, len(PRIMARY_LABELS))
    order = sorted(range(len(PRIMARY_LABELS)), key=lambda i: (-probabilities[i], i))
    i1, i2 = order[:2]
    rule = prediction.get('decision_source') == 'rule'
    if rule and prediction['primary_label'] != PRIMARY_LABELS[i1]:
        raise ValueError('Primary rule label and one-hot vector disagree')
    return {'route': 'rule' if rule else 'top1_pending', 'rule': rule,
            'top1': {'label': PRIMARY_LABELS[i1], 'probability': float(probabilities[i1])},
            'top2': {'label': PRIMARY_LABELS[i2], 'probability': float(probabilities[i2])},
            'gap': float(probabilities[i1] - probabilities[i2])}


def _real_distribution(probabilities, labels):
    real = [(label, float(probabilities[i])) for i, label in enumerate(labels)
            if label != REJECTION_LABEL]
    total = sum(value for _, value in real)
    if not real or total <= 0:
        raise ValueError('Secondary classifier has no usable emotion probability')
    normalized = {label: value / total for label, value in real}
    return normalized, max(real, key=lambda item: item[1])


def score_branch(candidate, secondary_prediction, secondary_labels, cfg,
                 checkpoint=None, issue=None):
    primary_label, p = candidate['label'], float(candidate['probability'])
    base = {'primary_label': primary_label, 'primary_probability': p,
            'checkpoint': str(checkpoint) if checkpoint else None}
    if primary_label == 'neutral':
        score = math.log(p + 1e-12)
        return {**base, 'status': 'complete', 'rejected': False,
                'rejection_probability': 0.0,
                'secondary_label_order': ['neutral'],
                'secondary_probabilities': {'neutral': 1.0},
                'best_leaf': {'primary_label': 'neutral', 'secondary_label': 'neutral',
                              'primary_probability': p, 'secondary_probability': 1.0,
                              'score': score, 'decision_source': 'terminal'}}
    if secondary_prediction is None:
        return {**base, 'status': 'missing_checkpoint', 'issue': issue,
                'best_leaf': None, 'secondary_label_order': [],
                'secondary_probabilities': None}
    if not secondary_labels or secondary_labels[-1] != REJECTION_LABEL:
        raise ValueError(f'{primary_label} checkpoint lacks final {REJECTION_LABEL} class')
    q = secondary_prediction['secondary_probabilities']
    validate_probabilities(q, len(secondary_labels))
    public_q, (best_label, best_raw) = _real_distribution(q, secondary_labels)
    q_out = float(q[-1])
    threshold = cfg['rejection_thresholds'][primary_label]
    rejected = (secondary_prediction.get('decision_source') != 'rule' and
                q_out >= threshold['probability'] and
                q_out - best_raw >= threshold['margin'])
    log_odds = math.log((1.0 - q_out + 1e-8) / (q_out + 1e-8))
    log_odds = max(-cfg['compatibility_log_odds_clip'],
                   min(cfg['compatibility_log_odds_clip'], log_odds))
    score = math.log(p + 1e-12) + cfg['compatibility_log_odds_weight'] * log_odds
    return {**base, 'status': 'complete', 'rejected': rejected,
            'rejection_probability': q_out, 'compatibility_log_odds': log_odds,
            'secondary_label_order': list(public_q),
            'secondary_probabilities': public_q,
            'best_leaf': {'primary_label': primary_label,
                          'secondary_label': best_label,
                          'primary_probability': p,
                          'secondary_probability': public_q[best_label],
                          'score': score,
                          'decision_source': secondary_prediction.get('decision_source', 'model')}}


def should_review_top2(choice, top1_branch, cfg):
    if not cfg['enable_correction'] or choice['rule']:
        return False
    if choice['top2']['probability'] < cfg['top2_min_probability']:
        return False
    return choice['top1']['label'] == 'neutral' or bool(top1_branch.get('rejected'))


def pick_winner(branches, minimum_reversal_log_gain=0.0):
    if not branches or branches[0]['status'] != 'complete':
        return None, 'missing_checkpoint'
    top1 = branches[0]['best_leaf']
    if len(branches) == 1:
        return top1, 'complete'
    if branches[1]['status'] != 'complete':
        return None, 'missing_checkpoint'
    challenger = branches[1]['best_leaf']
    if challenger['score'] >= top1['score'] + minimum_reversal_log_gain:
        return challenger, 'complete'
    return top1, 'complete'

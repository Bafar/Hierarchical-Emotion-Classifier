"""Legacy margin-routing reference; production uses rejection_routing.py."""
import math
import re

PRIMARY_LABELS = ['neutral', 'positive_acknowledgment', 'positive_affect',
                  'conflict_repair', 'emotional_support', 'cognitive_guidance',
                  'casual_engagement']

COGNITIVE_CHALLENGE = re.compile(
    r'(?:不|没|并不|并非)(?:认可|赞同|同意|相信|接受)|'
    r'(?:回答|答案|说法|观点|结论|解释|推导|数据|结果).{0,10}'
    r'(?:(?<!没)(?<!无)(?<!不是)(?<!并非)有问题|不对|错(?:了|的)?|有误|不准确|不合理|太绝对|站不住脚|说不通)|'
    r'(?:(?<!没)(?<!无)(?<!不是)(?<!并非)有问题|不对|错了|有误|不准确|不合理|太绝对|站不住脚|说不通|不可信|可疑|存疑|质疑)|'
    r'(?:依据|证据|来源|出处)(?:呢|是什么|在哪)|'
    r"\b(?:do not|don't|cannot|can't)\s+(?:agree|accept|believe)|"
    r'\b(?:answer|claim|view|conclusion|explanation).{0,30}(?:wrong|incorrect|inaccurate|flawed|too absolute)|'
    r'\b(?:wrong|incorrect|inaccurate|unconvincing|not credible|too absolute)\b', re.I)
CONFLICT_ATTACK = re.compile(
    r'呸|傻逼|傻[比笔]|蠢货|白痴|废物|垃圾|无耻|骗子|狗屁|混蛋|王八蛋|智障|脑残|眼瞎|'
    r'去死|滚|闭嘴|\b(?:idiot|moron|stupid|trash|garbage|bullshit|fuck(?:ing)?|shut up|get lost|useless)\b', re.I)
NEGATED_APPROVAL = re.compile(
    r'(?:不|没|并不|并非|从不|无法).{0,3}(?:认可|赞同|同意|正确|合理|有道理|相信|接受)|'
    r"\b(?:do not|don't|cannot|can't)\s+(?:agree|accept|believe)", re.I)
STRONG_ACKNOWLEDGMENT = re.compile(
    r'^(?:(?:(?:你(?:的)?|这个|这次|刚才)(?:回答|答案|解释|分析|表现)?'
    r'(?:说|讲|答|做)?得?)?(?:很|真|太|非常|特别)(?:棒|赞|好|优秀|完美|靠谱)|'
    r'(?:说得好|回答得(?:很|真|太|非常|特别)好|干得漂亮))'
    r'(?:了|啊|呀|呢|哦|！|!|~)*$', re.I)
POSITIVE_LAUGHTER = re.compile(
    r'(?:笑死(?:我)?了?|笑疯(?:我)?了?|笑不活了|笑麻了|乐死(?:我)?了?|'
    r'哈(?:哈){2,}|h(?:a){2,})', re.I)
PAYLOAD_TASK = re.compile(
    r'翻译|译成|译为|改写|润色|重写|总结|概括|摘要|创作|写一段|生成|台词|剧本|原文|以下内容|'
    r'\b(?:translate|rewrite|rephrase|summari[sz]e|analy[sz]e|quote|screenplay|dialogue)\b', re.I)


def validate_probabilities(values, expected):
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise ValueError(f'Expected {expected} probabilities')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v < 0 or v > 1 for v in values):
        raise ValueError('Probabilities must be finite numbers in [0,1]')
    if abs(sum(values)-1) > 1e-4:
        raise ValueError('Probabilities must sum to one')


def select_candidates(prediction, absolute_threshold, gap_threshold, enable_correction=False):
    if not isinstance(enable_correction, bool):
        raise ValueError('enable_correction must be boolean')
    probabilities = prediction['primary_probabilities']
    validate_probabilities(probabilities, len(PRIMARY_LABELS))
    ranked = sorted(range(len(PRIMARY_LABELS)), key=lambda i: (-probabilities[i], i))
    i1, i2 = ranked[:2]
    p1, p2 = float(probabilities[i1]), float(probabilities[i2])
    rule = prediction['decision_source'] == 'rule'
    if rule and prediction['primary_label'] != PRIMARY_LABELS[i1]:
        raise ValueError('Primary rule label and one-hot vector disagree')
    certain = p1 >= absolute_threshold and p1-p2 >= gap_threshold
    indices = [i1] if rule or not enable_correction or certain else [i1, i2]
    return {
        'route': 'rule' if rule else ('top1_only' if not enable_correction else
                                   ('single' if certain else 'dual')),
        'top1': {'label': PRIMARY_LABELS[i1], 'probability': p1},
        'top2': {'label': PRIMARY_LABELS[i2], 'probability': p2},
        'gap': p1-p2,
        'candidates': [{'label': PRIMARY_LABELS[i], 'probability': float(probabilities[i])}
                       for i in indices],
    }


def semantic_evidence(text, branch, odds):
    """Return a bounded log-likelihood contribution from high-precision speech acts."""
    cognitive = bool(COGNITIVE_CHALLENGE.search(text))
    attack = bool(CONFLICT_ATTACK.search(text))
    negated_approval = bool(NEGATED_APPROVAL.search(text))
    strong_acknowledgment = bool(STRONG_ACKNOWLEDGMENT.fullmatch(text.strip()))
    positive_laughter = bool(POSITIVE_LAUGHTER.search(text))
    if PAYLOAD_TASK.search(text):
        return 0.0, []
    if branch == 'conflict_repair' and attack:
        return math.log(odds['direct_attack']), ['direct_attack']
    if branch == 'cognitive_guidance' and cognitive and not attack:
        return math.log(odds['epistemic_challenge']), ['epistemic_challenge']
    if branch == 'positive_acknowledgment' and (cognitive or negated_approval):
        return math.log(odds['negated_or_challenged_acknowledgment']), [
            'negated_or_challenged_acknowledgment']
    if branch == 'positive_acknowledgment' and strong_acknowledgment:
        return math.log(odds['strong_acknowledgment']), ['strong_acknowledgment']
    if branch == 'positive_affect' and positive_laughter:
        return math.log(odds['positive_laughter']), ['positive_laughter']
    return 0.0, []


def score_branch(candidate, secondary_prediction, secondary_labels, cfg, text,
                 checkpoint=None, issue=None):
    primary_label = candidate['label']
    p = candidate['probability']
    base = {'primary_label': primary_label, 'primary_probability': p,
            'checkpoint': str(checkpoint) if checkpoint else None}
    if primary_label == 'neutral':
        score = math.log(p + 1e-12)
        return {**base, 'status': 'complete', 'decision_source': 'terminal',
                'secondary_label_order': ['neutral'],
                'secondary_probabilities': {'neutral': 1.0},
                'leaf_scores': {'neutral': score},
                'best_leaf': {'primary_label': 'neutral', 'secondary_label': 'neutral',
                              'primary_probability': p, 'secondary_probability': 1.0,
                              'score': score, 'decision_source': 'terminal'}}
    bonus, reasons = semantic_evidence(text, primary_label, cfg['semantic_evidence_odds'])
    if secondary_prediction is None:
        maximum = (math.log(p + 1e-12) + cfg['secondary_margin_weight'] * 3.0 + bonus)
        return {**base, 'status': 'missing_checkpoint', 'issue': issue,
                'secondary_label_order': secondary_labels,
                'secondary_probabilities': None, 'leaf_scores': None,
                'maximum_possible_score': maximum, 'best_leaf': None}
    q = secondary_prediction['secondary_probabilities']
    validate_probabilities(q, len(secondary_labels))
    if secondary_prediction['secondary_label'] not in secondary_labels:
        raise ValueError('Unknown secondary label')
    best_idx = max(range(len(q)), key=lambda i: (q[i], -i))
    # A direct rule's one-hot and a model softmax share the same scoring path.
    if secondary_prediction['secondary_label'] != secondary_labels[best_idx]:
        raise ValueError('Secondary label and probability vector disagree')
    ranked_q = sorted((float(value) for value in q), reverse=True)
    raw_margin = math.log((ranked_q[0] + 1e-12) / (ranked_q[1] + 1e-12))
    calibration = cfg['secondary_margin_calibration'][primary_label]
    raw_z = max(-3.0, min(3.0, (raw_margin - calibration['center']) / calibration['scale']))
    # Only unusually strong conditional confidence supports a reversal. A low
    # within-branch margin is not calibrated proof that the primary model is wrong.
    evidence_z = max(0.0, raw_z)
    branch_score = (math.log(p + 1e-12) +
                    cfg['secondary_margin_weight'] * evidence_z + bonus)
    # A high-precision final-routing rule may also resolve the leaf, without
    # modifying the trained secondary feature implementation or its hash.
    output_q = [float(value) for value in q]
    decision_source = secondary_prediction['decision_source']
    rule_id = secondary_prediction.get('rule_id')
    if 'strong_acknowledgment' in reasons and 'approval' in secondary_labels:
        best_idx = secondary_labels.index('approval')
        output_q = [float(i == best_idx) for i in range(len(secondary_labels))]
        decision_source = 'routing_rule'
        rule_id = 'R_ACK_STRONG_ACKNOWLEDGMENT'
    top_q = float(output_q[best_idx])
    scores = {label: branch_score + math.log((float(output_q[i]) + 1e-12) / (top_q + 1e-12))
              for i, label in enumerate(secondary_labels)}
    label = secondary_labels[best_idx]
    return {**base, 'status': 'complete',
            'decision_source': decision_source,
            'rule_id': rule_id,
            'correction_evidence': {'secondary_log_margin': raw_margin,
                                    'positive_standardized_margin': evidence_z,
                                    'semantic_reasons': reasons},
            'secondary_label_order': secondary_labels,
            'secondary_probabilities': {label: float(output_q[i]) for i, label in enumerate(secondary_labels)},
            'leaf_scores': scores,
            'best_leaf': {'primary_label': primary_label, 'secondary_label': label,
                          'primary_probability': p, 'secondary_probability': float(output_q[best_idx]),
                          'score': branch_score,
                          'decision_source': decision_source}}


def pick_winner(branches, minimum_reversal_log_gain=0.0):
    known = [branch['best_leaf'] for branch in branches if branch['status'] == 'complete']
    missing = [branch for branch in branches if branch['status'] == 'missing_checkpoint']
    if not known:
        return None, 'missing_checkpoint'
    winner = max(known, key=lambda item: item['score'])
    # Strict inequality avoids inventing a tie break for a missing branch.
    if any(winner['score'] <= branch['maximum_possible_score'] for branch in missing):
        return None, 'missing_checkpoint'
    top1 = branches[0]['best_leaf'] if branches and branches[0]['status'] == 'complete' else None
    if (top1 is not None and winner is not top1 and
            winner['score'] < top1['score'] + minimum_reversal_log_gain):
        winner = top1
    return winner, 'complete'

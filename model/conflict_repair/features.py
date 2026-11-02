"""Four-way conflict-repair rules and 94 interpretable current/history features."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import math
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('primary_features_conflict', ROOT/'primary_emotion_classifier/features.py')
primary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary)

LABELS = ['anger', 'frustration', 'impatience', 'disappointment', 'guardedness', 'disapproval', 'disgust', 'not_this_branch']
FEATURE_VERSION = 'secondary-conflict-features-v3.1-rejection'
# Each group contributes hit, first/last relative start position, log(1 + capped count).
# The order also defines the four learned feature projections in model.py.
LEXICON = OrderedDict([
 ('context_task', '|'.join('(?:'+primary.LEXICON[k]+')' for k in primary.TASKS)),
 ('context_report', primary.LEXICON['source_report']),
 ('context_technical', primary.LEXICON['source_technical']),
 ('anger_attack', r'八嘎|(?<![节体早课情做])操(?![作控练劳心守场纵办持盘演行履刀舵琴切课])|艹|傻逼|傻[比笔]|蠢货|白痴|废物|垃圾(?!\s*(?:运输|清运|回收|分类|处理|焚烧|填埋|桶|袋|站|场|车))|无耻|下作|无赖|骗子|诈骗|狗屁|混蛋|王八蛋|智障|脑残|眼瞎|混账|无良|卑鄙|可耻|无能|半吊子|傻子|去死|滚|闭嘴|\b(?:idiot|moron|stupid|trash|garbage(?!\s+(?:truck|bin|bag|collection|recycling|disposal|classification))|scam|fraud|bullshit|fuck(?:ing)?|shut up|get lost|insane|incompetent|useless|ridiculous|sabotage)\b'),
 ('anger_blame', r'你(?:又|总|还|根本|完全)?.{0,12}(?:错|乱答|胡说|瞎说|骗人|敷衍|忽悠|搞砸|害|删|毁|捏造)|怪你|都是你|甩锅|推卸责任|乱收费|乱扣|欺骗|欺负|坑(?:我|人|钱)|不负责任|赔得起|瞎编|胡编|捏造|狡辩|前言不搭后语|\b(?:your fault|you lied|you are wrong|you messed|you deleted|you ruined|you broke|you fabricated|blame you|rip.?off|cheat(?:ed|ing)?)\b'),
 ('frustration_repeat', r'还是.{0,12}(?:不对|不行|错|失败|报错|卡)|依然.{0,12}(?:不对|不行|错|失败|报错|卡)|又.{0,8}(?:错|失败|报错|不对|卡住)|反复|一遍又一遍|第[一二三四五六七八九十\d]+次|试了.{0,8}次|重试|越改越|\b(?:again|still (?:fails?|broken|wrong)|keeps? fail(?:ing)?|repeatedly|over and over|tried .{0,16}times)\b'),
 ('frustration_stuck', r'卡住|卡死|搞不定|弄不好|跑不通|调不通|修不好|越弄越乱|抓狂|烦死|烦透|崩溃|疯了|折磨|报错|白屏|没有进展|\b(?:stuck|frustrat(?:ed|ing|ion)|driving me crazy|losing my mind|cannot fix|can.t fix|doesn.t work|not working|broken)\b'),
 ('impatience_urgency', r'快点|赶紧|赶快|马上|现在就|立刻|来不及|急死|急用|火急|在线等|抓紧|速度|还要等多久|\b(?:hurry|asap|urgent|quickly|right now|immediately|no time|running out of time)\b'),
 ('impatience_directive', r'说重点|讲重点|直接给|直接说|只给|只要(?:答案|结论|代码|步骤)|别(?:解释|废话|啰嗦|绕)|不要(?:解释|废话|啰嗦|绕)|少废话|一句话|简短点|别磨蹭|\b(?:just (?:give|tell|show)|cut to the chase|no explanation|skip the explanation|stop rambling|be brief|straight answer)\b'),
 ('disappointment_exhaustion', r'失望|寒心|心凉|心累|疲惫|精疲力竭|心力交瘁|心灰意冷|死心|耗尽|消磨|白忙|白费|没意义|毫无意义|无力|辜负|不值得|太累了|无话可说|\b(?:disappoint(?:ed|ing|ment)|let me down|exhausted|worn out|waste of time|pointless|fed up|lost faith|drained|fatigue|crushed my motivation)\b'),
 ('disappointment_withdrawal', r'算了|不弄了|不折腾了?|放弃|不想(?:继续|再弄|再试|管了|再做|再为|再问)|不用了|就这样吧|爱咋咋地|彻底不|关电脑|关机|卸载|到此结束|就此打住|弃坑|不干了|不问了|自己重构|回滚|删了|不再(?:相信|指望|期待)|\b(?:give up|giving up|forget it|never mind|not worth it|I.m done|I am done|won.t try again|no longer trust|quitting|quit|surrender|abandon|rolling back|revert(?:ing)?)\b'),
 ('modifier_negation', primary.LEXICON['modifier_negation']),
 ('modifier_transition', r'还是|依然|仍然|又|再|已经|终于|结果|但是|却|反而|现在|\b(?:still|again|already|finally|but|however|yet|anymore)\b'),
 ('antecedent_failure', r'失败|报错|错误|不对|不行|无效|故障|漏洞|崩溃|卡|重试|\b(?:fail(?:ed|ure)?|error|bug|broken|crash|retry|wrong)\b'),
 ('antecedent_expectation', r'本来以为|原本以为|以为|原本|本来|期待|指望|说好|答应|承诺|保证|信誓旦旦|之前|\b(?:expected|expectation|promised|promise|supposed to|thought it would|counted on|trusted)\b'),
])
PATTERNS = OrderedDict((key, re.compile(pattern, re.I)) for key, pattern in LEXICON.items())

# Exact whole-message rules only. An isolated word inside a longer sentence is a feature,
# never a hard decision. Conflicts against the supplied labels are audited during splitting.
RULES = [
 ('S_CON_ANGER', r'(?:滚|闭嘴|傻逼|无耻|骗子)[!！。．.]*', 'anger'),
 ('S_CON_FRUSTRATION', r'(?:还是不对|又报错了|又错了|弄不好|搞不定)[!！。．.]*', 'frustration'),
 ('S_CON_IMPATIENCE', r'(?:快点|赶紧|说重点|别解释|不要解释|直接给答案|少废话)[!！。．.]*', 'impatience'),
 ('S_CON_DISAPPOINTMENT', r'(?:不弄了|不折腾了|放弃了|彻底失望了)[!！。．.]*', 'disappointment'),
]
COMPILED_RULES = [(key, re.compile(pattern, re.I), label) for key, pattern, label in RULES]
STRUCTURE = ['log_chars', 'log_clauses', 'last_clause_ratio', 'question_log', 'exclamation_log',
             'ellipsis_log', 'max_punct_run_log', 'quote_span_ratio', 'colon_present',
             'colon_first_position', 'first_person_present', 'second_person_present']
COMBO = ['attack_you_near', 'blame_you_near', 'repeat_failure_near', 'stuck_task_near',
         'urgent_directive_near', 'exhaustion_withdrawal_near', 'expectation_exhaustion_near',
          'conflict_tail_gap', 'attack_devaluation_near', 'attack_without_repeat']
HISTORY = ['nearest_char_jaccard', 'max_char_jaccard', 'weighted_char_jaccard',
           'nearest_word_jaccard', 'nearest_failure_hit', 'weighted_failure_hit',
           'nearest_expectation_hit', 'weighted_expectation_hit', 'nearest_task_hit',
           'weighted_task_hit', 'failure_to_repeat', 'expectation_to_withdrawal']
FEATURE_NAMES = ([f'{key}.{suffix}' for key in LEXICON for suffix in
                  ('hit', 'first_position', 'last_position', 'count_log')]
                 + ['structure.'+key for key in STRUCTURE]
                 + ['combo.'+key for key in COMBO]
                 + ['history.'+key for key in HISTORY])
assert len(FEATURE_NAMES) == 94


def secondary_rule(text):
    t = text.strip()
    if primary.direct_agent_insult(t):
        return {'secondary_label': 'anger', 'decision_source': 'rule',
                'rule_id': 'S_CON_AGENT_INSULT', 'secondary_probabilities': None}
    for rule_id, pattern, label in COMPILED_RULES:
        if pattern.fullmatch(t):
            return {'secondary_label': label, 'decision_source': 'rule', 'rule_id': rule_id,
                    'secondary_probabilities': None}
    return None


def extract(row):
    t = row['current_user_message'].strip()
    if not t:
        raise ValueError('Empty current_user_message')
    length = len(t)
    hits = {key: primary.spans(pattern, t) for key, pattern in PATTERNS.items()}
    result = OrderedDict()
    for key, spans in hits.items():
        values = (float(bool(spans)), spans[0][0]/length if spans else 0.,
                  spans[-1][0]/length if spans else 0., math.log1p(min(len(spans), 5)))
        for suffix, value in zip(('hit', 'first_position', 'last_position', 'count_log'), values):
            result[key+'.'+suffix] = value
    parent = primary.extract_features(row)
    for key in STRUCTURE:
        result['structure.'+key] = parent['structure.'+key]

    def union(keys):
        return sorted({span for key in keys for span in hits[key]})

    anger = union(['anger_attack', 'anger_blame'])
    frustration = union(['frustration_repeat', 'frustration_stuck'])
    impatience = union(['impatience_urgency', 'impatience_directive'])
    disappointment = union(['disappointment_exhaustion', 'disappointment_withdrawal'])
    all_conflict = sorted(set(anger+frustration+impatience+disappointment))
    you = primary.spans(primary.SECOND_PERSON, t)

    def near(a, b):
        return float(any(primary.gap(x, y) <= 24 for x in a for y in b))

    combos = (near(hits['anger_attack'], you), near(hits['anger_blame'], you),
              near(hits['frustration_repeat'], hits['antecedent_failure']),
              near(hits['frustration_stuck'], hits['context_task']),
              near(hits['impatience_urgency'], hits['impatience_directive']),
              near(hits['disappointment_exhaustion'], hits['disappointment_withdrawal']),
              near(hits['antecedent_expectation'], disappointment),
               (length-max(end for _, end in all_conflict))/length if all_conflict else 0.,
               near(primary.spans(primary.ATTACK_WORD,t), primary.spans(primary.ABILITY_DEVALUATION,t)),
               float(bool(primary.ATTACK_WORD.search(t)) and not primary.REPEAT.search(t)))
    result.update(('combo.'+key, value) for key, value in zip(COMBO, combos))

    hs = [h.strip() for h in row.get('history_user_messages', [])[:4]]

    def history_hits(key):
        return [float(bool(PATTERNS[key].search(h))) for h in hs]

    def weighted(values):
        return (sum(v*2**(-i) for i, v in enumerate(values)) /
                sum(2**(-i) for i in range(len(values)))) if values else 0.

    failure = history_hits('antecedent_failure')
    expectation = history_hits('antecedent_expectation')
    task = history_hits('context_task')
    hvalues = ([parent['history.'+key] for key in HISTORY[:4]]
               + [failure[0] if failure else 0., weighted(failure),
                  expectation[0] if expectation else 0., weighted(expectation),
                  task[0] if task else 0., weighted(task),
                  weighted(failure)*float(bool(hits['frustration_repeat'])),
                  weighted(expectation)*float(bool(hits['disappointment_withdrawal']))])
    result.update(('history.'+key, float(value)) for key, value in zip(HISTORY, hvalues))
    assert list(result) == BASE_FEATURE_NAMES and all(math.isfinite(v) for v in result.values())
    return result


BASE_FEATURE_NAMES = FEATURE_NAMES[:]
SCOPE_FEATURE_NAMES = primary.SCOPE_FEATURE_NAMES
_base_extract = extract
FEATURE_NAMES = BASE_FEATURE_NAMES + SCOPE_FEATURE_NAMES
def extract(row):
    values = _base_extract(row)
    values.update(primary.scope_features(row['current_user_message'].strip()))
    values['scope.material_history_short']=float(len(row['current_user_message'].strip())<=9 and primary.active_material_task(row.get('history_user_messages',[])))
    assert list(values) == FEATURE_NAMES
    return values

_legacy_secondary_rule=secondary_rule
def secondary_rule(row):
    record=row if isinstance(row,dict) else {'current_user_message':row,'history_user_messages':[]}
    label=primary.decision(record)
    if label is not None:
        return primary.context_secondary_rule(record,'conflict_repair')
    if primary.active_material_task(record.get('history_user_messages',[])):
        return None
    return _legacy_secondary_rule(record['current_user_message'])

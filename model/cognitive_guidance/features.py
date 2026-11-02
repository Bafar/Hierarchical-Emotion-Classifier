"""Three-way cognitive-guidance rules and 94 current/history text features."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import math
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('primary_features_cognitive', ROOT/'primary_emotion_classifier/features.py')
primary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary)

LABELS = ['confusion', 'skepticism', 'curiosity', 'not_this_branch']
FEATURE_VERSION = 'secondary-cognitive-features-v3-rejection'
# Three context, nine class-specific, three modifier groups. Each group has
# hit, first/last relative start position, and capped log-count.
LEXICON = OrderedDict([
 ('context_task', '|'.join('(?:'+primary.LEXICON[k]+')' for k in primary.TASKS)),
 ('context_report', primary.LEXICON['source_report']),
 ('context_technical', primary.LEXICON['source_technical']),
 ('confusion_explicit', r'不懂|看不懂|没看懂|听不懂|没听懂|不明白|没明白|不理解|糊涂|困惑|迷糊|[懵蒙]圈|[懵蒙]逼|懵了|绕晕|一头雾水|转不过弯|搞不清|脑子乱|看不出来|听不出来|\b(?:confus(?:ed|ing|ion)|lost me|totally lost|cannot follow|can.t follow|do not understand|don.t understand|cannot make sense|can.t make sense|makes no sense|baffled)\b'),
 ('confusion_inconsistency', r'为什么|为何|既然.{0,50}(?:为什么|怎么|为何)|明明.{0,50}(?:为什么|怎么|却|反而)|怎么(?:还|会|反而|又|就)|前后矛盾|逻辑(?:不通|混乱|颠倒)|对不上|说不通|到底(?:哪个|怎么|为何|为什么)|\b(?:if .{0,60} why|why (?:does|did|is|are|would|can)|how come|doesn.t add up|do not add up|contradict(?:s|ory|ion))\b'),
 ('confusion_reexplain', r'通俗|白话|大白话|讲清楚|解释清楚|再讲(?:一遍|一次|清楚)|重讲|重新解释|拆解|拆开|捋一捋|一步步|哪里没懂|什么意思|能不能解释|帮我理解|\b(?:explain again|explain simply|break it down|walk me through|what does .{0,35} mean|clarify|simpler terms)\b'),
 ('skepticism_challenge', r'你确定|您确定|确定吗|真的假的|不对|错了|有误|不准确|不可能|不靠谱|靠谱吗|站不住脚|瞎编|胡编|瞎扯|编造|幻觉|捏造|谣言|质疑|怀疑|有问题|算错|说错|搞错|不可信|\b(?:are you sure|really\?|doubt|skeptic(?:al|ism)?|hallucinat(?:e|ed|ion)|fabricat(?:e|ed|ion)|made up|not correct|you are wrong|unreliable|not true|inaccurate|false|guessing|wrong)\b'),
 ('skepticism_evidence', r'依据|证据|出处|来源|引用|文献|参考文献|官方|官网|原始数据|统计公报|报告|论文|实测|核实|验证|DOI|\b(?:evidence|citation|source|references?|official|documentation|docs|paper|study|verified?|fact.?check|published|records?)\b'),
 ('skepticism_counterevidence', r'我查(?:了|过)|我核对|我测(?:了|过)|我翻(?:了|过)|实际上|事实是|根本没|恰恰相反|正好相反|完全相反|反了|经不起推敲|自相矛盾|对不上|和.{0,15}不一样|与.{0,20}(?:不符|矛盾)|没写|查无此项|\b(?:I checked|I tested|actually|in fact|contradicts?|but the docs|on the contrary|does not match|doesn.t match|does not add up|doesn.t add up)\b'),
 ('curiosity_interest', r'好奇|有意思|真有趣|好神奇|奇妙|脑洞|想知道|想了解|探究|探索|探讨|聊聊|讲讲|说说|研究一下|\b(?:curious|curiosity|interesting|fascinat(?:ing|ed)|wonder|intriguing|what if|explore)\b'),
 ('curiosity_depth', r'底层|原理|机理|机制|深挖|深入|深讲|展开|进阶|剖析|源码|汇编|设计哲学|内部实现|背后|更进一步|细节|硬核|\b(?:under the hood|internals?|deep dive|deeper|in depth|mechanism|underlying|how does .{0,35} work|how it works|implementation details?)\b'),
 ('curiosity_hypothesis', r'如果|假设|要是|倘若|若(?:把|考虑|需要|发生|没有|换成)|会怎样|会怎么样|还能|能否进一步|极端|边界|变体|发散|扩展|换个场景|\b(?:what if|suppose|assuming|hypothetical|could we|what would happen|further|even more|different scenario)\b'),
 ('modifier_negation', primary.LEXICON['modifier_negation']),
 ('modifier_backref', r'刚才|前面|上面|之前|你说的|你给的|你的回答|这个方案|这个思路|顺着|基于|接着|进一步|\b(?:earlier|previous|you said|you gave|your answer|that approach|this approach|following up|building on)\b'),
 ('modifier_acceptance', r'懂了|明白了|理解了|学到了|认可|不错|很好|很妙|巧妙|有效|管用|解决了|跑通了|清晰|正确|对路|\b(?:makes sense|I understand|got it|nice|clever|works well|worked|solved|great approach|good idea)\b'),
])
PATTERNS = OrderedDict((key, re.compile(pattern, re.I)) for key, pattern in LEXICON.items())
HISTORY_DOUBT = re.compile(LEXICON['skepticism_challenge']+'|'+LEXICON['skepticism_counterevidence'], re.I)

RULES = [
 ('S_COG_CONFUSION', r'(?:看不懂|没看懂|听不懂|没听懂|一头雾水|彻底懵圈了|完全懵了|我没明白|[?？]+)[!！。．.]*', 'confusion'),
 ('S_COG_SKEPTICISM', r'(?:你确定吗|您确定吗|你确定|真的假的|证据呢|出处呢|来源呢|站不住脚|缺乏事实依据)[?？!！。．.]*', 'skepticism'),
 ('S_COG_CURIOSITY', r'(?:好神奇|真有意思|我很好奇|想深入了解|继续深挖)[!！。．.]*', 'curiosity'),
]
COMPILED_RULES = [(key, re.compile(pattern, re.I), label) for key, pattern, label in RULES]
STRUCTURE = ['log_chars', 'log_clauses', 'last_clause_ratio', 'question_log', 'exclamation_log',
             'ellipsis_log', 'max_punct_run_log', 'quote_span_ratio', 'colon_present',
             'colon_first_position', 'first_person_present', 'second_person_present']
COMBO = ['confusion_inconsistency_reexplain', 'confusion_explicit_reexplain',
         'skepticism_challenge_evidence', 'skepticism_counterevidence_challenge',
         'curiosity_interest_depth', 'curiosity_hypothesis_depth',
         'acceptance_depth', 'cognitive_tail_gap',
         'skepticism_target_challenge_near', 'skepticism_short_direct']
HISTORY = ['nearest_char_jaccard', 'max_char_jaccard', 'weighted_char_jaccard',
           'nearest_word_jaccard', 'nearest_acceptance_hit', 'weighted_acceptance_hit',
           'nearest_doubt_hit', 'weighted_doubt_hit', 'nearest_blocked_hit',
           'weighted_blocked_hit', 'acceptance_to_curiosity', 'doubt_to_skepticism']
FEATURE_NAMES = ([f'{key}.{suffix}' for key in LEXICON for suffix in
                  ('hit', 'first_position', 'last_position', 'count_log')]
                 + ['structure.'+key for key in STRUCTURE]
                 + ['combo.'+key for key in COMBO]
                 + ['history.'+key for key in HISTORY])
assert len(FEATURE_NAMES) == 94


def secondary_rule(text):
    t = text.strip()
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

    all_cognitive = union(list(LEXICON)[3:12])

    def near(a, b):
        return float(any(primary.gap(x, y) <= 24 for x in a for y in b))

    combos = (near(hits['confusion_inconsistency'], hits['confusion_reexplain']),
              near(hits['confusion_explicit'], hits['confusion_reexplain']),
              near(hits['skepticism_challenge'], hits['skepticism_evidence']),
              near(hits['skepticism_counterevidence'], hits['skepticism_challenge']),
              near(hits['curiosity_interest'], hits['curiosity_depth']),
              near(hits['curiosity_hypothesis'], hits['curiosity_depth']),
              near(hits['modifier_acceptance'], hits['curiosity_depth']),
              (length-max(end for _, end in all_cognitive))/length if all_cognitive else 0.,
              near(primary.spans(primary.SKEPTICISM_TARGET,t), primary.spans(primary.SKEPTICISM_CHALLENGE,t)),
              float(length <= 24 and bool(primary.SKEPTICISM_CHALLENGE.search(t)) and
                    not hits['context_task'] and not primary.QUOTE.search(t)))
    result.update(('combo.'+key, value) for key, value in zip(COMBO, combos))

    hs = [h.strip() for h in row.get('history_user_messages', [])[:4]]
    def history_hits(pattern):
        return [float(bool(pattern.search(h))) for h in hs]

    def weighted(values):
        return (sum(v*2**(-i) for i, v in enumerate(values)) /
                sum(2**(-i) for i in range(len(values)))) if values else 0.

    acceptance = history_hits(PATTERNS['modifier_acceptance'])
    doubt = history_hits(HISTORY_DOUBT)
    blocked = history_hits(PATTERNS['confusion_explicit'])
    hvalues = ([parent['history.'+key] for key in HISTORY[:4]]
               + [acceptance[0] if acceptance else 0., weighted(acceptance),
                  doubt[0] if doubt else 0., weighted(doubt),
                  blocked[0] if blocked else 0., weighted(blocked),
                  weighted(acceptance)*float(bool(union(['curiosity_interest', 'curiosity_depth', 'curiosity_hypothesis']))),
                  weighted(doubt)*float(bool(union(['skepticism_challenge', 'skepticism_evidence', 'skepticism_counterevidence'])))])
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
        return primary.context_secondary_rule(record,'cognitive_guidance')
    if primary.active_material_task(record.get('history_user_messages',[])):
        return None
    return _legacy_secondary_rule(record['current_user_message'])

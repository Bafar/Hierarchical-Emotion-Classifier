"""Seven-way positive-affect rules and 124 interpretable text features."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import math
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('primary_features_positive_affect', ROOT/'primary_emotion_classifier/features.py')
primary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary)

LABELS = ['affection', 'dependency', 'hope', 'joy', 'pride', 'touched', 'playfulness', 'caring', 'not_this_branch']
FEATURE_VERSION = 'secondary-positive-affect-features-v2-rejection'
LEXICON = OrderedDict([
 ('context_task', '|'.join('(?:'+primary.LEXICON[k]+')' for k in primary.TASKS)),
 ('context_report', primary.LEXICON['source_report']),
 ('context_technical', primary.LEXICON['source_technical']),
 ('affection_warmth', r'喜欢|喜爱|爱你|好爱|宠爱|暖心|贴心|温柔|亲切|心动|可爱|乖|萌|甜|\b(?:love|like|adore|fond of|affection|adorable|lovely|sweet)\b'),
 ('affection_appreciation', r'漂亮|美丽|好看|好吃|好喝|舒服|软糯|香|优雅|欣赏|赏心悦目|\b(?:beautiful|pretty|gorgeous|delicious|cozy|admire|appreciate)\b'),
 ('dependency_exclusive', r'只有你|唯有你|唯一|只剩你|全靠你|离不开你|不能没有你|非你不可|依赖你|精神寄托|\b(?:only you|my only|no one else|cannot live without you|depend on you|rely on you)\b'),
 ('dependency_separation', r'别离开|不要离开|别消失|永远陪|一直陪|别扔下|不要抛弃|失去你|\b(?:never leave|don.t leave|stay with me|lose you|abandon me)\b'),
 ('hope_wish', r'希望|期盼|期待|盼望|盼着|但愿|真盼|祈祷|祈求|保佑|祝我好运|心愿|愿望|\b(?:hope|hoping|hopeful|wish|pray|fingers crossed|good luck)\b'),
 ('hope_future', r'明天|下周|下个月|将来|未来|快要|即将|还没|尚未|要是能|如果能|过几天|\b(?:tomorrow|next week|future|upcoming|soon|might)\b'),
 ('joy_feeling', primary.LEXICON['expr_joy'] + r'|心情.*(?:好|亮)|乐出声|乐坏|小确幸|\b(?:good mood|made my day|grinning|pure joy)\b'),
 ('joy_laughter', primary.PURE_JOY_VOCALIZATION + r'|哈{2,}|笑得|逗笑|乐得|嘴角|\b(?:haha|hahaha|laughed out loud|cannot stop smiling)\b'),
 ('pride_achievement', r'拿到.{0,12}(?:offer|录取|奖|冠军)|考过|通过|入选|夺冠|得奖|获奖|赢了|破纪录|突破|达成|做到了|完成|成功|毕业|发表|上线|\b(?:got the job|accepted|won|champion|award|passed|achiev|graduated|personal best)\b'),
 ('pride_self_eval', r'骄傲|自豪|成就感|我真厉害|我真棒|佩服自己|证明自己|夸夸我|为自己鼓掌|\b(?:proud|pride|accomplished|impressed with myself|proved myself|credit)\b'),
 ('touched_trigger', r'善意|心意|惦记|记得我|默默|重逢|团聚|陪着|拥抱|手写|留着|等我|\b(?:kindness|remembered|reunion|stayed with me|saved for me|kept the)\b'),
 ('touched_reaction', r'感动|触动|动容|泪目|鼻酸|眼眶.{0,4}(?:热|湿|红)|热泪|感动哭|暖到|戳心|心里.{0,4}(?:暖|软)|\b(?:touched|moved|moving|heartwarming|teared up|eyes are wet|heart feels full)\b'),
 ('playfulness_tease', r'开玩笑|逗你|骗到你|上当|略略|调侃|捉弄|才不告诉你|有本事|叫声|\b(?:joking|teasing|fooled you|got you|just kidding)\b'),
 ('playfulness_persona', r'叫(?:爸爸|主人|师父)|假装|扮演|角色扮演|本王|小助手|外星人|\b(?:call me daddy|call me master|pretend|roleplay|your creator)\b'),
 ('modifier_negation', primary.LEXICON['modifier_negation']),
 ('modifier_hypothesis', primary.LEXICON['modifier_hypothesis']),
 ('antecedent_distress', r'孤独|寂寞|没人|害怕|焦虑|崩溃|痛苦|难过|\b(?:lonely|alone|afraid|anxious|despair|distress)\b'),
 ('antecedent_effort', r'努力|练习|准备|坚持|熬夜|备考|训练|排练|加班|反复|终于|\b(?:worked hard|practice|trained|studied|prepared|finally)\b'),
])
PATTERNS = OrderedDict((key, re.compile(pattern, re.I)) for key, pattern in LEXICON.items())
RULES = [
 ('S_AFFECT_PURE_JOY', primary.PURE_JOY_VOCALIZATION, 'joy'),
 ('S_AFFECT_PURE_PLAYFUL', primary.PURE_PLAYFUL_VOCALIZATION, 'playfulness'),
 ('S_AFFECT_AFFECTION', r'(?:喜欢你|好喜欢你|爱你|我爱你|你真好|好可爱|太可爱了)[!！。．.]*', 'affection'),
 ('S_AFFECT_DEPENDENCY', r'(?:只有你懂我|只有你理解我|不要离开我|别离开我|我不能没有你|你是我唯一的依靠)[!！。．.]*', 'dependency'),
 ('S_AFFECT_HOPE', r'(?:求保佑|求好运|祝我好运|保佑我吧|希望一切顺利|fingers crossed|wish me luck)[!！。．.]*', 'hope'),
]
COMPILED_RULES = [(key, re.compile(pattern, re.I), label) for key, pattern, label in RULES]
STRUCTURE = ['log_chars','log_clauses','last_clause_ratio','question_log','exclamation_log','ellipsis_log',
             'max_punct_run_log','quote_span_ratio','colon_present','colon_first_position',
             'first_person_present','second_person_present']
COMBO = ['wish_future_near','joy_laughter_near','achievement_pride_near','touch_trigger_reaction_near',
         'tease_laughter_near','love_exclusive_near','distress_separation_near','effort_achievement_near',
         'cry_touched_near','cry_without_touched','task_affect_near','negation_affect_near',
         'hypothesis_wish_near','affect_tail_gap']
HISTORY = ['nearest_char_jaccard','max_char_jaccard','weighted_char_jaccard','nearest_word_jaccard',
           'nearest_distress_hit','weighted_distress_hit','nearest_effort_hit','weighted_effort_hit',
           'nearest_touch_hit','weighted_touch_hit','nearest_task_hit','weighted_task_hit',
           'distress_to_dependency','effort_to_pride']
FEATURE_NAMES = ([f'{key}.{suffix}' for key in LEXICON for suffix in ('hit','first_position','last_position','count_log')]
                 + ['structure.'+key for key in STRUCTURE] + ['combo.'+key for key in COMBO]
                 + ['history.'+key for key in HISTORY])
assert len(FEATURE_NAMES) == 124

def secondary_rule(text):
    t = text.strip()
    for rule_id, pattern, label in COMPILED_RULES:
        if pattern.fullmatch(t):
            return {'secondary_label': label, 'decision_source': 'rule', 'rule_id': rule_id,
                    'secondary_probabilities': None}
    return None

def extract(row):
    t = row['current_user_message'].strip()
    if not t: raise ValueError('Empty current_user_message')
    length = len(t); hits = {key: primary.spans(pattern, t) for key, pattern in PATTERNS.items()}
    result = OrderedDict()
    for key, spans in hits.items():
        vals = (float(bool(spans)), spans[0][0]/length if spans else 0., spans[-1][0]/length if spans else 0., math.log1p(min(len(spans), 5)))
        for suffix, value in zip(('hit','first_position','last_position','count_log'), vals): result[key+'.'+suffix] = value
    parent = primary.extract_features(row)
    for key in STRUCTURE: result['structure.'+key] = parent['structure.'+key]
    def union(keys): return sorted({span for key in keys for span in hits[key]})
    def near(a, b): return float(any(primary.gap(x, y) <= 24 for x in a for y in b))
    affect = union(list(LEXICON)[3:17]); cry = primary.spans(re.compile(primary.CRY_EXPRESSION, re.I), t); touched = hits['touched_reaction']
    combos = (near(hits['hope_wish'],hits['hope_future']), near(hits['joy_feeling'],hits['joy_laughter']),
              near(hits['pride_achievement'],hits['pride_self_eval']), near(hits['touched_trigger'],touched),
              near(hits['playfulness_tease'],hits['joy_laughter']), near(hits['affection_warmth'],hits['dependency_exclusive']),
              near(hits['antecedent_distress'],hits['dependency_separation']), near(hits['antecedent_effort'],hits['pride_achievement']),
              near(cry,touched), float(bool(cry) and not touched), near(hits['context_task'],affect),
              near(hits['modifier_negation'],affect), near(hits['modifier_hypothesis'],hits['hope_wish']),
              (length-max(end for _,end in affect))/length if affect else 0.)
    result.update(('combo.'+key,float(value)) for key,value in zip(COMBO,combos))
    hs = [x.strip() for x in row.get('history_user_messages',[])[:4]]
    def hist(key): return [float(bool(PATTERNS[key].search(h))) for h in hs]
    def weighted(v): return sum(x*2**(-i) for i,x in enumerate(v))/sum(2**(-i) for i in range(len(v))) if v else 0.
    distress,effort,touch,task = hist('antecedent_distress'),hist('antecedent_effort'),hist('touched_trigger'),hist('context_task')
    hvals = ([parent['history.'+key] for key in HISTORY[:4]] +
             [distress[0] if distress else 0.,weighted(distress),effort[0] if effort else 0.,weighted(effort),
              touch[0] if touch else 0.,weighted(touch),task[0] if task else 0.,weighted(task),
              weighted(distress)*float(bool(hits['dependency_separation'])),weighted(effort)*float(bool(hits['pride_achievement']))])
    result.update(('history.'+key,float(value)) for key,value in zip(HISTORY,hvals))
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
        return primary.context_secondary_rule(record,'positive_affect')
    if primary.active_material_task(record.get('history_user_messages',[])):
        return None
    return _legacy_secondary_rule(record['current_user_message'])

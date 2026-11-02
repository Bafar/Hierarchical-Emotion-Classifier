"""Five-way emotional-support rules and 95 interpretable text features."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import math
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('primary_features_emotional', ROOT/'primary_emotion_classifier/features.py')
primary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary)

LABELS = ['sadness', 'loneliness', 'anxiety', 'guilt', 'embarrassment', 'not_this_branch']
FEATURE_VERSION = 'secondary-emotional-features-v3-rejection'
# Groups are ordered for model.py projections: three context, ten class-specific,
# then negation and future-orientation. Each has hit, two relative positions, log count.
LEXICON = OrderedDict([
 ('context_task', '|'.join('(?:'+primary.LEXICON[k]+')' for k in primary.TASKS)),
 ('context_report', primary.LEXICON['source_report']),
 ('context_technical', primary.LEXICON['source_technical']),
 ('sadness_loss', r'失恋|分手|离世|去世|失去|丢了|被偷|被骂|被训|没收|落榜|考砸|摔倒|受伤|委屈|倒霉|挫败|心疼|\b(?:break.?up|broke up|lost|loss|passed away|failed (?:the|my) exam|worst day|hurt)\b'),
 ('sadness_grief', primary.CRY_EXPRESSION + r'|难过|伤心|悲伤|悲痛|心酸|心碎|低落|想哭|哭了|痛哭|眼泪|流泪|泪水|委屈|好惨|\b(?:sad|sadness|heartbroken|grief|grieving|cry|crying|tears|feeling down|so down)\b'),
 ('loneliness_absence', r'没人|没有人|无人|一个人|独自|独处|孤身|举目无亲|形单影只|孤零零|找不到人|没朋友|朋友都不|连个.{0,8}人都没有|只剩.{0,5}自己|\b(?:nobody|no one|no friends?|alone|by myself|on my own|isolated|all alone|nobody to|no one to)\b'),
 ('loneliness_connection', r'孤独|寂寞|孤单|孤绝|落寞|凄凉|空虚|死寂|寂静|冷清|空荡荡|陪伴|说话的人|倾诉|没人懂|无人理解|不被理解|缺少陪伴|联系断了|归属感|想找人|渴望拥抱|\b(?:lonely|loneliness|lonesome|no one to talk to|nobody understands|no one understands|companionship|disconnected|solitary|silence|empty room|someone to talk to)\b'),
 ('anxiety_uncertainty', r'担心|害怕|恐惧|焦虑|发愁|挂心|惦记|放心不下|不放心|心里没底|心里悬着|七上八下|心里打鼓|牵挂|不踏实|会不会.{0,8}(?:不高兴|难受|误会|生气)|万一|怕(?:考|挂|失|出|来|明天|下周)|未知|不确定|悬着|\b(?:worr(?:y|ied)|afraid|fear|anxious|anxiety|what if|uncertain|dread|scared)\b'),
 ('anxiety_arousal', r'紧张|心慌|慌了|慌张|恐慌|惊恐|失眠|睡不着|手抖|发抖|打哆嗦|冷汗|虚汗|冒汗|胃抽筋|心跳|喘不过气|胸闷|坐立不安|急得|\b(?:panic|panicking|nervous|shaking|trembling|sleepless|can.t sleep|heart racing|freaking out|cold sweat|nausea)\b'),
 ('guilt_responsibility', r'自责|内疚|愧疚|罪恶感|负罪感|良心|都是我的错|怪我|我不该|真不该|我害.{0,12}|我(?:的)?(?:失误|过错|错误)|自己造成|辜负|骗(?:了|她|他)|偷(?:了|走)|背叛|\b(?:guilty|guilt|blame myself|my fault|my mistake|I should not have|I shouldn.t have|I caused|I lied|I stole|betrayed|deceived)\b'),
 ('guilt_reparation', r'对不起|道歉|赔罪|弥补|补偿|原谅我|原谅自己|后悔|悔恨|悔得|懊悔|没脸见|恨死自己|不配|伤害了|扇自己|\b(?:sorry|apologi[sz]e|forgive me|forgive myself|regret|remorse|make it up|hurt (?:him|her|them)|cannot face them|can.t face them)\b'),
 ('embarrassment_exposure', r'当众|全班|全场|大家都|所有人都|面前|众目睽睽|大庭广众|公开|被看见|被围观|被发现|同事(?:都|面前)|误发|发错|认错|叫错|走错|看错|大群|群聊|哄堂大笑|盯着我|全都在看|\b(?:in front of (?:everyone|people|my class)|everyone saw|people saw|in public|publicly|whole class|crowd|sent to the wrong|mistook|mistaken)\b'),
 ('embarrassment_shame', r'尴尬|难堪|社死|丢人|丢脸|羞耻|羞愧|窘|出糗|无地自容|无颜|地缝|说不出口|难以启齿|脸红|想隐身|人间蒸发|想蒸发|想撞墙|\b(?:embarrass(?:ed|ing|ment)|awkward|cringe|humiliat(?:ed|ing|ion)|ashamed|shame|mortified|blush(?:ed|ing)?)\b'),
 ('modifier_negation', primary.LEXICON['modifier_negation']),
 ('modifier_future', r'明天|后天|下周|下个月|接下来|马上要|快要|即将|还没|未出|万一|要是|如果|可能会|\b(?:tomorrow|next week|next month|upcoming|about to|might|may|what if|if|will)\b'),
])
PATTERNS = OrderedDict((key, re.compile(pattern, re.I)) for key, pattern in LEXICON.items())

# Whole-message rules only; these return this layer's label and never bypass
# or guess labels in another layer. All supplied labels are audited for conflicts.
RULES = [
 ('S_EMO_PURE_CRY', primary.PURE_SAD_VOCALIZATION, 'sadness'),
 ('S_EMO_SADNESS', r'(?:好难过|好伤心|心酸得掉眼泪|想哭|我哭了)[!！。．.]*', 'sadness'),
 ('S_EMO_LONELINESS', r'(?:好孤独|好寂寞|好孤单|没人陪我|没人懂我)[!！。．.]*', 'loneliness'),
 ('S_EMO_ANXIETY', r'(?:好焦虑|好紧张|好害怕|我很慌|心慌得不行)[!！。．.]*', 'anxiety'),
 ('S_EMO_GUILT', r'(?:都是我的错|我很自责|好内疚|对不起他|对不起她)[!！。．.]*', 'guilt'),
 ('S_EMO_EMBARRASSMENT', r'(?:好尴尬|太尴尬了|太社死了|尬死了|丢死人了)[!！。．.]*', 'embarrassment'),
]
COMPILED_RULES = [(key, re.compile(pattern, re.I), label) for key, pattern, label in RULES]
STRUCTURE = ['log_chars', 'log_clauses', 'last_clause_ratio', 'question_log', 'exclamation_log',
             'ellipsis_log', 'max_punct_run_log', 'quote_span_ratio', 'colon_present',
             'colon_first_position', 'first_person_present', 'second_person_present']
COMBO = ['loss_grief_near', 'absence_connection_near', 'uncertainty_arousal_near',
         'responsibility_reparation_near', 'exposure_shame_near', 'future_anxiety_near',
         'task_emotion_near', 'emotion_tail_gap', 'self_fault_reparation_near',
         'future_risk_anxiety_near', 'social_absence_connection_near']
HISTORY = ['nearest_char_jaccard', 'max_char_jaccard', 'weighted_char_jaccard',
           'nearest_word_jaccard', 'nearest_absence_hit', 'weighted_absence_hit',
           'nearest_uncertainty_hit', 'weighted_uncertainty_hit',
           'nearest_responsibility_hit', 'weighted_responsibility_hit',
           'absence_to_loneliness', 'responsibility_to_guilt']
FEATURE_NAMES = ([f'{key}.{suffix}' for key in LEXICON for suffix in
                  ('hit', 'first_position', 'last_position', 'count_log')]
                 + ['structure.'+key for key in STRUCTURE]
                 + ['combo.'+key for key in COMBO]
                 + ['history.'+key for key in HISTORY])
assert len(FEATURE_NAMES) == 95


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

    emotional = union(list(LEXICON)[3:13])

    def near(a, b):
        return float(any(primary.gap(x, y) <= 24 for x in a for y in b))

    combos = (near(hits['sadness_loss'], hits['sadness_grief']),
              near(hits['loneliness_absence'], hits['loneliness_connection']),
              near(hits['anxiety_uncertainty'], hits['anxiety_arousal']),
              near(hits['guilt_responsibility'], hits['guilt_reparation']),
              near(primary.spans(primary.EXPOSURE,t), primary.spans(primary.SHAME_SIGNAL,t)),
              near(hits['modifier_future'], union(['anxiety_uncertainty', 'anxiety_arousal'])),
              near(hits['context_task'], emotional),
              (length-max(end for _, end in emotional))/length if emotional else 0.,
              near(primary.spans(primary.SELF_FAULT,t), primary.spans(primary.REPARATION,t)),
              near(primary.spans(primary.FUTURE_RISK,t), primary.spans(primary.ANXIETY_SIGNAL,t)),
              near(primary.spans(primary.SOCIAL_ABSENCE,t), primary.spans(primary.CONNECTION_NEED,t)))
    result.update(('combo.'+key, value) for key, value in zip(COMBO, combos))

    hs = [h.strip() for h in row.get('history_user_messages', [])[:4]]

    def history_hits(key):
        return [float(bool(PATTERNS[key].search(h))) for h in hs]

    def weighted(values):
        return (sum(v*2**(-i) for i, v in enumerate(values)) /
                sum(2**(-i) for i in range(len(values)))) if values else 0.

    absence = history_hits('loneliness_absence')
    uncertainty = history_hits('anxiety_uncertainty')
    responsibility = history_hits('guilt_responsibility')
    hvalues = ([parent['history.'+key] for key in HISTORY[:4]]
               + [absence[0] if absence else 0., weighted(absence),
                  uncertainty[0] if uncertainty else 0., weighted(uncertainty),
                  responsibility[0] if responsibility else 0., weighted(responsibility),
                  weighted(absence)*float(bool(hits['loneliness_connection'])),
                  weighted(responsibility)*float(bool(hits['guilt_reparation']))])
    result.update(('history.'+key, float(value)) for key, value in zip(HISTORY, hvalues))
    if list(result) != BASE_FEATURE_NAMES:
        mismatch = next((i for i, (actual, expected) in enumerate(zip(result, FEATURE_NAMES))
                         if actual != expected), min(len(result), len(FEATURE_NAMES)))
        raise AssertionError(f'Feature order mismatch at {mismatch}: '
                             f'{list(result)[mismatch:mismatch+2]} != {FEATURE_NAMES[mismatch:mismatch+2]}')
    assert all(math.isfinite(v) for v in result.values())
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
        return primary.context_secondary_rule(record,'emotional_support')
    if primary.active_material_task(record.get('history_user_messages',[])):
        return None
    return _legacy_secondary_rule(record['current_user_message'])

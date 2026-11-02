"""Design-only feature coverage audit. Does not train, split or modify data."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util, json, re, math, hashlib, collections
ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('primary_features',ROOT/'primary_emotion_classifier/features.py')
primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
LABELS=['approval','gratitude','relief','realization','not_this_branch']
LEXICON={
 'context_task':'|'.join('(?:'+primary.LEXICON[k]+')' for k in primary.TASKS),
 'context_report':primary.LEXICON['source_report'],
 'context_technical':primary.LEXICON['source_technical'],
 'approval_judgment':r'认可|赞同|同意|有道理|在理|合理|正确|很对|不错|很好|很棒|完美|符合预期|符合.{0,24}(?:要求|规范)|正解|说得对|说的对|确实|^(?:好|好的|行|行吧|对|对的|可以|嗯|收到|OK|okay)[!！。．.]*$|\b(?:agree(?:d)?|correct|accurate|excellent|perfect|looks good|well done|spot on|makes sense)\b',
 'approval_adoption':r'就按|就这么办|听你的|采纳|采用|定稿|已合入|合并了|按这个来|就选|就买|先按|\b(?:approv(?:e|ed)|go with|go ahead|use this|take your advice|merged|finali[sz]ed|sounds good)\b',
 'gratitude_direct':r'谢谢|感谢|多谢|感激|感恩|辛苦(?:了|啦|你|您)?|谢了|\b(?:thanks|thank you|thankful|grateful|gratitude|appreciate(?:d)?|appreciation)\b',
 'gratitude_benefit':r'多亏|帮(?:了)?(?:我)?大忙|帮到我|帮我省|救了我|救我|救命恩人|欠你|请你(?:吃|喝)|全靠你|没有你|为你(?:点赞|鼓掌)|佩服你|欣赏你|\b(?:owe you|lifesaver|saved my day|saved me|without you|your help|your support|your patience|your guidance|drinks on me|admire your)\b',
 'relief_release':r'松(?:了)?(?:一)?口气|如释重负|虚惊一场|(?:石头|大石).{0,8}(?:落地|放下)|心.{0,6}放.{0,4}肚子|心放下|喘口气|一身轻|轻松|解脱|踏实|安心|放下心|放心|歇歇|休息|下班|收工|撤退|太不容易|舒坦|舒畅|放松|睡个安稳觉|还好|幸好|庆幸|\b(?:phew|relief|relieved|relax(?:ed)?|can breathe|weight off|false alarm|off my (?:mind|chest)|at ease|rest easy|finally safe)\b',
 'result_completed':r'跑通|搞定|解决|恢复|没(?:有)?报错|不报错|测试全绿|通过|过了|好了|好使|完工|完成|完赛|找到了|找回|批下|一切正常|没迟到|\b(?:work(?:s|ed|ing)?|fixed|resolved|passed|done|finished|completed|restored|recovered|intact|successful|successfully|no errors?)\b',
 'realization_understand':r'(?<![不没])懂了|明白了|理解了|想通|顿悟|我悟了|恍然大悟|茅塞顿开|豁然开朗|捋清|理清|搞懂|转过弯|\b(?:now (?:I )?understand|now I get it|now clear|now (?:it )?makes sense|got it|I see|clicked|clicks|understood)\b',
 'realization_discovery':r'原来|破案了|难怪|怪不得|我就说|问题在|原因是|关键在|也就是说|搞半天|早说|\b(?:aha|eureka|realiz(?:e|ed|ation)|no wonder|turns out|that explains|so that is|so that.s)\b',
 'modifier_negation':r'不|没|并非|别|不要|\b(?:not|never|no|without|don.t|isn.t|aren.t|didn.t|can.t)\b',
 'modifier_transition':r'终于|总算|可算|还好|幸好|不过|但是|现在|原先|之前|之前以为|\b(?:finally|at last|eventually|but|however|now|previously|used to|fortunately|luckily)\b',
 'antecedent_pressure':r'焦虑|紧张|担心|害怕|吓|慌|急|累|折腾|崩溃|抓狂|烦|卡住|失败|报错|没通过|\b(?:anxious|anxiety|nervous|worried|worry|panic|afraid|scared|exhausted|stress(?:ed)?|stuck|fail(?:ed|ure)?|error|on edge)\b',
 'antecedent_confusion':r'看不懂|没看懂|没听懂|不理解|不明白|糊涂|困惑|绕晕|[懵蒙][圈逼]|没搞懂|搞不懂|为什么|怎么会|\b(?:confus(?:ed|ing|ion)|baffled|bewildered|do not understand|don.t understand|completely lost|why)\b',
}
PATTERNS={k:re.compile(v,re.I) for k,v in LEXICON.items()}
STRUCTURE=['log_chars','log_clauses','last_clause_ratio','question_log','exclamation_log','ellipsis_log',
 'max_punct_run_log','quote_span_ratio','colon_present','colon_first_position','first_person_present','second_person_present']
COMBO=['negation_ack_near','task_ack_near',
 'approval_thanks_near','release_thanks_near','understanding_thanks_near','completion_release_near',
 'completion_understanding_near','ack_tail_gap']
HISTORY=['nearest_char_jaccard','max_char_jaccard','weighted_char_jaccard','nearest_word_jaccard',
 'nearest_pressure_hit','weighted_pressure_hit','nearest_confusion_hit','weighted_confusion_hit',
 'nearest_task_hit','weighted_task_hit','pressure_to_resolution','confusion_to_understanding']
RULES=[
 ('S_ACK_EXCLAMATION',r'[!！]+','realization'),
 ('S_ACK_THANKS',r'(?:谢谢(?:你|您)?|感谢(?:你|您)?|多谢|非常感谢(?:你|您)?|十分感谢|太感谢了|谢了|辛苦了|辛苦啦|thanks|thank you|thanks a lot)[!！。．.]*','gratitude'),
 ('S_ACK_REALIZATION',r'(?:懂了|我懂了|看懂了|明白了|我明白了|理解了|终于懂了|终于明白了|原来如此|原来是这回事|原来是这样啊|原来这么简单|原来问题在这|恍然大悟|豁然开朗)[!！。．.]*','realization'),
 ('S_ACK_APPROVAL',r'(?:同意|赞同|认可|完全同意|确实如此|有道理|好主意|就这么办|就按这个来|好的|可以|行|对|听你的|approved|agreed|I agree)[!！。．.]*','approval'),
 ('S_ACK_RELIEF',r'(?:松了口气|松了一口气|如释重负|虚惊一场|phew|what a relief)[!！。．.]*','relief'),
]
def route(t):
 for key,pattern,label in RULES:
  if re.fullmatch(pattern,t.strip(),re.I):return key,label
 return None
def extract(row):
 t=row['current_user_message'].strip();L=len(t)
 hits={k:primary.spans(p,t) for k,p in PATTERNS.items()}
 f={}
 for k,h in hits.items():
  for suffix,value in zip(['hit','first_position','last_position','count_log'],[float(bool(h)),h[0][0]/L if h else 0,h[-1][0]/L if h else 0,math.log1p(min(len(h),5))]):f[k+'.'+suffix]=value
 old=primary.extract_features(row)
 for k in STRUCTURE:f['structure.'+k]=old['structure.'+k]
 def union(keys):return sorted({s for k in keys for s in hits[k]})
 A=union(['approval_judgment','approval_adoption']);T=union(['gratitude_direct','gratitude_benefit'])
 R=hits['relief_release'];U=union(['realization_understand','realization_discovery']);C=hits['result_completed']
 ACK=sorted(set(A+T+R+U));quotes=list(primary.QUOTE.finditer(t))
 def near(a,b):return float(any(primary.gap(x,y)<=24 for x in a for y in b))
 def before(a,b):return float(any(x[1]<=y[0] and y[0]-x[1]<=40 for x in a for y in b))
 combos=[near(hits['modifier_negation'],ACK),near(hits['context_task'],ACK),
  near(A,T),near(R,T),near(U,T),near(C,R),near(C,U),(L-max(b for a,b in ACK))/L if ACK else 0]
 f.update(('combo.'+k,v) for k,v in zip(COMBO,combos))
 hs=[h.strip() for h in row['history_user_messages'][:4]]
 def hist(k):return [float(bool(PATTERNS[k].search(h))) for h in hs]
 def weighted(values):return sum(v*2**(-j) for j,v in enumerate(values))/sum(2**(-j) for j in range(len(values))) if values else 0
 hp=hist('antecedent_pressure');hc=hist('antecedent_confusion');ht=hist('context_task')
 history=[old['history.'+k] for k in HISTORY[:4]]+[hp[0] if hp else 0,weighted(hp),hc[0] if hc else 0,weighted(hc),ht[0] if ht else 0,weighted(ht),weighted(hp)*bool(C or R),weighted(hc)*bool(U)]
 f.update(('history.'+k,float(v)) for k,v in zip(HISTORY,history))
 assert len(f)==92 and all(math.isfinite(v) for v in f.values())
 return f

FEATURE_NAMES=list(extract({"current_user_message":"示例","history_user_messages":[]}).keys())
assert len(FEATURE_NAMES)==92

def secondary_rule(text):
    match=route(text)
    return ({"secondary_label":match[1],"decision_source":"rule","rule_id":match[0],"secondary_probabilities":None} if match else None)


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
        return primary.context_secondary_rule(record,'positive_acknowledgment')
    if primary.active_material_task(record.get('history_user_messages',[])):
        return None
    return _legacy_secondary_rule(record['current_user_message'])

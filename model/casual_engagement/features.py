"""Two-way casual-engagement rules and 70 interpretable text features."""
import sys
sys.dont_write_bytecode = True
import importlib.util, math, re
from collections import OrderedDict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('primary_features_casual',ROOT/'primary_emotion_classifier/features.py')
primary=importlib.util.module_from_spec(spec); spec.loader.exec_module(primary)
LABELS=['gaming','boredom','not_this_branch']
FEATURE_VERSION='secondary-casual-features-v4-rejection'
LEXICON=OrderedDict([
 ('context_task','|'.join('(?:'+primary.LEXICON[k]+')' for k in primary.TASKS)),
 ('context_report',primary.LEXICON['source_report']),('context_technical',primary.LEXICON['source_technical']),
 ('gaming_invitation',r'来玩|玩(?:一|个|局|把|轮)|开一局|开始玩|接龙|对战|你先|我先|到你了|轮到你|\b(?:let.s play|play a game|your turn|my turn)\b'),
 ('gaming_named',r'成语接龙|海龟汤|飞花令|二十个问题|猜谜|谁是卧底|跑团|TRPG|五子棋|象棋|围棋|扑克|麻将|王者荣耀|英雄联盟|原神|星穹铁道|\b(?:riddle|twenty questions|chess|poker|mahjong|role.?playing|word game)\b'),
 ('gaming_mechanic',r'规则|判定|回合|骰子|出牌|落子|棋|残局|将死|技能|冷却|伤害|副本|关卡|胜负|卡组|先手|\b(?:rule|turn|dice|move|checkmate|cooldown|damage|level|deck)\b'),
 ('boredom_idle',r'无聊|闲得|太闲|没事干|不知道干嘛|无所事事|百无聊赖|发呆|消磨时间|打发时间|空虚|\b(?:bored|boredom|nothing to do|idle|kill time|pass the time)\b'),
 ('boredom_chat',r'随便聊|聊聊天|唠嗑|扯两句|说点什么|找个话题|陪我聊|讲个笑话|解闷|\b(?:chat with me|talk to me|just chat|entertain me|tell me a joke)\b'),
 ('boredom_filler',r'[。．.,，…⋯~～]{2,}|在吗|有人吗|还在吗|干嘛呢|随便回|说啥都行|\b(?:anyone there|you there|whatever|just say something)\b'),
 ('modifier_invitation',r'来|一起|陪我|咱们|我们|给我|跟我|你来|开始|\b(?:let.s|we can|with me|join me)\b'),
 ('modifier_continuation',r'继续|接着|再来|下一(?:题|轮|局)|到你|轮到|上局|刚才|\b(?:continue|again|next round|your turn|previous round)\b'),
])
PATTERNS=OrderedDict((k,re.compile(v,re.I)) for k,v in LEXICON.items())
RULES=[('S_CAS_BOREDOM_PUNCT',r'[。．.,，…⋯~～]+','boredom'),
       ('S_CAS_BOREDOM_WORD',r'(?:好无聊(?:啊|呀|了)?|无聊死了|闲得发慌|无所事事|不知道干嘛|发呆中)[!！。．.]*','boredom'),
       ('S_CAS_GAMING',r'(?:成语接龙|海龟汤|飞花令|来玩成语接龙(?:吧)?|玩海龟汤|来玩海龟汤)[!！。．.]*','gaming')]
COMPILED_RULES=[(k,re.compile(p,re.I),y) for k,p,y in RULES]
STRUCTURE=['log_chars','log_clauses','last_clause_ratio','question_log','exclamation_log','ellipsis_log','max_punct_run_log','quote_span_ratio','colon_present','colon_first_position','first_person_present','second_person_present']
COMBO=['gaming_invite_named_near','gaming_named_mechanic_near','idle_chat_near','idle_filler_near','continuation_game_near','casual_tail_gap']
HISTORY=['nearest_char_jaccard','max_char_jaccard','weighted_char_jaccard','nearest_word_jaccard','nearest_game_hit','weighted_game_hit','nearest_idle_hit','weighted_idle_hit']
FEATURE_NAMES=([f'{k}.{s}' for k in LEXICON for s in ('hit','first_position','last_position','count_log')]+['structure.'+k for k in STRUCTURE]+['combo.'+k for k in COMBO]+['history.'+k for k in HISTORY])
assert len(FEATURE_NAMES)==70
def secondary_rule(text):
 t=text.strip()
 for rid,p,y in COMPILED_RULES:
  if p.fullmatch(t): return {'secondary_label':y,'decision_source':'rule','rule_id':rid,'secondary_probabilities':None}
 return None
def extract(row):
 t=row['current_user_message'].strip()
 if not t: raise ValueError('Empty current_user_message')
 n=len(t); hits={k:primary.spans(p,t) for k,p in PATTERNS.items()}; result=OrderedDict()
 for k,sp in hits.items():
  vals=(float(bool(sp)),sp[0][0]/n if sp else 0.,sp[-1][0]/n if sp else 0.,math.log1p(min(len(sp),5)))
  for s,v in zip(('hit','first_position','last_position','count_log'),vals): result[k+'.'+s]=v
 parent=primary.extract_features(row)
 for k in STRUCTURE: result['structure.'+k]=parent['structure.'+k]
 def union(keys): return sorted({x for k in keys for x in hits[k]})
 def near(a,b): return float(any(primary.gap(x,y)<=24 for x in a for y in b))
 game=union(['gaming_invitation','gaming_named','gaming_mechanic']); idle=union(['boredom_idle','boredom_chat','boredom_filler'])
 combos=(near(hits['gaming_invitation'],hits['gaming_named']),near(hits['gaming_named'],hits['gaming_mechanic']),near(hits['boredom_idle'],hits['boredom_chat']),near(hits['boredom_idle'],hits['boredom_filler']),near(hits['modifier_continuation'],game),(n-max(e for _,e in game+idle))/n if game or idle else 0.)
 result.update(('combo.'+k,float(v)) for k,v in zip(COMBO,combos))
 hs=[x.strip() for x in row.get('history_user_messages',[])[:4]]
 def hh(keys): return [float(any(PATTERNS[k].search(h) for k in keys)) for h in hs]
 def wt(v): return sum(x*2**(-i) for i,x in enumerate(v))/sum(2**(-i) for i in range(len(v))) if v else 0.
 gv=hh(['gaming_invitation','gaming_named','gaming_mechanic']); iv=hh(['boredom_idle','boredom_chat','boredom_filler'])
 hvals=[parent['history.'+k] for k in HISTORY[:4]]+[gv[0] if gv else 0.,wt(gv),iv[0] if iv else 0.,wt(iv)]
 result.update(('history.'+k,float(v)) for k,v in zip(HISTORY,hvals))
 assert list(result)==BASE_FEATURE_NAMES and all(math.isfinite(v) for v in result.values())
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
        return primary.context_secondary_rule(record,'casual_engagement')
    if primary.active_material_task(record.get('history_user_messages',[])):
        return None
    return _legacy_secondary_rule(record['current_user_message'])

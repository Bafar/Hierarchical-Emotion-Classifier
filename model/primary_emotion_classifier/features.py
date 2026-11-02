"""Primary-only rules, 211 base features and shared contextual scope evidence."""
import math
import re
from collections import OrderedDict

FEATURE_VERSION = 'primary7-features-v4.0-colloquial'

# Shared by the corresponding secondary lexicons. Greedy runs keep an elongated
# vocalization as one expression; structure.max_char_run_log captures its length.
# These are contextual features, never whole-message label overrides.
LAUGHTER_EXPRESSION = (r'哈{2,}|嘿{2,}|嘻{2,}|笑死(?:我)?(?:了)?|笑喷|笑疯|'
                      r'笑不活了|笑得停不下来|笑到肚子疼')
CRY_EXPRESSION = r'呜{2,}|嘤{2,}|呜咽|抽泣|泣不成声|哭得停不下来'

LABELS = ['neutral', 'positive_acknowledgment', 'positive_affect', 'conflict_repair',
          'emotional_support', 'cognitive_guidance', 'casual_engagement']

# Every pattern contributes hit, first_position, last_position, count_log.
# English alternatives use word boundaries; Chinese alternatives use phrases.
LEXICON = OrderedDict([
 ('task_translate', r'翻译|翻一下|译一下|怎么翻|译成|译为|翻成|(?:英语|英文|中文|日语|法语|德语)(?:里)?怎么(?:说|表达|讲)|怎么用(?:英语|英文|中文|日语|法语|德语)(?:说|讲|表达)|\b(?:translat(?:e|es|ed|ing|ion)|how (?:would you|do (?:I|you)|to) say)\b'),
 ('task_rewrite', r'改写|润色|扩写|缩写|重写|换(?:个|种)(?:礼貌|委婉)?说法|修改病句|改成(?:一句|标题|文案)|改成(?:正式|口语|委婉|书面).{0,8}表达|压成|缩成|转成(?:正式|口语|书面)|(?:说|改)得(?:口语|正式|书面|委婉|客气)|怎么(?:说|写)得更?(?:正式|口语|委婉|客气)|正式一点怎么说|近义表达|\b(?:rewrite|rewriting|paraphrase|rephrase|proofread|polish|make .{1,80} sound polite|make it sound (?:more )?(?:formal|casual|natural))\b'),
 ('task_extract', r'总结|概括|摘要|提取|整理|归档|录入|汇总|摘录|文本标注|是什么语气|\b(?:summari[sz]e|summary|extract|organize|organise|archive|record|label the emotion|what emotion does)\b'),
 ('task_create', r'编写|创作|构思|续写|拟写|草拟|起草|造(?:个)?(?:句|短句)|接着写|续一句|补(?:个|完整)|编个|剧本里怎么|剧本开头|在故事里写|设计.{0,15}(?:文案|剧本|台词)|写进台词|放到对白|台词里加|写成(?:短信|朋友圈文案)|朋友圈文案带上|写(?:一个|一段|一篇|个|篇|段|剧本|代码|程序|函数)|\b(?:write|draft|compose|create|continue (?:the|this) line|use .{1,80} in (?:a sentence|a movie scene)|put .{1,80} in(?:to)? (?:a short dialogue|a movie scene))\b'),
 ('task_define', r'什么是|是什么意思|什么含义|(?:这个|两个|这些)词有什么区别|名词解释|定义|解释(?:一下|概念|术语)|\b(?:define|definition|what does|meaning of)\b'),
 ('task_execute', r'帮我|请帮|麻烦|生成|计算|查询|查一下|检查|分析|解释|排查|修复|实现|\b(?:help me|please|generate|calculate|query|check|analy[sz]e|explain|debug|implement|fix)\b'),
 ('source_report', r'原文|文中|以下|如下|下面这段|这段话|这句话|上述|材料|摘录|记录|转述|他说|她说|客户反馈|买家留言|工单|台词|剧本|小说|角色|\b(?:excerpt|transcript|quoted?|quotation|source text|following|customer feedback|ticket|screenplay|dialogue|novel)\b'),
 ('source_technical', r'```|Traceback|\b[A-Za-z_]+Error\b|\[(?:ERROR|WARN|INFO)\]|\b(?:HTTP|SQL|JSON|API|SELECT|INSERT|def|class|return|Exception|stack trace)\b|报错|堆栈|日志|函数|代码|接口'),
 ('expr_approval', r'认可|赞同|同意|有道理|合理|正确|很对|不错|很好|很棒|完美|符合预期|无可挑剔|\b(?:agree|correct|accurate|excellent|perfect|looks good|well done|spot on)\b'),
 ('expr_gratitude', r'谢谢|感谢|多谢|感激|感恩|多亏|\b(?:thanks|thank you|thankful|grateful|gratitude|appreciate)\b'),
 ('expr_relief', r'松(?:了)?(?:一)?口气|如释重负|虚惊一场|终于(?:好了|解决|跑通|过了)|总算|没事了|\b(?:relief|relieved|sigh of relief|finally works|finally working)\b'),
 ('expr_realization', r'原来|懂了|明白了|想通|顿悟|恍然大悟|茅塞顿开|豁然开朗|\b(?:realiz(?:e|ed|ation)|eureka|aha|now I understand|makes sense)\b'),
 ('expr_affection', r'喜欢|喜爱|好感|可爱|温柔|暖心|亲切|爱你|\b(?:love|like|adore|affection|adorable|lovely|sweet)\b'),
 ('expr_dependency', r'依赖|依恋|离不开|不能没有|没有你.{0,2}不行|只有你|唯一.*(?:依靠|信任)|别离开|别扔下|\b(?:depend(?:ent|ency)?|rely|codependenc[ey]|cannot live without|only you)\b'),
 ('expr_joy', r'开心|高兴|快乐|好耶|幸福|愉快|乐坏|笑死|笑喷|嘴角.*(?:压不住|下来)|\b(?:happy|joy|joyful|delighted|cheerful|thrilled|cannot stop smiling)\b'),
 ('expr_pride', r'骄傲|自豪|成就感|我真厉害|我真棒|我(?:真是|真的).{0,3}厉害|拿了第一|全靠我|佩服自己|证明自己|夺冠|获奖|破纪录|录取|做到了|\b(?:proud|pride|accomplished|achievement|personal best|proved myself)\b'),
 ('expr_touched', r'感动|触动|动容|泪目|鼻酸|眼眶.*(?:热|湿|红)|热泪|暖到|戳心|心里.*(?:暖|软)|\b(?:touched|moved|moving|heartwarming|teared up|eyes are wet)\b'),
 ('expr_hope', r'希望|期盼|期待|盼望|盼着|但愿|真盼|祈祷|保佑|憧憬|\b(?:hope|hoping|hopeful|wish|wishing|look forward)\b'),
 ('expr_anger', r'愤怒|生气|气死|气炸|气疯|气坏|气得要命|火大|惹火|来气|想骂人|怒|八嘎|(?<![节体早课情做])操(?![作控练劳心守场纵办持盘演行履刀舵琴切课])|艹|傻逼|傻[比笔]|脑残|垃圾(?!\s*(?:运输|清运|回收|分类|处理|焚烧|填埋|桶|袋|站|场|车))|蠢|笨(?:蛋)?|废物|骗子|无耻|滚|\b(?:angry|anger|furious|stupid|idiot|garbage(?!\s+(?:truck|bin|bag|collection|recycling|disposal|classification))|fuck|bullshit)\b'),
 ('expr_frustration', r'抓狂|崩溃|烦死|搞心态|心态炸|卡住|卡死|又错|又失败|弄不好|没救了|折磨人|(?<![佩征说折心口])服了|还是.*(?:错|失败|不行|不对)|越改越乱|越弄越乱|\b(?:frustrat(?:ed|ing|ion)|losing my mind|stuck|failed again|still fails|still broken)\b'),
 ('expr_impatience', r'快点|快一点|赶紧|赶快|等不及|来不及|急死|别(?:解释|废话|磨蹭)|直接给|只给|说重点|太慢|\b(?:hurry|urgent|asap|impatient|right now|no more explanation|just give)\b'),
 ('expr_disappointment', r'失望|没达到预期|受够了|心凉|算了|不用了|不折腾|毫无长进|辜负|\b(?:disappoint(?:ed|ing|ment)|let me down|give up|forget it|never mind)\b'),
 ('expr_sadness', CRY_EXPRESSION + r'|难过|难受(?:死|了|得|到)|伤心|悲伤|悲痛|心碎|哭|眼泪|痛苦|绝望|疼死(?:我)?(?:了)?|痛死(?:我)?(?:了)?|好疼|好痛|疼得|痛得|疼到|痛到|疼哭|痛哭|疼(?!爱)|痛(?!快|点|恨|风)|\b(?:sad|sadness|grief|grieving|heartbroken|cry|crying|tears|despair|hurts? badly|in pain)\b'),
 ('expr_loneliness', r'孤独|寂寞|孤单|没人陪|无人倾诉|无人理解|独自|一个人|\b(?:lonely|loneliness|alone|isolated|isolation|nobody understands)\b'),
 ('expr_anxiety', r'焦虑|紧张|担心|害怕|恐慌|心慌|慌乱|睡不着|挂心|惦记|放心不下|不放心|心里没底|心里悬着|七上八下|会不会.{0,8}(?:不高兴|难受|误会|生气)|\b(?:anxious|anxiety|nervous|worried|worry|panic|afraid|scared|cannot sleep)\b'),
 ('expr_guilt', r'内疚|自责|愧疚|负罪|对不起|都是我的错|后悔|\b(?:guilt|guilty|regret|remorse|my fault|sorry)\b'),
 ('expr_embarrassment', r'尴尬|羞耻|羞愧|社死|丢人|难堪|无地自容|\b(?:embarrass(?:ed|ing|ment)|ashamed|shame|humiliat(?:ed|ing|ion)|awkward)\b'),
 ('expr_confusion', r'看不懂|没看懂|没听懂|不理解|不明白|糊涂|困惑|绕晕|[懵蒙]圈|[懵蒙]逼|懵|\b(?:confus(?:ed|ing|ion)|baffled|bewildered|do not understand|don.t understand|completely lost)\b'),
 ('expr_skepticism', r'怀疑|质疑|存疑|可疑|不相信|靠谱吗|站不住脚|你确定吗|确定吗|依据|出处|证据|瞎编|不可信|\b(?:skeptic(?:al|ism)?|doubt|evidence|source|are you sure|fabricated|verify)\b'),
 ('expr_curiosity', r'好奇|探究|探索|深入|底层|机理|原理|为什么|进一步|神奇|\b(?:curious|curiosity|explore|why|underlying|mechanism|fascinat(?:ed|ing))\b'),
 ('expr_playfulness', LAUGHTER_EXPRESSION + r'|开玩笑|调侃|逗你|戏谑|搞笑|角色扮演|\b(?:lol|lmao|haha|joking|teasing|playful|roleplay)\b'),
 ('expr_gaming', r'玩一局|玩个|来一局|成语接龙|接龙|海龟汤|跑团|猜谜|游戏|王者|原神|崩坏[：:·\s]*星穹铁道|星穹铁道|崩坏(?:3|三)?|\b(?:game|gaming|play|riddle|chess|dungeon)\b'),
 ('expr_boredom', r'无聊|没事干|不知道该干什么|闲得|发呆|打发时间|百无聊赖|消遣|\b(?:bored|boring|boredom|nothing to do|kill time)\b'),
 ('modifier_negation', r'不是|并不|没有|不再|并非|别|不要|\b(?:not|never|no|without|don.t|isn.t|aren.t)\b'),
 ('modifier_hypothesis', r'如果|假设|要是|倘若|假如|\b(?:if|suppose|assuming|imagine|hypothetical)\b'),
 ('modifier_ownership', r'这就是我|我的感受|我也是|说的就是我|我现在|我感觉|我觉得|\b(?:I feel|I am|I.m|my feelings|this is me|me too)\b'),
])
PATTERNS = OrderedDict((k, re.compile(v, re.I)) for k, v in LEXICON.items())
TASKS = tuple(k for k in LEXICON if k.startswith('task_'))
EXPRESSIONS = tuple(k for k in LEXICON if k.startswith('expr_'))

PURE_JOY_VOCALIZATION = r'(?:哈{2,}|啦{2,})[!！。．.…⋯~～]*'
PURE_PLAYFUL_VOCALIZATION = r'(?:嘿{2,}|嘻{2,}|略{2,})[!！。．.…⋯~～]*'
PURE_SAD_VOCALIZATION = r'(?:呜{2,}|嘤{2,})[!！。．.…⋯~～]*'

RULES = [
 ('P_AFFECT_PURE_JOY', PURE_JOY_VOCALIZATION, 'positive_affect', 'user_policy'),
 ('P_AFFECT_PURE_PLAYFUL', PURE_PLAYFUL_VOCALIZATION, 'positive_affect', 'user_policy'),
 ('P_EMO_PURE_CRY', PURE_SAD_VOCALIZATION, 'emotional_support', 'user_policy'),
 ('P_CASUAL_ELLIPSIS', r'[。．.,，…⋯]+', 'casual_engagement', 'user_policy'),
 ('P_COG_QUESTION', r'[?？]+', 'cognitive_guidance', 'user_policy'),
 ('P_ACK_EXCLAMATION', r'[!！]+', 'positive_acknowledgment', 'user_policy'),
 ('P_COG_WHAT', r'什么(?=[?？!！]*[?？])[?？!！]+', 'cognitive_guidance', 'user_policy'),
 ('P_ACK_THANKS', r'(?:谢谢(?:你|您)?|感谢(?:你|您)?|多谢|非常感谢(?:你|您)?|十分感谢|太感谢了|谢了|thanks|thank you|thanks a lot)[!！。．.]*', 'positive_acknowledgment', 'user_policy_extended'),
 ('P_ACK_REALIZATION', r'(?:懂了|我懂了|看懂了|明白了|我明白了|理解了|终于懂了|终于明白了|原来如此|原来是这回事|原来是这样啊|原来这么简单|原来问题在这|恍然大悟|豁然开朗)[!！。．.]*', 'positive_acknowledgment', 'user_policy_extended'),
 ('P_ACK_APPROVAL', r'(?:同意|赞同|认可|完全同意|确实如此|有道理|好主意|就这么办|就按这个来)[!！。．.]*', 'positive_acknowledgment', 'data_supported_extension'),
]
COMPILED_RULES = [(key, re.compile(pat, re.I), label) for key, pat, label, _ in RULES]

STRUCTURE_NAMES = ['log_chars', 'log_words', 'log_lines', 'log_clauses', 'first_clause_ratio',
 'last_clause_ratio', 'cjk_ratio', 'latin_ratio', 'digit_ratio', 'space_ratio', 'punct_ratio',
 'question_log', 'exclamation_log', 'ellipsis_log', 'max_punct_run_log', 'max_char_run_log',
 'code_fence_present', 'quote_span_ratio', 'colon_present', 'colon_first_position',
 'colon_last_position', 'first_person_present', 'second_person_present']
COMBO_NAMES = ['translation_language', 'creation_artifact', 'definition_frame', 'rewrite_object',
 'source_emotion_same_clause', 'task_emotion_same_clause', 'negation_emotion_near',
 'hypothesis_emotion_near', 'ownership_emotion_near', 'self_emotion_near', 'you_evaluation_near',
 'repeat_failure', 'task_after_emotion', 'task_before_emotion', 'task_emotion_pair',
 'task_emotion_min_distance', 'last_task_tail_gap', 'task_any_presence',
 'expression_inside_quote_ratio', 'first_second_same_clause',
 'skepticism_target_challenge_near', 'skepticism_short_direct',
 'attack_devaluation_near', 'attack_without_repeat',
 'self_fault_reparation_near', 'future_risk_anxiety_near',
 'social_absence_connection_near', 'exposure_shame_near',
 'gaming_invitation_named_near', 'idle_chat_near']
HISTORY_NAMES = ['nearest_char_jaccard', 'max_char_jaccard',
 'weighted_char_jaccard', 'nearest_word_jaccard', 'nearest_length_logratio', 'current_in_history',
 'history_in_current', 'current_backref', 'nearest_task_hit', 'any_history_task',
 'nearest_wait_material', 'history_task_current_no_task_emotion', 'history_task_current_no_task',
 'history_eval_current_task']
FEATURE_NAMES = [f'{k}.{suffix}' for k in LEXICON for suffix in ['hit', 'first_position', 'last_position', 'count_log']]
FEATURE_NAMES += ['structure.'+n for n in STRUCTURE_NAMES]
FEATURE_NAMES += ['combo.'+n for n in COMBO_NAMES]
FEATURE_NAMES += ['history.'+n for n in HISTORY_NAMES]
assert len(FEATURE_NAMES) == 211

CLAUSE_SEP = re.compile(r'[\n\r。！？!?；;，,]+')
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[\u4e00-\u9fff]|\d+")
QUOTE = re.compile(r'“[^”]*”|‘[^’]*’|"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')
LANGUAGE = re.compile(r'英文|英语|中文|汉语|日文|日语|法文|法语|德文|德语|西语|西班牙语|韩语|俄语|\b(?:English|Chinese|German|French|Japanese|Spanish|Korean|Russian)\b', re.I)
ARTIFACT = re.compile(r'剧本|台词|对白|故事|小说|诗|歌词|场景|\b(?:screenplay|dialogue|story|novel|poem|lyrics|scene)\b', re.I)
OBJECT = re.compile(r'这句|这段|原文|文案|文本|段落|口语|\b(?:this|text|sentence|paragraph|prose)\b', re.I)
SELF = re.compile(r'我|\b(?:I|my|me)\b', re.I)
YOU = re.compile(r'你|您|\b(?:you|your)\b', re.I)
FIRST_PERSON = re.compile(r'我|本人|\b(?:I|me|my|mine|we|us|our|ours)\b', re.I)
SECOND_PERSON = re.compile(r'你|您|\b(?:you|your|yours)\b', re.I)
BACKREF = re.compile(r'还是|刚才|前面|上面|继续|你说的|你给的|\b(?:still|previous|earlier|continue|you said|you gave)\b', re.I)
WAIT = re.compile(r'接下来|下一句|下一段|我发给你|稍后发|下面的内容|\b(?:next sentence|next message|I will send|following text)\b', re.I)
REPEAT = re.compile(r'(?:又|还是|反复|一直).{0,12}(?:错|失败|不行|报错|不对|弄不好)|(?:改|试|弄)了.{0,6}次|\b(?:still|again|repeatedly)\b.{0,35}\b(?:fail\w*|error|broken)\b', re.I)
SKEPTICISM_TARGET = re.compile(r'思路|逻辑|回答|答案|说法|结论|推导|计算|分析|观点|方案')
SKEPTICISM_CHALLENGE = re.compile(r'不对|有问题|错(?:了)?|有误|不合理|说不通|站不住脚|不可信')
ATTACK_WORD = re.compile(r'八嘎|(?<![节体早课情做])操(?![作控练劳心守场纵办持盘演行履刀舵琴切课])|艹|傻逼|傻[比笔]|脑残|垃圾(?!\s*(?:运输|清运|回收|分类|处理|焚烧|填埋|桶|袋|站|场|车))|废物|白痴|蠢货|智障')
ABILITY_DEVALUATION = re.compile(r'什么都不会|啥也不会|这都不会|一点用都没有|一点用没有|没一点用|会个屁|能干什么|干啥啥不行')
SELF_FAULT = re.compile(r'都是我的错|都怪我|怪我|我不该|是我(?:害|弄|搞|说错|做错)|我的(?:过错|错误|失误)|自责|内疚')
REPARATION = re.compile(r'对不起|道歉|弥补|补偿|后悔|原谅')
FUTURE_RISK = re.compile(r'明天|以后|接下来|还没|万一|可能会|会不会|怕|担心|害怕')
ANXIETY_SIGNAL = re.compile(r'焦虑|紧张|心慌|慌|恐慌|睡不着|手抖|担心|害怕')
SOCIAL_ABSENCE = re.compile(r'没人|没有人|一个人|独自|没朋友|无人|孤零零')
CONNECTION_NEED = re.compile(r'陪我|陪伴|说话的人|倾诉|懂我|抱抱|聊聊')
EXPOSURE = re.compile(r'当众|全班|全场|大家都|所有人都|面前|群聊|被看见|被围观')
SHAME_SIGNAL = re.compile(r'尴尬|社死|丢人|丢脸|羞耻|难堪|无地自容')
GAME_INVITATION = re.compile(r'来玩|玩一局|玩一把|开始玩|接龙|对战|轮到你|到你了')
GAME_NAMED = re.compile(r'成语接龙|海龟汤|飞花令|猜谜|跑团|五子棋|象棋|围棋|麻将|日麻')
IDLE_SIGNAL = re.compile(r'无聊|没事干|太闲|不知道干嘛|打发时间|解闷')
CHAT_REQUEST = re.compile(r'陪我聊|聊聊天|随便聊|找个话题|讲个笑话|说点什么')
AGENT_INSULT = re.compile(r'(?:傻逼|傻[比笔]|脑残|垃圾|废物|白痴|蠢货)\s*(?:ai|人工智能|机器人|智能助手)', re.I)
AGENT_INSULT_PAYLOAD = re.compile(
    r'翻译|译成|改写|润色|分析|解释|判断|识别|检测|提取|总结|概括|'
    r'这句话|这个词|原文|台词|剧本|例句|例如|比如|为什么|什么意思|怎么说|'
    r'是否(?:属于|算|是)|有人(?:说|称)|称为|叫做|所谓|[“”‘’"\x27]')


def direct_agent_insult(text):
    """High-precision short direct attack, excluding quoted/payload usage."""
    t = text.strip()
    return (len(t) <= 40 and bool(AGENT_INSULT.search(t)) and
            not AGENT_INSULT_PAYLOAD.search(t) and
            not any(PATTERNS[key].search(t) for key in TASKS))


def primary_rule(text):
    t = text.strip()
    if direct_agent_insult(t):
        return {'primary_label': 'conflict_repair', 'decision_source': 'rule',
                'rule_id': 'P_CON_AGENT_INSULT', 'primary_probabilities': None}
    for rule_id, pat, label in COMPILED_RULES:
        if pat.fullmatch(t):
            return {'primary_label': label, 'decision_source': 'rule', 'rule_id': rule_id,
                    'primary_probabilities': None}
    return None


def spans(pat, text):
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def gap(a, b):
    return max(0, max(a[0], b[0]) - min(a[1], b[1]))


def chargrams(text):
    s = re.sub(r'[^\w]', '', text.casefold())
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) > 1 else ({s} if s else set())


def jaccard(a, b):
    return len(a & b)/len(a | b) if a or b else 0.0


def extract_features(record):
    t = record['current_user_message'].strip()
    if not t:
        raise ValueError('Empty current_user_message')
    length = len(t)
    hits = {k: spans(p, t) for k,p in PATTERNS.items()}
    f = OrderedDict()
    for key, locations in hits.items():
        values = [float(bool(locations)), locations[0][0]/length if locations else 0.0,
                  locations[-1][0]/length if locations else 0.0, math.log1p(min(len(locations),5))]
        for suffix,value in zip(['hit','first_position','last_position','count_log'], values):
            f[key+'.'+suffix] = value
    clauses = [x for x in CLAUSE_SEP.split(t) if x.strip()]
    punct = r'[^\w\s]'
    runs = re.findall(punct+'+', t)
    repeated = [len(m.group()) for m in re.finditer(r'(.)\1*', t)]
    quotes = list(QUOTE.finditer(t))
    quote_chars = {i for m in quotes for i in range(m.start(),m.end())}
    colons = spans(re.compile(r'[:：]'), t)
    first_person = spans(FIRST_PERSON, t)
    second_person = spans(SECOND_PERSON, t)
    struct = [math.log1p(length), math.log1p(len(WORD.findall(t))), math.log1p(len(t.splitlines())),
              math.log1p(len(clauses)), len(clauses[0])/length if clauses else 0,
              len(clauses[-1])/length if clauses else 0,
              len(re.findall(r'[\u4e00-\u9fff]',t))/length, len(re.findall(r'[A-Za-z]',t))/length,
              sum(c.isdigit() for c in t)/length, sum(c.isspace() for c in t)/length,
              len(re.findall(punct,t))/length, math.log1p(len(re.findall(r'[?？]',t))),
              math.log1p(len(re.findall(r'[!！]',t))), math.log1p(len(re.findall(r'\.{2,}|。{2,}|[…⋯]+',t))),
              math.log1p(max(map(len,runs),default=0)), math.log1p(max(repeated,default=0)),
              float('```' in t), len(quote_chars)/length,
              bool(colons), colons[0][0]/length if colons else 0,
              colons[-1][0]/length if colons else 0, bool(first_person), bool(second_person)]
    f.update(('structure.'+k,float(v)) for k,v in zip(STRUCTURE_NAMES,struct))
    tasks = sorted({p for k in TASKS for p in hits[k]})
    emotions = sorted({p for k in EXPRESSIONS for p in hits[k]})
    evaluation = [p for k in ['expr_approval','expr_anger','expr_frustration','expr_disappointment','expr_skepticism'] for p in hits[k]]
    boundaries = [m.start() for m in CLAUSE_SEP.finditer(t)]
    def same_clause(a,b):
        return any(not any(min(x[0],y[0]) < z < max(x[0],y[0]) for z in boundaries) for x in a for y in b)
    def near(a,b):
        return any(gap(x,y)<=24 for x in a for y in b)
    both = bool(tasks and emotions)
    translation_payload = bool(hits['task_translate'] and (
        LANGUAGE.search(t) or (emotions and min(
            (gap(task, emotion) for task in hits['task_translate'] for emotion in emotions),
            default=length) <= 8 and length <= 40)))
    combos = [translation_payload, bool(hits['task_create'] and ARTIFACT.search(t)),
        bool(re.fullmatch(r'(?:什么是.{1,40}|.{1,40}(?:是什么意思|是什么含义)|(?:名词解释|定义)[：:].{1,40})[?？!！。.]?',t)), bool(hits['task_rewrite'] and OBJECT.search(t)),
        same_clause(hits['source_report'],emotions), same_clause(tasks,emotions),
        near(hits['modifier_negation'],emotions), near(hits['modifier_hypothesis'],emotions),
        near(hits['modifier_ownership'],emotions), near(spans(SELF,t),emotions), near(spans(YOU,t),evaluation),
        bool(REPEAT.search(t)), any(x[0]>y[0] for x in tasks for y in emotions),
        any(x[0]<y[0] for x in tasks for y in emotions), both,
        min((gap(x,y) for x in tasks for y in emotions),default=0)/length,
        (length-tasks[-1][1])/length if tasks else 0, bool(tasks),
        sum(any(m.start()+1 <= s and e <= m.end()-1 for m in quotes)
            for s,e in emotions)/len(emotions) if emotions else 0,
        same_clause(first_person, second_person),
        near(spans(SKEPTICISM_TARGET,t), spans(SKEPTICISM_CHALLENGE,t)),
        float(length <= 24 and bool(SKEPTICISM_CHALLENGE.search(t)) and not tasks and not quotes),
        near(spans(ATTACK_WORD,t), spans(ABILITY_DEVALUATION,t)),
        float(bool(ATTACK_WORD.search(t)) and not REPEAT.search(t)),
        near(spans(SELF_FAULT,t), spans(REPARATION,t)),
        near(spans(FUTURE_RISK,t), spans(ANXIETY_SIGNAL,t)),
        near(spans(SOCIAL_ABSENCE,t), spans(CONNECTION_NEED,t)),
        near(spans(EXPOSURE,t), spans(SHAME_SIGNAL,t)),
        near(spans(GAME_INVITATION,t), spans(GAME_NAMED,t)),
        near(spans(IDLE_SIGNAL,t), spans(CHAT_REQUEST,t))]
    f.update(('combo.'+k,float(v)) for k,v in zip(COMBO_NAMES,combos))
    hs = [h.strip() for h in record.get('history_user_messages',[])][:4]
    js = [jaccard(chargrams(t),chargrams(h)) for h in hs]
    hist_task = [any(PATTERNS[k].search(h) for k in TASKS) for h in hs]
    hist_eval = bool(hs and any(PATTERNS[k].search(hs[0]) for k in EXPRESSIONS))
    history = [js[0] if js else 0,max(js,default=0),
        sum(v*2**(-i) for i,v in enumerate(js))/sum(2**(-i) for i in range(len(js))) if js else 0,
        jaccard(set(WORD.findall(t.casefold())),set(WORD.findall(hs[0].casefold()))) if hs else 0,
        max(-5,min(5,math.log((length+1)/(len(hs[0])+1)))) if hs else 0,
        len(t)>=4 and any(t in h for h in hs), any(len(h)>=4 and h in t for h in hs),
        bool(BACKREF.search(t)), bool(hist_task and hist_task[0]), any(hist_task),
        bool(hs and WAIT.search(hs[0])), bool(any(hist_task) and not tasks and emotions),
        bool(any(hist_task) and not tasks), bool(hist_eval and tasks)]
    f.update(('history.'+k,float(v)) for k,v in zip(HISTORY_NAMES,history))
    assert list(f) == BASE_FEATURE_NAMES
    return f


# Scope signals are evidence for the classifier, never label overrides.
SCOPE_NAMES = ['negation_count_log', 'negated_emotion_ratio', 'negation_distance',
               'negated_emotion_position', 'deny_self_fault', 'fear_separation',
               'deny_dependency', 'litotes', 'contrast_present', 'contrast_position',
               'emotion_after_contrast_ratio', 'acceptance_depth', 'gaming_continue',
               'emotion_mixture_log', 'guard_resource_boundary', 'guard_motive',
               'guard_boundary', 'disapproval_evaluation', 'rhetorical_marker',
               'rhetorical_position', 'rhetorical_emotion_near', 'rhetorical_positive_frame']
SCOPE_NAMES += ['negated_approval', 'negated_doubt', 'negated_joy', 'trust_assertion',
                'refusal_disclosure', 'objective_inability', 'refusal_resource',
                'negative_stance_target', 'negative_fact_target',
                'personal_attack_question', 'rhetorical_denial_positive', 'explicit_unhappiness']
SCOPE_FEATURE_NAMES = ['scope.' + name for name in SCOPE_NAMES]
_RAW_SCOPE_NEGATION = re.compile(r'并不是|并不|并非|没有|不要|不能|不再|未曾|别|不|没|\b(?:not|never|no longer|don.t|doesn.t|isn.t|aren.t|without)\b', re.I)
SCOPE_CONTRAST = re.compile(r'反而|而是|但是|可是|不过|却|\b(?:but|instead|rather|however)\b', re.I)
SCOPE_LITOTES = re.compile(r'不是不|并非不|没有不|很难不|不能不|不只是|不但|好久没.{0,5}过|\b(?:not ungrateful|can.t help|cannot help|not unhappy|not just)\b', re.I)
DENY_FAULT = re.compile(r'(?:不是|并非|不该算|不全是).{0,3}(?:我|我的).{0,3}(?:错|责任|过失)|我没做错|别怪我|\b(?:not my fault|wasn.t my fault|didn.t do anything wrong)\b', re.I)
FEAR_SEPARATION = re.compile(r'(?:别|不要).{0,3}(?:离开|丢下|扔下|消失|不理)|不能没有你|离不开你|\b(?:don.t leave me|cannot live without you|can.t live without you)\b', re.I)
DENY_DEPENDENCY = re.compile(r'(?:不想|不再|没有|不要).{0,4}(?:依赖|依靠)|不需要你.{0,5}(?:陪|决定)|\b(?:don.t need you|no longer depend|don.t want to depend)\b', re.I)
ACCEPT_DEPTH = re.compile(r'(?:懂了|明白了|理解了|认可|有意思).{0,30}(?:深入|深处|再讲|展开|接着|往下|多讲)|(?:深入|深处|展开|继续追问|延伸问题)|\b(?:go deeper|one step deeper|tell me more|follow.up question)\b', re.I)
GAME_CONTINUE = re.compile(r'(?:继续|接着|再来|换个|下一(?:个|轮|局|关)).{0,12}(?:玩|游戏|猜|谜|出题)|(?:游戏|好玩|这局|一把).{0,15}(?:继续|接着|再来|下一)|\b(?:another round|next game|keep playing|play again|next level)\b', re.I)
GUARD_RESOURCE = re.compile(r'隐私|密码|验证码|账号|账户|权限|证件|住址|地址|通讯录|录音|相册|定位|位置|人脸|信息|私人|家人|孩子|授权|\b(?:privacy|password|code|account|permission|address|contacts|recording|photos|location|private|authorize)\b', re.I)
GUARD_BOUNDARY = re.compile(r'不提供|不愿意|不想.{0,8}(?:说|回答|透露|交|给|共享)|不能.{0,8}(?:给|交|信)|先别|别替我|没(?:有)?(?:同意|允许|授权)|越界|不该问|留个心眼|警觉|戒备|防备|谨慎|核实|先确认|我(?:想|得|需要|会).{0,5}(?:确认|保留|保护|掌握|亲自)|界限|边界|选择权|知情权|不放心|\b(?:not comfortable|won.t|don.t want|haven.t authorized|verify|wary|outside what|haven.t agreed|without asking)\b', re.I)
GUARD_MOTIVE = re.compile(r'(?:问|要|知道|读取|保存).{0,15}(?:干什么|做什么|什么意思|什么关系)|你.{0,8}(?:想害我|套取|试探|用意)|凭什么.{0,8}(?:问|查|替我)|\b(?:why (?:are you|do you|would you).{0,20}(?:ask|need|want|access|save|collect)|what .{0,12} for|your role|your intentions)\b', re.I)
DISAPPROVAL = re.compile(r'我反对|持反对|有(?:明确的)?(?:意见|异议)|不(?:认可|认同|赞成|赞同|同意|支持|接受|满意|欣赏|合适|公平|尊重)|不太(?:妥当|合适)|不舒服|反感|不满|敷衍|打发我|居高临下|轻视|说教|有点过分|不领情|\b(?:don.t appreciate|don.t accept|don.t agree|don.t approve|oppose|object to|objection|disagree|dismissive|not appropriate|not comfortable|talked down|not impressed|not satisfied|not respectful)\b', re.I)
RHETORICAL = re.compile(r'(?:这|这样)(?:也|就)(?:叫|算)|这就是你(?:说的|所谓的)|难道|你以为|你觉得|凭什么|怎么能|怎么会|哪能|哪会|谁会|谁能|还不(?:该|算|值得)|\b(?:am I supposed to|do you (?:really )?(?:think|expect|consider)|how could|how can|why would|who wouldn.t)\b', re.I)
RHETORICAL_POSITIVE = re.compile(r'谁(?:能|会|舍得).{0,5}不|怎么(?:能|会).{0,4}不|哪(?:能|会).{0,3}不|难道.{0,12}不(?:值得|该)|还不(?:值得|许我)|\b(?:who wouldn.t|how could I not|how can I not)\b', re.I)

# These are soft evidence, including contradictory signals, never hard routes.
DISCLOSURE_REFUSAL = re.compile(r'(?:不想|不愿|不能|不会|不便|不方便|没法).{0,5}(?:告诉|透露|说|讲|交代|回答|提供)|别(?:再)?(?:问|打听)|无可奉告|这(?:个|些)?不(?:能|想)说|\b(?:won.t tell|can.t tell|cannot tell|rather not say|not sharing|none of your business)\b', re.I)
OBJECTIVE_INABILITY = re.compile(r'不知道|不清楚|没查到|没有(?:这|相关|足够).{0,4}(?:信息|资料|数据)|我也不懂|不掌握|\b(?:don.t know|do not know|no information|not enough information|haven.t found)\b', re.I)
NEGATIVE_EVALUATION = re.compile(r'不对|不妥|不行|不合适|有问题|有毛病|错了|错的|不能接受|不认可|不赞成|不认同|反对|\b(?:wrong|inappropriate|disagree|object to|don.t approve)\b', re.I)
STANCE_TARGET = re.compile(r'观点|立场|主张|价值|做法|待人|对待|安排|态度|取笑|歧视|尊重|道德|\b(?:opinion|stance|attitude|treatment|approach|unfair|disrespect)\b', re.I)
FACT_TARGET = re.compile(r'计算|数字|数据|结果|公式|事实|推导|证据|来源|出处|引用|\b(?:calculation|number|data|result|formula|fact|evidence|source)\b', re.I)
PERSON_ATTACK_QUESTION = re.compile(r'(?:你|您).{0,3}(?:瞎(?:了)?|眼瞎|聋(?:了)?|脑残|傻(?:了)?|有病)(?:吗|么|啊|吧)|(?:你|您).{0,3}(?:没脑子|没长眼|长眼了吗)|\b(?:are you (?:blind|deaf|stupid)|what is wrong with you)\b', re.I)
TRUST_ASSERTION = re.compile(r'相信|确信|深信|信任|肯定.{0,4}(?:是|没问题)|\b(?:believe|trust|certain|confident)\b', re.I)
RHETORICAL_DENIAL_POSITIVE = re.compile(r'(?:难道|你以为|你觉得|还指望).{0,22}(?:高兴|开心|感谢|满意|喜欢)|\b(?:am I supposed to (?:thank|enjoy|like)|do you think.{0,15}(?:happy|pleased|grateful))\b', re.I)
UNHAPPINESS = re.compile(r'不(?:太|怎么|大|很)?(?:高兴|开心|快乐)|(?:高兴|开心)不起来|一点.{0,3}不(?:高兴|开心)|\b(?:unhappy|not happy|not pleased|not enjoying)\b', re.I)

def _negated_span(text, span):
    """Local clause scope/parity, retaining uncertainty for the learned model."""
    prefix = text[:span[0]]
    start = max([m.end() for m in re.finditer(r'[，,。！？!?；;\n]', prefix)], default=0)
    nearby = [m for m in SCOPE_NEGATION.finditer(text, start, span[0])
              if span[0]-m.end() <= 6 and not re.match(r'不(?:但|仅|只是)',text[m.start():])]
    return len(nearby) % 2 == 1

def _same_clause_near(text, left, right, distance=24):
    return any(gap(a,b)<=distance and not re.search(r'[，,。！？!?；;\n]',
               text[min(a[1],b[1]):max(a[0],b[0])]) for a in left for b in right)

def scope_features(text):
    n = max(1, len(text))
    neg = list(SCOPE_NEGATION.finditer(text))
    expr = sorted({span for key in EXPRESSIONS for span in spans(PATTERNS[key], text)})
    litotes = bool(SCOPE_LITOTES.search(text))
    pairs = [(m, e) for m in neg for e in expr
             if m.end() <= e[0] and e[0] - m.end() <= 6
             and not re.search(r'[，,。！？!?；;\n]', text[m.end():e[0]])]
    contrast = list(SCOPE_CONTRAST.finditer(text))
    last = contrast[-1] if contrast else None
    affected = {e for _, e in pairs}
    vals = [math.log1p(min(5, len(neg))),
            len(affected) / max(1, len(expr)) if not litotes else 0.,
            min((e[0]-m.end() for m,e in pairs), default=0)/n,
            min((m.start() for m,_ in pairs), default=0)/n,
            bool(DENY_FAULT.search(text)), bool(FEAR_SEPARATION.search(text)),
            bool(DENY_DEPENDENCY.search(text)), litotes, bool(last),
            last.start()/n if last else 0.,
            sum(s >= last.end() for s,_ in expr)/max(1,len(expr)) if last else 0.,
            bool(ACCEPT_DEPTH.search(text)), bool(GAME_CONTINUE.search(text)),
            math.log1p(sum(bool(PATTERNS[key].search(text)) for key in EXPRESSIONS))]
    resource=spans(GUARD_RESOURCE,text);boundary=spans(GUARD_BOUNDARY,text)
    rhetorical=spans(RHETORICAL,text)
    vals.extend([any(gap(a,b)<=24 for a in resource for b in boundary),
                 bool(GUARD_MOTIVE.search(text)),bool(boundary),bool(DISAPPROVAL.search(text)),
                 bool(rhetorical),rhetorical[0][0]/n if rhetorical else 0.,
                 any(gap(a,b)<=24 for a in rhetorical for b in expr),
                 bool(RHETORICAL_POSITIVE.search(text))])
    refused = spans(DISCLOSURE_REFUSAL,text)
    evaluation = spans(NEGATIVE_EVALUATION,text)
    negated = lambda key: any(_negated_span(text,s) for s in spans(PATTERNS[key],text))
    neg_doubt = negated('expr_skepticism')
    trust = any(not _negated_span(text,s) for s in spans(TRUST_ASSERTION,text)) or neg_doubt
    # Literal/quoted emotion words remain visible: task/source evidence and BERT
    # determine whether they describe the current speaker or material content.
    vals.extend([negated('expr_approval'),neg_doubt,negated('expr_joy'),trust,
                 bool(refused),bool(OBJECTIVE_INABILITY.search(text)),
                 _same_clause_near(text,refused,resource),
                 _same_clause_near(text,evaluation,spans(STANCE_TARGET,text)),
                 _same_clause_near(text,evaluation,spans(FACT_TARGET,text)),
                 bool(PERSON_ATTACK_QUESTION.search(text)),
                 bool(RHETORICAL_DENIAL_POSITIVE.search(text)) and not bool(RHETORICAL_POSITIVE.search(text)),
                 any(not _negated_span(text,s) for s in spans(UNHAPPINESS,text))])
    return OrderedDict((key, float(value)) for key,value in zip(SCOPE_FEATURE_NAMES, vals))

BASE_FEATURE_NAMES = FEATURE_NAMES[:]
_base_extract_features = extract_features
FEATURE_NAMES = BASE_FEATURE_NAMES + SCOPE_FEATURE_NAMES
def extract_features(record):
    values = _base_extract_features(record)
    values.update(scope_features(record['current_user_message'].strip()))
    assert list(values) == FEATURE_NAMES
    return values


class _WordAwareNegation:
    def finditer(self,text,*args):
        for match in _RAW_SCOPE_NEGATION.finditer(text,*args):
            if match.group()=='别' and match.start()>0 and text[match.start()-1] in '特区分识差个类鉴告离辨级':
                continue
            yield match
SCOPE_NEGATION = _WordAwareNegation()

_legacy_primary_rule=primary_rule
"""Context-aware short-expression policy used by all seven training projects."""
import re

PRIMARY={'joy':'positive_affect','affection':'positive_affect','touched':'positive_affect',
 'caring':'positive_affect','disgust':'conflict_repair','anger':'conflict_repair',
 'disappointment':'conflict_repair','confusion':'cognitive_guidance','frustration':'conflict_repair',
 'neutral':'neutral'}
WORDS={'吐了':'disgust','烦':'frustration','爽':'joy','高兴':'joy','开心':'joy',
 '喜欢':'affection','崇拜':'affection','爱你':'affection','感动':'touched','什么':'confusion','纳尼':'confusion',
 '八嘎':'anger','操':'anger','艹':'anger','呸':'disgust','下流':'disgust','恶心':'disgust',
 '同情你':'caring','节哀':'caring','心疼你':'caring','关心一下你':'caring'}

def active_material_task(history):
    for text in history[:4]:
        # Newest first: an explicit stop supersedes any older material task.
        if re.search(r'不用(?:再)?翻译|不需要翻译|不需要解释|不用解释|别解释|翻译完了|已经解释完|先不玩|不玩(?:了|字词)|别再翻译|不用再造句|造句结束|stop translating|done translating',text,re.I):
            return False
        task=re.search(r'翻译|译成|译为|造句|释义|解释.{0,10}(?:意思|含义|词义|词语|单词|名词|词)|(?:解释|分析)(?:这个|一下|下)?词|什么意思|怎么(?:说|翻译)|词语解释|玩.{0,5}(?:字词|词语)|词语接龙|成语接龙|逐字分析|\b(?:translate|define|word game|meaning of|make a sentence)\b',text,re.I)
        if task and (re.search(r'接下来|我发|我下面发|我接下来发|我给|下面|帮我|请|把|将|继续|我们|咱们|逐个|每个|收到|给我|麻烦|什么意思|怎么翻译|这句话|这个词|一下|\b(?:please|next|send|word|phrase)\b',text,re.I)
                     or re.match(r'\s*(?:翻译|释义|解释|造句|玩字词|成语接龙|translate|define)',text,re.I)
                     or re.fullmatch(r'(?:翻译|造句|解释意思|词语解释|玩字词游戏)[。！!?？ ]*',text)):
            return True
    return False

def decision(row):
    text=row['current_user_message'].strip()
    bare=re.sub(r'[。．.!！?？、，,\s]+$','',text)
    if re.fullmatch(r'呵{5,}',bare):return 'neutral'
    label='disappointment' if re.fullmatch(r'呵{2,3}',bare) else WORDS.get(bare)
    if label is None:return None
    if active_material_task(row.get('history_user_messages',[])):return 'neutral'
    return label

def context_primary_rule(row):
    label=decision(row)
    return None if label is None else {'primary_label':PRIMARY[label],'decision_source':'rule','rule_id':'ultrashort_context_v1'}

def context_secondary_rule(row,branch):
    label=decision(row)
    return None if label is None or PRIMARY[label]!=branch else {'secondary_label':label,'decision_source':'rule','rule_id':'ultrashort_context_v1'}

def negation_spans(text):
    out=[]
    for match in re.finditer(r'并不是|并不|并非|没有|不要|不能|不可以|不再|未曾|别|不|没',text):
        if match.group()=='别' and match.start()>0 and text[match.start()-1] in '特区分识差个类鉴告离辨级':continue
        if match.group()=='不' and re.match(r'不(?:但|仅|只是)',text[match.start():]):continue
        out.append((match.group(),match.start(),match.end()))
    return out


CONTEXT_RULE_VERSION='context-ultrashort-disgust-caring-v1'


def primary_rule(row):
    record=row if isinstance(row,dict) else {'current_user_message':row,'history_user_messages':[]}
    direct=context_primary_rule(record)
    if direct is not None:return direct
    legacy=_legacy_primary_rule(record['current_user_message'])
    if legacy is not None and active_material_task(record.get('history_user_messages',[])):
        return {'primary_label':'neutral','decision_source':'rule','rule_id':'material_history_fullmatch'}
    return legacy

_previous_names=FEATURE_NAMES[:]
_previous_extract=extract_features
EXTRA_SCOPE=['imperative_emotion','speaker_apology','soft_apology','rejection_other',
 'caring_expression','disgust_expression','deny_disgust','negation_phrase','permission_phrase',
 'material_history_short','ultrashort_ratio','comfort_with_fault']
SCOPE_FEATURE_NAMES += ['scope.'+name for name in EXTRA_SCOPE]
_old_scope=scope_features
def scope_features(text):
    result=_old_scope(text)
    imperative=bool(re.search(r'(?<![特区分识差个类鉴告离辨级])(?:别|不要)(?:再|太|这么|那么|不)?(?:高兴|开心|生气|难过|伤心|哭)',text))
    apology=bool(re.search(r'我(?:真的|确实|刚才)?(?:错了|不对|做错|说错)|对不起|是我的错',text)) and not bool(DENY_FAULT.search(text))
    disgust=bool(re.search(r'讨厌|厌恶|反感|恶心|(?<![往向朝])下流(?![水河])|作呕|膈应|不想再见|别来沾边|离我远点|吐了|呸',text))
    caring=bool(re.search(r'同情你|节哀|心疼|关心|爱护|体谅|我陪你|(?<![特区分识差个类鉴告离辨级])别硬撑|照顾好自己|(?<![特区分识差个类鉴告离辨级])别难过|(?<![特区分识差个类鉴告离辨级])别伤心',text))
    values=[imperative,apology,apology and bool(re.search(r'好不好|行不行|好嘛|原谅',text)),
      bool(re.search(r'不想(?:再)?(?:见|理|看见)你|你走|离我远点|别靠近我',text)),
      caring,disgust,bool(re.search(r'不是讨厌|并不讨厌|不反感|不觉得恶心',text)),
      bool(re.search(r'不要|不可以|不能|别',text)) and bool(list(SCOPE_NEGATION.finditer(text))),
      bool(re.search(r'(?<!不)可以|允许|同意',text)),False,1/min(max(len(text.strip()),1),20),imperative and apology]
    result.update(('scope.'+name,float(v)) for name,v in zip(EXTRA_SCOPE,values))
    return result
# _old_scope zips the original 34 values; its original name prefix is retained.
FEATURE_NAMES=_previous_names+['scope.'+name for name in EXTRA_SCOPE]
def extract_features(record):
    values=_previous_extract(record)
    values['scope.material_history_short']=float(len(record['current_user_message'].strip())<=9 and active_material_task(record.get('history_user_messages',[])))
    assert list(values)==FEATURE_NAMES
    return values

_old_spans=spans
def spans(pattern,text):
    result=_old_spans(pattern,text)
    if '别' in pattern.pattern:
        result=[(a,b) for a,b in result if not (text[a:b].startswith('别') and a>0 and text[a-1] in '特区分识差个类鉴告离辨级')]
    return result
PRIMARY['guilt']='emotional_support'
_short_decision=decision
def decision(row):
    result=_short_decision(row)
    if result is not None:return result
    t=row['current_user_message'].strip().rstrip('。！!?？')
    if re.fullmatch(r'别(?:难过|伤心|不高兴|不开心)(?:了|啦|呀|嘛)?',t):
        return 'neutral' if active_material_task(row.get('history_user_messages',[])) else 'caring'
    if re.fullmatch(r'别生气(?:了)?[，, ]*我(?:真的|确实)?错了(?:好不好[呀啊嘛]?)?',t):
        return 'neutral' if active_material_task(row.get('history_user_messages',[])) else 'guilt'
    return None


# The same ordered features are consumed by primary and every secondary model.
# Character offsets are divided by current-text length; absent matches are 0
# and their hit flag disambiguates absence from a match at position zero.
BOUNDARY_COMBO_NAMES=['disgust_first_position','disgust_count_log','caring_first_position',
 'caring_count_log','disgust_target_near','caring_other_near','caring_imperative_near','negated_caring']
_scope_before_boundary=scope_features
DISGUST_SIGNAL=re.compile(r'厌恶|讨厌|恶心|反感|膈应|作呕|(?<![往向朝])下流(?![水河])|猥琐|倒胃口|反胃|嫌恶|吐了|呸|\b(?:disgust(?:ed|ing)?|repuls(?:ed|ive)|gross|revolting|creepy|sleazy|off.putting|skin crawl|turns? my stomach)\b',re.I)
CARING_SIGNAL=re.compile(r'心疼|同情|关心|关怀|照顾|体谅|节哀|抱抱|别硬撑|我陪你|好好休息|慢慢来|吃点东西|歇一会|\b(?:care about|take care|get some rest|go easy on yourself|sit with you|here with you|sorry for your loss|hugs)\b',re.I)
OTHER_TARGET=re.compile(r'你|他|她|孩子|老人|朋友|对方|家人|\b(?:you|he|she|him|her|them|friend|child)\b',re.I)
AVERSION_TARGET=re.compile(r'你|他|她|人|事|行为|样子|味|东西|声音|画面|\b(?:you|him|her|person|behavior|smell|texture|sound|people|it)\b',re.I)
COMFORT_IMPERATIVE=re.compile(r'(?<![特区分识差个类鉴告离辨级])别(?:再|那么|太)?(?:哭|难过|伤心|硬撑|自责|勉强)|不要.{0,5}(?:硬撑|勉强|饿着)|不用.{0,8}(?:强迫|道歉|解释)|\b(?:don.t|do not|you don.t have to|please rest)\b',re.I)
SCOPE_FEATURE_NAMES += ['scope.'+k for k in BOUNDARY_COMBO_NAMES]
FEATURE_NAMES += ['scope.'+k for k in BOUNDARY_COMBO_NAMES]

def scope_features(text):
    result=_scope_before_boundary(text);length=max(1,len(text))
    disgust=spans(DISGUST_SIGNAL,text);caring=spans(CARING_SIGNAL,text)
    other=spans(OTHER_TARGET,text);target=spans(AVERSION_TARGET,text)
    comfort=spans(COMFORT_IMPERATIVE,text)
    values=[disgust[0][0]/length if disgust else 0,math.log1p(min(len(disgust),5)),
            caring[0][0]/length if caring else 0,math.log1p(min(len(caring),5)),
            _same_clause_near(text,disgust,target),_same_clause_near(text,caring,other),
            _same_clause_near(text,comfort,other),any(_negated_span(text,z) for z in caring)]
    result.update(('scope.'+k,float(v)) for k,v in zip(BOUNDARY_COMBO_NAMES,values))
    return result


# Common named chat emoji: soft lexical evidence, never direct labels.
EMOJI_LEXICON = {'neutral': ['OK', '收到'], 'approval': ['握手', '强', '赞', 'OK'], 'gratitude': ['玫瑰', '抱拳', '爱心', '双手合十', '害羞', '愉快'], 'relief': ['擦汗', '愉快', '破涕为笑', '合十'], 'realization': ['灯泡', 'OK', '抱拳', '愉快'], 'affection': ['爱心', '比心', '亲亲', '玫瑰', '害羞', '愉快'], 'dependency': ['抱抱', '委屈', '可怜', '爱心'], 'hope': ['双手合十', '祈祷', '许愿', '四叶草', '可怜'], 'joy': ['笑脸', '哈哈', '大笑', '开心', '跳舞', '笑哭', '偷笑', '龇牙', '嘻嘻', '捂脸', '愉快'], 'pride': ['得意', '酷', '胜利', '强', '耶'], 'touched': ['流泪', '爱心', '拥抱'], 'playfulness': ['调皮', '吐舌', '偷笑', '坏笑', '吃瓜', 'doge', '旺柴'], 'caring': ['抱抱', '拥抱', '爱心', '玫瑰'], 'anger': ['发怒', '抓狂', '咒骂'], 'frustration': ['抓狂', '敲打', '擦汗'], 'impatience': ['白眼', '无语', '汗', '困'], 'disappointment': ['叹气', '心碎', '撇嘴', '苦涩', '无奈'], 'guardedness': ['闭嘴', '嘘', '警告'], 'disapproval': ['摇头', '白眼', '撇嘴', '皱眉'], 'disgust': ['呕吐', '恶心', '嫌弃', '鄙视'], 'sadness': ['流泪', '大哭', '难过', '心碎', '裂开'], 'loneliness': ['凋谢', '叹气', '流泪'], 'anxiety': ['冷汗', '惊恐', '皱眉', '汗'], 'guilt': ['委屈', '难过', '流泪', '抱歉'], 'embarrassment': ['尴尬', '捂脸', '害羞', '汗', '抠鼻', '再见', '无语'], 'confusion': ['疑问', '晕', '敲打', '思考'], 'skepticism': ['疑问', '思考', '皱眉', '放大镜'], 'curiosity': ['双眼', '疑问', '思考', '放大镜'], 'gaming': [], 'boredom': ['发呆', '困', '睡觉', '哈欠']}
EMOJI_EXCLUDED = frozenset({'微笑'})
EMOJI_NAMES = sorted({x for names in EMOJI_LEXICON.values() for x in names} - EMOJI_EXCLUDED)
EMOJI_PATTERN = re.compile(r'\[(' + '|'.join(re.escape(x) for x in sorted(EMOJI_NAMES,key=len,reverse=True)) + r')\]')
EMOJI_FEATURE_NAMES = ['scope.emoji_hit','scope.emoji_count_log','scope.emoji_only'] + ['scope.emoji_'+k+'_hit' for k,icons in EMOJI_LEXICON.items() if icons]
_scope_before_emoji = scope_features
SCOPE_FEATURE_NAMES += EMOJI_FEATURE_NAMES
FEATURE_NAMES += EMOJI_FEATURE_NAMES
def scope_features(text):
    result = _scope_before_emoji(text)
    matches = list(EMOJI_PATTERN.finditer(text))
    names = {m.group(1) for m in matches}
    only = bool(matches) and not EMOJI_PATTERN.sub('',text).strip()
    values = [bool(matches),math.log1p(min(len(matches),5)),only]
    values += [bool(names.intersection(icons)) for icons in EMOJI_LEXICON.values() if icons]
    result.update((key,float(value)) for key,value in zip(EMOJI_FEATURE_NAMES,values))
    return result

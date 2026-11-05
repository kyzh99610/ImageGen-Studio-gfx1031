"""zh_tags.py — common Chinese words → Danbooru tags, for 一键出图's "what to draw" box.

The checkpoints' text encoders (CLIP) read English: Chinese words in a prompt are mostly ignored (the Help says so). This table turns the
everyday words of a scene into the tags the anime checkpoints were trained on — places, time, weather, light, framing and camera, poses,
expressions, hair, eyes, basic clothes and colours. A word that is not in the table is reported, never guessed. English text passes
through unchanged. Only everyday (SFW) words are here; anything else is for the user to write themself.
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[一-鿿]")

# colours: combined with the clothing / hair / eye word that follows ("红色裙子" -> "red dress", "银发" via HAIR)
COLOURS = {
    "红色": "red", "红": "red", "蓝色": "blue", "蓝": "blue", "浅蓝": "light blue", "深蓝": "dark blue", "绿色": "green", "绿": "green",
    "黄色": "yellow", "黄": "yellow", "白色": "white", "白": "white", "黑色": "black", "黑": "black", "粉色": "pink", "粉红": "pink", "粉": "pink",
    "紫色": "purple", "紫": "purple", "橙色": "orange", "橙": "orange", "灰色": "grey", "灰": "grey", "棕色": "brown", "棕": "brown",
    "金色": "gold", "银色": "silver", "米色": "beige", "藏青": "navy blue",
}

# nouns a colour can go in front of
CLOTHES = {
    "连衣裙": "dress", "裙子": "skirt", "短裙": "miniskirt", "长裙": "long skirt", "百褶裙": "pleated skirt", "衬衫": "shirt", "T恤": "t-shirt",
    "毛衣": "sweater", "开衫": "cardigan", "外套": "jacket", "夹克": "jacket", "大衣": "coat", "风衣": "trench coat", "卫衣": "hoodie",
    "背心": "camisole", "吊带": "camisole", "短裤": "shorts", "牛仔裤": "jeans", "裤子": "pants", "西装": "suit", "制服": "uniform",
    "校服": "school uniform", "水手服": "serafuku", "和服": "kimono", "浴衣": "yukata", "旗袍": "china dress", "女仆装": "maid",
    "泳装": "swimsuit", "泳衣": "swimsuit", "比基尼": "bikini", "睡衣": "pajamas", "婚纱": "wedding dress", "礼服": "evening gown",
    "围巾": "scarf", "帽子": "hat", "贝雷帽": "beret", "草帽": "straw hat", "蝴蝶结": "bow", "发带": "hairband", "丝带": "ribbon",
    "手套": "gloves", "长筒袜": "thighhighs", "丝袜": "pantyhose", "袜子": "socks", "靴子": "boots", "高跟鞋": "high heels", "运动鞋": "sneakers",
    "眼镜": "glasses", "耳环": "earrings", "项链": "necklace", "领带": "necktie", "围裙": "apron", "斗篷": "cape", "披风": "cape",
}

HAIR = {
    "长发": "long hair", "短发": "short hair", "中长发": "medium hair", "马尾": "ponytail", "双马尾": "twintails", "辫子": "braid",
    "麻花辫": "braid", "丸子头": "hair bun", "刘海": "bangs", "齐刘海": "blunt bangs", "卷发": "wavy hair", "直发": "straight hair",
    "金发": "blonde hair", "银发": "silver hair", "白发": "white hair", "黑发": "black hair", "红发": "red hair", "蓝发": "blue hair",
    "粉发": "pink hair", "紫发": "purple hair", "棕发": "brown hair", "灰发": "grey hair", "绿发": "green hair",
    "头发": "hair",
}

EYES = {"红眼": "red eyes", "蓝眼": "blue eyes", "绿眼": "green eyes", "金眼": "yellow eyes", "紫眼": "purple eyes", "黑眼": "black eyes",
        "眼睛": "eyes", "异色瞳": "heterochromia"}

WORDS = {
    # who
    "女孩": "1girl", "一个女孩": "1girl, solo", "少女": "1girl", "女性": "1girl", "男孩": "1boy", "两个女孩": "2girls", "单人": "solo",
    # places
    "海边": "beach", "沙滩": "beach", "大海": "ocean", "海": "ocean", "海浪": "waves", "海滩": "beach", "游泳池": "pool", "泳池": "pool",
    "教室": "classroom", "学校": "school", "走廊": "hallway", "图书馆": "library", "卧室": "bedroom", "房间": "room", "床": "bed",
    "厨房": "kitchen", "浴室": "bathroom", "咖啡馆": "cafe", "咖啡厅": "cafe", "餐厅": "restaurant", "街道": "street", "城市": "city",
    "城市夜景": "city lights, night", "屋顶": "rooftop", "天台": "rooftop", "公园": "park", "花园": "garden", "森林": "forest", "树林": "forest",
    "草地": "grass field", "花田": "flower field", "山": "mountain", "湖": "lake", "河": "river", "桥": "bridge", "神社": "shrine",
    "寺庙": "temple", "樱花": "cherry blossoms", "樱花树": "cherry blossom tree", "雪地": "snow", "沙漠": "desert", "车站": "train station",
    "电车": "train interior", "船": "ship", "甲板": "ship deck", "军舰": "warship", "港口": "harbor", "窗边": "by window", "窗户": "window",
    "沙发": "couch", "椅子": "chair", "楼梯": "stairs", "舞台": "stage", "天空": "sky", "云": "cloud", "星空": "starry sky", "宇宙": "space",
    "室内": "indoors", "室外": "outdoors", "户外": "outdoors",
    # time, weather, light
    "白天": "day", "夜晚": "night", "晚上": "night", "夜景": "night, city lights", "早晨": "morning", "黄昏": "dusk", "傍晚": "evening",
    "日落": "sunset", "夕阳": "sunset", "日出": "sunrise", "晴天": "clear sky", "下雨": "rain", "雨": "rain", "雨天": "rain", "雪": "snow",
    "下雪": "snowing", "雾": "fog", "彩虹": "rainbow", "月亮": "moon", "满月": "full moon", "阳光": "sunlight", "逆光": "backlighting",
    "侧光": "sidelighting", "柔光": "soft lighting", "霓虹": "neon lights", "霓虹灯": "neon lights", "烛光": "candlelight",
    "光斑": "lens flare", "景深": "depth of field", "背景虚化": "blurry background, bokeh", "花瓣": "petals", "落叶": "falling leaves",
    "萤火虫": "fireflies", "烟花": "fireworks", "春天": "spring", "夏天": "summer", "秋天": "autumn", "冬天": "winter",
    # framing, camera
    "头像": "portrait", "特写": "close-up", "半身": "upper body", "七分身": "cowboy shot", "全身": "full body", "远景": "wide shot",
    "从上方": "from above", "俯视": "from above", "从下方": "from below", "仰视": "from below", "侧面": "from side", "背面": "from behind",
    "背影": "from behind", "正面": "facing viewer", "看镜头": "looking at viewer", "看着镜头": "looking at viewer", "回头": "looking back",
    "回头看": "looking back", "看向远方": "looking afar", "看旁边": "looking to the side", "自拍": "selfie",
    # poses
    "站着": "standing", "站立": "standing", "坐着": "sitting", "坐": "sitting", "躺着": "lying", "躺": "lying", "跪坐": "seiza",
    "蹲": "squatting", "蹲下": "squatting", "走路": "walking", "跑步": "running", "跳": "jumping", "跳跃": "jumping", "跳舞": "dancing",
    "伸懒腰": "stretching", "抱膝": "hugging own legs", "双手叉腰": "hands on hips", "叉腰": "hand on hip", "托腮": "hand on own cheek",
    "比耶": "v", "剪刀手": "v", "挥手": "waving", "双手合十": "own hands together", "抱着": "holding", "拿着": "holding",
    "撑伞": "holding umbrella", "伞": "umbrella", "看书": "reading, book", "喝茶": "drinking, cup", "喝咖啡": "coffee, cup",
    "吃东西": "eating", "睡觉": "sleeping", "倚靠": "leaning", "靠墙": "against wall", "撩头发": "hand in own hair", "张开双臂": "outstretched arms",
    "举手": "arm up", "双手举起": "arms up", "交叉双臂": "crossed arms", "歪头": "head tilt", "眨眼": "one eye closed", "闭眼": "closed eyes",
    # expressions
    "微笑": "smile", "笑": "smile", "大笑": "laughing, open mouth", "开心": "happy, smile", "害羞": "blush, shy", "脸红": "blush",
    "生气": "angry", "哭": "crying, tears", "流泪": "tears", "惊讶": "surprised", "困": "sleepy", "无表情": "expressionless",
    "得意": "smug", "认真": "serious", "温柔": "gentle smile", "调皮": "playful, tongue out", "吐舌": "tongue out", "张嘴": "open mouth",
    "闭嘴": "closed mouth", "撅嘴": "pout", "嘟嘴": "pout",
    # things
    "猫": "cat", "狗": "dog", "鸟": "bird", "蝴蝶": "butterfly", "花": "flower", "花束": "bouquet", "玫瑰": "rose", "向日葵": "sunflower",
    "书": "book", "手机": "phone", "相机": "camera", "耳机": "headphones", "蛋糕": "cake", "冰淇淋": "ice cream", "气球": "balloon",
    "灯笼": "lantern", "剑": "sword", "枪": "gun",
}

# whole words that would otherwise start with a one-character word (猫耳 is cat ears, not a cat)
WORDS.update({"猫耳": "cat ears", "兽耳": "animal ears", "狐耳": "fox ears", "兔耳": "rabbit ears", "猫娘": "cat girl", "狐狸": "fox",
              "海军服": "naval uniform", "军装": "military uniform", "看书": "reading, book", "花园": "garden", "雨伞": "umbrella",
              "雪人": "snowman", "云朵": "cloud", "山顶": "mountain top", "湖边": "lakeside", "河边": "riverbank", "桥上": "on bridge"})
CLOTHES.update({"体恤": "t-shirt"})

_TABLE: dict[str, str] = {}
for _d in (WORDS, CLOTHES, HAIR, EYES):
    _TABLE.update(_d)
_KEYS = sorted(set(_TABLE) | set(COLOURS), key=len, reverse=True)
# small words that carry no picture of their own ("in", "wearing", "of" …)
_FILLER = "的了和与在着很非常一点有个地得里中上是也又把被正穿戴"
# Chinese punctuation separates tags; ASCII punctuation (weights, emoticons such as ;) or !?) is left as it is
_PUNCT = str.maketrans({"，": ",", "、": ",", "。": ",", "；": ",", "！": ",", "？": ",", "\n": ",", "（": ",", "）": ",", "：": ","})


def _key_at(piece: str, i: int) -> str | None:
    return next((k for k in _KEYS if piece.startswith(k, i)), None)


def _segment(piece: str) -> tuple[list[str], list[str]]:
    """Greedy longest match of one Chinese piece over the table: (tags, words not understood)."""
    tags, unknown, i, junk = [], [], 0, ""

    def flush():
        nonlocal junk
        if junk:
            unknown.append(junk)
            junk = ""
    while i < len(piece):
        hit = _key_at(piece, i)
        # a one-character word counts only when it stands alone (the end, a small word or another known word follows): 猫耳 is not 'cat'
        if hit and len(hit) == 1 and i + 1 < len(piece) and piece[i + 1] not in _FILLER and _key_at(piece, i + 1) is None:
            hit = None
        if hit is None:
            ch = piece[i]
            if ch.strip() and ch not in _FILLER:
                junk += ch
            else:
                flush()
            i += 1
            continue
        flush()
        i += len(hit)
        if hit in COLOURS:
            colour, j = COLOURS[hit], i
            while j < len(piece) and piece[j] == "的":            # 红色的裙子
                j += 1
            cloth = next((k for k in _KEYS if piece.startswith(k, j) and k in CLOTHES), None)
            hair = next((k for k in _KEYS if piece.startswith(k, j) and k in HAIR), None)
            part = next((k for k in ("头发", "眼睛", "发", "眼", "瞳") if piece.startswith(k, j)), None)
            if cloth:
                tags.append(f"{colour} {CLOTHES[cloth]}")
                i = j + len(cloth)
            elif hair:                                               # 黑色长发 -> black hair, long hair
                tags += [f"{colour} hair"] + ([HAIR[hair]] if HAIR[hair] != "hair" else [])
                i = j + len(hair)
            elif part:
                tags.append(f"{colour} {'hair' if '发' in part else 'eyes'}")
                i = j + len(part)
            else:
                tags.append(colour)
            continue
        tags.append(_TABLE[hit])
        if hit.endswith("眼") and piece.startswith("睛", i):          # 蓝眼睛
            i += 1
    flush()
    return tags, unknown


def to_tags(text: str) -> tuple[str, list[str]]:
    """'海边，回头看，微笑, cowboy shot' -> ('beach, looking back, smile, cowboy shot', []); Chinese words not in the table come back in the list.
    Tags without Chinese pass through unchanged (weights, wildcards like __set/name__, {a|b} groups, emoticons)."""
    from backend.prompt_tools import split_tags
    text = re.sub(r"[Tt]恤", "体恤", text or "").translate(_PUNCT)
    out, unknown = [], []

    def words(run: str) -> list[str]:
        t, u = _segment(run)
        unknown.extend(u)
        return [x.strip() for tag in t for x in tag.split(",") if x.strip()]
    for tag in split_tags(text):
        if not _CJK.search(tag):
            out.append(tag)
        elif any(c in tag for c in "{}()[]|<>") or "__" in tag:
            # a group or a weight: translate the Chinese inside and keep the structure ({微笑|大笑} -> {smile|laughing, open mouth})
            new = re.sub(r"[一-鿿]+", lambda m: ", ".join(words(m.group(0))), tag)
            new = re.sub(r"([{(\[|])\s*,\s*", r"\1", new)
            new = re.sub(r"\s*,\s*([})\]|:])", r"\1", new).strip(" ,")
            if new:
                out.append(new)
        else:
            for run in re.findall(r"[一-鿿]+|[^一-鿿]+", tag):          # mixed text: the English runs stay, the Chinese runs are looked up
                if _CJK.search(run):
                    out += words(run)
                elif run.strip(" ,"):
                    out.append(run.strip(" ,"))
    seen, tags = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            tags.append(t)
    return ", ".join(tags), unknown

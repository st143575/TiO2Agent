"""Clean up text extracted from Chinese technical PDFs.

pypdf reproduces the PDF's glyph positions, so the text comes with artefacts that
break both keyword search and number extraction. Each rule below fixes one seen in
the sample patent:

    分散剂0 . 5‑0 .9份     -> 分散剂0.5-0.9份      (split decimals, U+2011 hyphen)
    温度设为50 ℃‑80℃      -> 温度设为50℃-80℃
    去离子水 40 ‑50份      -> 去离子水40-50份
    二氧化钛 、去离子水    -> 二氧化钛、去离子水   (space before CJK punctuation)
    以 及 / 30 名          -> 以及 / 30名
    将PH值调节至5          -> 将pH值调节至5

Latin-only spans are left alone ("C09C 1/36(2006.01)" keeps its space). The text
is never rewritten beyond spacing, dashes and the pH casing: no NFKC, so Chinese
full-width punctuation (，：；) stays as written.
"""

import re

# Dashes that pypdf emits for ranges; U+2014 (em dash, "——" in Chinese) is kept.
_HYPHENS = str.maketrans({c: "-" for c in "‐‑‒−﹣－"})   # 把各种奇怪的短横线（连字符、减号、全角减号等，如 ‑, −, －）统一翻译为标准的半角减号 -
_SPACES = str.maketrans({"　": " ", " ": " ", " ": " ", " ": " ", "\t": " "})   # 把全角空格、不换行空格、制表符（\t）等统一翻译为普通的半角空格

_CJK = r"㐀-䶿一-鿿豈-﫿"   # 囊括了中日韩（CJK）统一表意文字的所有主流区块编码
_CJK_PUNCT = r"、-〿！-／：-＠［-｀｛-･“”‘’…—"   # 常见的中文全角标点符号
_UNITS = r"℃%‰°"   # 常见的物理单位符号

_MULTI_SPACE = re.compile(r" {2,}")  # 匹配两个或多个连续空格
_DECIMAL = re.compile(r"(?<=\d) ?\. ?(?=\d)")  # 匹配数字之间的点号，用于修复断裂的小数，如：0 . 5 -> 0.5
_SPACE_NEAR_CJK = re.compile(rf"(?<=[{_CJK}{_CJK_PUNCT}]) +| +(?=[{_CJK}{_CJK_PUNCT}])")  # 匹配CJK字符或标点符号附近的空格，用于消除中文标点符号前后的空格
_SPACE_BEFORE_UNIT = re.compile(rf" +(?=[{_UNITS}])")  # 匹配单位符号前的空格，用于消除单位符号前的空格，如：50 ℃ -> 50℃
_RANGE = re.compile(rf"(?<=[\d{_UNITS}]) *[-~～] *(?=\d)")  # 匹配单位符号后的范围符号，用于统一范围符号
_EN_DASH_RANGE = re.compile(r"(?<=\d) *– *(?=\d)")  # 匹配数字间的短横线，用于统一短横线
_PH = re.compile(r"(?<![A-Za-z])P[Hh](?=\s*(?:值|=|＝|为|调|至|在|\d))")  # 匹配pH值，用于统一pH值的大小写


def normalize_text(text: str) -> str:
    text = text.translate(_SPACES).translate(_HYPHENS)
    text = _EN_DASH_RANGE.sub("-", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _DECIMAL.sub(".", text)
    text = _SPACE_NEAR_CJK.sub("", text)
    text = _SPACE_BEFORE_UNIT.sub("", text)
    text = _RANGE.sub(lambda m: "~" if "~" in m.group() or "～" in m.group() else "-", text)
    text = _PH.sub("pH", text)
    return text.strip()


_ALNUM_END = re.compile(r"[A-Za-z]$")
_ALNUM_START = re.compile(r"^[A-Za-z]")


def join_lines(lines: list[str], *, hard_break: re.Pattern[str] | None = None) -> str:
    """Join wrapped lines of one paragraph.

    Chinese text wraps at any character, so lines are concatenated directly; only two
    Latin words that meet at a line break get a space. Lines matching ``hard_break``
    (e.g. process steps "S2、...") start on a new line.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        # 如果当前行匹配硬换行模式（如工艺步骤"S1、..."），则在当前行前添加换行符，保证每个工艺步骤独立成行。
        elif hard_break is not None and hard_break.match(line):
            out += "\n" + line
        # 如果上一行的结尾是英文字母，且下一行的开头也是英文字母，则在两者中间加一个空格。
        elif _ALNUM_END.search(out) and _ALNUM_START.search(line):
            out += " " + line
        # 中文可以从任意字符处换行，不需要空格连接，直接将两行拼接。
        else:
            out += line
    return out


def squash(text: str) -> str:
    """
    Remove all whitespace; used to compare headings like "说　明　书".
    有的PDF排版时为了美观，会在标题中添加空格，如"说　明　书"，需要消除这些空格。
    """
    return re.sub(r"\s+", "", text.translate(_SPACES))


# --- numbering ---------------------------------------------------------------------

CN_DIGITS = "零一二两三四五六七八九十"
_CN_VALUES = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def chinese_number(text: str) -> int | None:
    """"三" -> 3, "十" -> 10, "十二" -> 12, "二十一" -> 21; None if not a number up to 99."""
    if not text or any(c not in _CN_VALUES and c != "十" for c in text):
        return None
    if "十" not in text:
        return _CN_VALUES[text] if len(text) == 1 else None
    tens, _, ones = text.partition("十")
    if len(tens) > 1 or len(ones) > 1:
        return None
    return (_CN_VALUES[tens] if tens else 1) * 10 + (_CN_VALUES[ones] if ones else 0)


def parse_number(text: str) -> int | None:
    """"3" or "三" -> 3."""
    text = text.strip()
    return int(text) if text.isdigit() else chinese_number(text)


# Step markers at the start of a paragraph or claim line: "S1、", "步骤1：", "步骤一、",
# "第一步，". Not "步骤1中…" (a reference to a step), hence the required separator.
STEP_MARKER = re.compile(
    rf"^(?:S\s*(?P<s>\d+)|步骤\s*(?P<b>\d+|[{CN_DIGITS}]+)|第\s*(?P<d>\d+|[{CN_DIGITS}]+)\s*步)\s*[、.．:：，,]"
)


def step_id(text: str) -> str | None:
    """The step a paragraph or claim line starts with, normalized to "S1", "S2", …"""
    m = STEP_MARKER.match(text)
    if not m:
        return None
    n = parse_number(m.group("s") or m.group("b") or m.group("d"))
    return f"S{n}" if n is not None else None
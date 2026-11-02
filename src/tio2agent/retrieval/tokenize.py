"""Tokenizer for keyword search over Chinese technical text.

No dictionary and no segmentation model: Chinese runs become character bigrams, a
common and robust baseline for Chinese IR that handles unseen technical terms
("锆盐", "二叔丁基甲酚") as well as common words. On top of that:

* references are kept whole: "实施例1" -> "实施例1", "权利要求8" -> "权利要求8",
  "S2" -> "s2", so a question about 实施例2 does not match every chunk containing "2";
* a range is one token plus its endpoints: "45-65份" -> "45-65", "45", "65", and
  "50℃-80℃" -> "50-80", "50", "80", so both "45-65" and "45" find it;
* Latin words are lower-cased: "pH" -> "ph".

Input should already be normalized (normalize_text), which index and query both do.
"""

import re

_CJK = r"㐀-䶿一-鿿豈-﫿"
_REFERENCE = re.compile(r"(实施例|对比例|比较例|权利要求)\s*(\d+)")
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[℃%]?\s*[-~]\s*(\d+(?:\.\d+)?)")
_TOKEN = re.compile(rf"[{_CJK}]+|[A-Za-z]+\d*|\d+(?:\.\d+)?")

# Bigrams that occur in nearly every patent paragraph and carry no topic.
STOP_BIGRAMS = frozenset({"所述", "其特", "特征", "征在", "在于", "根据", "本发", "发明", "一种"})


def tokenize(text: str) -> list[str]:
    """
    中文分词。采用bigram策略，将连续的中文字符切分为双字节组合。
    同时处理专利中的特殊格式，如"实施例1"、"权利要求8"等。
    """
    tokens: list[str] = []

    for m in _REFERENCE.finditer(text):
        tokens.append(f"{m.group(1)}{m.group(2)}")
    for m in _RANGE.finditer(text):
        tokens.append(f"{m.group(1)}-{m.group(2)}")

    for m in _TOKEN.finditer(text):
        piece = m.group()
        if re.match(rf"[{_CJK}]", piece):
            if len(piece) == 1:
                tokens.append(piece)
            else:
                tokens.extend(
                    bigram for i in range(len(piece) - 1)
                    if (bigram := piece[i : i + 2]) not in STOP_BIGRAMS
                )
        else:
            tokens.append(piece.lower())
    return tokens

"""Checking a quoted piece of evidence against the original text.

An LLM is asked to copy its evidence verbatim, but it may change what it copies in
small ways: drop or add a space, write a half-width "%" for the full-width "％", or
"°C" for "℃". ``locate`` finds the quote anyway and returns the text *as it is in the
original*, so a stored fact always quotes the document exactly.
"""

import re
import unicodedata

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_STEP_REF = re.compile(r"(?:步骤\s*S?|S)(\d+)")


def _folded(text: str) -> tuple[str, list[int]]:
    """NFKC-folded, case-folded text without whitespace, and for each of its
    characters the index of the original character it came from."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).casefold():
            if not c.isspace():
                chars.append(c)
                index.append(i)
    return "".join(chars), index


def squash(text: str) -> str:
    return _folded(text)[0]


def locate(text: str, quote: str) -> str | None:
    """The span of ``text`` that ``quote`` copies (exactly, or up to whitespace,
    full-/half-width forms and case); None if it is not there."""
    quote = quote.strip()
    if not quote:
        return None
    if quote in text:
        return quote
    folded_text, index = _folded(text)
    folded_quote = squash(quote)
    if not folded_quote:
        return None
    pos = folded_text.find(folded_quote)
    if pos < 0:
        return None
    start, end = index[pos], index[pos + len(folded_quote) - 1] + 1
    return text[start:end]


def contains(haystack: str, needle: str) -> bool:
    """``needle`` occurs in ``haystack`` up to whitespace, width and case."""
    return bool(needle.strip()) and squash(needle) in squash(haystack)


def numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(unicodedata.normalize("NFKC", text))}


def step_references(text: str) -> set[str]:
    """Steps a text names: "S2、…", "步骤S2中", "步骤2：", "所述步骤4中" -> {"S2", "S4"}."""
    return {f"S{int(n)}" for n in _STEP_REF.findall(unicodedata.normalize("NFKC", text))}


def spans(text: str, quote: str) -> list[tuple[int, int]]:
    """Every exact occurrence of ``quote`` in ``text``."""
    out = []
    start = text.find(quote) if quote else -1
    while start >= 0:
        out.append((start, start + len(quote)))
        start = text.find(quote, start + 1)
    return out

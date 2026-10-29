"""Turn table output from OCR engines into a plain cell grid.

Two formats are supported:

* HTML ``<table>`` with ``rowspan``/``colspan``. This is what the PaddleOCR-VL
  pipeline service returns in ``block_content`` for table blocks.
* OTSL (``<fcel>``, ``<ecel>``, ``<lcel>``, ``<ucel>``, ``<xcel>``, ``<nl>``), the
  raw output of the PaddleOCR-VL model. The pipeline converts it to HTML, but falls
  back to the raw OTSL string when conversion fails, so we accept both.

Merged cells are expanded: every grid position covered by a span repeats the text
of the cell that spans it. That keeps "row header x column header -> value" lookups
simple for downstream stages.
"""

import re
import unicodedata
from html.parser import HTMLParser

Table = list[list[str]]

_WHITESPACE = re.compile(r"\s+")
_OTSL_TOKEN = re.compile(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>|<nl>)")


def normalize_cell(text: str) -> str:
    """Canonical form for comparing cell text.

    NFKC folds full-width digits and punctuation ("１．２１" -> "1.21"); whitespace is
    dropped entirely because it carries no meaning inside Chinese table cells and OCR
    often inserts it inside numbers ("1. 21").
    """
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text))


def parse_table(content: str) -> Table | None:
    """Parse HTML or OTSL table content. Returns None if neither format is present."""
    if "<table" in content.lower():
        return parse_html_table(content)
    if _OTSL_TOKEN.search(content):
        return parse_otsl(content)
    return None


# --- HTML -------------------------------------------------------------------------


class _HTMLTableParser(HTMLParser):
    """Collects (text, rowspan, colspan) cells of the first top-level table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._depth = 0
        self._done = False
        self._cell: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        if tag == "table":
            self._depth += 1
        elif self._depth != 1:
            return  # ignore nested tables
        elif tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            values = dict(attrs)
            self._span = (_span(values.get("rowspan")), _span(values.get("colspan")))
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._done:
            return
        if tag == "table":
            self._depth -= 1
            if self._depth == 0:
                self._done = True
        elif self._depth == 1 and tag in ("td", "th") and self._cell is not None:
            text = _WHITESPACE.sub(" ", "".join(self._cell)).strip()
            self.rows[-1].append((text, *self._span))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None and self._depth == 1:
            self._cell.append(data)


def _span(value: str | None) -> int:
    try:
        return max(1, min(int(value or 1), 1000))
    except ValueError:
        return 1


def parse_html_table(html: str) -> Table:
    parser = _HTMLTableParser()
    parser.feed(html)
    parser.close()

    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(parser.rows):
        c = 0
        for text, rowspan, colspan in row:
            while (r, c) in grid:  # skip positions filled by a rowspan from above
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = text
            c += colspan
    return _grid_to_rows(grid)


# --- OTSL -------------------------------------------------------------------------


def parse_otsl(content: str) -> Table:
    rows: Table = []
    row: list[str] = []
    pending_text: list[str] | None = None

    def flush_text() -> None:
        nonlocal pending_text
        if pending_text is not None:
            row[-1] = "".join(pending_text).strip()
            pending_text = None

    for part in _OTSL_TOKEN.split(content):
        if not part:
            continue
        if part == "<nl>":
            flush_text()
            rows.append(row)
            row = []
        elif part == "<fcel>":
            flush_text()
            row.append("")
            pending_text = []
        elif part == "<ecel>":
            flush_text()
            row.append("")
        elif part in ("<lcel>", "<xcel>"):  # merged with the cell to the left
            flush_text()
            row.append(row[-1] if row else "")
        elif part == "<ucel>":  # merged with the cell above
            flush_text()
            col = len(row)
            above = rows[-1][col] if rows and col < len(rows[-1]) else ""
            row.append(above)
        elif pending_text is not None:
            pending_text.append(part)
    flush_text()
    if row:
        rows.append(row)

    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows if r]


# --- output -----------------------------------------------------------------------


def _grid_to_rows(grid: dict[tuple[int, int], str]) -> Table:
    if not grid:
        return []
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def table_to_markdown(rows: Table) -> str:
    if not rows:
        return ""

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    lines = [fmt(rows[0]), "| " + " | ".join("---" for _ in rows[0]) + " |"]
    lines += [fmt(row) for row in rows[1:]]
    return "\n".join(lines)

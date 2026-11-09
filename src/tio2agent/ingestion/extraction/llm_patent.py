"""LLM fact extraction for patents, with every proposed fact checked against the text.

The model reads the same authoritative places as the rule extractor: each claim, and
each example (实施例) as a whole. Restating paragraphs and comparative sections are
not sent. Tables are read from their cells as before (``table_results``): they are
already structured, and an LLM adds nothing there.

What the model decides: kind, name, qualifier, the numbers, and which words are the
evidence. What it does not decide: the subject (claim N / example E1), the location,
and, where the structure knows it, the step. Each proposed fact must pass:

    evidence   a contiguous quote of the claim / one paragraph of the example
               (``evidence.locate``: whitespace, width and case may differ; the stored
               evidence is the original text)
    numbers    low and high both occur in the evidence
    name       components and options: the material name occurs in the evidence
               (or a known alias of it: "去离子" for 去离子水)
    options    every option occurs in the evidence
    step       paragraph with a step marker: that step; claim with step lines: the
               line holding the evidence; otherwise the model's step only if the text
               names it ("所述步骤S2中"), else none
    unique     one fact per (subject, step, kind, name, qualifier)

Failures are dropped and reported (``ExtractionResult.rejected``, one warning per
claim or example). An LLM error on one claim or example is a warning; the others are
still extracted. After ``max_failures`` failed requests in a row (the service is down,
or every request times out) the rest of the document is not sent to the LLM, so an
import does not hang for hours; in hybrid mode the rules still cover it.

What this cannot catch: a parameter given the wrong name ("温度" for a number that is
a time) while its number and quote are right. Comparing with the claimed range
(``validate_facts``) and with the rule extractor (``HybridExtractor``) catches some of
these; the gold standard (evals/run_extraction_eval.py) measures the rest.
"""

import json
import logging
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from typing import Any

from tio2agent.domain import EmbodimentKind, FactKind, StructuredDocument
from tio2agent.ingestion.extraction import evidence as ev
from tio2agent.ingestion.extraction.base import ExtractionIssue, ExtractionResult, FactBuilder, Rejection, Source
from tio2agent.ingestion.extraction.llm_prompts import (
    RECORD_FACTS,
    RETRY_NOTE,
    SYSTEM_PROMPT,
    TOOL_NAME,
    claim_message,
    document_line,
    embodiment_message,
)
from tio2agent.ingestion.extraction.patent_rules import table_results
from tio2agent.ingestion.extraction.quantities import NAME_ALIASES, canonical_unit, normalize_chemical_name
from tio2agent.ingestion.structure.normalize import step_id
from tio2agent.llm import ChatMessage, ChatModel, LlmError, Usage

logger = logging.getLogger(__name__)

_KINDS = {"component": FactKind.COMPONENT, "parameter": FactKind.PARAMETER, "option": FactKind.OPTION}
_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_STEP = re.compile(r"^(?:S|步骤\s*S?)?\s*(\d+)$")
_PARAGRAPH_ID = re.compile(r"^\[\d{4}\]\s*")
_MAX_NAME = 30


class Unit:
    """One request to the model: a claim, or all paragraphs of one example."""

    def __init__(self, subject: str, label: str, message: str, sources: list[Source], claim_mode: bool) -> None:
        self.subject = subject
        self.label = label
        self.message = message
        self.sources = sources        # a claim: one source; an example: one per paragraph
        self.claim_mode = claim_mode


class LlmPatentExtractor:
    name = "llm"

    def __init__(
        self,
        model: ChatModel,
        *,
        retries: int = 1,
        include_tables: bool = True,
        max_failures: int = 2,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.retries = retries
        self.include_tables = include_tables
        self.max_failures = max_failures
        self.progress = progress

    # --- driver ---------------------------------------------------------------------

    def extract(self, doc: StructuredDocument) -> ExtractionResult:
        builder = FactBuilder(doc.doc_id, self.name)
        result = ExtractionResult()
        stats = {"model": self.model.model, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                 "failed_units": 0, "skipped_units": 0, "seconds": 0.0, "failures_in_a_row": 0}
        started = time.monotonic()
        doc_line = document_line(
            doc.bibliographic.publication_number if doc.bibliographic else None, doc.title
        )

        for claim in doc.claims:
            source = Source(
                text=claim.text, subject=f"claim:{claim.number}", subject_label=f"权利要求{claim.number}",
                claim_number=claim.number, pages=claim.pages,
            )
            message = claim_message(doc_line, claim.number, claim.category, claim.text)
            self._run(Unit(source.subject, source.subject_label, message, [source], True), builder, result, stats)

        names = list(dict.fromkeys(
            f.name for f in builder.facts if f.kind is FactKind.COMPONENT and f.claim_number is not None
        ))
        paragraphs = {p.id: p for p in doc.paragraphs}
        for emb in doc.embodiments:
            if emb.kind is not EmbodimentKind.EXAMPLE:
                continue
            members = [paragraphs[pid] for pid in emb.paragraph_ids if pid in paragraphs and paragraphs[pid].text]
            if not members:
                continue
            sources = [
                Source(text=p.text, subject=emb.id, subject_label=emb.label, paragraph_ids=[p.id],
                       pages=p.pages, step=p.step)
                for p in members
            ]
            message = embodiment_message(doc_line, emb.id, emb.label, [(p.id, p.text) for p in members], names)
            self._run(Unit(emb.id, emb.label, message, sources, False), builder, result, stats)

        if self.include_tables:
            labels = {e.label: e.id for e in doc.embodiments}
            for table in doc.tables:
                table_results(builder, table, labels)

        stats["seconds"] = round(time.monotonic() - started, 1)
        stats.pop("failures_in_a_row")
        result.facts = builder.facts
        result.issues = builder.rejected + result.issues
        result.stats = stats
        logger.info(
            "document %s: LLM extraction %d facts, %d rejected, %d calls, %ss",
            doc.doc_id, len(result.facts), len(result.rejected), stats["llm_calls"], stats["seconds"],
        )
        return result

    def _run(self, unit: Unit, builder: FactBuilder, result: ExtractionResult, stats: dict) -> None:
        if stats["failures_in_a_row"] >= self.max_failures:
            stats["skipped_units"] += 1
            self._report(f"{unit.label}：跳过（大模型已连续失败 {self.max_failures} 次）")
            return
        started = time.monotonic()
        items = self._ask(unit, result, stats)
        if items is None:
            self._report(f"{unit.label}：失败，{time.monotonic() - started:.0f}s（{result.issues[-1].message.split('：', 1)[-1]}）")
            if stats["failures_in_a_row"] >= self.max_failures:
                result.issues.append(ExtractionIssue(
                    "EXTRACT_LLM_STOPPED",
                    f"大模型连续 {self.max_failures} 次调用失败，本文档其余部分不再调用大模型",
                ))
            return
        facts_before, rejected_before = len(builder.facts), len(result.rejected)
        seen: dict[tuple, tuple] = {}
        for item in items:
            checked = verify(item, unit) if isinstance(item, dict) else "不是对象"
            if isinstance(checked, str):
                result.rejected.append(Rejection(unit.label, checked, item if isinstance(item, dict) else {"raw": item}))
                continue
            source, fields = checked
            key = (unit.subject, fields["step"], fields["kind"], fields["name"], fields["qualifier"])
            value = (fields.get("low"), fields.get("high"), tuple(fields.get("options") or ()))
            if key in seen:
                if seen[key] != value:
                    result.rejected.append(Rejection(unit.label, "与同一位置的另一条事实重复但数值不同", item))
                continue
            seen[key] = value
            kind = fields.pop("kind")
            name = fields.pop("name")
            quote = fields.pop("evidence")
            builder.add(source, kind, name, quote, **fields)
        new = result.rejected[rejected_before:]
        self._report(
            f"{unit.label}：{len(builder.facts) - facts_before} 条，丢弃 {len(new)} 条，{time.monotonic() - started:.0f}s"
        )
        if new:
            counts = Counter(r.reason for r in new)
            summary = "，".join(f"{reason} {n}" for reason, n in counts.most_common())
            page = unit.sources[0].pages[0] if unit.sources[0].pages else None
            result.issues.append(ExtractionIssue(
                "EXTRACT_LLM_REJECTED", f"{unit.label}：大模型给出的 {len(new)} 条事实未通过核对，已丢弃（{summary}）", page=page,
            ))

    def _ask(self, unit: Unit, result: ExtractionResult, stats: dict) -> list | None:
        problem = ""
        for attempt in range(self.retries + 1):
            messages = [ChatMessage.system(SYSTEM_PROMPT), ChatMessage.user(unit.message)]
            if attempt:
                messages.append(ChatMessage.user(RETRY_NOTE.format(problem=problem)))
            try:
                response = self.model.chat(messages, tools=[RECORD_FACTS], tool_choice=TOOL_NAME)
            except LlmError as exc:
                stats["failed_units"] += 1
                stats["failures_in_a_row"] += 1
                result.issues.append(ExtractionIssue("EXTRACT_LLM_FAILED", f"{unit.label}：大模型调用失败（{exc.message}）"))
                return None
            stats["failures_in_a_row"] = 0
            stats["llm_calls"] += 1
            usage: Usage = response.usage
            stats["prompt_tokens"] += usage.prompt_tokens
            stats["completion_tokens"] += usage.completion_tokens
            items, problem = parse_response(response.message)
            if items is not None:
                return items
            if response.finish_reason == "length":
                problem += "；输出达到 max_tokens 上限"
            logger.info("%s: unparseable LLM output (%s), attempt %d", unit.label, problem, attempt + 1)
        stats["failed_units"] += 1
        result.issues.append(ExtractionIssue("EXTRACT_LLM_FAILED", f"{unit.label}：大模型的输出无法解析（{problem}）"))
        return None

    def _report(self, line: str) -> None:
        logger.info("LLM extraction %s", line)
        if self.progress is not None:
            self.progress(line)


# --- reading the model's answer ------------------------------------------------------------


def parse_response(message: ChatMessage) -> tuple[list | None, str]:
    """The list of proposed facts, or (None, why not). Accepts a real tool call, or
    the JSON written into the text (```json fences, <tool_call> tags, bare JSON)."""
    problem = "没有调用工具，也没有 JSON"
    for call in message.tool_calls:
        try:
            facts = _facts_from(call.parsed_arguments())
        except ValueError as exc:
            problem = f"工具参数不是合法 JSON：{exc}"
            continue
        if facts is not None:
            return facts, ""
        problem = "工具参数里没有 facts 列表"
    text = _THINK.sub("", message.content or "").strip()
    candidates = _TOOL_TAG.findall(text) + _FENCE.findall(text)
    first = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if first >= 0:
        candidates.append(text[first:max(text.rfind("}"), text.rfind("]")) + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            problem = "正文中的 JSON 不合法"
            continue
        facts = _facts_from(data)
        if facts is not None:
            return facts, ""
        problem = "JSON 里没有 facts 列表"
    return None, problem


def _facts_from(data: Any) -> list | None:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("facts"), list):
            return data["facts"]
        if isinstance(data.get("arguments"), dict):  # {"name": "record_facts", "arguments": {...}}
            return _facts_from(data["arguments"])
    return None


# --- verifying one proposed fact ------------------------------------------------------------


def verify(item: dict, unit: Unit) -> tuple[Source, dict] | str:
    """(source, Fact fields) for a fact that passes every check, else the reason."""
    kind = _KINDS.get(str(item.get("kind", "")).strip())
    if kind is None:
        return "类型无效"
    name = _text(item.get("name"))
    if not name or len(name) > _MAX_NAME:
        return "名称为空或过长"
    quote = _PARAGRAPH_ID.sub("", _text(item.get("evidence")))   # "[0028] S3、…" -> "S3、…"
    if not quote:
        return "缺少证据"

    found = _locate(quote, unit, _text(item.get("paragraph")))
    if found is None:
        return "证据不在原文中"
    source, quote = found
    fields: dict[str, Any] = {"kind": kind, "evidence": quote, "qualifier": _text(item.get("qualifier")) or None}

    if kind is FactKind.OPTION:
        options = [normalize_chemical_name(o) for o in item.get("options") or [] if _text(o)]
        if len(options) < 2:
            return "可选材料少于两个"
        if not all(ev.contains(quote, o) for o in options):
            return "可选材料不在证据中"
        fields.update(options=options, text_value="、".join(options))
    else:
        low, high = _number(item.get("low")), _number(item.get("high"))
        if low is None and high is None:
            return "缺少数值"
        low, high = (low if low is not None else high), (high if high is not None else low)
        low, high = min(low, high), max(low, high)
        if not {low, high} <= ev.numbers(quote):
            return "数值不在证据中"
        unit_text = _text(item.get("unit"))
        is_ph = name.casefold() in ("ph", "ph值")
        if is_ph:
            name = "pH"
        fields.update(
            low=low, high=high, value=low if low == high else None,
            unit=None if is_ph or not unit_text else canonical_unit(unicodedata.normalize("NFKC", unit_text)),
        )

    if kind in (FactKind.COMPONENT, FactKind.OPTION):
        if not ev.contains(quote, name):
            alias = next((raw for raw, canon in NAME_ALIASES.items() if canon == name and ev.contains(quote, raw)), None)
            if alias is None:
                return "名称不在证据中"
            fields["name_raw"] = alias
    fields["name"] = name
    fields["step"] = _step(item.get("step"), source, quote, unit.claim_mode)
    return source, fields


def _locate(quote: str, unit: Unit, paragraph: str) -> tuple[Source, str] | None:
    sources = sorted(unit.sources, key=lambda s: s.paragraph_ids != [paragraph.lstrip("[").rstrip("]")])
    for source in sources:
        original = ev.locate(source.text, quote)
        if original is not None:
            return source, original
    return None


def _step(proposed: Any, source: Source, quote: str, claim_mode: bool) -> str | None:
    if not claim_mode and source.step:            # the paragraph's own step marker
        return source.step
    if claim_mode:
        start = source.text.find(quote)
        line = source.text[source.text.rfind("\n", 0, start) + 1:].split("\n", 1)[0]
        marked = step_id(line)
        if marked:                                  # a step line of a method claim
            return marked
    match = _STEP.match(_text(proposed).upper())
    if not match:
        return None
    step = f"S{int(match.group(1))}"
    return step if step in ev.step_references(source.text) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None

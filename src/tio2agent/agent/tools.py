"""The Agent's tools: read-only access to the stores plus a calculator.

    list_documents     which documents can be used
    search_documents   hybrid retrieval over text chunks           -> T ids
    query_facts        formulations, process parameters, results   -> F ids
    get_formulation    all components of one example / claim + sum -> F ids, C id
    calculate          arithmetic over source values               -> C id

Arguments are pydantic models; their JSON Schema (inlined, no $ref, so smaller
models and stricter servers accept it) is what the LLM sees. Each tool returns
compact JSON and registers what it returned in the run's SourceLedger.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from tio2agent.agent.calculator import CalculationError, evaluate
from tio2agent.agent.catalog import DocumentCatalog
from tio2agent.agent.ledger import SourceLedger
from tio2agent.domain import ChunkKind, Fact, FactKind, group_formulations
from tio2agent.errors import AppError
from tio2agent.llm import ToolCall, ToolSpec
from tio2agent.retrieval import SearchService
from tio2agent.storage import FactRepository

TEXT_EXCERPT_CHARS = 900


@dataclass
class ToolContext:
    ledger: SourceLedger
    doc_ids: list[str] | None              # documents this question may use (None: all)
    labels: dict[str, str] = field(default_factory=dict)  # doc_id -> display label

    def allowed(self, doc_id: str | None) -> list[str] | None:
        """Combine a tool's own doc_id argument with the request's restriction."""
        if doc_id is None:
            return self.doc_ids
        if self.doc_ids is not None and doc_id not in self.doc_ids:
            return []
        return [doc_id]


@dataclass
class ToolResult:
    content: dict[str, Any]   # sent to the model as JSON
    summary: str              # one line for the trace
    sources: list[str] = field(default_factory=list)


@dataclass
class ToolOutcome:
    ok: bool
    text: str                 # tool message content
    summary: str
    arguments: dict[str, Any] | str
    sources: list[str] = field(default_factory=list)


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, json_schema(self.Args))

    @abstractmethod
    def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...


# --- tools -----------------------------------------------------------------------------------


class ListDocumentsArgs(BaseModel):
    pass


class ListDocumentsTool(Tool):
    name = "list_documents"
    description = "列出可用的文档：doc_id、编号（申请公布号）、标题、切块数和事实数。问题涉及“哪篇专利/哪些文档”时使用。"
    Args = ListDocumentsArgs

    def __init__(self, catalog: DocumentCatalog) -> None:
        self.catalog = catalog

    def run(self, args: ListDocumentsArgs, ctx: ToolContext) -> ToolResult:
        entries = self.catalog.entries(ctx.doc_ids)
        docs = [
            {"doc_id": e.doc_id, "label": e.label, "title": e.title or e.filename, "chunks": e.chunks, "facts": e.facts}
            for e in entries
        ]
        return ToolResult({"documents": docs}, f"{len(docs)} 篇文档")


class SearchDocumentsArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="检索词或问题，中文")
    top_k: int = Field(5, ge=1, le=10, description="返回条数")
    kinds: list[Literal["bibliographic", "abstract", "claim", "paragraph", "embodiment", "table"]] | None = Field(
        None, description="只检索这些类型；不填则全部"
    )
    doc_id: str | None = Field(None, description="只检索这一篇文档")


class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = (
        "在文档原文中检索（关键词 + 语义混合）。用于事实库里没有的内容：背景、原理、作用、"
        "申请人等著录项、对比例、检测方法。返回片段 id（T1…）、出处和原文。"
    )
    Args = SearchDocumentsArgs

    def __init__(self, search: SearchService) -> None:
        self.search = search

    def run(self, args: SearchDocumentsArgs, ctx: ToolContext) -> ToolResult:
        doc_ids = ctx.allowed(args.doc_id)
        if doc_ids == []:
            return ToolResult({"results": [], "note": "该文档不在本次问答的范围内"}, "文档不在范围内")
        result = self.search.search(
            args.query, top_k=args.top_k, doc_ids=doc_ids,
            kinds=[ChunkKind(k) for k in args.kinds] if args.kinds else None,
        )
        items, ids = [], []
        for hit in result.hits:
            sid = ctx.ledger.add_chunk(hit.chunk)
            ids.append(sid)
            text = hit.chunk.text
            items.append({
                "id": sid,
                "doc": hit.chunk.doc_label,
                "citation": hit.chunk.citation,
                "context": hit.chunk.context,
                "text": text if len(text) <= TEXT_EXCERPT_CHARS else text[:TEXT_EXCERPT_CHARS] + "…",
            })
        content: dict[str, Any] = {"results": items}
        if result.warnings:
            content["warnings"] = result.warnings
        return ToolResult(content, f"检索“{args.query}”：{len(items)} 条", ids)


class QueryFactsArgs(BaseModel):
    name: str | None = Field(None, description="名称的一部分，如 二氧化钛、温度、pH、色差")
    subject: str | None = Field(None, description="实施例或权利要求：实施例1 / E1 / 权利要求1 / claim:1")
    kind: Literal["component", "parameter", "option", "result"] | None = Field(
        None, description="component 配方组分，parameter 工艺参数，option 可选材料，result 测试结果"
    )
    step: str | None = Field(None, description="工艺步骤，如 S1")
    doc_id: str | None = Field(None, description="只查这一篇文档")
    limit: int = Field(30, ge=1, le=60)


class QueryFactsTool(Tool):
    name = "query_facts"
    description = (
        "查询事实库：从专利的权利要求、实施例和表格中抽取的配方组分、工艺参数、可选材料和测试结果，"
        "每条带数值、单位和出处。问具体数值时优先用它。返回事实 id（F1…）。"
    )
    Args = QueryFactsArgs

    def __init__(self, facts: FactRepository) -> None:
        self.facts = facts

    def run(self, args: QueryFactsArgs, ctx: ToolContext) -> ToolResult:
        doc_ids = ctx.allowed(args.doc_id)
        found = self.facts.query(
            doc_ids=doc_ids, kind=FactKind(args.kind) if args.kind else None,
            subject=_subject(args.subject), name=args.name, step=args.step, limit=args.limit + 1,
        )
        truncated = len(found) > args.limit
        found = found[: args.limit]
        items = [_fact_item(f, ctx) for f in found]
        content: dict[str, Any] = {"facts": items}
        if truncated:
            content["truncated"] = f"结果超过 {args.limit} 条，请加 subject / kind / step 缩小范围"
        if not items:
            content["hint"] = "没有匹配的事实。试试更短的名称、去掉 step，或用 search_documents 检索原文。"
        return ToolResult(content, f"事实 {len(items)} 条" + ("（已截断）" if truncated else ""), [i["id"] for i in items])


class GetFormulationArgs(BaseModel):
    subject: str = Field(description="实施例或权利要求：实施例1 / E1 / 权利要求2")
    doc_id: str | None = Field(None, description="只查这一篇文档")


class GetFormulationTool(Tool):
    name = "get_formulation"
    description = "取一个实施例（或权利要求）的完整配方：全部组分及用量，单值时附合计（计算 id C…）。"
    Args = GetFormulationArgs

    def __init__(self, facts: FactRepository) -> None:
        self.facts = facts

    def run(self, args: GetFormulationArgs, ctx: ToolContext) -> ToolResult:
        found = self.facts.query(
            doc_ids=ctx.allowed(args.doc_id), kind=FactKind.COMPONENT, subject=_subject(args.subject)
        )
        out, ids = [], []
        for formulation in group_formulations(found):
            components = [_fact_item(c, ctx) for c in formulation.components]
            ids += [c["id"] for c in components]
            item: dict[str, Any] = {
                "doc": ctx.labels.get(formulation.doc_id, formulation.doc_id),
                "subject": formulation.subject_label,
                "unit": formulation.unit,
                "components": components,
            }
            if formulation.total is not None:
                names = {f"x{i + 1}": c["id"] for i, c in enumerate(components)}
                total_id = ctx.ledger.add_calculation(
                    "+".join(names), names, formulation.total, formulation.unit
                )
                ids.append(total_id)
                item["total"] = {"id": total_id, "value": formulation.total, "unit": formulation.unit}
            out.append(item)
        content: dict[str, Any] = {"formulations": out}
        if not out:
            content["hint"] = "没有找到配方。用 query_facts(kind='component') 看看有哪些 subject。"
        return ToolResult(content, f"配方 {len(out)} 个", ids)


class CalculateArgs(BaseModel):
    expression: str = Field(description="算式，变量名用字母，如 a / b * 100；支持 + - * / ** % 和 min max abs round sqrt avg sum")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="变量 -> 来源 id，如 {\"a\": \"F3\", \"b\": \"C1\"}；范围用 F2.low / F2.high",
    )
    unit: str | None = Field(None, description="结果的单位，如 %、份、min")


class CalculateTool(Tool):
    name = "calculate"
    description = (
        "计算工艺指标（比例、占比、差值、平均值、换算等）。数值必须来自已返回的来源 id，"
        "不要把文档里的数字直接写进算式；常数（如 100）可以直接写。返回计算 id（C…）。"
    )
    Args = CalculateArgs

    def run(self, args: CalculateArgs, ctx: ToolContext) -> ToolResult:
        try:
            values = {name: ctx.ledger.number(ref) for name, ref in args.variables.items()}
            result = evaluate(args.expression, values)
        except (CalculationError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        cid = ctx.ledger.add_calculation(args.expression, args.variables, result, args.unit)
        shown = round(result, 6)
        content = {
            "id": cid,
            "value": shown,
            "unit": args.unit,
            "expression": args.expression,
            "inputs": {k: {"source": ref, "value": values[k]} for k, ref in args.variables.items()},
        }
        return ToolResult(content, f"{args.expression} = {shown:g}", [cid])


# --- registry -----------------------------------------------------------------------------


class ToolError(AppError):
    """A tool could not do what was asked; the message goes back to the model."""

    code = "TOOL_ERROR"
    message = "工具调用失败"


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        """Run one call. Every failure becomes an error message for the model."""
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(call.arguments, f"没有名为 {call.name} 的工具，可用：{', '.join(self._tools)}")
        try:
            raw = call.parsed_arguments()
        except ValueError as exc:
            return _error(call.arguments, f"参数不是合法的 JSON 对象：{exc}")
        try:
            args = tool.Args.model_validate(raw)
        except ValidationError as exc:
            return _error(raw, "参数不合法：" + "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '(参数)'}: {e['msg']}" for e in exc.errors()
            ))
        try:
            result = tool.run(args, ctx)
        except AppError as exc:
            return _error(raw, exc.message)
        return ToolOutcome(True, json.dumps(result.content, ensure_ascii=False), result.summary, raw, result.sources)


def default_tools(*, catalog: DocumentCatalog, search: SearchService, facts: FactRepository) -> list[Tool]:
    return [
        QueryFactsTool(facts),
        GetFormulationTool(facts),
        SearchDocumentsTool(search),
        CalculateTool(),
        ListDocumentsTool(catalog),
    ]


# --- helpers ------------------------------------------------------------------------------


def _error(arguments: dict[str, Any] | str, message: str) -> ToolOutcome:
    return ToolOutcome(False, json.dumps({"error": message}, ensure_ascii=False), message, arguments)


def _subject(subject: str | None) -> str | None:
    """Accept "实施例1", "E1", "权利要求1", "claim:1", "claim 1"."""
    if not subject:
        return None
    s = subject.strip().replace(" ", "")
    if s.lower().startswith("claim") and not s.lower().startswith("claim:"):
        s = "claim:" + s[5:]
    return s


def _fact_item(f: Fact, ctx: ToolContext) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": ctx.ledger.add_fact(f, ctx.labels.get(f.doc_id)),
        "doc": ctx.labels.get(f.doc_id, f.doc_id),
        "subject": f.subject_label,
        "name": f.name,
    }
    if f.step:
        item["step"] = f.step
    if f.qualifier:
        item["qualifier"] = f.qualifier
    if f.kind is FactKind.OPTION:
        item["options"] = f.options
    elif f.value is not None:
        item["value"] = f.value
    elif f.low is not None:
        item["range"] = [f.low, f.high]
    elif f.text_value:
        item["text"] = f.text_value
    if f.unit:
        item["unit"] = f.unit
    item["citation"] = f.citation
    item["evidence"] = f.evidence
    failed = [c.message for c in f.checks if not c.ok]
    if failed:
        item["check_failed"] = failed
    return item


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema without titles and null defaults, $defs/$ref inlined and
    Optional[X] shown as X (optional = not in "required")."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return clean(defs[node["$ref"].split("/")[-1]])
            if "anyOf" in node:
                options = [o for o in node["anyOf"] if o != {"type": "null"}]
                if len(options) == 1:
                    merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **options[0]}
                    return clean(merged)
            return {
                k: clean(v) for k, v in node.items()
                if k != "title" and not (k == "default" and v is None)
            }
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    out = clean(schema)
    out.setdefault("properties", {})
    out["type"] = "object"
    return out

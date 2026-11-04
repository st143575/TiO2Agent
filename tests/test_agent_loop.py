"""AgentRunner mechanics with a scripted model and a tiny in-memory tool."""

import json

from typing import ClassVar

from pydantic import BaseModel, Field

from llm_fakes import ScriptedChatModel, call, raw_call, reply, text, tool_results
from tio2agent.agent import AgentRunner, SourceLedger, Tool, ToolContext, ToolRegistry, ToolResult
from tio2agent.agent.prompts import LAST_STEP, NUDGE_FINAL_ANSWER
from tio2agent.agent.tools import CalculateTool, ToolError
from tio2agent.domain import AnswerStatus, Fact, FactKind


class LookupArgs(BaseModel):
    name: str = Field(min_length=1)


class LookupTool(Tool):
    """Returns one fact per known name; 'boom' raises a ToolError; 'big' returns a lot."""

    name = "lookup"
    description = "查一个组分"
    Args = LookupArgs
    VALUES: ClassVar[dict[str, float]] = {"二氧化钛": 55.0, "去离子水": 45.0}

    def run(self, args: LookupArgs, ctx: ToolContext) -> ToolResult:
        if args.name == "boom":
            raise ToolError("数据库暂时不可用")
        if args.name == "big":
            return ToolResult({"rows": ["x" * 100] * 100}, "big")
        value = self.VALUES[args.name]
        f = Fact(id=f"d-{args.name}", doc_id="d", kind=FactKind.COMPONENT, subject="E1", subject_label="实施例1",
                 name=args.name, value=value, low=value, high=value, unit="份", evidence=f"{value:g}份{args.name}",
                 paragraph_ids=["0026"], extractor="test")
        sid = ctx.ledger.add_fact(f, "CN1")
        return ToolResult({"id": sid, "value": value}, f"{args.name} = {value:g}", [sid])


def runner(model: ScriptedChatModel, **kw) -> AgentRunner:
    return AgentRunner(model, ToolRegistry([LookupTool(), CalculateTool()]), **kw)


def ctx() -> ToolContext:
    return ToolContext(ledger=SourceLedger(), doc_ids=None)


def test_tools_then_final_answer() -> None:
    model = ScriptedChatModel(
        reply(call("lookup", name="二氧化钛"), call("lookup", name="去离子水"), content="先查两个组分"),
        reply(call("calculate", expression="a / b", variables={"a": "F2", "b": "F1"})),
        reply(call("final_answer", answer="水与二氧化钛之比为 0.818[C1]",
                   values=[{"label": "比值", "value": 0.818, "source": "C1"}])),
    )
    run = runner(model).run("比值？", ctx(), system_prompt="sys", history=[("上一个问题", "上一个回答[F9]")])

    assert run.answer.status is AnswerStatus.ANSWERED
    assert run.answer.values[0].verified is True
    assert [c.id for c in run.answer.citations] == ["C1", "F2", "F1"]   # calculation + its inputs
    assert run.warnings == []
    assert [(s.index, s.tool, s.ok, s.sources) for s in run.steps] == [
        (1, "lookup", True, ["F1"]), (2, "lookup", True, ["F2"]), (3, "calculate", True, ["C1"]),
    ]
    assert run.steps[0].thought == "先查两个组分" and run.steps[1].thought is None
    assert (run.usage.llm_calls, run.usage.prompt_tokens, run.usage.completion_tokens) == (3, 300, 60)

    first = model.calls[0]
    assert [m.role for m in first.messages] == ["system", "user", "assistant", "user"]
    assert first.messages[2].content == "上一个回答"          # old citation ids stripped from history
    assert [t.name for t in first.tools] == ["lookup", "calculate", "final_answer"]
    assert first.tool_choice is None
    # every tool call got its tool message, in order
    third = model.calls[2].messages
    assert [m.role for m in third[-4:]] == ["tool", "tool", "assistant", "tool"]
    assert json.loads(third[-1].content)["value"] == 0.818182


def test_tool_errors_go_back_to_the_model() -> None:
    model = ScriptedChatModel(
        reply(
            call("lookup", name="boom"),
            call("lookup"),                          # missing argument
            raw_call("lookup", "{name: 二氧化钛"),     # invalid JSON
            call("nope"),
            call("calculate", expression="a+1", variables={"a": "F7"}),
        ),
        reply(call("final_answer", answer="查不到", status="insufficient_evidence")),
    )
    run = runner(model).run("q", ctx(), system_prompt="sys")
    assert run.answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert [s.ok for s in run.steps] == [False] * 5
    errors = [r["error"] for r in tool_results(model.calls[1].messages)]
    assert errors[0] == "数据库暂时不可用"
    assert errors[1].startswith("参数不合法：name: Field required")
    assert errors[2].startswith("参数不是合法的 JSON 对象")
    assert errors[3].startswith("没有名为 nope 的工具，可用：lookup, calculate")
    assert errors[4] == "来源 F7 不存在：只能引用本轮工具返回过的 id"
    assert run.steps[2].arguments == "{name: 二氧化钛"


def test_invalid_final_answer_is_retried() -> None:
    model = ScriptedChatModel(
        reply(call("final_answer", values=[{"label": "x"}])),            # no answer, value without source
        reply(call("final_answer", answer="55 份", status="answered")),
    )
    run = runner(model).run("q", ctx(), system_prompt="sys")
    assert run.answer.answer == "55 份"
    assert run.steps[0].tool == "final_answer" and run.steps[0].ok is False
    error = tool_results(model.calls[1].messages)[0]["error"]
    assert "answer: Field required" in error and "values.0.source: Field required" in error


def test_plain_text_is_nudged_then_accepted() -> None:
    model = ScriptedChatModel(text("二氧化钛 55 份"), text("二氧化钛 55 份。"))
    run = runner(model).run("q", ctx(), system_prompt="sys")
    assert model.calls[1].messages[-1].content == NUDGE_FINAL_ANSWER
    assert run.answer.answer == "二氧化钛 55 份。"
    assert run.warnings[0] == "模型没有调用 final_answer，直接使用了它的文字回答"


def test_last_step_forces_final_answer_and_gives_up_cleanly() -> None:
    model = ScriptedChatModel(
        reply(call("lookup", name="二氧化钛")),
        reply(call("lookup", name="去离子水")),
        reply(call("lookup", name="二氧化钛")),   # ignores the forced tool_choice
    )
    run = runner(model, max_steps=3).run("q", ctx(), system_prompt="sys")
    assert [c.tool_choice for c in model.calls] == [None, None, "final_answer"]
    assert model.calls[2].messages[-1].content == LAST_STEP
    assert run.answer.status is AnswerStatus.INCOMPLETE
    assert [c.id for c in run.answer.citations] == ["F1", "F2"]
    assert run.warnings == ["在 3 轮内没有得到回答"]


def test_long_tool_results_are_capped() -> None:
    model = ScriptedChatModel(reply(call("lookup", name="big")), reply(call("final_answer", answer="ok")))
    runner(model, tool_result_chars=500).run("q", ctx(), system_prompt="sys")
    content = model.calls[1].messages[-1].content
    assert len(content) < 600 and content.endswith("（结果过长已截断，请缩小查询范围）")

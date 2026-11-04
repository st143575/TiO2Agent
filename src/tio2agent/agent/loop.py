"""The Agent loop: plan with the LLM, call tools, stop at ``final_answer``.

One iteration = one LLM call. The model sees the system prompt, earlier turns and
every tool result so far, and either calls tools (results are appended and the loop
continues) or calls ``final_answer`` (checked and returned). Guard rails:

    - at most ``max_steps`` LLM calls; the last one forces ``final_answer``;
    - a reply without any tool call is nudged once towards ``final_answer``, and
      accepted as plain text the second time;
    - tool errors and invalid arguments go back to the model as tool results, so
      it can correct itself instead of the request failing;
    - each tool result is capped at ``tool_result_chars``.
"""

import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from tio2agent.agent.final_answer import FINAL_ANSWER, FINAL_ANSWER_SPEC, FinalAnswerArgs, build_answer
from tio2agent.agent.prompts import LAST_STEP, NUDGE_FINAL_ANSWER, history_messages
from tio2agent.agent.tools import ToolContext, ToolRegistry
from tio2agent.domain import AgentAnswer, AgentStep, AgentUsage, AnswerStatus, SourceKind
from tio2agent.llm import ChatMessage, ChatModel, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    answer: AgentAnswer
    steps: list[AgentStep]
    usage: AgentUsage
    warnings: list[str] = field(default_factory=list)
    model: str = ""


class AgentRunner:
    def __init__(
        self,
        model: ChatModel,
        registry: ToolRegistry,
        *,
        max_steps: int = 8,
        tool_result_chars: int = 6000,
    ) -> None:
        self.model = model
        self.registry = registry
        self.max_steps = max_steps
        self.tool_result_chars = tool_result_chars

    def run(
        self,
        question: str,
        ctx: ToolContext,
        *,
        system_prompt: str,
        history: list[tuple[str, str]] | None = None,
        max_steps: int | None = None,
    ) -> AgentRun:
        max_steps = max_steps or self.max_steps
        messages = [ChatMessage.system(system_prompt), *history_messages(history or []), ChatMessage.user(question)]
        specs = [*self.registry.specs(), FINAL_ANSWER_SPEC]
        steps: list[AgentStep] = []
        warnings: list[str] = []
        usage = AgentUsage()
        model_name = self.model.model
        nudged = False

        for turn in range(max_steps):
            last = turn == max_steps - 1
            if last and turn > 0:
                messages.append(ChatMessage.user(LAST_STEP))
            response = self.model.chat(messages, tools=specs, tool_choice=FINAL_ANSWER if last else None)
            usage.llm_calls += 1
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens
            model_name = response.model or model_name
            message = response.message
            messages.append(message)

            if not message.tool_calls:
                text = (message.content or "").strip()
                if text and (nudged or last):
                    answer, problems = build_answer(FinalAnswerArgs(answer=text), ctx.ledger)
                    warnings += ["模型没有调用 final_answer，直接使用了它的文字回答", *problems]
                    return AgentRun(answer, steps, usage, warnings, model_name)
                nudged = True
                messages.append(ChatMessage.user(NUDGE_FINAL_ANSWER))
                continue

            thought = (message.content or "").strip()[:500] or None
            final_calls = [c for c in message.tool_calls if c.name == FINAL_ANSWER]
            for call in message.tool_calls:
                if call.name != FINAL_ANSWER:
                    steps.append(self._run_tool(call, ctx, messages, len(steps) + 1, thought))
                    thought = None
            if not final_calls:
                continue

            final, error = self._parse_final(final_calls[0])
            if final is not None:
                answer, problems = build_answer(final, ctx.ledger)
                warnings += problems
                logger.info("agent answered in %d LLM calls, %d tool calls", usage.llm_calls, len(steps))
                return AgentRun(answer, steps, usage, warnings, model_name)
            # invalid final answer: tell the model what is wrong and let it try again
            for i, call in enumerate(final_calls):
                content = error if i == 0 else "重复调用，已忽略"
                messages.append(ChatMessage.tool(call.id, json.dumps({"error": content}, ensure_ascii=False)))
            steps.append(AgentStep(
                index=len(steps) + 1, tool=FINAL_ANSWER, arguments=final_calls[0].arguments,
                ok=False, summary=error, thought=thought,
            ))

        warnings.append(f"在 {max_steps} 轮内没有得到回答")
        consulted = [r.to_citation() for r in ctx.ledger.records() if r.kind is not SourceKind.CALCULATION]
        answer = AgentAnswer(
            status=AnswerStatus.INCOMPLETE,
            answer="未能在限定的步数内完成回答。引用中列出了已经查到的资料。",
            citations=consulted,
        )
        return AgentRun(answer, steps, usage, warnings, model_name)

    def _run_tool(
        self, call: ToolCall, ctx: ToolContext, messages: list[ChatMessage], index: int, thought: str | None
    ) -> AgentStep:
        start = time.perf_counter()
        outcome = self.registry.execute(call, ctx)
        elapsed = int((time.perf_counter() - start) * 1000)
        text = outcome.text
        if len(text) > self.tool_result_chars:
            text = text[: self.tool_result_chars] + "…（结果过长已截断，请缩小查询范围）"
        messages.append(ChatMessage.tool(call.id, text))
        logger.debug("tool %s(%s) -> %s", call.name, call.arguments, outcome.summary)
        return AgentStep(
            index=index, tool=call.name, arguments=outcome.arguments, ok=outcome.ok,
            summary=outcome.summary, sources=outcome.sources, thought=thought, duration_ms=elapsed,
        )

    @staticmethod
    def _parse_final(call: ToolCall) -> tuple[FinalAnswerArgs | None, str]:
        try:
            return FinalAnswerArgs.model_validate(call.parsed_arguments()), ""
        except ValueError as exc:  # also pydantic.ValidationError
            if isinstance(exc, ValidationError):
                detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            else:
                detail = str(exc)
            return None, f"final_answer 参数不合法：{detail}。请修正后重新调用"

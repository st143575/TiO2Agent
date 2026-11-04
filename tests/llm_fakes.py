"""A scripted ChatModel: replays responses so the Agent can be tested without an LLM.

Each script entry is a ChatResponse or a function(messages) -> ChatResponse; the
function form lets a step use ids that earlier tool results returned:

    model = ScriptedChatModel(
        reply(call("query_facts", subject="实施例1", name="二氧化钛")),
        lambda msgs: reply(call("final_answer", answer=f"55 份[{fact_id(msgs, '二氧化钛')}]")),
    )
"""

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tio2agent.llm import ChatMessage, ChatResponse, LlmError, ToolCall, ToolSpec, Usage

Step = ChatResponse | Exception | Callable[[list[ChatMessage]], ChatResponse]
_ids = itertools.count(1)


@dataclass
class RecordedCall:
    messages: list[ChatMessage]
    tools: list[ToolSpec] | None
    tool_choice: str | None


class ScriptedChatModel:
    def __init__(self, *script: Step, model: str = "scripted") -> None:
        self.model = model
        self.script: list[Step] = list(script)
        self.calls: list[RecordedCall] = []
        self.closed = False

    def load(self, *script: Step) -> "ScriptedChatModel":
        self.script = list(script)
        self.calls = []
        return self

    def chat(self, messages, *, tools=None, tool_choice=None) -> ChatResponse:
        self.calls.append(RecordedCall(list(messages), tools, tool_choice))
        if not self.script:
            raise AssertionError("ScriptedChatModel: script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step

    def close(self) -> None:
        self.closed = True


def call(tool: str, /, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=tool, arguments=json.dumps(arguments, ensure_ascii=False))


def raw_call(name: str, arguments: str) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)


def reply(*calls: ToolCall, content: str | None = None, usage: tuple[int, int] = (100, 20)) -> ChatResponse:
    return ChatResponse(ChatMessage.assistant(content, list(calls)), "tool_calls" if calls else "stop", Usage(*usage))


def text(content: str) -> ChatResponse:
    return reply(content=content)


def fail(message: str = "boom") -> LlmError:
    return LlmError(message)


# --- reading tool results back out of the conversation ---------------------------------------


def tool_results(messages: list[ChatMessage]) -> list[dict]:
    return [json.loads(m.content) for m in messages if m.role == "tool" and m.content and m.content.startswith("{")]


def last_result(messages: list[ChatMessage]) -> dict:
    return tool_results(messages)[-1]


def facts_in(messages: list[ChatMessage]) -> list[dict]:
    out = []
    for r in tool_results(messages):
        out += r.get("facts", [])
        for f in r.get("formulations", []):
            out += f["components"]
    return out


def fact_id(messages: list[ChatMessage], name: str, subject: str | None = None) -> str:
    for f in facts_in(messages):
        if f["name"] == name and (subject is None or f["subject"] == subject):
            return f["id"]
    raise AssertionError(f"no fact {name!r} ({subject}) in tool results")

"""Chat messages, tool specs and the ChatModel protocol.

The shapes follow the OpenAI chat-completions format, which almost every serving
stack speaks (vLLM, SGLang, Ollama, LM Studio, DeepSeek, Qwen/DashScope, OpenAI…), so
switching models is a configuration change. The Agent only depends on ``ChatModel``;
tests replace it with a scripted model.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tio2agent.errors import AppError

Role = Literal["system", "user", "assistant", "tool"]


class LlmError(AppError):
    code = "LLM_FAILED"
    message = "大模型服务调用失败"


class LlmNotConfiguredError(AppError):
    code = "LLM_NOT_CONFIGURED"
    message = "未配置大模型服务（TIO2_LLM_BACKEND=none），无法使用 Agent 问答"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, as the model wrote it

    def parsed_arguments(self) -> dict[str, Any]:
        """Raises ValueError if the model wrote invalid JSON or not an object."""
        data = json.loads(self.arguments or "{}")
        if not isinstance(data, dict):
            raise ValueError("参数必须是 JSON 对象")
        return data


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments object


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens)


@dataclass
class ChatMessage:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None                           # tool only

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: list[ToolCall] | None = None) -> "ChatMessage":
        return cls("assistant", content, tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "ChatMessage":
        return cls("tool", content, tool_call_id=tool_call_id)


@dataclass
class ChatResponse:
    message: ChatMessage
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class ChatModel(Protocol):
    model: str

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        """One completion. ``tool_choice``: None/"auto", "required", or a tool name
        to force that tool. Raises LlmError."""
        ...

    def close(self) -> None: ...

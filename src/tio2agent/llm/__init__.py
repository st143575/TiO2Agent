"""LLM access for the Agent: one protocol, one OpenAI-compatible implementation."""

from tio2agent.config import LlmBackend, Settings
from tio2agent.llm.base import (
    ChatMessage,
    ChatModel,
    ChatResponse,
    LlmError,
    LlmNotConfiguredError,
    ToolCall,
    ToolSpec,
    Usage,
)
from tio2agent.llm.openai_compatible import OpenAICompatibleChatModel


def build_chat_model(settings: Settings) -> ChatModel | None:
    if settings.llm_backend is LlmBackend.OPENAI_COMPATIBLE:
        return OpenAICompatibleChatModel(
            settings.llm_base_url,
            settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    return None


__all__ = [
    "ChatMessage",
    "ChatModel",
    "ChatResponse",
    "LlmError",
    "LlmNotConfiguredError",
    "OpenAICompatibleChatModel",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "build_chat_model",
]

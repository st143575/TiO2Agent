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


def build_extraction_model(settings: Settings) -> ChatModel | None:
    """
    The LLM used for fact extraction: the same service as the Agent's, with 
    its own model name (optional), output budget, timeout and reasoning setting. 
    A request that times out is not repeated (it would take as long again).
    """
    if settings.llm_backend is LlmBackend.OPENAI_COMPATIBLE:
        effort = settings.extract_llm_reasoning_effort
        return OpenAICompatibleChatModel(
            settings.llm_base_url,
            settings.extract_llm_model or settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.extract_llm_temperature,
            max_tokens=settings.extract_llm_max_tokens,
            timeout_s=settings.extract_llm_timeout_s,
            max_retries=0,
            extra_body={"reasoning_effort": effort} if effort else None,
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
    "build_extraction_model",
]

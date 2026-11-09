"""ChatModel over any OpenAI-compatible ``/v1/chat/completions`` endpoint with tools.

Tested against the request/response format only (tests use a mock transport); which
model to run is a deployment choice. It must support function/tool calling, e.g.
Qwen3 or Qwen2.5-Instruct on Ollama/vLLM, DeepSeek-V3, GLM-4, or hosted APIs.
"""

import json
import logging
import time
import uuid
from typing import Any

import httpx

from tio2agent.llm.base import ChatMessage, ChatResponse, LlmError, ToolCall, ToolSpec, Usage

logger = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleChatModel:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
        max_retries: int = 1,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.extra_body = extra_body or {}  # sent with every request, e.g. {"reasoning_effort": "none"}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_encode_message(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
            if tool_choice in (None, "auto", "required", "none"):
                body["tool_choice"] = tool_choice or "auto"
            else:  # force one tool
                body["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        return _decode_response(self._post(body), self.model)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = self._client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                if last:
                    raise LlmError(f"无法连接大模型服务：{exc}") from exc
                logger.warning("LLM request failed (%s), retrying", exc)
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code in _RETRY_STATUS and not last:
                logger.warning("LLM returned %s, retrying", response.status_code)
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code != 200:
                raise LlmError(
                    f"大模型服务返回 {response.status_code}：{response.text[:300]}",
                    details={"status": response.status_code},
                )
            try:
                return response.json()
            except ValueError as exc:
                raise LlmError("大模型服务的响应不是 JSON") from exc
        raise AssertionError("unreachable")


def _encode_message(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    return out


def _decode_response(data: dict[str, Any], default_model: str) -> ChatResponse:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError("大模型服务的响应缺少 choices[0].message") from exc
    calls = []
    for raw in msg.get("tool_calls") or []:
        fn = raw.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):  # some servers send an object instead of JSON text
            args = json.dumps(args or {}, ensure_ascii=False)
        calls.append(ToolCall(id=raw.get("id") or f"call_{uuid.uuid4().hex[:12]}", name=fn.get("name", ""), arguments=args))
    usage = data.get("usage") or {}
    return ChatResponse(
        message=ChatMessage.assistant(msg.get("content"), calls),
        finish_reason=choice.get("finish_reason"),
        usage=Usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)),
        model=data.get("model") or default_model,
    )

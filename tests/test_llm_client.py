"""OpenAICompatibleChatModel: request and response format, retries, errors."""

import json

import httpx
import pytest

from tio2agent.config import LlmBackend, Settings
from tio2agent.llm import (
    ChatMessage,
    LlmError,
    OpenAICompatibleChatModel,
    ToolCall,
    ToolSpec,
    build_chat_model,
)

TOOL = ToolSpec("query_facts", "查事实", {"type": "object", "properties": {"name": {"type": "string"}}})


def completion(message: dict, *, finish: str = "stop", usage: dict | None = None) -> dict:
    return {
        "id": "x", "model": "qwen3:8b", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 50, "completion_tokens": 7},
    }


class Server:
    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        self.requests.append({"body": json.loads(request.content), "headers": dict(request.headers)})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def model(server: Server, **kw) -> OpenAICompatibleChatModel:
    return OpenAICompatibleChatModel(
        "http://llm/v1/", "qwen3:8b", api_key="sk-test", transport=httpx.MockTransport(server), **kw
    )


def test_request_format_and_tool_call_parsing() -> None:
    server = Server(httpx.Response(200, json=completion(
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "query_facts", "arguments": "{\"name\": \"温度\"}"}},
            {"type": "function", "function": {"name": "calculate", "arguments": {"expression": "1+1"}}},
        ]},
        finish="tool_calls",
    )))
    history = [
        ChatMessage.system("sys"),
        ChatMessage.user("问题"),
        ChatMessage.assistant(None, [ToolCall("c0", "list_documents", "{}")]),
        ChatMessage.tool("c0", '{"documents": []}'),
    ]
    response = model(server, temperature=0.2, max_tokens=512).chat(history, tools=[TOOL])

    req = server.requests[0]
    assert req["headers"]["authorization"] == "Bearer sk-test"
    body = req["body"]
    assert body["model"] == "qwen3:8b" and body["temperature"] == 0.2 and body["max_tokens"] == 512
    assert body["tool_choice"] == "auto"
    assert body["tools"] == [{"type": "function", "function": {
        "name": "query_facts", "description": "查事实", "parameters": TOOL.parameters}}]
    assert body["messages"][2]["tool_calls"] == [
        {"id": "c0", "type": "function", "function": {"name": "list_documents", "arguments": "{}"}}
    ]
    assert body["messages"][3] == {"role": "tool", "content": '{"documents": []}', "tool_call_id": "c0"}

    calls = response.message.tool_calls
    assert (calls[0].id, calls[0].name) == ("c1", "query_facts")
    assert calls[0].parsed_arguments() == {"name": "温度"}
    assert calls[1].id.startswith("call_")                       # missing id is generated
    assert calls[1].parsed_arguments() == {"expression": "1+1"}  # object arguments are accepted
    assert (response.finish_reason, response.usage.prompt_tokens, response.model) == ("tool_calls", 50, "qwen3:8b")


def test_forced_tool_choice() -> None:
    server = Server(httpx.Response(200, json=completion({"role": "assistant", "content": "ok"})))
    model(server).chat([ChatMessage.user("q")], tools=[TOOL], tool_choice="final_answer")
    assert server.requests[0]["body"]["tool_choice"] == {"type": "function", "function": {"name": "final_answer"}}


def test_no_tools_no_tool_choice() -> None:
    server = Server(httpx.Response(200, json=completion({"role": "assistant", "content": "你好"})))
    response = model(server).chat([ChatMessage.user("q")])
    assert "tools" not in server.requests[0]["body"] and "tool_choice" not in server.requests[0]["body"]
    assert response.message.content == "你好" and response.message.tool_calls == []


def test_retries_once_on_overload_and_connection_errors() -> None:
    ok = httpx.Response(200, json=completion({"role": "assistant", "content": "ok"}))
    server = Server(httpx.Response(503, text="busy"), ok)
    assert model(server).chat([ChatMessage.user("q")]).message.content == "ok"
    server = Server(httpx.ConnectError("refused"), httpx.Response(200, json=completion({"role": "assistant", "content": "ok"})))
    assert model(server).chat([ChatMessage.user("q")]).message.content == "ok"
    assert len(server.requests) == 2


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([httpx.Response(400, text="tools not supported")], "返回 400"),
        ([httpx.Response(503), httpx.Response(503)], "返回 503"),
        ([httpx.ConnectError("refused"), httpx.ConnectError("refused")], "无法连接"),
        ([httpx.Response(200, text="<html>")], "不是 JSON"),
        ([httpx.Response(200, json={"choices": []})], "缺少 choices"),
    ],
)
def test_errors(responses, message: str) -> None:
    with pytest.raises(LlmError, match=message):
        model(Server(*responses)).chat([ChatMessage.user("q")])


def test_build_chat_model() -> None:
    assert build_chat_model(Settings(_env_file=None)) is None
    built = build_chat_model(Settings(_env_file=None, llm_backend=LlmBackend.OPENAI_COMPATIBLE, llm_model="m"))
    assert isinstance(built, OpenAICompatibleChatModel) and built.model == "m"
    built.close()

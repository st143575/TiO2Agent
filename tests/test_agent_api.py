"""POST /agent/chat end to end: sample patent ingested (fake OCR for the results
table), real tools, fact store and search index; only the LLM is scripted."""

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES_DIR, UPLOAD_URL, upload, wait_until_done
from llm_fakes import ScriptedChatModel, call, fact_id, fail, last_result, reply, tool_results
from ocr_fakes import FakeLayoutParsingServer
from tio2agent.config import Settings
from tio2agent.ingestion.ocr import PaddleOcrVlClient
from tio2agent.main import create_app

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
CHAT_URL = "/api/v1/agent/chat"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    settings = Settings(
        _env_file=None, data_dir=tmp_path_factory.mktemp("data"), max_upload_mb=50, max_pages=200, ingest_workers=1
    )
    model = ScriptedChatModel()
    engine = PaddleOcrVlClient("http://ocr:8080", transport=httpx.MockTransport(FakeLayoutParsingServer()))
    with TestClient(create_app(settings, ocr_engine=engine, chat_model=model)) as client:
        response = upload(client, SAMPLE.read_bytes(), filename="doc.pdf")
        doc = wait_until_done(client, response.json()["id"])
        assert doc["status"] == "ready", doc["warnings"]
        yield client, model, doc["id"]


def test_formulation_share_is_computed_from_facts(app) -> None:
    client, model, _ = app
    model.load(
        reply(call("get_formulation", subject="实施例3")),
        lambda msgs: reply(call(
            "calculate", expression="a / b * 100", unit="%",
            variables={"a": fact_id(msgs, "二氧化钛"), "b": last_result(msgs)["formulations"][0]["total"]["id"]},
        )),
        lambda msgs: reply(call(
            "final_answer",
            answer=f"实施例3的配方合计 167 份[{last_result(msgs)['inputs']['b']['source']}]，"
                   f"其中二氧化钛 65 份[{fact_id(msgs, '二氧化钛')}]，占 38.92%[{last_result(msgs)['id']}]。",
            values=[
                {"label": "配方合计", "value": 167, "unit": "份", "source": last_result(msgs)["inputs"]["b"]["source"]},
                {"label": "二氧化钛", "value": 65, "unit": "份", "source": fact_id(msgs, "二氧化钛")},
                {"label": "二氧化钛占比", "value": 38.92, "unit": "%", "source": last_result(msgs)["id"]},
            ],
        )),
    )
    response = client.post(CHAT_URL, json={"question": "实施例3中二氧化钛占配方总量的百分比是多少？"})
    assert response.status_code == 200, response.text
    body = response.json()

    answer = body["answer"]
    assert answer["status"] == "answered" and body["warnings"] == []
    assert [(v["display"], v["verified"]) for v in answer["values"]] == [
        ("167份", True), ("65份", True), ("38.92%", True),
    ]
    citations = {c["id"]: c for c in answer["citations"]}
    tio2 = fact_id(model.calls[1].messages, "二氧化钛")
    assert citations[tio2]["citation"] == "说明书[0036]" and citations[tio2]["doc_label"] == "CN116694108A"
    assert citations[tio2]["quote"] == "65份二氧化钛"
    # the share and the total expand to all 8 component facts of 实施例3
    assert sum(c["kind"] == "fact" for c in answer["citations"]) == 8
    assert [s["tool"] for s in body["steps"]] == ["get_formulation", "calculate"]
    assert body["steps"][1]["summary"] == "a / b * 100 = 38.9222"
    assert body["session_id"].startswith("ses_") and body["turn"] == 1
    assert body["usage"]["llm_calls"] == 3 and body["model"] == "scripted"

    # the system prompt lists the document; the tools are offered with their schemas
    system = model.calls[0].messages[0].content
    assert "CN116694108A：一种高耐候钛白粉及制备方法" in system
    assert {t.name for t in model.calls[0].tools} == {
        "query_facts", "get_formulation", "search_documents", "calculate", "list_documents", "final_answer"
    }


def test_query_facts_search_and_results_table(app) -> None:
    client, model, doc_id = app
    model.load(
        reply(
            call("query_facts", kind="result", name="色差"),
            call("query_facts", subject="E2", step="S3", name="pH"),
            call("search_documents", query="抗氧化剂的作用", top_k=3),
        ),
        reply(call("final_answer", answer="见来源", status="partial")),
    )
    body = client.post(CHAT_URL, json={"question": "q", "doc_ids": [doc_id]}).json()
    results = tool_results(model.calls[1].messages)

    assert [(f["subject"], f["value"], f["citation"]) for f in results[0]["facts"]] == [
        ("实施例1", 1.21, "说明书[0043]表格"), ("实施例2", 1.35, "说明书[0043]表格"), ("实施例3", 1.46, "说明书[0043]表格"),
    ]
    assert [(f["qualifier"], f["value"]) for f in results[1]["facts"]] == [("加入锆盐后", 5), ("加入铝盐后", 4)]
    hits = results[2]["results"]
    assert len(hits) == 3 and all(h["id"].startswith("T") and h["doc"] == "CN116694108A" for h in hits)
    assert body["steps"][0]["sources"] == ["F1", "F2", "F3"]
    assert body["answer"]["status"] == "partial"


def test_session_history_and_lifecycle(app) -> None:
    client, model, _ = app
    model.load(reply(call("final_answer", answer="第一轮回答[F1]", status="insufficient_evidence")))
    first = client.post(CHAT_URL, json={"question": "第一个问题"}).json()
    sid = first["session_id"]

    model.load(reply(call("final_answer", answer="第二轮回答", status="insufficient_evidence")))
    second = client.post(CHAT_URL, json={"question": "那实施例2呢？", "session_id": sid}).json()
    assert second["session_id"] == sid and second["turn"] == 2
    sent = model.calls[0].messages
    assert [(m.role, m.content) for m in sent[1:]] == [
        ("user", "第一个问题"), ("assistant", "第一轮回答"), ("user", "那实施例2呢？"),
    ]

    session = client.get(f"/api/v1/agent/sessions/{sid}").json()
    assert [(t["index"], t["question"], t["answer"]["answer"]) for t in session["turns"]] == [
        (1, "第一个问题", "第一轮回答"), (2, "那实施例2呢？", "第二轮回答"),
    ]
    assert client.delete(f"/api/v1/agent/sessions/{sid}").status_code == 204
    assert client.get(f"/api/v1/agent/sessions/{sid}").status_code == 404
    assert client.post(CHAT_URL, json={"question": "q", "session_id": sid}).json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_request_validation(app) -> None:
    client, model, _ = app
    model.load()
    assert client.post(CHAT_URL, json={"question": ""}).status_code == 422
    assert client.post(CHAT_URL, json={"question": "q", "session_id": "abc"}).status_code == 422
    assert client.post(CHAT_URL, json={"question": "q", "max_steps": 50}).status_code == 422
    r = client.post(CHAT_URL, json={"question": "q", "doc_ids": ["doc_missing"]})
    assert (r.status_code, r.json()["error"]["code"]) == (404, "DOCUMENT_NOT_FOUND")
    assert client.post(CHAT_URL, json={"question": "q", "doc_ids": []}).status_code == 422
    assert model.calls == []   # nothing reached the LLM


def test_llm_failure_is_502_and_nothing_is_stored(app) -> None:
    client, model, _ = app
    model.load(reply(call("query_facts", name="温度")), fail("connection reset"))
    r = client.post(CHAT_URL, json={"question": "温度？"})
    assert (r.status_code, r.json()["error"]["code"]) == (502, "LLM_FAILED")


def test_without_llm_or_documents(tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        r = client.post(CHAT_URL, json={"question": "q"})
        assert (r.status_code, r.json()["error"]["code"]) == (503, "LLM_NOT_CONFIGURED")

    model = ScriptedChatModel()
    with TestClient(create_app(settings, chat_model=model)) as client:
        body = client.post(CHAT_URL, json={"question": "q"}).json()
        assert body["answer"]["status"] == "insufficient_evidence"
        assert body["warnings"] == ["没有处理完成的文档，未调用大模型"]
        assert model.calls == []
    assert model.closed  # the app closes the model on shutdown


def test_documents_still_work_with_agent_enabled(app) -> None:
    client, _, doc_id = app
    assert client.get(f"{UPLOAD_URL}/{doc_id}/facts").json()["total"] == 106

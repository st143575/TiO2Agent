"""Agent evaluation: correct values, correct sources, honest refusals.

Runs the real pipeline in-process on the PDFs named in the dataset, then asks every
question through POST /api/v1/agent/chat with the LLM configured in .env / TIO2_*:

    TIO2_LLM_BACKEND=openai_compatible TIO2_LLM_BASE_URL=http://localhost:11434/v1 \\
    TIO2_LLM_MODEL=qwen3:8b uv run python evals/run_agent_eval.py --out evals/reports/agent.json

Each item in evals/datasets/agent_gold.jsonl may expect:
    values     numbers the answer must report as verified values (value, or low/high;
               ``tolerance`` absolute, default 1e-6 relative)
    citations  locations the citations must cover: "0026" (description paragraph),
               "权利要求8", "扉页著录项", "摘要"
    keywords   strings the answer text must contain
    status     default "answered" (partial also accepted); "insufficient_evidence"
               for questions the documents do not answer
Items with ``"requires": ["ocr"]`` are skipped when TIO2_OCR_BACKEND=none.

A question passes if every expectation holds. The report also counts values that
appear in the text but not as verified values, and answers with grounding warnings.
"""

import argparse
import json
import math
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals/datasets/agent_gold.jsonl"


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


# --- scoring ---------------------------------------------------------------------------------


def citation_tokens(citation: str) -> set[str]:
    """"说明书[0026]-[0029]、[0031]" -> {"0026", …, "0029", "0031"}; "权利要求1、2" ->
    {"权利要求1", "权利要求2"}; other labels as they are."""
    tokens: set[str] = set()
    for a, b in re.findall(r"\[(\d{4})\](?:-\[(\d{4})\])?", citation):
        tokens.update(f"{n:04d}" for n in range(int(a), int(b or a) + 1))
    if citation.startswith("权利要求"):
        tokens.update(f"权利要求{n}" for n in re.findall(r"\d+", citation))
    if not tokens:
        tokens.add(citation)
    return tokens


def _close(actual: float | None, expected: float, tolerance: float | None) -> bool:
    if actual is None:
        return False
    if tolerance is not None:
        return abs(actual - expected) <= tolerance
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)


def _value_matches(v: dict, exp: dict) -> bool:
    tol = exp.get("tolerance")
    if "value" in exp:
        return _close(v.get("value"), exp["value"], tol)
    return _close(v.get("low"), exp["low"], tol) and _close(v.get("high"), exp["high"], tol)


def _in_text(text: str, exp: dict) -> bool:
    numbers = [exp["value"]] if "value" in exp else [exp["low"], exp["high"]]
    found = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    return all(any(_close(f, n, exp.get("tolerance", 0.5 * 10 ** -_decimals(n))) for f in found) for n in numbers)


def _decimals(x: float) -> int:
    s = repr(float(x)).rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def score(item: dict, response: dict) -> dict:
    expected = item["expected"]
    answer = response["answer"]
    text = answer["answer"]
    checks: dict[str, bool] = {}

    want_status = expected.get("status", "answered")
    accepted = {"answered", "partial"} if want_status == "answered" else {want_status}
    checks["status"] = answer["status"] in accepted

    verified = [v for v in answer["values"] if v.get("verified")]
    if expected.get("values"):
        checks["values"] = all(any(_value_matches(v, e) for v in verified) for e in expected["values"])
    if expected.get("citations"):
        covered = set().union(*(citation_tokens(c["citation"]) for c in answer["citations"])) if answer["citations"] else set()
        checks["citations"] = all(c in covered for c in expected["citations"])
    if expected.get("keywords"):
        checks["keywords"] = all(k in text for k in expected["keywords"])

    grounding = [w for w in response.get("warnings", []) if "不一致" in w or "没有出现过的来源" in w]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values_in_text_only": bool(expected.get("values")) and not checks.get("values")
        and all(_in_text(text, e) for e in expected["values"]),
        "grounding_warnings": grounding,
    }


# --- running ---------------------------------------------------------------------------------


def check_llm(settings) -> str | None:
    """Reach the configured LLM before ingesting anything. Returns a problem, or None.

    GET {base_url}/models answers on Ollama, vLLM, LM Studio and most hosted APIs; a
    server that does not implement it is let through (the questions will show).
    """
    import httpx

    from tio2agent.config import LlmBackend

    if settings.llm_backend is LlmBackend.NONE:
        return "未配置大模型：在 .env 中设置 TIO2_LLM_BACKEND=openai_compatible 及 TIO2_LLM_*"
    url = settings.llm_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        r = httpx.get(f"{url}/models", headers=headers, timeout=10)
    except httpx.HTTPError as exc:
        return (
            f"连不上大模型服务 {url}（{exc}）。\n"
            "  · Ollama：先启动服务（打开 Ollama 应用，或运行 ollama serve），再 ollama pull "
            f"{settings.llm_model}\n"
            "  · 其他服务：检查 TIO2_LLM_BASE_URL 和端口"
        )
    if r.status_code in (401, 403):
        return f"大模型服务拒绝访问（{r.status_code}）：检查 TIO2_LLM_API_KEY"
    if r.status_code == 200:
        try:
            ids = [m.get("id") for m in r.json().get("data", [])]
        except (ValueError, AttributeError):
            ids = []
        if ids and settings.llm_model not in ids:
            return (
                f"服务上没有模型 {settings.llm_model}，现有：{', '.join(map(str, ids[:10]))}。\n"
                f"  · Ollama：ollama pull {settings.llm_model}，或把 TIO2_LLM_MODEL 改成上面的某一个"
            )
    return None


def evaluate(
    dataset_path: Path = DEFAULT_DATASET,
    *,
    only: set[str] | None = None,
    max_steps: int | None = None,
    env_file: str | None = ".env",
    chat_model=None,
) -> dict:
    """``chat_model`` overrides the configured LLM (tests)."""
    from fastapi.testclient import TestClient

    from tio2agent.config import OcrBackend, Settings
    from tio2agent.main import create_app

    items = [i for i in load_dataset(dataset_path) if only is None or i["id"] in only]
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(_env_file=env_file, data_dir=Path(tmp), max_upload_mb=200, max_pages=2000)
        if chat_model is None and (problem := check_llm(settings)):
            raise SystemExit(f"错误：{problem}")
        has_ocr = settings.ocr_backend is not OcrBackend.NONE
        with TestClient(create_app(settings, chat_model=chat_model)) as client:
            for pdf in sorted({i["pdf"] for i in items}):
                r = client.post("/api/v1/documents", files={"file": (Path(pdf).name, (ROOT / pdf).read_bytes(), "application/pdf")})
                _wait(client, r.json()["id"])

            rows = []
            for item in items:
                if "ocr" in item.get("requires", []) and not has_ocr:
                    rows.append({"id": item["id"], "type": item["type"], "question": item["question"], "skipped": "需要 OCR"})
                    continue
                body = {"question": item["question"]}
                if max_steps:
                    body["max_steps"] = max_steps
                start = time.perf_counter()
                r = client.post("/api/v1/agent/chat", json=body)
                elapsed = time.perf_counter() - start
                if r.status_code != 200:
                    error = r.json().get("error", {})
                    if error.get("code") == "LLM_NOT_CONFIGURED":
                        raise SystemExit("未配置大模型：设置 TIO2_LLM_BACKEND=openai_compatible 及 TIO2_LLM_* 后再运行")
                    rows.append({"id": item["id"], "type": item["type"], "question": item["question"],
                                 "passed": False, "error": f"{r.status_code} {error.get('code')}: {error.get('message')}"})
                    continue
                response = r.json()
                rows.append({
                    "id": item["id"], "type": item["type"], "question": item["question"],
                    **score(item, response),
                    "status": response["answer"]["status"],
                    "answer": response["answer"]["answer"],
                    "tools": [s["tool"] for s in response["steps"]],
                    "llm_calls": response["usage"]["llm_calls"],
                    "tokens": response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"],
                    "seconds": round(elapsed, 1),
                    "warnings": response["warnings"],
                })
            model = client.app.state.agent_service.runner.model.model if client.app.state.agent_service.enabled else None
    return {"dataset": str(dataset_path.relative_to(ROOT)), "model": model, "summary": summarize(rows), "items": rows}


def summarize(rows: list[dict]) -> dict:
    ran = [r for r in rows if "skipped" not in r]
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in ran:
        by_type[r["type"]].append(bool(r.get("passed")))
    answered = [r for r in ran if "error" not in r]
    return {
        "questions": len(rows),
        "skipped": len(rows) - len(ran),
        "passed": sum(bool(r.get("passed")) for r in ran),
        "pass_rate": round(sum(bool(r.get("passed")) for r in ran) / len(ran), 3) if ran else 0.0,
        "by_type": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_type.items())},
        "errors": len(ran) - len(answered),
        "values_in_text_only": sum(bool(r.get("values_in_text_only")) for r in answered),
        "with_grounding_warnings": sum(bool(r.get("grounding_warnings")) for r in answered),
        "avg_llm_calls": round(sum(r["llm_calls"] for r in answered) / len(answered), 2) if answered else 0,
        "avg_tokens": round(sum(r["tokens"] for r in answered) / len(answered)) if answered else 0,
        "avg_seconds": round(sum(r["seconds"] for r in answered) / len(answered), 1) if answered else 0,
    }


def _wait(client, doc_id: str, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/documents/{doc_id}").json()["status"] in {"ready", "ready_with_warnings", "failed"}:
            return
        time.sleep(0.2)
    raise TimeoutError(doc_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--only", help="逗号分隔的题目 id，如 a01,a07")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", type=Path, help="把完整报告写成 JSON")
    args = parser.parse_args()

    report = evaluate(args.dataset, only=set(args.only.split(",")) if args.only else None, max_steps=args.max_steps)
    print(f"\nAgent 评测 · 模型 {report['model']} · {report['dataset']}\n")
    for r in report["items"]:
        if "skipped" in r:
            print(f"  -    {r['id']} [{r['type']}] 跳过（{r['skipped']}）")
            continue
        mark = "PASS" if r.get("passed") else "FAIL"
        detail = r.get("error") or " ".join(f"{k}={'✓' if v else '✗'}" for k, v in r["checks"].items())
        extra = f"  {r['llm_calls']} 次调用 {r['seconds']}s" if "llm_calls" in r else ""
        print(f"  {mark} {r['id']} [{r['type']}] {detail}{extra}")
    print("\n" + json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

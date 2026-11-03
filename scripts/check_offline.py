"""End-to-end check of the offline pipeline on real PDFs.

Uploads every PDF (default: data/samples/*.pdf), waits for parse -> ocr -> structure
-> index -> extract to finish, reads back the structure, chunks and facts through the
API, and checks invariants that must hold for any patent, without needing hand-made
expected values:

    FAIL  processing failed; no claims; claim numbers not 1..n; no independent claim;
          no paragraphs; a claim that is in no chunk
    WARN  not detected as a patent; missing publication number / title; gaps in
          paragraph numbering; no 具体实施方式 section; no examples; paragraphs or
          tables in no chunk; no facts; an example without components; tables but no
          result facts; failed fact checks; every warning the document itself reports

Two modes:

    # in process, with a throw-away data directory; services from .env / TIO2_* vars
    uv run python scripts/check_offline.py

    # against a running server (uv run tio2agent, or docker compose up)
    uv run python scripts/check_offline.py --base-url http://localhost:8000

Exit code 1 if any check FAILs (with --strict, also on WARN).
"""

import argparse
import contextlib
import json
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "/api/v1"
TERMINAL = {"ready", "ready_with_warnings", "failed"}


class ServerUnavailable(RuntimeError):
    """--base-url given but nothing answers there."""


# --- talking to the API ------------------------------------------------------------------


@contextlib.contextmanager
def open_client(base_url: str | None, keep_data: Path | None):
    """An httpx-compatible client: a real one for --base-url, else an in-process app."""
    if base_url:
        import httpx

        with httpx.Client(base_url=base_url, timeout=120) as client:
            try:
                client.get("/health", timeout=5).raise_for_status()
            except httpx.HTTPError as exc:
                raise ServerUnavailable(
                    f"连不上 {base_url}（{exc}）。\n"
                    "  · 先在另一个终端启动服务：uv run tio2agent，或 docker compose up -d\n"
                    "  · 或者去掉 --base-url，在进程内运行，不需要启动服务"
                ) from exc
            yield client, f"server {base_url}"
        return

    from fastapi.testclient import TestClient

    from tio2agent.config import Settings
    from tio2agent.main import create_app

    with contextlib.ExitStack() as stack:
        data_dir = keep_data or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="tio2-offline-")))
        settings = Settings(data_dir=data_dir)  # .env and TIO2_* still apply, except data_dir
        client = stack.enter_context(TestClient(create_app(settings)))
        services = (
            f"ocr={settings.ocr_backend.value} embedding={settings.embedding_backend.value} "
            f"data_dir={data_dir}"
        )
        yield client, f"in-process ({services})"


def ingest(client, pdf: Path, *, reindex: bool, timeout: float) -> tuple[dict, float]:
    response = client.post(f"{API}/documents", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    if response.status_code not in (200, 202):
        raise RuntimeError(f"上传失败 {response.status_code}: {response.text[:300]}")
    doc = response.json()
    if doc.get("duplicate") and reindex:
        client.post(f"{API}/documents/{doc['id']}/reindex").raise_for_status()
    start = time.monotonic()
    while True:
        doc = client.get(f"{API}/documents/{doc['id']}").json()
        if doc["status"] in TERMINAL:
            return doc, time.monotonic() - start
        if time.monotonic() - start > timeout:
            raise RuntimeError(f"{timeout:.0f}s 内未处理完（当前状态 {doc['status']}）")
        time.sleep(0.2)


def fetch(client, doc_id: str) -> dict:
    def get(path: str):
        r = client.get(f"{API}/documents/{doc_id}{path}")
        return r.json() if r.status_code == 200 else None

    return {
        "structure": get("/structure"),
        "chunks": (get("/chunks") or {}).get("items", []),
        "facts": (get("/facts") or {}).get("items", []),
        "formulations": (get("/formulations") or {}).get("items", []),
    }


# --- checks --------------------------------------------------------------------------------


def check(doc: dict, data: dict) -> list[tuple[str, str]]:
    """[(level, message)] with level FAIL / WARN."""
    out: list[tuple[str, str]] = []
    fail = lambda m: out.append(("FAIL", m))  # noqa: E731
    warn = lambda m: out.append(("WARN", m))  # noqa: E731

    if doc["status"] == "failed":
        err = doc.get("error") or {}
        fail(f"处理失败 [{err.get('stage')}] {err.get('code')}: {err.get('message')}")
        return out
    for w in doc.get("warnings", []):
        page = f" p{w['page']}" if w.get("page") else ""
        warn(f"文档警告 [{w.get('stage')}] {w['code']}{page}: {w['message']}")

    s = data["structure"]
    if s is None:
        fail("读不到结构识别结果")
        return out
    if s["doc_type"] != "patent":
        warn(f"文档类型识别为 {s['doc_type']}，不是 patent")
    bib = s.get("bibliographic") or {}
    if not bib.get("publication_number"):
        warn("没有识别出公布号")
    if not (bib.get("title") or s.get("title")):
        warn("没有识别出标题")

    claims = s["claims"]
    numbers = [c["number"] for c in claims]
    if not claims:
        fail("没有识别出权利要求")
    elif numbers != list(range(1, len(numbers) + 1)):
        fail(f"权利要求编号不连续：{numbers}")
    if claims and not any(c["kind"] == "independent" for c in claims):
        fail("没有独立权利要求")

    paragraphs = s["paragraphs"]
    if not paragraphs:
        fail("没有识别出说明书段落")
    numeric = sorted(int(p["id"]) for p in paragraphs if p["id"].isdigit())
    if numeric:
        missing = sorted(set(range(numeric[0], numeric[-1] + 1)) - set(numeric))
        if missing or numeric[0] != 1:
            warn(f"段落编号有缺口：从 [{numeric[0]:04d}] 开始，缺 {[f'{m:04d}' for m in missing][:10]}")
    if not any(sec["kind"] == "detailed_description" for sec in s["sections"]):
        warn("没有识别出“具体实施方式”章节")
    examples = [e for e in s["embodiments"] if e["kind"] == "example"]
    if not examples:
        warn("没有识别出实施例")

    chunks = data["chunks"]
    in_chunks = {
        "claims": {n for c in chunks for n in c["claim_numbers"]},
        "paragraphs": {p for c in chunks for p in c["paragraph_ids"]},
        "tables": {t for c in chunks for t in c["table_ids"]},
    }
    lost_claims = sorted(set(numbers) - in_chunks["claims"])
    if lost_claims:
        fail(f"权利要求 {lost_claims} 不在任何切块里")
    lost = [
        p["id"] for p in paragraphs
        # headings ("实施例1", "(二)技术方案") go into the chunk breadcrumb, not its body
        if p["text"].strip() and p["role"] == "body" and p["id"] not in in_chunks["paragraphs"]
    ]
    if lost:
        warn(f"{len(lost)} 个段落不在任何切块里：{lost[:10]}")
    lost_tables = [t["id"] for t in s["tables"] if t["id"] not in in_chunks["tables"]]
    if lost_tables:
        warn(f"表格 {lost_tables} 不在任何切块里")

    facts = data["facts"]
    if s["doc_type"] == "patent" and doc["options"].get("extract_facts", True):
        if not facts:
            warn("没有抽取到任何事实")
        with_components = {f["subject"] for f in facts if f["kind"] == "component"}
        bare = [e["label"] for e in examples if e["id"] not in with_components]
        if facts and bare:
            warn(f"实施例没有抽到组分：{bare}")
        if s["tables"] and not any(f["kind"] == "result" for f in facts):
            warn(f"有 {len(s['tables'])} 个表格，但没有抽到测试结果")
        failed = [c["message"] for f in facts for c in f["checks"] if not c["ok"]]
        if failed:
            warn(f"{len(failed)} 项事实校验未通过，例如：{failed[:3]}")
    return out


# --- report --------------------------------------------------------------------------------


def summarize(pdf: Path, doc: dict, data: dict, seconds: float, issues: list) -> dict:
    s = data["structure"] or {}
    facts = data["facts"]
    checks = Counter(c["ok"] for f in facts for c in f["checks"])
    return {
        "file": pdf.name,
        "id": doc["id"],
        "status": doc["status"],
        "seconds": round(seconds, 1),
        "publication_number": (s.get("bibliographic") or {}).get("publication_number"),
        "title": s.get("title"),
        "stats": doc["stats"],
        "claims": len(s.get("claims", [])),
        "independent_claims": [c["number"] for c in s.get("claims", []) if c["kind"] == "independent"],
        "embodiments": [e["label"] for e in s.get("embodiments", [])],
        "tables": len(s.get("tables", [])),
        "facts_by_kind": dict(Counter(f["kind"] for f in facts)),
        "fact_checks": {"passed": checks[True], "failed": checks[False]},
        "formulation_totals": {f["subject_label"]: f["total"] for f in data["formulations"]},
        "issues": [{"level": lvl, "message": msg} for lvl, msg in issues],
    }


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _cell(value, width: int, left: bool = False) -> str:
    text = str(value)
    if left and _width(text) > width:
        while _width(text) > width - 1:
            text = text[:-1]
        text += "…"
    pad = " " * max(0, width - _width(text))
    return text + pad if left else pad + text


COLUMNS = [("文件", 32), ("状态", 20), ("秒", 6), ("页", 4), ("段落", 5), ("权利", 5), ("实施例", 7),
           ("表", 3), ("切块", 5), ("事实", 5), ("FAIL", 5), ("WARN", 5)]


def print_report(rows: list[dict], where: str) -> None:
    print(f"\n离线流水线检查 · {where}\n")
    print(" ".join(_cell(name, w, left=i < 2) for i, (name, w) in enumerate(COLUMNS)))
    print("-" * (sum(w for _, w in COLUMNS) + len(COLUMNS) - 1))
    for r in rows:
        st, levels = r["stats"], Counter(i["level"] for i in r["issues"])
        values = [r["file"], r["status"], r["seconds"], st["pages"], st["paragraphs"], r["claims"],
                  len(r["embodiments"]), r["tables"], st["chunks"], st["facts"], levels["FAIL"], levels["WARN"]]
        print(" ".join(_cell(v, w, left=i < 2) for i, (v, (_, w)) in enumerate(zip(values, COLUMNS, strict=True))))
    for r in rows:
        print(f"\n■ {r['file']}  ({r['publication_number'] or '无公布号'}, id={r['id']})")
        print(f"  标题: {r['title']}")
        print(f"  独立权利要求: {r['independent_claims']}  实施例: {r['embodiments']}")
        print(f"  事实: {r['facts_by_kind']}  校验: {r['fact_checks']}")
        if r["formulation_totals"]:
            print(f"  配方合计: {r['formulation_totals']}")
        for i in r["issues"]:
            print(f"  [{i['level']}] {i['message']}")


def run_checks(pdfs: list[Path], args) -> tuple[list[dict], str]:
    rows = []
    with open_client(args.base_url, args.keep_data) as (client, where):
        for pdf in pdfs:
            print(f"处理 {pdf.name} …", file=sys.stderr, flush=True)
            try:
                doc, seconds = ingest(client, pdf, reindex=args.reindex, timeout=args.timeout)
            except Exception as exc:  # report and go on with the next file
                rows.append({"file": pdf.name, "id": None, "status": "error", "seconds": 0, "publication_number": None,
                             "title": None, "stats": {k: 0 for k in ("pages", "paragraphs", "chunks", "facts")},
                             "claims": 0, "independent_claims": [], "embodiments": [], "tables": 0,
                             "facts_by_kind": {}, "fact_checks": {}, "formulation_totals": {},
                             "issues": [{"level": "FAIL", "message": str(exc)}]})
                continue
            data = fetch(client, doc["id"])
            rows.append(summarize(pdf, doc, data, seconds, check(doc, data)))
    return rows, where


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="*", type=Path, help="默认 data/samples/*.pdf")
    parser.add_argument("--base-url", help="检查正在运行的服务，例如 http://localhost:8000")
    parser.add_argument("--reindex", action="store_true", help="服务上已有同一文件时重新处理（用当前代码）")
    parser.add_argument("--keep-data", type=Path, help="进程内模式：数据目录保留在这里，便于查看中间文件")
    parser.add_argument("--timeout", type=float, default=900, help="单个文档的处理超时（秒）")
    parser.add_argument("--out", type=Path, help="把报告写成 JSON")
    parser.add_argument("--strict", action="store_true", help="有 WARN 也返回非零退出码")
    args = parser.parse_args()

    pdfs = args.pdfs or sorted((ROOT / "data" / "samples").glob("*.pdf"))
    if not pdfs:
        print("没有找到 PDF", file=sys.stderr)
        return 2

    try:
        rows, where = run_checks(pdfs, args)
    except ServerUnavailable as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print_report(rows, where)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入 {args.out}")

    levels = Counter(i["level"] for r in rows for i in r["issues"])
    print(f"\n合计：{len(rows)} 个文件，FAIL {levels['FAIL']}，WARN {levels['WARN']}")
    return 1 if levels["FAIL"] or (args.strict and levels["WARN"]) else 0


if __name__ == "__main__":
    sys.exit(main())

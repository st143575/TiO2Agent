"""Retrieval evaluation: Hit@k and MRR on hand-labelled questions.

Runs the real pipeline in-process (upload -> parse -> structure -> index) on the PDFs
named in the dataset, then queries POST /api/v1/search for every question.

    uv run python evals/run_retrieval_eval.py                       # BM25
    TIO2_EMBEDDING_BACKEND=openai_compatible \\
    TIO2_EMBEDDING_BASE_URL=http://localhost:11434/v1 TIO2_EMBEDDING_MODEL=bge-m3 \\
    uv run python evals/run_retrieval_eval.py --mode bm25 --mode vector --mode hybrid
    # with TIO2_RERANK_* configured, also score every mode with reranking:
    uv run python evals/run_retrieval_eval.py --mode bm25 --mode hybrid --rerank

Relevance labels (``relevant`` in the JSONL):
    "bibliographic" / "abstract"   a chunk of that kind
    "claim:N"                      the chunk of claim N, or a hit that absorbed it (also_in)
    "p:NNNN"                       any chunk containing description paragraph [NNNN]
A hit is relevant if it matches any label. Hit@k: a relevant hit within the top k.
MRR: 1 / rank of the first relevant hit (0 if none in the top ``--k``).
"""

import argparse
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals/datasets/retrieval_gold.jsonl"


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def is_relevant(hit: dict, labels: list[str]) -> bool:
    chunk = hit["chunk"]
    absorbed = {int(n) for cite in hit.get("also_in", []) for n in re.findall(r"权利要求(\d+)", cite)}
    for label in labels:
        if label in ("bibliographic", "abstract") and chunk["kind"] == label:
            return True
        if label.startswith("claim:") and (int(label[6:]) in chunk["claim_numbers"] or int(label[6:]) in absorbed):
            return True
        if label.startswith("p:") and label[2:] in chunk["paragraph_ids"]:
            return True
    return False


def evaluate(
    dataset_path: Path = DEFAULT_DATASET, 
    modes: tuple[str, ...] = ("bm25",), 
    k: int = 10,
    rerank: bool = False,
    env_file: str | None = ".env",
) -> dict:
    from fastapi.testclient import TestClient
    from tio2agent.config import Settings
    from tio2agent.main import create_app

    items = load_dataset(dataset_path)
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            _env_file=env_file,
            data_dir=Path(tmp),
            max_upload_mb=200,
            max_pages=2000,
            ingest_workers=2,
        )
        with TestClient(create_app(settings)) as client:
            doc_ids = {}
            for pdf in sorted({item["pdf"] for item in items}):
                response = client.post(
                    "/api/v1/documents", files={"file": (Path(pdf).name, (ROOT / pdf).read_bytes(), "application/pdf")}
                )
                doc_ids[pdf] = response.json()["id"]
            for doc_id in doc_ids.values():
                _wait(client, doc_id)

            report = {"dataset": str(dataset_path.relative_to(ROOT)), "k": k, "questions": len(items), "modes": {}}
            runs = [(mode, False) for mode in modes] + ([(mode, True) for mode in modes] if rerank else [])
            for mode, with_rerank in runs:
                rows = []
                for item in items:
                    body = client.post(
                        "/api/v1/search",
                        json={
                            "query": item["question"], "top_k": k, "doc_ids": [doc_ids[item["pdf"]]], 
                            "mode": mode, "rerank": with_rerank
                        },
                    ).json()
                    rank = next(
                        (h["rank"] for h in body["hits"] if is_relevant(h, item["relevant"])), None
                    )
                    rows.append(
                        {
                            "id": item["id"],
                            "type": item.get("type", ""),
                            "question": item["question"],
                            "first_relevant_rank": rank,
                            "top3": [h["chunk"]["citation"] for h in body["hits"][:3]],
                            "mode_used": body["mode_used"] + ("+rerank" if body["reranked"] else ""),
                        }
                    )
                label = f"{mode}+rerank" if with_rerank else mode
                report["modes"][label] = {"metrics": _metrics(rows, k), "by_type": _by_type(rows, k), "rows": rows}
    return report


def _wait(client, doc_id: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/documents/{doc_id}").json()["status"] in {"ready", "ready_with_warnings", "failed"}:
            return
        time.sleep(0.1)
    raise TimeoutError(doc_id)


def _metrics(rows: list[dict], k: int) -> dict:
    n = len(rows) or 1
    ranks = [r["first_relevant_rank"] for r in rows]
    return {
        "hit@1": round(sum(1 for r in ranks if r and r <= 1) / n, 3),
        "hit@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
        "hit@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        f"mrr@{k}": round(sum(1 / r for r in ranks if r) / n, 3),
        "mode_used": sorted({r["mode_used"] for r in rows}),
    }


def _by_type(rows: list[dict], k: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["type"]].append(row)
    return {t: {**_metrics(g, k), "n": len(g)} for t, g in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", action="append", choices=["bm25", "vector", "hybrid"], help="repeatable; default bm25")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true", help="also run every mode with reranking (needs TIO2_RERANK_*)")
    parser.add_argument("--out", type=Path, default=ROOT / "data/eval/retrieval")
    args = parser.parse_args()

    report = evaluate(args.dataset, tuple(args.mode or ["bm25"]), args.k, rerank=args.rerank)
    for mode, result in report["modes"].items():
        m = result["metrics"]
        print(f"\n=== {mode} (used: {', '.join(m['mode_used'])})  questions={report['questions']}")
        print("  " + "  ".join(f"{key}={value}" for key, value in m.items() if key != "mode_used"))
        for t, tm in result["by_type"].items():
            print(f"  {t:<14} n={tm['n']:<3} hit@1={tm['hit@1']:<5} hit@5={tm['hit@5']:<5} mrr={tm[f'mrr@{args.k}']}")
        misses = [r for r in result["rows"] if not r["first_relevant_rank"] or r["first_relevant_rank"] > 3]
        for r in misses:
            print(f"  ✗ {r['id']} rank={r['first_relevant_rank']} {r['question']}  top3={r['top3']}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

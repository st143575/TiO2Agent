"""Extraction evaluation: rules vs LLM vs hybrid against the hand-labelled gold standard.

    uv run python evals/run_extraction_eval.py                       # rules only
    uv run python evals/run_extraction_eval.py --mode rules --mode llm --mode hybrid \\
        --out evals/reports/extraction.json --errors                  # LLM from .env / TIO2_*

Gold: evals/datasets/extraction_gold/<公开号>.json (see the README there). The PDFs
are read from data/samples (file names in the gold files), without OCR; results that
are only in images (``requires_ocr``) are therefore not scored.

A predicted fact is correct if subject, step, kind, name, qualifier and value all
equal a gold fact (names compared after the alias table, numbers to 1e-9; options as
sets). Each wrong or missing fact is listed; a prediction that has the right number
at the right place but a different name / step / qualifier is listed as a near miss,
with the fields that differ. Predictions from places outside the gold scope (e.g. a
paragraph the gold excludes as a restatement) count as wrong and are marked.

CN116694108A is not scored: the LLM prompt's worked examples come from it.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "evals/datasets/extraction_gold"
SAMPLES_DIR = ROOT / "data/samples"
MODES = ("rules", "llm", "hybrid")
_EPS = 1e-9


# --- scoring ------------------------------------------------------------------------------


def _canon(name: str) -> str:
    from tio2agent.ingestion.extraction.quantities import canonical_name

    return canonical_name(name)[0]


def gold_items(gold: dict) -> list[dict]:
    return [f for f in gold["facts"] if f["kind"] != "result"]


def predicted_items(facts) -> list[dict]:
    """Facts (pydantic) -> the gold shape; results (from tables) are not scored."""
    out = []
    for f in facts:
        if f.kind.value == "result":
            continue
        where = {"claim": f.claim_number} if f.claim_number is not None else {"paragraph": (f.paragraph_ids or [""])[0]}
        out.append({
            "id": f.id, "kind": f.kind.value, "subject": f.subject, "step": f.step, "name": f.name,
            "qualifier": f.qualifier, "low": f.low, "high": f.high, "unit": f.unit, "options": f.options,
            "evidence": f.evidence, "where": where, "extractor": f.extractor,
        })
    return out


def _key(f: dict) -> tuple:
    return (f["subject"], f.get("step"), f["kind"], _canon(f["name"]), f.get("qualifier"))


def _value_equal(a: dict, b: dict) -> bool:
    if a["kind"] == "option" or b["kind"] == "option":
        return set(a.get("options") or []) == set(b.get("options") or [])
    return all(
        x is not None and y is not None and abs(x - y) < _EPS
        for x, y in ((a.get("low"), b.get("low")), (a.get("high"), b.get("high")))
    )


def _in_scope(f: dict, scope: dict) -> bool:
    w = f["where"]
    return w.get("claim") in scope["claims"] if "claim" in w else w.get("paragraph") in scope["paragraphs"]


def score(gold: dict, predicted: list[dict]) -> dict:
    golds = gold_items(gold)
    unused = list(range(len(predicted)))
    matched_gold: set[int] = set()
    for gi, g in enumerate(golds):
        for pi in unused:
            p = predicted[pi]
            if _key(p) == _key(g) and _value_equal(p, g):
                matched_gold.add(gi)
                unused.remove(pi)
                break
    tp = len(matched_gold)
    missed = [g for gi, g in enumerate(golds) if gi not in matched_gold]

    near, extra = [], []
    for pi in unused:
        p = predicted[pi]
        twin = next(
            (g for g in missed
             if g["subject"] == p["subject"] and g["kind"] == p["kind"] and g["where"] == p["where"]
             and _value_equal(g, p) and not any(n["gold"] is g for n in near)),
            None,
        )
        if twin is not None:
            fields = [k for k in ("name", "step", "qualifier")
                      if (_canon(p[k]) if k == "name" else p.get(k)) != (_canon(twin[k]) if k == "name" else twin.get(k))]
            near.append({"gold": twin, "predicted": p, "fields": fields})
        else:
            extra.append({**p, "out_of_scope": not _in_scope(p, gold["scope"])})
    near_gold = {id(n["gold"]) for n in near}
    missed = [g for g in missed if id(g) not in near_gold]

    n_gold, n_pred = len(golds), len(predicted)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold": n_gold, "predicted": n_pred, "correct": tp,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "near_misses": near, "missed": missed, "extra": extra,
    }


def total(rows: list[dict]) -> dict:
    g = sum(r["gold"] for r in rows)
    p = sum(r["predicted"] for r in rows)
    c = sum(r["correct"] for r in rows)
    precision, recall = (c / p if p else 0.0), (c / g if g else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gold": g, "predicted": p, "correct": c, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


# --- running ------------------------------------------------------------------------------


def structure_pdf(pdf: Path):
    from tio2agent.domain import DocumentRecord
    from tio2agent.ingestion.pipeline import IngestionContext
    from tio2agent.ingestion.stages.parse import ParseStage
    from tio2agent.ingestion.structure import StructureInput, structure_document

    ctx = IngestionContext(
        record=DocumentRecord(id=pdf.stem[:8], filename=pdf.name, sha256="x", size_bytes=1),
        pdf_path=pdf,
        work_dir=Path(tempfile.mkdtemp()),
    )
    ParseStage().run(ctx)
    return structure_document(StructureInput(ctx.record.id, "patent", ctx.artifacts["pages"], [])).document


def load_gold(only: list[str] | None = None) -> list[dict]:
    golds = [json.loads(p.read_text("utf-8")) for p in sorted(GOLD_DIR.glob("CN*.json"))]
    return [g for g in golds if not only or g["doc"] in only]


class _Replay:
    """
    An extractor that returns a stored LLM result: llm and hybrid then 
    share one run of the LLM per document instead of asking it twice.
    """

    def __init__(self, result) -> None:
        self.result = result
        self.name = "llm"

    def extract(self, doc):
        from tio2agent.ingestion.extraction import ExtractionResult

        r = self.result
        return ExtractionResult(facts=[f.model_copy(deep=True) for f in r.facts], issues=list(r.issues),
                                rejected=list(r.rejected), stats=dict(r.stats))


def evaluate(
    golds: list[dict],
    modes: list[str],
    model=None,
    samples_dir: Path = SAMPLES_DIR,
    *,
    max_failures: int = 2,
    progress=None,
) -> dict:
    from tio2agent.ingestion.extraction import (
        HybridExtractor,
        LlmPatentExtractor,
        PatentRuleExtractor,
        extract_facts,
    )

    report: dict = {"modes": {}, "model": getattr(model, "model", None)}
    docs = {}
    for g in golds:
        pdf = samples_dir / g["file"]
        if not pdf.exists():
            print(f"跳过 {g['doc']}：找不到 {pdf}", file=sys.stderr)
            continue
        docs[g["doc"]] = (g, structure_pdf(pdf))

    llm_runs: dict[str, object] = {}

    def llm_result(name, doc):
        if name not in llm_runs:
            if progress:
                progress(f"── {name}：大模型抽取")
            llm_runs[name] = LlmPatentExtractor(model, max_failures=max_failures, progress=progress).extract(doc)
        return llm_runs[name]
    
    for mode in modes:
        rows = []
        for name, (g, doc) in docs.items():
            started = time.monotonic()
            if mode == "rules":
                extractor = PatentRuleExtractor()
            elif mode == "llm":
                extractor = _Replay(llm_result(name, doc))
            else:
                extractor = HybridExtractor(PatentRuleExtractor(), _Replay(llm_result(name, doc)))
            result = extract_facts(doc, extractor)
            seconds = result.stats.get("seconds", round(time.monotonic() - started, 1))
            row = {"doc": name, "seconds": seconds,
                   **score(g, predicted_items(result.facts)),
                   "issues": [{"code": i.code, "message": i.message} for i in result.issues],
                   "rejected": [{"where": r.subject_label, "reason": r.reason, "item": r.item} for r in result.rejected],
                   "stats": result.stats}
            rows.append(row)
        report["modes"][mode] = {"documents": rows, "total": total(rows)}
    return report


def print_report(report: dict, *, errors: bool = False) -> None:
    print(f"{'模式':<8}{'专利':<15}{'标准':>5}{'抽出':>6}{'正确':>6}{'精确':>8}{'召回':>8}{'F1':>8}{'近似':>6}{'用时':>8}")
    for mode, data in report["modes"].items():
        for r in data["documents"]:
            print(f"{mode:<8}{r['doc']:<15}{r['gold']:>5}{r['predicted']:>6}{r['correct']:>6}"
                  f"{r['precision']:>8.2f}{r['recall']:>8.2f}{r['f1']:>8.2f}{len(r['near_misses']):>6}{r['seconds']:>7}s")
        t = data["total"]
        print(f"{mode:<8}{'合计':<15}{t['gold']:>5}{t['predicted']:>6}{t['correct']:>6}"
              f"{t['precision']:>8.2f}{t['recall']:>8.2f}{t['f1']:>8.2f}")
    if not errors:
        return
    for mode, data in report["modes"].items():
        for r in data["documents"]:
            print(f"\n== {mode} {r['doc']}")
            for n in r["near_misses"]:
                g, p = n["gold"], n["predicted"]
                print(f"  近似 {_show(g)}  ← 抽成 {_show(p)}  不同：{'、'.join(n['fields'])}")
            for g in r["missed"]:
                print(f"  漏抽 {_show(g)}  「{g['evidence']}」")
            for p in r["extra"]:
                mark = "（范围外）" if p["out_of_scope"] else ""
                print(f"  多抽{mark} {_show(p)}  「{p['evidence']}」")
            for x in r["rejected"]:
                print(f"  核对未通过 {x['where']}：{x['reason']}  {json.dumps(x['item'], ensure_ascii=False)[:120]}")


def _show(f: dict) -> str:
    if f["kind"] == "option":
        value = "/".join(f.get("options") or [])
    else:
        low, high = f.get("low"), f.get("high")
        value = f"{low:g}" if low == high else f"{low:g}-{high:g}"
    where = f"权利要求{f['where']['claim']}" if "claim" in f["where"] else f"[{f['where'].get('paragraph')}]"
    parts = [f["subject"], f.get("step"), f.get("qualifier"), f["name"], value + (f.get("unit") or "")]
    return " ".join(p for p in parts if p) + f" @{where}"


def _llm_problem(settings) -> str | None:
    """Reuse the Agent evaluation's reachability check, for the extraction model."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_agent_eval", ROOT / "evals/run_agent_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_llm(settings.model_copy(update={"llm_model": settings.extract_llm_model or settings.llm_model}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", action="append", choices=MODES, help="可重复；默认 rules")
    parser.add_argument("--doc", action="append", help="只评测这些公开号")
    parser.add_argument("--out", type=Path, help="把完整报告写入 JSON")
    parser.add_argument("--errors", action="store_true", help="列出近似、漏抽、多抽和未通过核对的事实")
    args = parser.parse_args(argv)
    modes = args.mode or ["rules"]

    from tio2agent.config import Settings
    from tio2agent.llm import build_extraction_model

    model = None
    settings = None
    if any(m != "rules" for m in modes):
        settings = Settings()
        problem = _llm_problem(settings)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        model = build_extraction_model(settings)
        effort = settings.extract_llm_reasoning_effort
        print(f"大模型：{model.model}（reasoning_effort={effort or '不发送'}，max_tokens={settings.extract_llm_max_tokens}，"
              f"超时 {settings.extract_llm_timeout_s:g}s）；每篇约 11 次调用，llm 与 hybrid 共用一次结果", flush=True)
    report = evaluate(
        load_gold(args.doc), modes, model,
        max_failures=settings.extract_llm_max_failures if settings else 2,
        progress=lambda line: print("  " + line, flush=True),
    )
    print_report(report, errors=args.errors)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf-8")
        print(f"\n报告：{args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())

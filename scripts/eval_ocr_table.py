"""Evaluate a PaddleOCR-VL service on a table image against a hand-labelled ground truth.

Default target: the image-only results table on page 6 of the sample patent
(CN116694108A, paragraph [0043]).

    # 1. Start PaddleOCR-VL (see README: Docker Compose on NVIDIA, or paddlex --serve on CPU/Mac)
    # 2. Run:
    uv run python scripts/eval_ocr_table.py --base-url http://localhost:8080 --runs 3

Cells are matched by (row header, column header), not by grid position, so a table that
comes back with a different shape (e.g. the diagonal header split into two cells) is
still scored on its values. Numeric cells are the ones that matter for process metrics.

Exit code: 0 if every numeric cell matched in every run, 1 otherwise, 2 on errors.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

from pypdf import PdfReader

from tio2agent.ingestion.ocr import OcrError, PaddleOcrVlClient, normalize_cell, table_to_markdown
from tio2agent.ingestion.ocr.paddleocr_vl import parse_layout_parsing_result

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = ROOT / "tests/fixtures/CN116694108A_p6_table.expected.json"

Table = list[list[str]]


def load_image(expected: dict, image_path: Path | None) -> bytes:
    if image_path:
        return image_path.read_bytes()
    pdf = ROOT / expected["source"]
    page = PdfReader(pdf).pages[expected["page"] - 1]
    want = (expected["image"]["width"], expected["image"]["height"])
    for image in page.images:
        if image.image.size == want:
            buffer = io.BytesIO()
            image.image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
    raise SystemExit(f"no {want[0]}x{want[1]} image on page {expected['page']} of {pdf}")


def keyed_cells(rows: Table) -> dict[tuple[str, str], str]:
    """{(row header, column header): value} using the first row and column as headers."""
    if len(rows) < 2 or len(rows[0]) < 2:
        return {}
    header = [normalize_cell(h) for h in rows[0]]
    cells = {}
    for row in rows[1:]:
        if not row:
            continue
        label = normalize_cell(row[0])
        for col in range(1, min(len(row), len(header))):
            cells[(label, header[col])] = row[col]
    return cells


def score(expected: dict, predicted: Table) -> dict:
    truth = expected["rows"]
    numeric_keys = {
        (normalize_cell(truth[r][0]), normalize_cell(truth[0][c])) for r, c in expected["numeric_cells"]
    }
    want = keyed_cells(truth)
    got = keyed_cells(predicted)

    details = []
    for key, value in want.items():
        pred = got.get(key)
        details.append(
            {
                "row": key[0],
                "col": key[1],
                "expected": value,
                "predicted": pred,
                "numeric": key in numeric_keys,
                "match": pred is not None and normalize_cell(pred) == normalize_cell(value),
            }
        )
    numeric = [d for d in details if d["numeric"]]
    return {
        "shape_expected": [len(truth), len(truth[0])],
        "shape_predicted": [len(predicted), max((len(r) for r in predicted), default=0)],
        "cells_matched": sum(d["match"] for d in details),
        "cells_total": len(details),
        "numeric_matched": sum(d["match"] for d in numeric),
        "numeric_total": len(numeric),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--image", type=Path, help="use this image instead of extracting it from the PDF")
    parser.add_argument("--runs", type=int, default=1, help="repeat to check output stability")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=ROOT / "data/eval/ocr", help="where raw responses go")
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text("utf-8"))
    image = load_image(expected, args.image)
    client = PaddleOcrVlClient(args.base_url, timeout_s=args.timeout)
    if not client.health():
        print(f"OCR service not reachable at {args.base_url}/health", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    reports, outputs = [], []

    for run in range(1, args.runs + 1):
        try:
            body, elapsed_ms = client.layout_parsing(image)
        except OcrError as exc:
            print(f"run {run}: {exc.code}: {exc.message}", file=sys.stderr)
            return 2
        (args.out / f"{stamp}-run{run}.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = parse_layout_parsing_result(body.get("result") or {}, elapsed_ms=elapsed_ms)
        tables = result.tables
        predicted = tables[0] if tables else []
        report = score(expected, predicted)
        report.update(run=run, elapsed_ms=elapsed_ms, labels=[b.label for b in result.blocks])
        reports.append(report)
        outputs.append(json.dumps(predicted, ensure_ascii=False))

        print(f"\n=== run {run}/{args.runs}  {elapsed_ms} ms  blocks={report['labels']}")
        if not predicted:
            print("no table block in the response; markdown was:\n" + (result.markdown or "(empty)"))
        else:
            print(table_to_markdown(predicted))
        print(
            f"shape {report['shape_predicted']} (expected {report['shape_expected']}), "
            f"cells {report['cells_matched']}/{report['cells_total']}, "
            f"numeric {report['numeric_matched']}/{report['numeric_total']}"
        )
        for d in report["details"]:
            if not d["match"]:
                kind = "NUMERIC " if d["numeric"] else ""
                print(f"  {kind}mismatch [{d['row']} x {d['col']}]: expected {d['expected']!r}, got {d['predicted']!r}")

    stable = len(set(outputs)) == 1
    all_numeric = all(r["numeric_matched"] == r["numeric_total"] for r in reports)
    summary = {
        "base_url": args.base_url,
        "runs": args.runs,
        "identical_outputs": stable,
        "all_numeric_correct": all_numeric,
        "latency_ms": [r["elapsed_ms"] for r in reports],
        "reports": reports,
    }
    (args.out / f"{stamp}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nsummary: numeric all correct={all_numeric}, identical across runs={stable}, "
        f"latency ms={summary['latency_ms']}\nraw responses and summary: {args.out}"
    )
    return 0 if all_numeric else 1


if __name__ == "__main__":
    sys.exit(main())

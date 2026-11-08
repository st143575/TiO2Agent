"""Render the extraction gold as review tables (Markdown).

    uv run python evals/render_gold_review.py            # -> evals/datasets/extraction_gold/REVIEW.md
    uv run python evals/render_gold_review.py --out x.md

One section per document: open notes, excluded paragraphs, then every fact with its
evidence and location, so a reviewer can check the labels against the PDF.
"""

import argparse
import json
from pathlib import Path

GOLD_DIR = Path(__file__).parent / "datasets" / "extraction_gold"
KIND_CN = {"component": "组分", "parameter": "工艺参数", "option": "可选项", "result": "检测结果"}


def _num(v) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def display(f: dict) -> str:
    if f["kind"] == "option":
        return "、".join(f["options"])
    low, high, unit = f.get("low"), f.get("high"), f.get("unit") or ""
    if low is None and high is None:
        return f.get("text") or ""
    if f.get("text") and (low is None or high is None):  # one-sided bound such as ">92"
        return f"{f['text']}{unit}"
    value = _num(low) if low == high else f"{_num(low)}-{_num(high)}"
    return f"{value}{unit}"


def where(w: dict) -> str:
    if "claim" in w:
        return f"权利要求{w['claim']}"
    if "figure" in w:
        return f"{w['figure']}（第{w['page']}页图片）"
    return f"[{w['paragraph']}]"


def render(gold: dict) -> str:
    out = [f"## {gold['doc']} {gold['title']}", "",
           f"状态：`{gold['status']}`　事实数：{len(gold['facts'])}　文件：{gold['file']}", ""]
    scope = gold["scope"]
    out += [f"范围：权利要求 {scope['claims'][0]}-{scope['claims'][-1]}；实施例段落 "
            f"[{scope['paragraphs'][0]}]-[{scope['paragraphs'][-1]}]；图表 {'、'.join(scope.get('figures', []))}", ""]
    out += ["### 原文问题与待确认", ""]
    out += [f"- [ ] **{n['type']}** {where(n['where'])}：{n['text']}" for n in gold["notes"]]
    out += ["", "### 不纳入的段落", ""]
    out += [f"- {e['paragraphs']}：{e['reason']}" for e in gold["excluded"]]
    out += ["", "### 事实", "",
            "| ✓ | id | 类型 | 主体 | 步骤 | 名称 | 限定 | 值 | 证据 | 位置 | 备注 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in gold["facts"]:
        note = f.get("issue", "")
        if f.get("requires_ocr"):
            note = ("需OCR；" + note).rstrip("；")
        cells = ["", f["id"], KIND_CN[f["kind"]], f["subject"], f["step"] or "", f["name"],
                 f["qualifier"] or "", display(f), f["evidence"], where(f["where"]), note]
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=GOLD_DIR / "REVIEW.md")
    args = parser.parse_args()
    docs = [json.loads(p.read_text("utf-8")) for p in sorted(GOLD_DIR.glob("CN*.json"))]
    head = ["# 抽取标准答案审阅表", "",
            "由 `evals/render_gold_review.py` 从 JSON 生成，改 JSON 后重新生成；不要手改本文件。", ""]
    args.out.write_text("\n".join(head) + "\n" + "\n".join(render(d) for d in docs), "utf-8")
    print(f"wrote {args.out} ({sum(len(d['facts']) for d in docs)} facts in {len(docs)} documents)")


if __name__ == "__main__":
    main()

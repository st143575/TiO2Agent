"""Prompt, tool schema and messages for LLM fact extraction.

The worked examples come from CN116694108A (claims 1 and 5, example 1 [0028]), the
development document. The three other sample patents are the evaluation set
(evals/datasets/extraction_gold), so nothing in this prompt is taken from them.

Naming follows what the rest of the system queries: materials by the name the
document uses, process parameters by a small fixed vocabulary, canonical units.
"""

import json

from tio2agent.domain import ClaimCategory
from tio2agent.llm import ToolSpec

TOOL_NAME = "record_facts"

RECORD_FACTS = ToolSpec(
    name=TOOL_NAME,
    description="一次提交从原文中抽取的全部事实；没有可抽取的内容时提交空列表。",
    parameters={
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["component", "parameter", "option"]},
                        "name": {"type": "string", "description": "物料名（按原文）或参数名"},
                        "step": {"type": "string", "description": "S1、S2…；没有步骤时省略"},
                        "qualifier": {"type": "string", "description": "区分同一步骤里重复的同名参数"},
                        "low": {"type": "number"},
                        "high": {"type": "number"},
                        "unit": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "evidence": {"type": "string", "description": "从原文逐字复制的连续片段"},
                        "paragraph": {"type": "string", "description": "证据所在段号，如 0028"},
                    },
                    "required": ["kind", "name", "evidence"],
                },
            }
        },
        "required": ["facts"],
    },
)

_EXAMPLE_CLAIM = "二氧化钛45-65份、去离子水40-50份……所述分散剂为聚乙二醇、三聚磷酸钠、六偏磷酸钠或焦磷酸钠"
_EXAMPLE_CLAIM_FACTS = [
    {"kind": "component", "name": "二氧化钛", "low": 45, "high": 65, "unit": "份", "evidence": "二氧化钛45-65份"},
    {"kind": "component", "name": "去离子水", "low": 40, "high": 50, "unit": "份", "evidence": "去离子水40-50份"},
    {"kind": "option", "name": "分散剂", "options": ["聚乙二醇", "三聚磷酸钠", "六偏磷酸钠", "焦磷酸钠"],
     "evidence": "所述分散剂为聚乙二醇、三聚磷酸钠、六偏磷酸钠或焦磷酸钠"},
]
_EXAMPLE_PARAGRAPH = (
    "[0028] S3、然后向步骤S2中得到的钛白粉浆液内分批次加入30份包膜剂，反应釜对其混合的同时进行超声处理，"
    "反应釜的温度控制在35℃，搅拌时间为15分钟，同时反应釜在搅拌6分钟时，缓慢加入7份锆盐，将pH值调节至6，"
    "在搅拌至15分钟时，缓慢加入6份铝盐，将pH值调节至5；"
)
_EXAMPLE_PARAGRAPH_FACTS = [
    {"kind": "component", "name": "包膜剂", "low": 30, "high": 30, "unit": "份", "step": "S3",
     "evidence": "30份包膜剂", "paragraph": "0028"},
    {"kind": "parameter", "name": "温度", "low": 35, "high": 35, "unit": "℃", "step": "S3",
     "evidence": "温度控制在35℃", "paragraph": "0028"},
    {"kind": "parameter", "name": "搅拌时间", "low": 15, "high": 15, "unit": "min", "step": "S3",
     "evidence": "搅拌时间为15分钟", "paragraph": "0028"},
    {"kind": "parameter", "name": "加入时间", "qualifier": "锆盐", "low": 6, "high": 6, "unit": "min", "step": "S3",
     "evidence": "在搅拌6分钟时，缓慢加入7份锆盐", "paragraph": "0028"},
    {"kind": "component", "name": "锆盐", "low": 7, "high": 7, "unit": "份", "step": "S3",
     "evidence": "7份锆盐", "paragraph": "0028"},
    {"kind": "parameter", "name": "pH", "qualifier": "加入锆盐后", "low": 6, "high": 6, "step": "S3",
     "evidence": "将pH值调节至6", "paragraph": "0028"},
    {"kind": "parameter", "name": "加入时间", "qualifier": "铝盐", "low": 15, "high": 15, "unit": "min", "step": "S3",
     "evidence": "在搅拌至15分钟时，缓慢加入6份铝盐", "paragraph": "0028"},
    {"kind": "component", "name": "铝盐", "low": 6, "high": 6, "unit": "份", "step": "S3",
     "evidence": "6份铝盐", "paragraph": "0028"},
    {"kind": "parameter", "name": "pH", "qualifier": "加入铝盐后", "low": 5, "high": 5, "step": "S3",
     "evidence": "将pH值调节至5", "paragraph": "0028"},
]


def _json(facts: list[dict]) -> str:
    return json.dumps({"facts": facts}, ensure_ascii=False)


SYSTEM_PROMPT = f"""你从中文工艺专利中抽取结构化事实，调用 {TOOL_NAME} 一次提交全部结果。

事实分三类：
- component 配方组分：物料及其用量。"55份二氧化钛"、"二氧化钛45-65份"。
- parameter 工艺参数：温度、转速、时间、pH、浓度等数值条件。
- option 可选材料："所述X为A、B或C" → name=X，options=[A, B, C]。

规则：
1. 只抽原文明确写出的数值，不推断、不补全、不换算。原文没写用量的物料不要抽。
2. evidence 必须从原文逐字复制一个连续片段，包含这条事实的全部数字；组分和可选材料的 evidence 还要包含物料名称。不要改写，不要用省略号。
3. 物料名称按原文写法；如果它就是"权利要求中的组分名称"之一，用那个名称。
4. 参数名称用：温度、转速、搅拌时间、研磨时间、加入时间、pH；其他参数用"操作+时间/温度/浓度"命名，如 干燥温度、保温时间。
5. 单位用：份、%、℃、rpm、min、h、g/L、μm。"转/分钟""r/min"记 rpm，"分钟"记 min，"小时"记 h。pH 不填单位。
6. 范围填 low 和 high；单个值 low 和 high 相同。
7. step：原文按步骤书写时填 S1、S2…（"步骤1"也记 S1）；"所述步骤2中…"填所指的步骤；没有步骤时省略。
8. qualifier 只用来区分同一步骤里重复出现的同名参数：加入时间的 qualifier 是加入的物料；加入某物料后调节的 pH，qualifier 写"加入X后"。其他情况省略。
9. 实施例原文每段以段号开头，如 [0028]；把证据所在的段号填进 paragraph。

示例一（权利要求）：
原文：{_EXAMPLE_CLAIM}
提交：{_json(_EXAMPLE_CLAIM_FACTS)}

示例二（实施例1的一段）：
原文：{_EXAMPLE_PARAGRAPH}
提交：{_json(_EXAMPLE_PARAGRAPH_FACTS)}
"""


def document_line(publication_number: str | None, title: str | None) -> str:
    return f"文档：{publication_number or '（无公开号）'} {title or ''}".rstrip()


def claim_message(doc_line: str, number: int, category: ClaimCategory, text: str) -> str:
    kind = "方法权利要求" if category is ClaimCategory.METHOD else "权利要求"
    return f"{doc_line}\n来源：claim:{number}（{kind}{number}）\n原文：\n{text}"


def embodiment_message(
    doc_line: str, subject: str, label: str, paragraphs: list[tuple[str, str]], component_names: list[str]
) -> str:
    lines = [doc_line, f"来源：{subject}（{label}）"]
    if component_names:
        lines.append("权利要求中的组分名称：" + "、".join(component_names))
    lines.append("原文：")
    lines += [f"[{pid}] {text}" for pid, text in paragraphs]
    return "\n".join(lines)


RETRY_NOTE = "上一次的输出无法解析（{problem}）。请调用 record_facts 提交，参数格式为 {{\"facts\": [...]}}。"

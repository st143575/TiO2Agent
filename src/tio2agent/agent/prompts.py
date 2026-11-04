"""System prompt and the fixed messages the loop adds."""

from tio2agent.agent.catalog import CatalogEntry
from tio2agent.agent.ledger import CITATION_GROUP
from tio2agent.llm import ChatMessage

MAX_DOCS_IN_PROMPT = 30

# system prompt
_SYSTEM = """你是钛白粉（TiO2）生产工艺技术文档的问答助手。只根据工具返回的信息回答用户问题；文档里的数值、范围、结论都必须来自工具结果，不要用自己的记忆或常识补充。

## 可用文档
{documents}

## 怎样使用工具
1. 配方用量、工艺参数（温度、转速、时间、pH 等）、测试结果：先用 query_facts；要一个实施例的完整配方或合计，用 get_formulation。
2. 事实库里没有的内容（作用、原理、背景、申请人等著录项、检测方法、对比例）：用 search_documents 检索原文。
3. 任何计算（占比、比值、差值、合计、平均、换算）都用 calculate，变量绑定来源 id，不要心算。
4. 一次没查到就换关键词、去掉过严的条件再查；最多 {max_steps} 轮工具调用。
5. 最后必须调用 final_answer 提交回答。

## 术语
- 实施例N 的 subject 写“实施例N”（或 EN）；权利要求N 写“权利要求N”。
- S1、S2、S3 是工艺步骤，不是来源 id；来源 id 是 T…（原文片段）、F…（事实）、C…（计算）。
- “份”指重量份；范围写作 45-65。

## 回答要求（final_answer）
- answer：中文，直接回答问题；每个数值或结论后面用方括号标来源 id，如“二氧化钛 55 份[F3]”“占 38.9%[C1]”。
- values：列出回答里的关键数值，每个填 source；数值要和来源一致（计算结果可以四舍五入）。
- status：answered 已回答；partial 只回答了一部分；insufficient_evidence 文档中没有依据，说明查了什么、没找到什么。
- 只能引用本轮工具返回过的 id，不要编造。"""

NUDGE_FINAL_ANSWER = "请调用 final_answer 工具提交回答：answer 中用 [F…][C…][T…] 标注来源，关键数值写进 values。"
LAST_STEP = "已到工具调用次数上限。请根据已经查到的结果立即调用 final_answer；没有依据的部分在 answer 中说明，并把 status 设为 partial 或 insufficient_evidence。"


def build_system_prompt(entries: list[CatalogEntry], *, max_steps: int) -> str:
    if entries:
        lines = [
            f"- {e.label}：{e.title or e.filename}（doc_id={e.doc_id}，事实 {e.facts} 条）"
            for e in entries[:MAX_DOCS_IN_PROMPT]
        ]
        if len(entries) > MAX_DOCS_IN_PROMPT:
            lines.append(f"- ……共 {len(entries)} 篇，用 list_documents 查看全部")
        documents = "\n".join(lines)
    else:
        documents = "（没有可用文档）"
    return _SYSTEM.format(documents=documents, max_steps=max_steps)


def history_messages(history: list[tuple[str, str]]) -> list[ChatMessage]:
    """Earlier turns as plain Q&A. Citation ids are removed: they belonged to earlier
    runs and cannot be cited again without querying the tools anew."""
    out: list[ChatMessage] = []
    for question, answer in history:
        out.append(ChatMessage.user(question))
        out.append(ChatMessage.assistant(CITATION_GROUP.sub("", answer).strip()))
    return out

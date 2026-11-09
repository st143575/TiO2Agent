"""A ChatModel that answers extraction requests from the hand-labelled gold standard.

It reads the "文档：<公开号>" and "来源：<subject>" lines of the request (and, for an
example, the paragraph ids) and submits exactly the gold facts for that place, the
way a perfect model would. Used to test that verification accepts every correct fact
and that merging and scoring work end to end.
"""

import json
import re
from pathlib import Path

from llm_fakes import call, reply
from tio2agent.llm import ChatMessage, ChatResponse

GOLD_DIR = Path(__file__).resolve().parents[1] / "evals" / "datasets" / "extraction_gold"
_FIELDS = ("kind", "name", "step", "qualifier", "low", "high", "unit", "options", "evidence")


class GoldChatModel:
    def __init__(self, gold_dir: Path = GOLD_DIR, *, edit=None) -> None:
        self.model = "gold"
        self.gold = {p.stem: json.loads(p.read_text("utf-8")) for p in gold_dir.glob("CN*.json")}
        self.edit = edit  # optional function(subject, facts) -> facts, to inject mistakes
        self.requests: list[str] = []

    def chat(self, messages: list[ChatMessage], *, tools=None, tool_choice=None) -> ChatResponse:
        text = messages[1].content or ""
        self.requests.append(text)
        doc = re.search(r"^文档：(\S+)", text, re.M).group(1)
        subject = re.search(r"^来源：(\S+?)（", text, re.M).group(1)
        paragraphs = set(re.findall(r"^\[(\d{4})\]", text, re.M))
        facts = []
        for f in self.gold.get(doc, {}).get("facts", []):
            if f["kind"] == "result" or f["subject"] != subject:
                continue
            if paragraphs and f["where"].get("paragraph") not in paragraphs:
                continue
            item = {k: f[k] for k in _FIELDS if f.get(k) is not None}
            if "paragraph" in f["where"]:
                item["paragraph"] = f["where"]["paragraph"]
            facts.append(item)
        if self.edit is not None:
            facts = self.edit(subject, facts)
        return reply(call("record_facts", facts=facts))

    def close(self) -> None:
        pass

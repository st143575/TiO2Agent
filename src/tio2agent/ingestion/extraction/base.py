"""Extractor interface and the helper that builds traceable facts."""

from dataclasses import dataclass, field
from typing import Protocol

from tio2agent.domain import Fact, FactKind, StructuredDocument


@dataclass
class ExtractionIssue:
    code: str
    message: str
    page: int | None = None


@dataclass
class ExtractionResult:
    facts: list[Fact] = field(default_factory=list)
    issues: list[ExtractionIssue] = field(default_factory=list)


class FactExtractor(Protocol):
    """
    定义了一个事实抽取器的统一接口。
    系统未来可能会有 TemperatureExtractor（专抽温度）、MaterialExtractor（专抽材料配比）。
    任何开发者想写一个新的抽取器，都必须实现这个协议：提供一个名字（name），并接受一个 StructuredDocument 大树，
    吐出 ExtractionResult。这让调用层的代码可以对所有抽取器一视同仁。
    """
    name: str

    def extract(self, doc: StructuredDocument) -> ExtractionResult: ...


@dataclass
class Source:
    """Where a span of text comes from; facts read from it inherit this provenance."""

    text: str  # original text
    subject: str
    subject_label: str
    # parameters for provenance tracking
    paragraph_ids: list[str] = field(default_factory=list)
    claim_number: int | None = None
    table_id: str | None = None
    pages: list[int] = field(default_factory=list)
    step: str | None = None


class FactBuilder:
    """Numbers facts and refuses any fact whose evidence is not in its source text."""

    def __init__(self, doc_id: str, extractor: str) -> None:
        self.doc_id = doc_id
        self.extractor = extractor
        self.facts: list[Fact] = []
        self.rejected: list[ExtractionIssue] = []

    def add(self, source: Source, kind: FactKind, name: str, evidence: str, **fields) -> Fact | None:
        # 强制要求传入的证据字符串必须绝对存在于原文中 (完全字符串匹配)，
        # 如果传入的证据字符串不在原文中，则丢弃该事实，并生成一条 EXTRACT_EVIDENCE_MISMATCH 的 Issue。
        # 此步骤用于防止大模型幻觉。
        if evidence not in source.text:
            self.rejected.append(
                ExtractionIssue(
                    "EXTRACT_EVIDENCE_MISMATCH",
                    f"{source.subject_label}：证据“{evidence}”不在原文中，已丢弃",
                    page=source.pages[0] if source.pages else None,
                )
            )
            return None
        fields.setdefault("step", source.step)
        fact = Fact(
            id=f"{self.doc_id}-f{len(self.facts) + 1:04d}",  # 如：doc123-f0001
            doc_id=self.doc_id,
            kind=kind,
            subject=source.subject,
            subject_label=source.subject_label,
            name=name,
            evidence=evidence,
            paragraph_ids=source.paragraph_ids,
            claim_number=source.claim_number,
            table_id=source.table_id,
            pages=source.pages,
            extractor=self.extractor,
            **fields,
        )
        self.facts.append(fact)
        return fact

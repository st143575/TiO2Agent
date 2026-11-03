"""Response models for facts and formulations."""

from pydantic import BaseModel

from tio2agent.domain import Fact, Formulation


class FactListOut(BaseModel):
    doc_id: str | None = None   # None for the cross-document query
    total: int
    items: list[Fact]


class FormulationListOut(BaseModel):
    doc_id: str
    total: int
    items: list[Formulation]

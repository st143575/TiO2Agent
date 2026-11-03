"""Read-only access to extracted facts: per document, as formulations, and across documents."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from tio2agent.api.deps import get_document_service, get_fact_repo
from tio2agent.api.schemas import ErrorResponse
from tio2agent.api.schemas.facts import FactListOut, FormulationListOut
from tio2agent.domain import FactKind, group_formulations
from tio2agent.ingestion import DocumentService
from tio2agent.storage import FactRepository

router = APIRouter(tags=["facts"])

ServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RepoDep = Annotated[FactRepository, Depends(get_fact_repo)]
_404 = {404: {"model": ErrorResponse, "description": "文档不存在"}}

KindQ = Annotated[FactKind | None, Query(description="component / parameter / result / option")]
SubjectQ = Annotated[str | None, Query(description="E1、实施例1、claim:1 或 权利要求1")]
NameQ = Annotated[str | None, Query(description="名称子串，如 温度、锆盐、pH")]
StepQ = Annotated[str | None, Query(description="工艺步骤，如 S1")]


@router.get(
    "/documents/{doc_id}/facts", response_model=FactListOut, responses=_404, summary="文档的结构化事实"
)
def list_document_facts(
    doc_id: str,
    service: ServiceDep,
    repo: RepoDep,
    kind: KindQ = None,
    subject: SubjectQ = None,
    name: NameQ = None,
    step: StepQ = None,
) -> FactListOut:
    service.get(doc_id)  # 404 if the document does not exist
    items = repo.query(doc_ids=[doc_id], kind=kind, subject=subject, name=name, step=step)
    return FactListOut(doc_id=doc_id, total=len(items), items=items)


@router.get(
    "/documents/{doc_id}/formulations",
    response_model=FormulationListOut,
    responses=_404,
    summary="文档的配方（按实施例 / 权利要求分组的组分）",
)
def list_document_formulations(
    doc_id: str, service: ServiceDep, repo: RepoDep, subject: SubjectQ = None
) -> FormulationListOut:
    service.get(doc_id)
    items = group_formulations(repo.query(doc_ids=[doc_id], kind=FactKind.COMPONENT, subject=subject))
    return FormulationListOut(doc_id=doc_id, total=len(items), items=items)


@router.get("/facts", response_model=FactListOut, summary="跨文档查询结构化事实")
def query_facts(
    repo: RepoDep,
    doc_id: Annotated[list[str] | None, Query(description="可重复；不传则查全部文档")] = None,
    kind: KindQ = None,
    subject: SubjectQ = None,
    name: NameQ = None,
    step: StepQ = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> FactListOut:
    items = repo.query(doc_ids=doc_id, kind=kind, subject=subject, name=name, step=step, limit=limit)
    return FactListOut(total=len(items), items=items)

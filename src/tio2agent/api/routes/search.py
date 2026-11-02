"""POST /search: hybrid retrieval over all indexed documents."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from tio2agent.api.deps import get_search_service
from tio2agent.api.schemas.search import SearchRequest, SearchResponse
from tio2agent.retrieval import SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse, summary="检索文档片段")
def search(
    body: SearchRequest,
    request: Request,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse:
    top_k = body.top_k or request.app.state.settings.search_top_k
    result = service.search(
        body.query, 
        top_k=top_k, 
        doc_ids=body.doc_ids, 
        kinds=body.kinds, 
        mode=body.mode, 
        rerank=body.rerank,
    )
    return SearchResponse.from_result(result)

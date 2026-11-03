from fastapi import Request

from tio2agent.ingestion import DocumentService
from tio2agent.retrieval import SearchService
from tio2agent.storage import ChunkRepository, FactRepository


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_chunk_repo(request: Request) -> ChunkRepository:
    return request.app.state.chunk_repo


def get_fact_repo(request: Request) -> FactRepository:
    return request.app.state.fact_repo

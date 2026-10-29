from fastapi import Request

from tio2agent.ingestion import DocumentService


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service

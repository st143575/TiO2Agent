"""POST /agent/chat and session endpoints.

Handlers are plain ``def``: a chat blocks on several LLM calls, and FastAPI runs
sync handlers in its thread pool, so other requests keep being served.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status

from tio2agent.agent import AgentService
from tio2agent.api.deps import get_agent_service
from tio2agent.api.schemas import ErrorResponse
from tio2agent.api.schemas.agent import SESSION_ID_PATTERN, ChatRequest, ChatResponse, SessionOut

router = APIRouter(prefix="/agent", tags=["agent"])

ServiceDep = Annotated[AgentService, Depends(get_agent_service)]
SessionId = Annotated[str, Path(pattern=SESSION_ID_PATTERN)]


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse, "description": "会话或文档不存在"},
        409: {"model": ErrorResponse, "description": "指定的文档正在处理中"},
        422: {"model": ErrorResponse, "description": "参数不合法"},
        502: {"model": ErrorResponse, "description": "大模型服务调用失败"},
        503: {"model": ErrorResponse, "description": "未配置大模型服务"},
    },
    summary="Agent 问答：检索、查事实、计算，返回带引用的结构化回答",
)
def chat(body: ChatRequest, service: ServiceDep) -> ChatResponse:
    result = service.chat(
        body.question, session_id=body.session_id, doc_ids=body.doc_ids, max_steps=body.max_steps
    )
    return ChatResponse.from_result(result)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionOut,
    responses={404: {"model": ErrorResponse, "description": "会话不存在"}},
    summary="查看会话的全部轮次",
)
def get_session(session_id: SessionId, service: ServiceDep) -> SessionOut:
    return SessionOut.from_record(service.get_session(session_id))


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse, "description": "会话不存在"}},
    summary="删除会话",
)
def delete_session(session_id: SessionId, service: ServiceDep) -> Response:
    service.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

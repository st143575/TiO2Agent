"""AgentService: what the API calls. Validates the request, runs the Agent, keeps the session."""

import logging
import time
from dataclasses import dataclass

from tio2agent.agent.catalog import DocumentCatalog
from tio2agent.agent.ledger import SourceLedger
from tio2agent.agent.loop import AgentRun, AgentRunner
from tio2agent.agent.prompts import build_system_prompt
from tio2agent.agent.session import SessionRecord, SessionStore, new_session_id
from tio2agent.agent.tools import ToolContext, ToolRegistry, default_tools
from tio2agent.config import Settings
from tio2agent.domain import AgentAnswer, AgentUsage, AnswerStatus, DocumentStatus
from tio2agent.errors import DocumentBusyError, DocumentNotFoundError, InvalidParameterError, SessionNotFoundError
from tio2agent.llm import ChatModel, LlmNotConfiguredError
from tio2agent.retrieval import SearchService
from tio2agent.storage import ChunkRepository, DocumentRepository, FactRepository

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    session_id: str
    turn: int
    run: AgentRun
    elapsed_ms: int


class AgentService:
    """
    负责：
    - 检查会话和 doc_ids 是否存在
    - 运行 Agent，并保存这一轮
    - 没有可用文档时，直接回答"证据不足"，不调用大模型
    """
    def __init__(
        self,
        *,
        chat_model: ChatModel | None,
        search: SearchService,
        facts: FactRepository,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        sessions: SessionStore,
        settings: Settings,
    ) -> None:
        self.chat_model = chat_model
        self.documents = documents
        self.sessions = sessions
        self.settings = settings
        self.catalog = DocumentCatalog(documents, chunks, facts)
        self.registry = ToolRegistry(default_tools(catalog=self.catalog, search=search, facts=facts))
        self.runner = (
            AgentRunner(
                chat_model, self.registry,
                max_steps=settings.agent_max_steps, tool_result_chars=settings.agent_tool_result_chars,
            )
            if chat_model is not None
            else None
        )

    @property
    def enabled(self) -> bool:
        return self.runner is not None

    def chat(
        self,
        question: str,
        *,
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        max_steps: int | None = None,
    ) -> ChatResult:
        if self.runner is None:
            raise LlmNotConfiguredError()
        if session_id is not None and not self.sessions.exists(session_id):
            raise SessionNotFoundError(details={"session_id": session_id})
        self._check_documents(doc_ids)

        start = time.perf_counter()
        entries = self.catalog.entries(doc_ids)
        if not entries:
            run = AgentRun(
                answer=AgentAnswer(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    answer="还没有可用的文档。请先上传 PDF，并等待处理完成（status 为 ready）。",
                ),
                steps=[], usage=AgentUsage(), warnings=["没有处理完成的文档，未调用大模型"],
                model=self.runner.model.model,
            )
        else:
            history = self.sessions.history(session_id, self.settings.agent_history_turns) if session_id else []
            ctx = ToolContext(ledger=SourceLedger(), doc_ids=doc_ids, labels={e.doc_id: e.label for e in entries})
            run = self.runner.run(
                question, ctx,
                system_prompt=build_system_prompt(entries, max_steps=max_steps or self.runner.max_steps),
                history=history, max_steps=max_steps,
            )
        elapsed = int((time.perf_counter() - start) * 1000)

        session_id = session_id or new_session_id()
        turn = self.sessions.append_turn(session_id, question, self.response_payload(run, elapsed))
        logger.info(
            "agent %s turn %d: %s, %d tool calls, %d LLM calls, %d ms",
            session_id, turn, run.answer.status, len(run.steps), run.usage.llm_calls, elapsed,
        )
        return ChatResult(session_id, turn, run, elapsed)

    def get_session(self, session_id: str) -> SessionRecord:
        record = self.sessions.get(session_id)
        if record is None:
            raise SessionNotFoundError(details={"session_id": session_id})
        return record

    def delete_session(self, session_id: str) -> None:
        if not self.sessions.delete(session_id):
            raise SessionNotFoundError(details={"session_id": session_id})

    @staticmethod
    def response_payload(run: AgentRun, elapsed_ms: int) -> dict:
        """What is stored per turn (and returned by GET /agent/sessions/{id})."""
        return {
            "answer": run.answer.model_dump(mode="json"),
            "steps": [s.model_dump(mode="json") for s in run.steps],
            "usage": run.usage.model_dump(mode="json"),
            "warnings": run.warnings,
            "model": run.model,
            "elapsed_ms": elapsed_ms,
        }

    def _check_documents(self, doc_ids: list[str] | None) -> None:
        if doc_ids is None:
            return
        if not doc_ids:
            raise InvalidParameterError("doc_ids 不能为空列表；不限文档时不要传这个字段")
        for doc_id in doc_ids:
            record = self.documents.get(doc_id)
            if record is None:
                raise DocumentNotFoundError(details={"id": doc_id})
            if not record.status.is_terminal:
                raise DocumentBusyError(details={"id": doc_id, "status": record.status})
            if record.status is DocumentStatus.FAILED:
                raise InvalidParameterError(f"文档 {doc_id} 处理失败，不能用于问答", details={"id": doc_id})

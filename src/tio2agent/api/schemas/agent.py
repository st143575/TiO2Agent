"""Request/response models for /agent."""

from datetime import datetime

from pydantic import BaseModel, Field

from tio2agent.agent import AgentService, ChatResult, SessionRecord
from tio2agent.domain import AgentAnswer, AgentStep, AgentUsage

SESSION_ID_PATTERN = r"^ses_[0-9a-f]{12}$"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(None, pattern=SESSION_ID_PATTERN, description="继续已有会话；不传则新建")
    doc_ids: list[str] | None = Field(None, max_length=50, description="只用这些文档回答；不传则用全部已处理文档")
    max_steps: int | None = Field(None, ge=1, le=20, description="最多调用大模型的次数，默认 TIO2_AGENT_MAX_STEPS")


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    model: str
    answer: AgentAnswer
    steps: list[AgentStep]
    usage: AgentUsage
    warnings: list[str]
    elapsed_ms: int

    @classmethod
    def from_result(cls, result: ChatResult) -> "ChatResponse":
        run = result.run
        return cls(
            session_id=result.session_id, turn=result.turn, model=run.model, answer=run.answer,
            steps=run.steps, usage=run.usage, warnings=run.warnings, elapsed_ms=result.elapsed_ms,
        )


class TurnOut(BaseModel):
    index: int
    question: str
    created_at: datetime
    answer: AgentAnswer
    steps: list[AgentStep]
    usage: AgentUsage
    warnings: list[str]
    model: str


class SessionOut(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    turns: list[TurnOut]

    @classmethod
    def from_record(cls, record: SessionRecord) -> "SessionOut":
        turns = [
            TurnOut(
                index=t.index, question=t.question, created_at=t.created_at,
                answer=t.response["answer"], steps=t.response.get("steps", []),
                usage=t.response.get("usage", {}), warnings=t.response.get("warnings", []),
                model=t.response.get("model", ""),
            )
            for t in record.turns
        ]
        return cls(id=record.id, created_at=record.created_at, updated_at=record.updated_at, turns=turns)


__all__ = ["AgentService", "ChatRequest", "ChatResponse", "SessionOut", "TurnOut"]

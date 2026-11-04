"""Multi-step Agent: an LLM plans, calls tools over the fact store and the retrieval
index, computes with a safe calculator, and returns a checked, structured answer."""

from tio2agent.agent.calculator import CalculationError, evaluate
from tio2agent.agent.catalog import CatalogEntry, DocumentCatalog
from tio2agent.agent.final_answer import FINAL_ANSWER, FinalAnswerArgs, FinalValue, build_answer
from tio2agent.agent.ledger import SourceLedger, SourceRecord
from tio2agent.agent.loop import AgentRun, AgentRunner
from tio2agent.agent.service import AgentService, ChatResult
from tio2agent.agent.session import SessionRecord, SessionStore, TurnRecord
from tio2agent.agent.tools import Tool, ToolContext, ToolRegistry, ToolResult, default_tools

__all__ = [
    "FINAL_ANSWER",
    "AgentRun",
    "AgentRunner",
    "AgentService",
    "CalculationError",
    "CatalogEntry",
    "ChatResult",
    "DocumentCatalog",
    "FinalAnswerArgs",
    "FinalValue",
    "SessionRecord",
    "SessionStore",
    "SourceLedger",
    "SourceRecord",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "TurnRecord",
    "build_answer",
    "default_tools",
    "evaluate",
]

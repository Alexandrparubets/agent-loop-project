from dataclasses import dataclass, field
from typing import Any

replan_count: int = 0
max_replans: int = 1


@dataclass
class AgentStep:
    step_id: int
    thought: str | None = None
    action: str | None = None
    tool_call: dict[str, Any] | None = None
    observation: str | None = None
    status: str = "success"
    error: str | None = None


@dataclass
class AgentState:
    query: str
    plan: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    memory_hits: list[str] = field(default_factory=list)
    final_answer: str | None = None
    needs_replan: bool = False
    final_answer: str | None = None
    replan: list[str] | None = None
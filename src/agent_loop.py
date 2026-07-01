from dataclasses import dataclass, field
from typing import Any
from src.tools import TOOLS
from src.context import ToolContext
from sentence_transformers import SentenceTransformer
from src.faiss_index import load_index, load_chunks
from src.memory import load_memory
from src.llm_decision import llm_decide_next_action
from src.decision_context import build_decision_context
from src.llm_openai import openai_llm
from src.summary_context import build_summary_context
from src.progress import calculate_progress
from src.state_manager import update_no_progress_count, update_blocked_actions, get_available_actions




@dataclass
class AgentStep:
    step_id: int
    thought: str | None = None
    action: str | None = None
    tool_call: dict[str, Any] | None = None
    observation: Any = None
    similarity_to_previous: float | None = None
    is_new_information: bool | None = None
    no_progress_count: int = 0
    status: str = "success"
    error: str | None = None
    blocked_actions: set[str] | None = None
    available_actions: list[str] = field(default_factory=list)



@dataclass
class AgentState:
    query: str
    steps: list[AgentStep] = field(default_factory=list)

    step_count: int = 0
    max_steps: int = 20
    min_retrieval_score: float = 0.75

    is_done: bool = False
    stop_reason: str | None = None

    
    no_progress_counts: dict[str, int] = field(default_factory=dict)
    no_progress_limit: int = 3
    blocked_actions: set[str] = field(default_factory=set)

    final_answer: str | None = None


def mock_llm(prompt: str) -> str:
    if "Last action:\nmemory_search" in prompt:
        return """
Thought: Memory search was completed. I can answer now.
Action: final_answer
"""

    if "Last action:\nretrieval" in prompt:
        return """
Thought: Retrieval was already used. Need to search memory.
Action: memory_search
"""

    return """
Thought: Need to search relevant context.
Action: retrieval
"""

def build_tool_call(action: str, state: AgentState) -> dict[str, Any] | None:
    if action == "final_answer":
        return None

    if action == "retrieval":
        return {
            "tool_name": "retrieval",
            "args": {"query": state.query},
        }

    if action == "memory_search":
        return {
            "tool_name": "memory_search",
            "args": {"query": state.query},
        }

    raise ValueError(f"Cannot build tool_call for action: {action}")


# def decide_next_action(state: AgentState) -> dict[str, Any]:
#     if not state.steps:
#         return {
#             "thought": "Need to search relevant context",
#             "action": "retrieval",
#             "tool_call": {
#                 "tool_name": "retrieval",
#                 "args": {"query": state.query},
#             },
#         }

#     last_step = state.steps[-1]

#     if last_step.action == "retrieval":
#         best_score = last_step.observation.get("best_score", 0)

#         if best_score >= state.min_retrieval_score:
#             return {
#                 "thought": "Retrieval score is high enough. I can answer.",
#                 "action": "final_answer",
#                 "tool_call": None,
#             }

#         return {
#             "thought": "Retrieval score is too low. Need memory search.",
#             "action": "memory_search",
#             "tool_call": {
#                 "tool_name": "memory_search",
#                 "args": {"query": state.query},
#             },
#         }
    
#     if last_step.action == "memory_search":
#         return {
#             "thought": "Memory search completed. I can answer with available context.",
#             "action": "final_answer",
#             "tool_call": None,
#         }
    
#     return {
#     "thought": "No useful next action. Stop the loop.",
#     "action": "final_answer",
#     "tool_call": None,
#     }



def execute_tool(tool_call: dict[str, Any] | None,
    context: ToolContext,) -> Any:
    if tool_call is None:
        return None

    tool_name = tool_call.get("tool_name")
    args = tool_call.get("args", {})
    

    tool_func = TOOLS.get(tool_name)

    if tool_func is None:
        return {
            "error": f"Unknown tool: {tool_name}"
        }

    return tool_func(context=context, **args)
    


def synthesize_response(state: AgentState) -> str:
    all_results = []

    for step in state.steps:
        observation = step.observation

        if not isinstance(observation, dict):
            continue

        all_results.extend(
            observation.get("results", [])
        )

    if not all_results:
        return "I don't have enough information to answer."

    best_result = max(
        all_results,
        key=lambda item: item.get("score", 0),
    )

    return best_result["text"]
    


def run_agent_loop(query: str, context: ToolContext, llm, max_steps: int = 20) -> AgentState:
    state = AgentState(query=query, max_steps=max_steps)

    while not state.is_done and state.step_count < state.max_steps:
        state.step_count += 1

        decision_context = build_decision_context(state)

        available_actions = get_available_actions(state)

        decision = llm_decide_next_action(
            llm=llm,
            available_actions = available_actions,
            **decision_context,
        )
        action = decision.get("action")

        tool_call = build_tool_call(
            action=decision["action"],
            state=state,
        )

        observation = execute_tool(
            tool_call,
            context=context,
        )

        summary_context = build_summary_context(
            state=state,
            current_action=decision.get("action"),
            current_observation=observation,
        )

        progress = calculate_progress(
            model=context.model,
            previous_summary=summary_context["previous_summary"],
            current_summary=summary_context["current_summary"],
        )

        update_no_progress_count(
            state=state,
            action=action,
            progress=progress,
        )

        update_blocked_actions(
            state=state,
            action=action,
        )

        step = AgentStep(
            step_id=state.step_count,
            thought=decision.get("thought"),
            action=action,
            tool_call=tool_call,
            observation=observation,
            similarity_to_previous=progress["similarity_to_previous"],
            is_new_information=progress["is_new_information"],
            no_progress_count=state.no_progress_counts.get(action, 0),
            blocked_actions=set(state.blocked_actions),
            available_actions=available_actions.copy(),
        )

        state.steps.append(step)

        

        if decision.get("action") == "final_answer":
            state.is_done = True
            state.stop_reason = "final_answer"
            state.final_answer = synthesize_response(state)

    if not state.is_done:
        state.is_done = True
        state.stop_reason = "max_steps"

    return state


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = load_index()
    chunks = load_chunks()

    load_memory()

    context = ToolContext(
        model=model,
        index=index,
        chunks=chunks,
    )
    state = run_agent_loop(
        "What is Feature Stop?",
        context=context,
        llm=openai_llm,
    )

    print("Final answer:", state.final_answer)
    print("Stop reason:", state.stop_reason)
    print("Steps:")

    for step in state.steps:
        print(step)
        print("-" * 60)
        print(f"Step {step.step_id}")
        print(f"Action: {step.action}")
        print(f"Available: {step.available_actions}")
        print(f"Blocked: {step.blocked_actions}")
        print(f"No progress: {step.no_progress_count}")
        print("-" * 60)
        

    #----------------------------------------------------python -m src.agent_loop
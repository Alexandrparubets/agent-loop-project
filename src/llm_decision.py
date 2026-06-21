from typing import Any

from src.prompts.decision_prompt import build_decision_prompt


def parse_llm_decision(text: str) -> dict[str, str | None]:
    thought = None
    action = None

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("Thought:"):
            thought = line.replace("Thought:", "", 1).strip()

        if line.startswith("Action:"):
            action = line.replace("Action:", "", 1).strip()

    return {
        "thought": thought,
        "action": action,
    }


def validate_decision(
    decision: dict[str, str | None],
    available_actions: list[str],
) -> None:
    thought = decision.get("thought")
    action = decision.get("action")

    if not thought:
        raise ValueError("LLM decision has empty Thought.")

    if not action:
        raise ValueError("LLM decision has empty Action.")

    if action not in available_actions:
        raise ValueError(
            f"Invalid action: {action}. "
            f"Available actions: {available_actions}"
        )


def llm_decide_next_action(
    llm: Any,
    query: str,
    last_action: str | None,
    last_observation: dict[str, Any] | None,
    step_count: int,
    max_steps: int,
    available_actions: list[str],
) -> dict[str, str | None]:
    prompt = build_decision_prompt(
        query=query,
        last_action=last_action,
        last_observation=last_observation,
        step_count=step_count,
        max_steps=max_steps,
        available_actions=available_actions,
    )

    raw_response = llm(prompt)
    
    decision = parse_llm_decision(str(raw_response))

    validate_decision(
        decision=decision,
        available_actions=available_actions,
    )

    return decision
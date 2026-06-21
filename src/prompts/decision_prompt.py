from typing import Any


def build_decision_prompt(
    query: str,
    last_action: str | None,
    last_observation: dict[str, Any] | None,
    step_count: int,
    max_steps: int,
    available_actions: list[str],
) -> str:
    actions_text = "\n".join(
        f"- {action}" for action in available_actions
    )

    if last_observation is None:
        context_text = "This is the first step. No previous observation exists."
    else:
        context_text = f"""
Last action:
{last_action}

Last observation summary:
{last_observation}
"""

    return f"""
You are a decision module inside an AI agent loop.

Your task is to choose the next action.

User query:
{query}

Current step:
{step_count} / {max_steps}

Available actions:
{actions_text}

Context:
{context_text}

Rules:
- Choose exactly one action from the available actions.
- Do not invent new actions.
- Return only this format:

Thought: <short reasoning>
Action: <one available action>
""".strip()
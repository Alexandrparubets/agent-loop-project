# src/summary_context.py

from typing import Any


def build_summary_context(
    state,
    current_action: str | None,
    current_observation: dict[str, Any] | None,
) -> dict[str, str | None]:
    current_summary = None

    if isinstance(current_observation, dict):
        current_summary = current_observation.get("summary")

    previous_summary = None

    for step in reversed(state.steps):
        if step.action != current_action:
            continue

        previous_observation = step.observation

        if isinstance(previous_observation, dict):
            previous_summary = previous_observation.get("summary")
            break

    return {
        "previous_summary": previous_summary,
        "current_summary": current_summary,
    }
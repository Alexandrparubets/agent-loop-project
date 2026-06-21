from src.observation_snapshot import (
    build_observation_snapshot,
)


def build_decision_context(state):
    if not state.steps:
        return {
            "query": state.query,
            "last_action": None,
            "last_observation": None,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
        }

    last_step = state.steps[-1]

    return {
        "query": state.query,
        "last_action": last_step.action,
        "last_observation": build_observation_snapshot(
            last_step.observation
        ),
        "step_count": state.step_count,
        "max_steps": state.max_steps,
    }
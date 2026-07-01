from src.tools import TOOLS


def update_no_progress_count(
    state,
    action: str,
    progress: dict,
) -> None:
    count = state.no_progress_counts.get(action, 0)

    if progress["is_new_information"]:
        state.no_progress_counts[action] = 0
    else:
        state.no_progress_counts[action] = count + 1


def update_blocked_actions(
    state,
    action: str,
) -> None:
    count = state.no_progress_counts.get(action, 0)

    if count < state.no_progress_limit:
        return

    if action == "final_answer":
        return

    state.blocked_actions.add(action)
    state.no_progress_counts[action] = 0


def get_available_actions(state) -> list[str]:
    search_actions = [
        action
        for action in TOOLS.keys()
        if action not in state.blocked_actions
    ]

    if search_actions:
        return search_actions

    return ["final_answer"]
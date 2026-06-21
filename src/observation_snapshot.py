from typing import Any


def build_observation_snapshot(
    observation: Any,
) -> dict[str, Any] | None:

    if observation is None:
        return None

    if not isinstance(observation, dict):
        return {
            "summary": str(observation)
        }

    results = observation.get("results", [])

    best_score = observation.get(
        "best_score",
        0,
    )

    summary = None

    if results:
        summary = results[0].get("text")

    return {
        "best_score": best_score,
        "summary": summary,
    }
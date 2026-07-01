from typing import Any

from sklearn.metrics.pairwise import cosine_similarity


def calculate_progress(
    model: Any,
    previous_summary: str | None,
    current_summary: str | None,
    similarity_threshold: float = 0.95,
) -> dict[str, Any]:
    if not current_summary:
        return {
            "similarity_to_previous": None,
            "is_new_information": False,
        }

    if not previous_summary:
        return {
            "similarity_to_previous": None,
            "is_new_information": True,
        }

    previous_embedding = model.encode([previous_summary])
    current_embedding = model.encode([current_summary])

    similarity = cosine_similarity(
        previous_embedding,
        current_embedding,
    )[0][0]

    is_new_information = bool(similarity < similarity_threshold)

    return {
        "similarity_to_previous": float(similarity),
        "is_new_information": is_new_information,
    }
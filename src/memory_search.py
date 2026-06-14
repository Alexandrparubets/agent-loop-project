#memory_search.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.memory import get_memory


def search_memory(query, model, top_k=1):
    memory = get_memory()

    # только записи с summary
    memory_items = [
        item
        for item in memory
        if item.get("summary")
    ]

    if not memory_items:
        return []

    memory_texts = [
        item["summary"]
        for item in memory_items
    ]

    # embeddings memory
    memory_embeddings = model.encode(memory_texts)

    # embedding query
    query_embedding = model.encode([query])

    # similarity
    sims = cosine_similarity(
        query_embedding,
        memory_embeddings
    )[0]

    # top memory
    sorted_idx = np.argsort(sims)[::-1][:top_k]

    results = []

    for i in sorted_idx:
        results.append({
            "memory": memory_items[i],
            "score": float(sims[i])
        })

    return results
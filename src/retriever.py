#retriever.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query: str, model, index, chunks, top_k=2, fetch_k=5):
    queries = expand_query(query)

    all_candidates = []

    for q in queries:
        q_emb = model.encode([q])
        q_vec = np.array(q_emb).astype("float32")

        distances, indices = index.search(q_vec, fetch_k)

        for idx in indices[0]:
            all_candidates.append(idx)

    # уникальные кандидаты
    unique_indices = list(set(all_candidates))

    candidate_chunks = [chunks[i] for i in unique_indices]
    candidate_texts = [c["text"] for c in candidate_chunks]
    candidate_embeddings = model.encode(candidate_texts)
    

    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, candidate_embeddings)[0]

    sorted_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in sorted_idx:
        results.append({
        "text": candidate_chunks[i]["text"],
        "source": candidate_chunks[i]["source"],
        "score": float(sims[i])
    })

    return results


def expand_query(query: str) -> list[str]:
    expansions = [
        query,
        query + " data",
        query + " machine learning",
        query.replace("what is", "explain"),
        query.replace("what is", "definition of"),
        query.replace("model registry", "ml model registry"),
        query.replace("model registry", "model versioning"),
        query.replace("model registry", "model management"),
    ]
    return list(set(expansions))
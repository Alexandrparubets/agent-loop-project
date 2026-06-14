# tools.py

from typing import Any
from src.retriever import retrieve
from src.context import ToolContext
from src.memory_search import search_memory


def retrieval_tool(
    context: ToolContext,
    query: str,
    top_k: int = 2,
) -> dict[str, Any]:
    results = retrieve(
        query=query,
        model=context.model,
        index=context.index,
        chunks=context.chunks,
        top_k=top_k,
    )

    best_score = 0.0

    if results:
        best_score = max(item["score"] for item in results)

    return {
        "results": results,
        "best_score": best_score,
    }


def memory_search_tool(
    context: ToolContext,
    query: str,
    top_k: int = 1,
) -> dict[str, Any]:
    results = search_memory(
        query=query,
        model=context.model,
        top_k=top_k,
    )

    best_score = max(
        (item["score"] for item in results),
        default=0.0,
    )

    formatted_results = []

    for item in results:
        memory = item["memory"]

        formatted_results.append({
            "text": memory.get("summary", ""),
            "score": item["score"],
            "source": "memory",
            "raw_memory": memory,
        })

    return {
        "results": formatted_results,
        "best_score": best_score,
    }


TOOLS = {
    "retrieval": retrieval_tool,
    "memory_search": memory_search_tool,
}
from src.tools import calculator_tool, retrieval_tool, summarizer_tool
from src.memory_search import search_memory


def execute_tool(tool_call: dict, model, index, chunks):
    tool_name = tool_call["tool"]
    arguments = tool_call.get("arguments", {})

    if tool_name == "calculator":
        expression = arguments["expression"]
        return calculator_tool(expression)

    if tool_name == "retrieval":
        query = arguments["query"]
        return retrieval_tool(query, model, index, chunks)

    if tool_name == "summarizer":
        texts = arguments["texts"]
        return summarizer_tool(texts)

    if tool_name == "memory_search":
        query = arguments["query"]
        top_k = arguments.get("top_k", 1)
        return search_memory(query, model, top_k=top_k)

    return {
        "error": f"Unknown tool: {tool_name}"
    }
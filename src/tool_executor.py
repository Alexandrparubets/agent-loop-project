from typing import Any

from src.tools import TOOLS
from src.context import ToolContext


def execute_tool(
    tool_call: dict[str, Any] | None,
    context: ToolContext,
) -> Any:
    if tool_call is None:
        return None

    tool_name = tool_call.get("tool_name")
    arguments = tool_call.get("args", {})

    tool_func = TOOLS.get(tool_name)

    if tool_func is None:
        return {
            "error": f"Unknown tool: {tool_name}"
        }

    return tool_func(
        context=context,
        **arguments,
    )
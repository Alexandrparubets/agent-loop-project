from dataclasses import dataclass
from typing import Any


@dataclass
class ToolContext:
    model: Any
    index: Any
    chunks: list[dict[str, Any]]
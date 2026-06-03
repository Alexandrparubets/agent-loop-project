import json
from pathlib import Path


MEMORY_FILE = "data/memory.json"

memory = []


def load_memory():
    global memory

    path = Path(MEMORY_FILE)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            memory = json.load(f)

    else:
        memory = []


def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_to_memory(item):
    memory.append(item)

    save_memory()


def get_memory():
    return memory


def get_last_memory():
    if memory:
        return memory[-1]

    return None
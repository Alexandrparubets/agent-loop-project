from src.llm_openai import openai_llm

print(
    openai_llm(
        "Return exactly:\nThought: test\nAction: retrieval"
    )
)
import re

from src.retriever import retrieve
from openai import OpenAI

client = OpenAI()


def summarizer_tool(texts):
    context = "\n\n".join(texts)

    prompt = f"""
Summarize the following text:

{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def retrieval_tool(query, model, index, chunks):
    results = retrieve(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=2
    )

    best_score = results[0]["score"] if results else 0

    return {
        "results": results,
        "best_score": best_score
    }


def calculator_tool(query):
    # оставляем только цифры и операторы
    expression = re.sub(r"[^0-9+\-*/(). ]", "", query)

    try:
        result = eval(expression)
        return {"result": result}

    except Exception:
        return {"error": "invalid expression"}
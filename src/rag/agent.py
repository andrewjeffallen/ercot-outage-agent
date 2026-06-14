"""
RAG agent: retrieves relevant chunks and calls the LLM to answer.
"""

from __future__ import annotations

from typing import Generator

from src.inference.client import get_client
from src.rag.store import VectorStore

SYSTEM_PROMPT = """You are an expert on ERCOT (Electric Reliability Council of Texas) grid operations.
You answer questions about grid conditions, outages, transmission constraints, and market conditions
using the provided context from official ERCOT data sources.

Guidelines:
- Be specific and cite the source/date of information when available.
- If the context doesn't contain enough information to answer confidently, say so.
- Use correct ERCOT terminology (LMP, NOPR, TOLAR, hub prices, etc.).
- Keep answers concise and factual.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source_label = f"[{i}] {meta.get('title', 'Unknown')} ({meta.get('published_at', 'N/A')})"
        context_blocks.append(f"{source_label}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    return f"""Use the following ERCOT data context to answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


class GridAgent:
    def __init__(self):
        self.store = VectorStore()
        self.client = get_client()

    def answer(self, question: str, top_k: int = 5) -> str:
        """Blocking RAG answer."""
        chunks = self.store.query(question, top_k=top_k)
        if not chunks:
            return "No relevant ERCOT data found. Try running the ingestion script first."
        prompt = build_prompt(question, chunks)
        return self.client.complete(prompt, system=SYSTEM_PROMPT)

    def stream_answer(self, question: str, top_k: int = 5) -> Generator[str, None, None]:
        """Streaming RAG answer — yields text chunks."""
        chunks = self.store.query(question, top_k=top_k)
        if not chunks:
            yield "No relevant ERCOT data found. Try running the ingestion script first."
            return
        prompt = build_prompt(question, chunks)
        yield from self.client.stream(prompt, system=SYSTEM_PROMPT)

    def get_context(self, question: str, top_k: int = 5) -> list[dict]:
        """Expose retrieved chunks for inspection/debugging."""
        return self.store.query(question, top_k=top_k)

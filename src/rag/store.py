"""
RAG pipeline: chunk → embed → store → retrieve.

Uses ChromaDB as the local vector store.
Embeddings are generated via Ollama (nomic-embed-text).
"""

from __future__ import annotations

import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from src.ingestion.ercot import GridDocument

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ercot_grid")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K = int(os.getenv("TOP_K", 5))


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_document(doc: GridDocument) -> list[dict]:
    """
    Split a GridDocument into overlapping text chunks.
    Returns list of dicts with 'text' and 'metadata'.
    
    Simple character splitter for now — replace with a recursive or
    semantic splitter as you experiment.
    """
    text = doc.content
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": doc.source,
                "doc_type": doc.doc_type,
                "title": doc.title,
                "chunk_idx": idx,
                "published_at": doc.published_at.isoformat() if doc.published_at else "",
                **{k: str(v) for k, v in doc.metadata.items()},
            },
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1

    return chunks


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None  # lazy init

    def _get_embedder(self):
        """Lazy-load the embedding function. Uses Ollama."""
        if self._embedder is None:
            from src.inference.client import OllamaClient
            ollama = OllamaClient()
            # Wrap as a callable ChromaDB embedding function
            class OllamaEmbedder:
                def __call__(self, input: list[str]) -> list[list[float]]:
                    return [ollama.embed(text) for text in input]
            self._embedder = OllamaEmbedder()
        return self._embedder

    def add_documents(self, docs: list[GridDocument]) -> int:
        """Chunk all docs and upsert into ChromaDB. Returns total chunks added."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_document(doc))

        if not all_chunks:
            return 0

        embedder = self._get_embedder()
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder(texts)
        ids = [f"{c['metadata']['source']}::chunk_{c['metadata']['chunk_idx']}" for c in all_chunks]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in all_chunks],
        )
        return len(all_chunks)

    def query(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for a question.
        Returns list of dicts with 'text', 'metadata', 'distance'.
        """
        embedder = self._get_embedder()
        q_embedding = embedder([question])[0]

        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({"text": text, "metadata": meta, "distance": dist})

        return chunks

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Wipe and recreate the collection. Useful during development."""
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

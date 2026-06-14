"""
Local inference client using Ollama.

Provides LLM completions and embeddings via Ollama running locally.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Generator

import httpx
from dotenv import load_dotenv

load_dotenv()


# ── Base ──────────────────────────────────────────────────────────────────────

class InferenceClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Blocking completion."""
        ...

    @abstractmethod
    def stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        """Streaming completion — yields text chunks."""
        ...


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaClient(InferenceClient):
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def _build_payload(self, prompt: str, system: str, stream: bool) -> dict:
        payload: dict = {"model": self.model, "stream": stream}
        if system:
            payload["messages"] = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        else:
            payload["prompt"] = prompt
        return payload

    def complete(self, prompt: str, system: str = "") -> str:
        payload = self._build_payload(prompt, system, stream=False)
        endpoint = "chat" if system else "generate"
        resp = httpx.post(
            f"{self.base_url}/api/{endpoint}",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # /api/chat returns message.content; /api/generate returns response
        return data.get("message", {}).get("content") or data.get("response", "")

    def stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        payload = self._build_payload(prompt, system, stream=True)
        endpoint = "chat" if system else "generate"
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/{endpoint}",
            json=payload,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                import json
                chunk = json.loads(line)
                text = (
                    chunk.get("message", {}).get("content")
                    or chunk.get("response", "")
                )
                if text:
                    yield text
                if chunk.get("done"):
                    break

    def embed(self, text: str) -> list[float]:
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ── Factory ───────────────────────────────────────────────────────────────────

def get_client() -> InferenceClient:
    """Return the Ollama inference client."""
    return OllamaClient()

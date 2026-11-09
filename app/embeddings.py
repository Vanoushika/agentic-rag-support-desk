"""Embedding providers behind a small interface.

The default provider is :class:`HashingEmbedder`, a deterministic,
dependency-free embedder based on signed feature hashing of word tokens.
It needs no API key and no network, which makes tests and demos fully
reproducible offline. If ``OPENAI_API_KEY`` is set, :func:`get_embedder`
tries to return an OpenAI-backed embedder via LangChain and falls back
to the hashing embedder if that is unavailable.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List, Protocol, Sequence


class Embedder(Protocol):
    """Anything that can turn texts into fixed-size vectors."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic signed-hashing embedder (bag of words).

    Tokens are hashed into a fixed-dimension space; documents that share
    vocabulary with a query land close together under cosine similarity.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = digest % self.dim
            sign = 1.0 if (digest >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Thin wrapper around LangChain's OpenAI embeddings (lazy import)."""

    def __init__(self, model: str | None = None) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._impl = OpenAIEmbeddings(
            model=model or os.environ.get(
                "SUPPORT_DESK_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        )

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self._impl.embed_documents(list(texts))


def get_embedder() -> Embedder:
    """Return the best available embedder for this environment."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception:
            pass
    return HashingEmbedder()

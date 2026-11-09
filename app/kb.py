"""Knowledge base: document loading, chunking and FAISS retrieval.

Documents are plain Markdown files. They are chunked with a small
paragraph-aware splitter, embedded through the :mod:`app.embeddings`
interface, and indexed with FAISS (inner-product over normalised
vectors = cosine similarity). If FAISS is not installed, retrieval
falls back to an equivalent NumPy cosine search so the app still runs.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence

import numpy as np

from .embeddings import Embedder

try:  # FAISS is the primary index; NumPy is the fallback.
    import faiss

    _HAS_FAISS = True
except Exception:  # pragma: no cover - exercised only without faiss
    faiss = None
    _HAS_FAISS = False


def chunk_text(text: str, max_chars: int = 600, overlap: int = 80) -> List[str]:
    """Split text into paragraph-aware chunks of at most ``max_chars``.

    Paragraphs are packed together until the size limit is reached.
    A single oversized paragraph is hard-split with a small overlap so
    no content is lost at chunk boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                chunks.append(para[start : start + max_chars])
                start += max_chars - overlap
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def load_documents(docs_dir: str) -> List[Dict[str, str]]:
    """Load every Markdown file in ``docs_dir`` as {source, text}."""
    documents = []
    for name in sorted(os.listdir(docs_dir)):
        if name.endswith(".md"):
            with open(os.path.join(docs_dir, name), encoding="utf-8") as fh:
                documents.append({"source": name, "text": fh.read()})
    return documents


class KnowledgeBase:
    """An in-memory chunk index with FAISS (or NumPy) search."""

    def __init__(self, chunks: List[Dict[str, str]], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("KnowledgeBase needs at least one chunk")
        self.chunks = chunks
        self.embedder = embedder
        vectors = np.array(
            embedder.embed([c["text"] for c in chunks]), dtype="float32"
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = vectors / norms
        self._index = None
        if _HAS_FAISS:
            index = faiss.IndexFlatIP(self._vectors.shape[1])
            index.add(self._vectors)
            self._index = index

    @classmethod
    def from_docs(cls, docs_dir: str, embedder: Embedder) -> "KnowledgeBase":
        chunks: List[Dict[str, str]] = []
        for doc in load_documents(docs_dir):
            for piece in chunk_text(doc["text"]):
                chunks.append({"source": doc["source"], "text": piece})
        return cls(chunks, embedder)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Return the top-``k`` chunks for ``query`` with cosine scores."""
        q = np.array(self.embedder.embed([query]), dtype="float32")
        q_norm = np.linalg.norm(q) or 1.0
        q = q / q_norm
        k = min(k, len(self.chunks))
        if self._index is not None:
            scores, idxs = self._index.search(q, k)
            pairs = [(float(s), int(i)) for s, i in zip(scores[0], idxs[0])]
        else:  # NumPy fallback: identical cosine ranking
            sims = (self._vectors @ q[0]).tolist()
            order = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
            pairs = [(float(sims[i]), i) for i in order]
        return [
            {
                "source": self.chunks[i]["source"],
                "text": self.chunks[i]["text"],
                "score": round(score, 4),
            }
            for score, i in pairs
            if i >= 0
        ]

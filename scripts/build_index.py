"""Build a persistent FAISS index from the Markdown knowledge base.

Usage:
    python scripts/build_index.py [--docs docs] [--out index]

Writes ``faiss.index`` and ``chunks.json`` (chunk text + source) into
the output directory. Uses the embedder selected by
:func:`app.embeddings.get_embedder` — the deterministic offline
embedder unless OPENAI_API_KEY is set. If FAISS is not installed, the
script reports that and exits without writing (the app itself falls
back to NumPy search at runtime).

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.embeddings import get_embedder
from app.kb import chunk_text, load_documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="docs", help="knowledge base directory")
    parser.add_argument("--out", default="index", help="output directory")
    args = parser.parse_args()

    try:
        import faiss
    except ImportError:
        print("faiss is not installed; skipping persistent index build.")
        print("The app will use its NumPy fallback search at runtime.")
        return 0

    chunks = []
    for doc in load_documents(args.docs):
        for piece in chunk_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": piece})

    embedder = get_embedder()
    vectors = np.array(embedder.embed([c["text"] for c in chunks]), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(args.out, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out, "faiss.index"))
    with open(os.path.join(args.out, "chunks.json"), "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, indent=2)

    print(f"Indexed {len(chunks)} chunks from {args.docs} into {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

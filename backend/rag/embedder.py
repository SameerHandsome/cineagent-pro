"""
backend/rag/embedder.py
────────────────────────
Generates sentence embeddings using sentence-transformers (runs locally,
no API cost).  Used by the indexer and retriever.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Small, fast model — 384-dim vectors, runs on CPU
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> list[float]:
    model = get_embedder()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()

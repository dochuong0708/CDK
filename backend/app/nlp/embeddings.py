"""Embedding câu tiếng Việt dùng chung cho phân loại hình thức và RAG fact-check.

Mặc định dùng bi-encoder tiếng Việt (sentence-transformers) nếu thư viện đã được cài;
nếu không, rơi về TF-IDF n-gram ký tự để vẫn tính được cosine similarity.
"""

import math
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DISABLED_VALUES = {"", "0", "false", "off", "none", "disabled"}


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _segment(text: str) -> str:
    """vietnamese-bi-encoder được huấn luyện trên văn bản đã tách từ (PyVi)."""
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text)
    except Exception:  # noqa: BLE001 - thư viện/model tùy chọn
        return text


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            [_segment(text) for text in texts], normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, vector)) for vector in vectors]


class TfidfEmbedder:
    """Fallback từ vựng: chỉ đo độ trùng lặp bề mặt, không hiểu ngữ nghĩa sâu."""

    name = "tfidf-char-ngram"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        cleaned = [text.casefold() for text in texts]
        if not any(text.strip() for text in cleaned):
            return [[0.0] for _ in cleaned]
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(cleaned)
        return matrix.toarray().tolist()


@lru_cache(maxsize=1)
def get_semantic_embedder() -> Embedder | None:
    """Bi-encoder ngữ nghĩa, hoặc None khi bị tắt / chưa cài sentence-transformers."""
    model_name = os.getenv("DEEPGUARD_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if model_name.strip().casefold() in DISABLED_VALUES:
        return None
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception:  # noqa: BLE001 - thư viện/model tùy chọn
        return None


def get_embedder() -> Embedder:
    return get_semantic_embedder() or TfidfEmbedder()


def similarities(query: str, documents: Sequence[str], embedder: Embedder | None = None) -> list[float]:
    """Cosine similarity giữa query và từng document (cùng một không gian embedding)."""
    if not documents:
        return []
    vectors = (embedder or get_embedder()).embed([query, *documents])
    return [max(0.0, min(1.0, cosine(vectors[0], vector))) for vector in vectors[1:]]

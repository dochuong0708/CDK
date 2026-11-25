"""Optional local retrieval and ONNX NLI checks over trusted reference snippets."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.nlp.embeddings import Embedder, cosine, get_embedder

TRUSTED_DOMAINS = (
    "baochinhphu.vn",
    "tuoitre.vn",
    "vnexpress.net",
    "vtv.vn",
)
NLI_LABELS = {"entailment", "neutral", "contradiction"}


class NliScorer(Protocol):
    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]: ...


@dataclass(frozen=True)
class Reference:
    text: str
    title: str
    url: str
    published_at: str | None = None

    @property
    def trusted(self) -> bool:
        host = self.url.split("//", 1)[-1].split("/", 1)[0].casefold().removeprefix("www.")
        return any(host == domain or host.endswith(f".{domain}") for domain in TRUSTED_DOMAINS)


@dataclass(frozen=True)
class LocalRagResult:
    factuality_risk: float | None
    verified_trusted_source: bool
    matched_reference_urls: list[str]
    verdict: str | None


class OnnxNliScorer:
    def __init__(self, model_dir: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = Path(model_dir)
        onnx_path = next(
            path
            for path in (model_path / "model_quantized.onnx", model_path / "model.onnx")
            if path.exists()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        self.labels = {
            int(index): str(label).casefold().replace("label_", "")
            for index, label in config.get("id2label", {}).items()
        }

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="np",
            truncation=True,
            max_length=384,
        )
        inputs = {item.name: encoded[item.name] for item in self.session.get_inputs()}
        logits = self.session.run(None, inputs)[0][0]
        maximum = max(float(value) for value in logits)
        exponentials = [math.exp(float(value) - maximum) for value in logits]
        probabilities = [value / sum(exponentials) for value in exponentials]
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        label = self.labels.get(index, "neutral")
        return label, float(probabilities[index])


class LocalRagFactChecker:
    def __init__(
        self,
        corpus_path: str | None = None,
        embedder: Embedder | None = None,
        nli_scorer: NliScorer | None = None,
        similarity_threshold: float = 0.35,
    ):
        configured_path = corpus_path or os.getenv("DEEPGUARD_REFERENCE_CORPUS")
        self.references = self._load_references(Path(configured_path)) if configured_path else []
        self.embedder = embedder or (get_embedder() if self.references else None)
        self.nli_scorer = nli_scorer or self._load_nli_scorer()
        self.similarity_threshold = similarity_threshold
        self._uses_tfidf_fallback = (
            self.embedder is not None and self.embedder.name == "tfidf-char-ngram"
        )
        self._vectors = (
            self.embedder.embed([item.text for item in self.references])
            if self.embedder is not None and not self._uses_tfidf_fallback
            else []
        )
        self._faiss_index: Any | None = None
        self._load_faiss_index()

    @staticmethod
    def _load_references(path: Path) -> list[Reference]:
        if not path.is_file():
            return []
        records: list[Reference] = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                    text = str(item.get("text", "")).strip()
                    url = str(item.get("url", "")).strip()
                    if text and url.startswith("https://"):
                        records.append(
                            Reference(
                                text=text,
                                title=str(item.get("title", "")),
                                url=url,
                                published_at=item.get("published_at"),
                            )
                        )
                except (json.JSONDecodeError, AttributeError):
                    continue
        return records

    @staticmethod
    def _load_nli_scorer() -> NliScorer | None:
        model_dir = os.getenv("DEEPGUARD_NLI_ONNX_DIR")
        if not model_dir:
            return None
        try:
            return OnnxNliScorer(model_dir)
        except (ImportError, OSError, ValueError, RuntimeError, StopIteration):
            return None

    def _load_faiss_index(self) -> None:
        if not self.references or not self._vectors or self._uses_tfidf_fallback:
            return
        try:
            import faiss
            import numpy as np

            vectors = np.asarray(self._vectors, dtype="float32")
            faiss.normalize_L2(vectors)
            self._faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss_index.add(vectors)
        except (ImportError, ValueError):
            self._faiss_index = None

    def check(self, text: str, top_k: int = 3) -> LocalRagResult:
        if (
            not text.strip()
            or not self.references
            or self.nli_scorer is None
            or self.embedder is None
        ):
            return LocalRagResult(None, False, [], None)

        if self._uses_tfidf_fallback:
            vectors = self.embedder.embed([text, *(item.text for item in self.references)])
            query_vector, reference_vectors = vectors[0], vectors[1:]
            ranked = sorted(
                (
                    (index, cosine(query_vector, vector))
                    for index, vector in enumerate(reference_vectors)
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
        elif self._faiss_index is not None:
            import faiss
            import numpy as np

            query_vector = self.embedder.embed([text])[0]
            query = np.asarray([query_vector], dtype="float32")
            faiss.normalize_L2(query)
            scores, indices = self._faiss_index.search(query, min(top_k, len(self.references)))
            ranked = [(int(index), float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
        else:
            query_vector = self.embedder.embed([text])[0]
            ranked = sorted(
                ((index, cosine(query_vector, vector)) for index, vector in enumerate(self._vectors)),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]

        matched: list[str] = []
        trusted_entailment = False
        strongest_contradiction = 0.0
        strongest_entailment = 0.0
        for index, similarity in ranked:
            reference = self.references[index]
            if similarity < self.similarity_threshold:
                continue
            label, confidence = self.nli_scorer.predict(reference.text, text)
            if label not in NLI_LABELS:
                continue
            if reference.trusted:
                matched.append(reference.url)
            if label == "contradiction":
                strongest_contradiction = max(strongest_contradiction, confidence)
            elif label == "entailment":
                strongest_entailment = max(strongest_entailment, confidence)
                trusted_entailment |= reference.trusted and confidence >= 0.75

        verified_trusted_source = trusted_entailment and strongest_contradiction < 0.75
        if strongest_contradiction >= 0.75:
            risk, verdict = 0.85, "contradiction"
        elif verified_trusted_source:
            risk, verdict = 0.0, "verified"
        elif strongest_entailment >= 0.75:
            risk, verdict = 0.15, "entailment"
        else:
            risk, verdict = None, None
        return LocalRagResult(risk, verified_trusted_source, list(dict.fromkeys(matched)), verdict)
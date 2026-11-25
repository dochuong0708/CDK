"""Optional CPU inference for a quantized PhoBERT text classifier."""

import json
import math
import os
from pathlib import Path
from typing import Any

from app.detectors.language.preprocessing import preprocess_text


class PhoBertOnnxClassifier:
    RISK_LABELS = {"risky", "risk", "fraud", "scam", "1", "label_1"}

    def __init__(self, model_dir: str | None = None):
        configured_dir = model_dir or os.getenv("DEEPGUARD_PHOBERT_ONNX_DIR")
        self.session: Any | None = None
        self.tokenizer: Any | None = None
        self.risky_index = 1
        if not configured_dir:
            return

        model_path = Path(configured_dir)
        if model_path.is_dir():
            model_path = next(
                (
                    path
                    for path in (
                        model_path / "model_quantized.onnx",
                        model_path / "model.onnx",
                    )
                    if path.exists()
                ),
                model_path / "model_quantized.onnx",
            )
        if not model_path.exists():
            return

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path.parent))
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            config_path = model_path.parent / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                id2label = config.get("id2label", {})
                labels = {int(index): str(label).casefold() for index, label in id2label.items()}
                self.risky_index = next(
                    (index for index, label in labels.items() if label in self.RISK_LABELS),
                    1,
                )
        except (ImportError, OSError, ValueError, RuntimeError):
            self.session = None
            self.tokenizer = None

    def predict_score(self, text: str) -> float | None:
        if self.session is None or self.tokenizer is None or not text.strip():
            return None
        model_text = " ".join(preprocess_text(text)["tokens"])
        encoded = self.tokenizer(
            model_text, return_tensors="np", truncation=True, max_length=256
        )
        inputs = {item.name: encoded[item.name] for item in self.session.get_inputs()}
        logits = self.session.run(None, inputs)[0][0]
        maximum = max(float(value) for value in logits)
        exponentials = [math.exp(float(value) - maximum) for value in logits]
        probabilities = [value / sum(exponentials) for value in exponentials]
        if self.risky_index >= len(probabilities):
            return None
        return float(probabilities[self.risky_index])
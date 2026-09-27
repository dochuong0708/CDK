import os
from pathlib import Path
from typing import Any


class LanguageModel:
    """Mô hình joblib tùy chọn; mặc định xác suất lớp 1 là xác suất rủi ro."""

    RISK_LABELS = {
        "1", "true", "risky", "scam", "phishing", "spam", "toxic", "hate",
        "fake", "fake_news", "dangerous", "lừa đảo",
    }

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("DEEPGUARD_LANGUAGE_MODEL_PATH")
        self.model: Any | None = None
        if self.model_path:
            import joblib

            self.model = joblib.load(Path(self.model_path))

    def predict_score(self, text: str) -> float | None:
        if self.model is None:
            return None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([text])[0]
            classes = [str(value).casefold() for value in getattr(self.model, "classes_", [])]
            risky_index = next(
                (index for index, label in enumerate(classes) if label in self.RISK_LABELS),
                1 if len(probabilities) > 1 else 0,
            )
            return float(probabilities[risky_index])
        if hasattr(self.model, "decision_function"):
            import math

            decision = float(self.model.decision_function([text])[0])
            return 1.0 / (1.0 + math.exp(-decision))
        raise TypeError("Mô hình ngôn ngữ cần hỗ trợ predict_proba hoặc decision_function")

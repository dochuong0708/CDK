from datetime import datetime
from uuid import uuid4

from app.detectors.language.heuristics import inspect_rules
from app.detectors.language.model import LanguageModel
from app.detectors.language.preprocessing import preprocess_text
from app.models.schemas import Analysis


class LanguageDetector:
    def __init__(self, model: LanguageModel | None = None):
        self.model = model or LanguageModel()

    def analyze(self, text: str) -> Analysis:
        processed = preprocess_text(text)
        rule_score, evidence = inspect_rules(processed)
        model_score = self.model.predict_score(str(processed["text"]))
        final_score = max(rule_score, model_score or 0.0)

        if final_score < 0.3:
            label = "SAFE"
        elif final_score < 0.7:
            label = "SUSPICIOUS"
        else:
            label = "DANGEROUS"

        if model_score is not None:
            evidence.append(f"Mô hình NLP dự đoán rủi ro: {model_score:.0%}")
            confidence = max(model_score, 1 - model_score)
            detector = "text-rules+model-v1"
        else:
            evidence.append("Chưa cấu hình mô hình NLP; kết quả chỉ dựa trên luật heuristic")
            confidence = min(0.95, 0.6 + len(evidence) * 0.05)
            detector = "text-rules-v1"

        evidence.append(f"Điểm luật: {rule_score:.0%}; điểm rủi ro cuối: {final_score:.0%}")
        return Analysis(
            analysisId=f"text-{uuid4().hex[:8]}",
            modality="Text",
            label=label,
            riskScore=round(final_score * 100),
            confidence=round(confidence, 2),
            evidence=evidence,
            detector=detector,
            input=str(processed["text"]),
            createdAt=datetime.now(),
        )

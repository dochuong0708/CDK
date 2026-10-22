from datetime import datetime
from uuid import uuid4

from app.detectors.language.heuristics import inspect_rules_detailed
from app.detectors.language.model import LanguageModel
from app.detectors.language.preprocessing import preprocess_text
from app.detectors.language.style_classifier import EmbeddingStyleClassifier
from app.models.fact_checking import FactCheckReport
from app.models.schemas import Analysis
from app.services.risk_fusion import combine_factuality, fuse, noisy_or, report_factuality


class LanguageDetector:
    def __init__(
        self,
        model: LanguageModel | None = None,
        style_classifier: EmbeddingStyleClassifier | None = None,
    ):
        self.model = model or LanguageModel()
        self.style_classifier = style_classifier or EmbeddingStyleClassifier()

    def analyze(self, text: str, fact_report: FactCheckReport | None = None) -> Analysis:
        processed = preprocess_text(text)
        rules = inspect_rules_detailed(processed)
        evidence = list(rules.evidence)
        detector_parts = ["text-rules"]

        style_prediction = self.style_classifier.predict(str(processed["text"]))
        if style_prediction is not None:
            detector_parts.append("embedding-style")
            evidence.append(
                "Bộ phân loại ngữ nghĩa (embedding): giật gân "
                f"{style_prediction.sensational:.0%}, lừa đảo {style_prediction.scam:.0%}"
            )
        style_risk = noisy_or(
            rules.style_score, style_prediction.sensational if style_prediction else None
        )
        fraud_risk = noisy_or(rules.fraud_score, style_prediction.scam if style_prediction else None)

        model_score = self.model.predict_score(str(processed["text"]))
        if model_score is not None:
            detector_parts.append("model")
            evidence.append(f"Mô hình NLP dự đoán rủi ro tin giả: {model_score:.0%}")
        else:
            evidence.append("Chưa cấu hình mô hình tin giả; độ chính xác thông tin chưa được mô hình đánh giá")

        fact_risk, fact_confidence = report_factuality(fact_report)
        if fact_report is not None:
            detector_parts.append("rag-factcheck")
            evidence.append(f"Fact-check: {fact_report.summary}")
        factuality_risk = combine_factuality(model_score, fact_risk, fact_confidence)

        risk = fuse(style_risk, fraud_risk, factuality_risk)
        factuality_text = (
            "chưa xác minh" if factuality_risk is None else f"{factuality_risk:.0%}"
        )
        evidence.append(
            f"Rủi ro hình thức: {style_risk:.0%}; lừa đảo: {fraud_risk:.0%}; "
            f"sai sự thật: {factuality_text}; điểm rủi ro cuối: {risk.overall:.0%}"
        )

        if factuality_risk is not None:
            confidence = max(factuality_risk, 1 - factuality_risk)
        else:
            confidence = min(0.95, 0.6 + len(evidence) * 0.05)

        return Analysis(
            analysisId=f"text-{uuid4().hex[:8]}",
            modality="Text",
            label=risk.label,
            riskScore=round(risk.overall * 100),
            confidence=round(confidence, 2),
            evidence=evidence,
            detector="+".join(detector_parts) + "-v2",
            styleRisk=round(style_risk * 100),
            styleLabel=risk.style_label,
            fraudRisk=round(fraud_risk * 100),
            factualityRisk=None if factuality_risk is None else round(factuality_risk * 100),
            factualityLabel=risk.factuality_label,
            factCheckId=fact_report.reportId if fact_report else None,
            input=str(processed["text"]),
            createdAt=datetime.now(),
        )

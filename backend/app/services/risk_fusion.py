"""Tách và hợp nhất rủi ro theo hai trục độc lập.

- Style risk (hình thức/nguồn): giật gân, nặc danh, thuyết âm mưu, kêu gọi lan truyền.
- Factuality risk (độ chính xác thông tin): mô hình tin giả + kết quả fact-check (RAG).
- Fraud risk (lừa đảo/phishing) được giữ riêng vì đã đủ nguy hiểm dù nội dung "đúng".

Điểm tổng KHÔNG lấy max(rule, model): bài viết giật gân nhẹ nhưng đúng sự thật sẽ có
style risk cao, factuality risk thấp và điểm tổng thấp. Khi đã có factuality, style chỉ
cộng thêm theo trọng số; sai sự thật rõ ràng tự nó đủ để bị đánh giá nguy hiểm.
"""

from dataclasses import dataclass

from app.models.fact_checking import FactCheckReport

STYLE_WEIGHT = 0.35
FACTUALITY_WEIGHT = 0.65
NLP_EVIDENCE_WEIGHT = 0.6
MAX_FACT_CHECK_WEIGHT = 0.8
SUSPICIOUS_THRESHOLD = 0.3
DANGEROUS_THRESHOLD = 0.7


@dataclass(frozen=True)
class RiskBreakdown:
    style_risk: float
    fraud_risk: float
    factuality_risk: float | None
    overall: float

    @property
    def style_label(self) -> str:
        return risk_label(self.style_risk)

    @property
    def fraud_label(self) -> str:
        return risk_label(self.fraud_risk)

    @property
    def factuality_label(self) -> str:
        return "UNVERIFIED" if self.factuality_risk is None else risk_label(self.factuality_risk)

    @property
    def label(self) -> str:
        return risk_label(self.overall)


def risk_label(score: float) -> str:
    if score < SUSPICIOUS_THRESHOLD:
        return "SAFE"
    if score < DANGEROUS_THRESHOLD:
        return "SUSPICIOUS"
    return "DANGEROUS"


def noisy_or(rule_score: float, nlp_score: float | None) -> float:
    """Gộp hai nguồn bằng chứng độc lập; NLP được giảm trọng số vì chưa hiệu chuẩn."""
    if nlp_score is None:
        return rule_score
    return min(0.95, 1 - (1 - rule_score) * (1 - NLP_EVIDENCE_WEIGHT * nlp_score))


def combine_factuality(
    model_score: float | None,
    fact_check_risk: float | None,
    fact_check_confidence: float = 0.0,
) -> float | None:
    if fact_check_risk is None:
        return model_score
    if model_score is None:
        return fact_check_risk
    weight = min(MAX_FACT_CHECK_WEIGHT, max(0.0, fact_check_confidence))
    return weight * fact_check_risk + (1 - weight) * model_score


def fuse(style_risk: float, fraud_risk: float, factuality_risk: float | None) -> RiskBreakdown:
    if factuality_risk is None:
        content_risk = style_risk
    else:
        blended = STYLE_WEIGHT * style_risk + FACTUALITY_WEIGHT * factuality_risk
        content_risk = max(factuality_risk, blended)
    overall = min(0.95, max(fraud_risk, content_risk))
    return RiskBreakdown(style_risk, fraud_risk, factuality_risk, overall)


def report_factuality(report: FactCheckReport | None) -> tuple[float | None, float]:
    if report is None or report.factuality_risk is None:
        return None, 0.0
    return report.factuality_risk, report.factuality_confidence

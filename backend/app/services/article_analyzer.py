import re
from typing import Any
from uuid import uuid4

from app.detectors.language.heuristics import find_suspicious_phrases, inspect_rules_detailed
from app.detectors.language.model import LanguageModel
from app.detectors.language.preprocessing import preprocess_text
from app.detectors.language.style_classifier import EmbeddingStyleClassifier
from app.models.fact_checking import FactCheckReport
from app.services.fact_checking import FactChecker
from app.services.risk_fusion import (
    combine_factuality,
    fuse,
    noisy_or,
    report_factuality,
    risk_label,
)

CONTRADICTED_CLAIM_RISK = 0.9


class ArticleAnalyzer:
    def __init__(
        self,
        model: LanguageModel | None = None,
        fact_checker: FactChecker | None = None,
        store: Any | None = None,
        style_classifier: EmbeddingStyleClassifier | None = None,
    ):
        self.model = model or LanguageModel()
        self.fact_checker = fact_checker or FactChecker()
        self.store = store
        self.style_classifier = style_classifier or EmbeddingStyleClassifier()

    def split_into_paragraphs(self, text: str) -> list[str]:
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [p.strip() for p in text.splitlines() if p.strip()]

        if not raw_paragraphs and text.strip():
            raw_paragraphs = [text.strip()]

        # Nếu một đoạn quá dài (> 1200 ký tự) mà không có ngắt dòng, chia theo câu
        result: list[str] = []
        for p in raw_paragraphs:
            if len(p) > 1200:
                sentences = re.split(r"(?<=[.!?…])\s+", p)
                current_chunk = []
                current_len = 0
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    current_chunk.append(s_clean)
                    current_len += len(s_clean)
                    if current_len >= 600:
                        result.append(" ".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                if current_chunk:
                    result.append(" ".join(current_chunk))
            else:
                result.append(p)

        return result

    def analyze_article(
        self,
        text: str,
        title: str = "",
        check_facts: bool = True,
    ) -> dict[str, Any]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Nội dung bài viết không được để trống.")

        paragraphs_raw = self.split_into_paragraphs(cleaned_text)
        paragraph_analyses: list[dict[str, Any]] = []

        for idx, p_text in enumerate(paragraphs_raw, start=1):
            preprocessed = preprocess_text(p_text)
            rules = inspect_rules_detailed(preprocessed)
            suspicious_phrases = find_suspicious_phrases(p_text)
            style_prediction = self.style_classifier.predict(str(preprocessed["text"]))
            model_score = self.model.predict_score(str(preprocessed["text"]))

            style_risk = noisy_or(
                rules.style_score, style_prediction.sensational if style_prediction else None
            )
            fraud_risk = noisy_or(
                rules.fraud_score, style_prediction.scam if style_prediction else None
            )

            issues = [ev for ev in rules.evidence if "Không tìm thấy" not in ev]
            if style_prediction is not None and style_prediction.risk >= 0.5:
                issues.append(
                    "Bộ phân loại ngữ nghĩa nhận diện văn phong "
                    f"{'giật gân' if style_prediction.top_category == 'sensational' else 'lừa đảo'}"
                    f" ({style_prediction.risk:.0%})"
                )
            if model_score is not None and model_score >= 0.4:
                issues.append(f"Mô hình AI nhận diện xác suất tin giả: {model_score:.0%}")

            paragraph_analyses.append({
                "index": idx,
                "text": p_text,
                "model_score": round(model_score, 3) if model_score is not None else None,
                "rule_score": round(rules.combined, 3),
                "style_score": round(style_risk, 3),
                "fraud_score": round(fraud_risk, 3),
                "factuality_score": model_score,
                "suspicious_phrases": suspicious_phrases,
                "issues": issues,
                "claims": [],
            })

        # Fact checking các luận điểm
        fact_report: FactCheckReport | None = None
        claims_data: list[dict[str, Any]] = []
        fact_check_error: str | None = None

        if check_facts:
            try:
                fact_report = self.fact_checker.check(cleaned_text[:25000])
                if self.store is not None:
                    fact_report = self.store.add_fact_check(fact_report, cleaned_text[:25000])
                for claim in fact_report.claims:
                    claims_data.append(claim.model_dump(mode="json"))
                    paragraph = self._paragraph_for_claim(claim.claim, paragraph_analyses)
                    if paragraph is None:
                        continue
                    paragraph["claims"].append({
                        "claim": claim.claim,
                        "verdict": claim.verdict,
                        "explanation": claim.explanation,
                    })
                    if claim.verdict == "contradicted":
                        paragraph["factuality_score"] = max(
                            paragraph["factuality_score"] or 0.0, CONTRADICTED_CLAIM_RISK
                        )
                        paragraph["issues"].append(f"Chứa luận điểm bị bác bỏ: \"{claim.claim}\"")
                        paragraph["suspicious_phrases"].append({
                            "phrase": claim.claim,
                            "start": 0,
                            "end": len(claim.claim),
                            "category": "claim_contradicted",
                            "reason": f"Luận điểm sai lệch: {claim.explanation}",
                        })
            except Exception as err:
                fact_check_error = str(err)

        for paragraph in paragraph_analyses:
            factuality = paragraph.pop("factuality_score")
            risk = fuse(paragraph["style_score"], paragraph["fraud_score"], factuality)
            paragraph["risk_score"] = round(risk.overall * 100)
            paragraph["label"] = risk.label
            paragraph["style_risk"] = round(risk.style_risk * 100)
            paragraph["style_label"] = risk.style_label
            paragraph["factuality_risk"] = (
                None if factuality is None else round(factuality * 100)
            )
            paragraph["factuality_label"] = risk.factuality_label
            paragraph["fraud_risk"] = round(risk.fraud_risk * 100)
            del paragraph["style_score"], paragraph["fraud_score"]

        # Thống kê tổng quan
        num_paragraphs = len(paragraph_analyses)
        dangerous_count = sum(1 for p in paragraph_analyses if p["label"] == "DANGEROUS")
        suspicious_count = sum(1 for p in paragraph_analyses if p["label"] == "SUSPICIOUS")
        safe_count = sum(1 for p in paragraph_analyses if p["label"] == "SAFE")

        contradicted_claims_count = sum(1 for c in claims_data if c.get("verdict") == "contradicted")
        supported_claims_count = sum(1 for c in claims_data if c.get("verdict") == "supported")
        unverified_claims_count = sum(
            1 for c in claims_data if c.get("verdict") in {"insufficient_evidence", "mixed"}
        )

        # Hai trục độc lập: hình thức (style) và độ chính xác thông tin (factuality)
        style_scores = [p["style_risk"] / 100 for p in paragraph_analyses]
        article_style = 0.6 * max(style_scores) + 0.4 * sum(style_scores) / num_paragraphs
        model_scores = [p["model_score"] for p in paragraph_analyses if p["model_score"] is not None]
        article_model = sum(model_scores) / len(model_scores) if model_scores else None
        fact_risk, fact_confidence = report_factuality(fact_report)
        article_factuality = combine_factuality(article_model, fact_risk, fact_confidence)
        article_fraud = max(p["fraud_risk"] / 100 for p in paragraph_analyses)
        breakdown = fuse(article_style, article_fraud, article_factuality)
        overall_risk = round(breakdown.overall * 100)
        style_risk_score = round(article_style * 100)
        factuality_risk_score = (
            None if article_factuality is None else round(article_factuality * 100)
        )
        credibility_score = 100 - overall_risk
        overall_label = risk_label(breakdown.overall)

        if overall_label == "DANGEROUS":
            verdict_title = "NGUY CƠ TIN GIẢ RẤT CAO / SAI SỰ THẬT"
        elif overall_label == "SUSPICIOUS":
            verdict_title = "CẢNH BÁO NGHI VẤN / THÔNG TIN CHƯA RÕ RÀNG"
        else:
            verdict_title = "ĐỘ TIN CẬY CAO / CHƯA THẤY DẤU HIỆU BẤT THƯỜNG"

        # Tổng hợp kết luận chi tiết của Agent
        conclusion_parts: list[str] = []
        if overall_label == "DANGEROUS":
            conclusion_parts.append(
                f"Bài viết có mức độ rủi ro nghiêm trọng ({overall_risk}%). "
                f"Phát hiện {dangerous_count} đoạn văn có dấu hiệu sai lệch hoặc thổi phồng cao."
            )
            if contradicted_claims_count > 0:
                conclusion_parts.append(
                    f"Đặc biệt, hệ thống đã đối chiếu và phát hiện {contradicted_claims_count} luận điểm "
                    "bị các nguồn tin báo chí chính thống bác bỏ hoàn toàn."
                )
        elif overall_label == "SUSPICIOUS":
            conclusion_parts.append(
                f"Bài viết thuộc diện nghi vấn ({overall_risk}% rủi ro). "
                f"Có {suspicious_count + dangerous_count} phần chứa câu chữ giật gân, thiếu dẫn chứng "
                "hoặc nguồn tin chưa được kiểm chứng độc lập."
            )
        else:
            conclusion_parts.append(
                f"Bài viết có độ tin cậy tương đối tốt ({credibility_score}% tin cậy). "
                "Các câu chữ chuẩn mực, không phát hiện dấu hiệu lừa đảo, giật gân hay mâu thuẫn sự thật lớn."
            )

        conclusion_parts.append(self._axes_sentence(breakdown))

        recommendations: list[str] = []
        if overall_label == "DANGEROUS":
            recommendations.append("KHÔNG chia sẻ hoặc lan truyền bài viết này lên mạng xã hội.")
            recommendations.append("Cảnh giác trước các thông tin kích động cảm xúc, kêu gọi chuyển tiền hoặc chữa bệnh thần tốc.")
            recommendations.append("Chỉ tin tưởng thông tin từ các cơ quan thông tấn chính thức (VnExpress, Tuổi Trẻ, VTV, Bộ Y tế, Bộ Công an).")
        elif overall_label == "SUSPICIOUS":
            recommendations.append("Cần đối chiếu thêm với ít nhất 2 nguồn tin báo chí uy tín khác trước khi tin tưởng.")
            recommendations.append("Chú ý các cụm từ nặc danh ('nghe nói', 'dân mạng đồn') thường là dấu hiệu tin đồn thất thiệt.")
        else:
            recommendations.append("Bài viết có văn phong chuẩn mực và thông tin khách quan.")

        return {
            "article_id": f"art-{uuid4().hex[:8]}",
            "title": title or "Bài viết được phân tích",
            "summary": {
                "overall_risk_score": overall_risk,
                "credibility_score": credibility_score,
                "label": overall_label,
                "verdict_title": verdict_title,
                "conclusion": " ".join(conclusion_parts),
                "style_risk_score": style_risk_score,
                "style_label": breakdown.style_label,
                "fraud_risk_score": round(article_fraud * 100),
                "factuality_risk_score": factuality_risk_score,
                "factuality_label": breakdown.factuality_label,
                "stats": {
                    "total_paragraphs": num_paragraphs,
                    "dangerous_paragraphs": dangerous_count,
                    "suspicious_paragraphs": suspicious_count,
                    "safe_paragraphs": safe_count,
                    "total_claims": len(claims_data),
                    "contradicted_claims": contradicted_claims_count,
                    "supported_claims": supported_claims_count,
                    "unverified_claims": unverified_claims_count,
                },
            },
            "paragraphs": paragraph_analyses,
            "claims": claims_data,
            "fact_check_error": fact_check_error,
            "fact_check_report_id": fact_report.reportId if fact_report else None,
            "recommendations": recommendations,
        }

    @staticmethod
    def _paragraph_for_claim(
        claim: str, paragraphs: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        claim_lower = claim.casefold()
        for paragraph in paragraphs:
            text_lower = paragraph["text"].casefold()
            if claim_lower in text_lower or any(
                word in text_lower for word in claim_lower.split()[:4] if len(word) > 3
            ):
                return paragraph
        return None

    @staticmethod
    def _axes_sentence(breakdown: Any) -> str:
        style = {
            "SAFE": "văn phong chuẩn mực",
            "SUSPICIOUS": "văn phong có dấu hiệu giật gân/câu view",
            "DANGEROUS": "văn phong kích động, thao túng mạnh",
        }[breakdown.style_label]
        factuality = {
            "UNVERIFIED": "nội dung chưa được xác minh",
            "SAFE": "nội dung phù hợp với nguồn đối chiếu",
            "SUSPICIOUS": "nội dung có phần chưa chính xác",
            "DANGEROUS": "nội dung mâu thuẫn với nguồn đối chiếu",
        }[breakdown.factuality_label]
        return f"Đánh giá tách biệt: hình thức — {style}; sự thật — {factuality}."

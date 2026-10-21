import re
from typing import Any
from uuid import uuid4

from app.detectors.language.heuristics import find_suspicious_phrases, inspect_rules
from app.detectors.language.model import LanguageModel
from app.detectors.language.preprocessing import preprocess_text
from app.models.fact_checking import FactCheckReport
from app.services.fact_checking import FactChecker


class ArticleAnalyzer:
    def __init__(
        self,
        model: LanguageModel | None = None,
        fact_checker: FactChecker | None = None,
    ):
        self.model = model or LanguageModel()
        self.fact_checker = fact_checker or FactChecker()

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

        total_risk_sum = 0.0
        max_risk = 0.0

        for idx, p_text in enumerate(paragraphs_raw, start=1):
            preprocessed = preprocess_text(p_text)
            rule_score, rule_evidence = inspect_rules(preprocessed)
            suspicious_phrases = find_suspicious_phrases(p_text)

            model_score = self.model.predict_score(str(preprocessed["text"]))
            effective_model_score = model_score if model_score is not None else 0.0

            # Điểm rủi ro kết hợp
            combined_risk = max(rule_score, effective_model_score)
            risk_percent = round(combined_risk * 100)

            if risk_percent >= 65:
                label = "DANGEROUS"
            elif risk_percent >= 35:
                label = "SUSPICIOUS"
            else:
                label = "SAFE"

            total_risk_sum += risk_percent
            if risk_percent > max_risk:
                max_risk = risk_percent

            issues: list[str] = []
            for ev in rule_evidence:
                if "Không tìm thấy" not in ev:
                    issues.append(ev)
            if model_score is not None and model_score >= 0.4:
                issues.append(f"Mô hình AI nhận diện xác suất tin giả: {model_score:.0%}")

            paragraph_analyses.append({
                "index": idx,
                "text": p_text,
                "risk_score": risk_percent,
                "label": label,
                "model_score": round(model_score, 3) if model_score is not None else None,
                "rule_score": round(rule_score, 3),
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
                # Kiểm chứng toàn bộ hoặc phần văn bản chính
                fact_report = self.fact_checker.check(cleaned_text[:25000])
                for claim in fact_report.claims:
                    c_dict = claim.model_dump(mode="json")
                    claims_data.append(c_dict)

                    # Gán claim về đoạn văn liên quan nhất
                    claim_text_lower = claim.claim.casefold()
                    for p in paragraph_analyses:
                        if (
                            claim_text_lower in p["text"].casefold()
                            or any(w in p["text"].casefold() for w in claim_text_lower.split()[:4] if len(w) > 3)
                        ):
                            p["claims"].append({
                                "claim": claim.claim,
                                "verdict": claim.verdict,
                                "explanation": claim.explanation,
                            })
                            # Nếu luận điểm bị mâu thuẫn/bác bỏ, tăng điểm rủi ro cho đoạn đó
                            if claim.verdict == "contradicted":
                                p["risk_score"] = min(100, p["risk_score"] + 25)
                                p["label"] = "DANGEROUS"
                                p["issues"].append(f"Chứa luận điểm bị bác bỏ: \"{claim.claim}\"")
                                p["suspicious_phrases"].append({
                                    "phrase": claim.claim,
                                    "start": 0,
                                    "end": len(claim.claim),
                                    "category": "claim_contradicted",
                                    "reason": "Luận điểm mâu thuẫn với báo chí chính thống",
                                })
                            break
            except Exception as err:
                fact_check_error = str(err)

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

        # Tính điểm rủi ro tổng hợp cho toàn bài viết
        avg_risk = total_risk_sum / num_paragraphs if num_paragraphs > 0 else 0.0
        # Trọng số: 50% max_risk, 30% avg_risk, 20% phạt thêm nếu có luận điểm contradicted
        claim_penalty = min(30, contradicted_claims_count * 15)
        overall_risk = round(0.45 * max_risk + 0.35 * avg_risk + claim_penalty)
        overall_risk = max(0, min(100, overall_risk))
        credibility_score = 100 - overall_risk

        if overall_risk >= 65:
            overall_label = "DANGEROUS"
            verdict_title = "NGUY CƠ TIN GIẢ RẤT CAO / SAI SỰ THẬT"
        elif overall_risk >= 35:
            overall_label = "SUSPICIOUS"
            verdict_title = "CẢNH BÁO NGHI VẤN / THÔNG TIN CHƯA RÕ RÀNG"
        else:
            overall_label = "SAFE"
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
            "recommendations": recommendations,
        }

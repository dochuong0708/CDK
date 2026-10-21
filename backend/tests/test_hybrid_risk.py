import json
from types import SimpleNamespace

import pytest

from app.db.store import AnalysisStore
from app.detectors.language.detector import LanguageDetector
from app.detectors.language.domains import inspect_host, levenshtein, normalize_homoglyphs
from app.detectors.language.style_classifier import EmbeddingStyleClassifier
from app.models.fact_checking import ClaimFactCheck, FactCheckReport
from app.services.fact_checking import FactChecker, factuality_from_claims
from app.services.risk_fusion import fuse


class FixedModel:
    def __init__(self, score):
        self.score = score

    def predict_score(self, text):
        return self.score


class KeywordEmbedder:
    """Embedder giả lập: vector theo số lần xuất hiện từ khóa."""

    name = "keyword-test"
    KEYWORDS = ("sốc", "kinh", "otp", "khóa", "họp", "thống", "kê")

    def embed(self, texts):
        return [[text.casefold().count(word) + 0.01 for word in self.KEYWORDS] for text in texts]


def test_levenshtein_and_homoglyphs():
    assert levenshtein("vietcombankk", "vietcombank") == 1
    assert levenshtein("kitten", "sitting") == 3
    assert normalize_homoglyphs("v1etc0mbank") == "vietcombank"


@pytest.mark.parametrize(
    "host",
    ["vietcombankk.com", "v1etcombank.com", "vietcornbank.net", "techcombank-xacminh.com", "teckcombank.vn"],
)
def test_typosquatted_bank_domains_are_flagged(host):
    finding = inspect_host(host)
    assert finding is not None
    assert finding.brand in {"vietcombank", "techcombank"}


@pytest.mark.parametrize(
    "host", ["www.vietcombank.com.vn", "ebanking.techcombank.com.vn", "vnexpress.net", "abc.com.vn"]
)
def test_official_or_unrelated_domains_are_not_flagged(host):
    assert inspect_host(host) is None


def test_typosquat_url_in_text_raises_fraud_risk():
    result = LanguageDetector().analyze("Đăng nhập tại https://v1etcombank.com để nhận quà")

    assert result.fraudRisk >= 50
    assert any("giả mạo vietcombank" in item for item in result.evidence)


def test_mild_clickbait_with_true_content_is_not_flagged_by_max():
    detector = LanguageDetector(model=FixedModel(0.1))
    result = detector.analyze("Cực sốc: đội tuyển Việt Nam thắng 2-0 trong trận giao hữu tối qua.")

    assert result.styleLabel == "SUSPICIOUS"
    assert result.factualityLabel == "SAFE"
    assert result.label == "SAFE"
    assert result.riskScore < 30


def test_neutral_style_but_contradicted_facts_is_flagged():
    report = FactCheckReport(
        reportId="fc-1",
        summary="1 luận điểm bị mâu thuẫn",
        claims=[],
        factuality_risk=1.0,
        factuality_confidence=0.8,
    )
    result = LanguageDetector(model=FixedModel(0.5)).analyze(
        "Bộ Y tế cho biết nước chanh chữa khỏi ung thư.", fact_report=report
    )

    assert result.styleLabel == "SAFE"
    assert result.factualityRisk == 90
    assert result.label == "DANGEROUS"
    assert result.factCheckId == "fc-1"


def test_fraud_risk_dominates_even_without_factuality():
    breakdown = fuse(style_risk=0.0, fraud_risk=0.8, factuality_risk=0.0)
    assert breakdown.label == "DANGEROUS"


def test_embedding_style_classifier_uses_semantic_neighbours():
    classifier = EmbeddingStyleClassifier(embedder=KeywordEmbedder())

    sensational = classifier.predict("Sốc! kinh hoàng, sốc nặng")
    scam = classifier.predict("Gửi OTP ngay kẻo khóa tài khoản, OTP")
    neutral = classifier.predict("Cuộc họp về số liệu thống kê")

    assert sensational.top_category == "sensational"
    assert scam.top_category == "scam"
    assert neutral.top_category == "neutral"


def test_embedding_style_classifier_unavailable_without_backend(monkeypatch):
    classifier = EmbeddingStyleClassifier()
    assert classifier.available is False
    assert classifier.predict("Cực sốc") is None


def _claim(verdict, confidence):
    return ClaimFactCheck(
        claim="c", search_query="c", verdict=verdict, confidence=confidence, explanation="", evidence=[]
    )


def test_factuality_from_claims_ignores_undecided_claims():
    risk, confidence = factuality_from_claims(
        [_claim("contradicted", 0.8), _claim("supported", 0.8), _claim("insufficient_evidence", 0)]
    )
    assert risk == 0.5
    assert confidence == pytest.approx(0.533, abs=1e-3)
    assert factuality_from_claims([_claim("insufficient_evidence", 0)]) == (None, 0.0)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, **kwargs):
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(self.responses))))


class FakeSearch:
    def __init__(self, results):
        self.results = results

    def search(self, query, limit):
        return self.results


def test_rag_gating_downgrades_low_similarity_snippet_and_ranks_sources():
    sources = [
        {
            "title": "Giá vàng hôm nay",
            "url": "https://vnexpress.net/gia-vang",
            "snippet": "Giá vàng tăng mạnh lên 80 triệu đồng mỗi lượng.",
            "source": "vnexpress.net",
            "published_at": "",
        },
        {
            "title": "Bộ Y tế bác bỏ tin nước chanh chữa ung thư",
            "url": "https://tuoitre.vn/nuoc-chanh",
            "snippet": "Bộ Y tế khẳng định uống nước chanh không chữa khỏi ung thư.",
            "source": "tuoitre.vn",
            "published_at": "",
        },
    ]
    client = FakeClient(
        [
            {"claims": [{"claim": "Uống nước chanh chữa khỏi ung thư", "search_query": "nước chanh ung thư"}]},
            {
                "verdict": "contradicted",
                "confidence": 0.9,
                "explanation": "",
                "source_findings": [
                    {"source_index": 0, "stance": "contradicts"},
                    {"source_index": 1, "stance": "supports"},
                ],
            },
        ]
    )

    report = FactChecker(client=client, searcher=FakeSearch(sources)).check("Nước chanh chữa ung thư.")
    evidence = report.claims[0].evidence

    assert evidence[0].url == "https://tuoitre.vn/nuoc-chanh"
    assert evidence[0].similarity > evidence[1].similarity
    # Sau khi xếp hạng, index 0 là bài liên quan; bài giá vàng bị hạ xuống context.
    assert evidence[0].stance == "contradicts"
    assert evidence[1].stance == "context"
    assert report.claims[0].verdict == "contradicted"
    assert report.factuality_risk == 1.0
    assert report.retrieval_backend == "tfidf-char-ngram"


def test_fact_check_reports_are_persisted(tmp_path):
    store = AnalysisStore(str(tmp_path / "deepguard.db"))
    saved = store.add_fact_check(
        FactCheckReport(summary="ok", claims=[_claim("supported", 0.7)], factuality_risk=0.0),
        "input text",
    )

    assert saved.reportId.startswith("fc-")
    assert store.get_fact_check(saved.reportId).claims[0].verdict == "supported"
    assert [item.reportId for item in store.list_fact_checks()] == [saved.reportId]

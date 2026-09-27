from app.detectors.language.detector import LanguageDetector
from app.detectors.language import preprocessing
from app.detectors.language.preprocessing import preprocess_text, vectorize_text


def test_preprocessing_normalizes_unicode_and_extracts_indicators():
    result = preprocess_text("Cha\u0300o  https://bit.ly/abc  0912345678")

    assert result["text"].startswith("Chào")
    assert result["urls"] == ["https://bit.ly/abc"]
    assert result["phone_numbers"]


def test_preprocessing_falls_back_when_tokenizers_are_unavailable(monkeypatch):
    monkeypatch.setattr(preprocessing, "word_tokenize", None)
    monkeypatch.setattr(preprocessing, "ViTokenizer", None)

    assert preprocess_text("Xin chào")["tokens"] == ["xin", "chào"]


def test_vectorize_text_uses_vietnamese_word_tokenizer(monkeypatch):
    monkeypatch.setattr(preprocessing, "word_tokenize", lambda text, format: "học_sinh")

    assert vectorize_text("Học sinh") == "học_sinh"


def test_normal_message_is_safe_without_model():
    result = LanguageDetector().analyze("Cuộc họp sẽ diễn ra vào lúc 9 giờ sáng mai.")

    assert result.label == "SAFE"
    assert result.riskScore == 0
    assert result.detector == "text-rules-v1"


def test_single_urgent_request_is_suspicious():
    result = LanguageDetector().analyze("Xác minh ngay để tiếp tục sử dụng dịch vụ.")

    assert result.label == "SUSPICIOUS"
    assert 30 <= result.riskScore < 70


def test_official_bank_subdomain_is_not_flagged_as_impersonation():
    result = LanguageDetector().analyze("Thông tin tại https://www.vietcombank.com.vn")

    assert result.label == "SAFE"
    assert not any("giả mạo" in item for item in result.evidence)


def test_urgent_otp_and_shortened_url_is_dangerous():
    result = LanguageDetector().analyze(
        "Tài khoản sẽ bị khóa trong vòng 24h. Xác minh ngay và gửi OTP tại https://bit.ly/abc"
    )

    assert result.label == "DANGEROUS"
    assert result.riskScore >= 70
    assert any("URL rút gọn" in item for item in result.evidence)

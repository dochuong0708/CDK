from app.detectors.language.detector import LanguageDetector
from app.detectors.language.model import LanguageModel
from app.detectors.language import preprocessing
from app.detectors.language.preprocessing import preprocess_text, vectorize_text


def test_preprocessing_normalizes_unicode_and_extracts_indicators():
    result = preprocess_text("Cha\u0300o  https://bit.ly/abc  0912345678")

    assert result["text"].startswith("Chào")
    assert result["urls"] == ["https://bit.ly/abc"]
    assert result["phone_numbers"]


def test_language_model_does_not_auto_load_unvalidated_local_artifact(monkeypatch):
    monkeypatch.delenv("DEEPGUARD_LANGUAGE_MODEL_PATH", raising=False)

    assert LanguageModel().model is None


def test_official_news_sample_is_safe_without_an_explicit_classifier(monkeypatch):
    from app.services.article_analyzer import ArticleAnalyzer

    monkeypatch.delenv("DEEPGUARD_LANGUAGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("DEEPGUARD_PHOBERT_ONNX_DIR", raising=False)
    monkeypatch.delenv("DEEPGUARD_REFERENCE_CORPUS", raising=False)
    monkeypatch.delenv("DEEPGUARD_NLI_ONNX_DIR", raising=False)
    text = (
        "UBND thành phố Hà Nội vừa ban hành kế hoạch cao điểm về kiểm tra, kiểm soát thị trường "
        "và bảo đảm an toàn vệ sinh thực phẩm dịp cuối năm và Tết Nguyên đán.\n\n"
        "Theo đó, các đoàn kiểm tra liên ngành sẽ thanh tra cơ sở kinh doanh thực phẩm. "
        "Cơ quan chức năng yêu cầu xử lý nghiêm các trường hợp vi phạm về nhãn mác.\n\n"
        "Sở Y tế Hà Nội khuyến cáo người tiêu dùng chọn mua thực phẩm tại cửa hàng có giấy phép."
    )

    result = ArticleAnalyzer().analyze_article(text, check_facts=False)

    assert result["summary"]["label"] == "SAFE"
    assert result["summary"]["overall_risk_score"] < 30
    assert result["ai_components"] == []


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
    assert result.detector == "text-rules-v2"


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

from datetime import datetime

from fastapi.testclient import TestClient
import app.main as main
from app.main import app
from app.models.fact_checking import ClaimFactCheck, FactCheckReport
from app.models.schemas import Analysis

client = TestClient(app)


def test_health(): assert client.get('/api/health').json()['status'] == 'healthy'


def test_text_analysis():
    response = client.post('/api/analyze/text', json={'text':'The committee approved the revised climate brief.'})
    assert response.status_code == 200
    assert response.json()['modality'] == 'Text'


def test_fact_check_text(monkeypatch):
    report = FactCheckReport(
        summary='Đã đối chiếu 1 luận điểm.',
        claims=[
            ClaimFactCheck(
                claim='Sự kiện X xảy ra năm 2024.',
                search_query='Sự kiện X 2024',
                verdict='supported',
                confidence=0.8,
                explanation='Nguồn tìm được hỗ trợ luận điểm.',
                evidence=[],
            )
        ],
    )
    monkeypatch.setattr(main.fact_checker, 'check', lambda text: report)

    response = client.post('/api/fact-check/text', json={'text': 'Sự kiện X xảy ra năm 2024.'})

    assert response.status_code == 200
    assert response.json()['claims'][0]['verdict'] == 'supported'


def test_corrupt_image_upload_returns_400():
    response = client.post(
        '/api/analyze/media',
        files={'file': ('broken.png', b'not an image', 'image/png')},
        data={'modality': 'Image'},
    )

    assert response.status_code == 400
    assert 'not a valid supported image' in response.json()['detail']


def test_image_upload_is_analyzed_from_bytes_without_saving(
    monkeypatch, tmp_path
):
    uploaded = {}
    result = Analysis(
        analysisId="image-upload-test",
        modality="Image",
        label="Uncertain",
        riskScore=50,
        confidence=0.5,
        evidence=[],
        detector="test-detector",
        input="sample.png",
        createdAt=datetime.now(),
    )

    def analyze_image(file_bytes: bytes, filename: str) -> Analysis:
        uploaded["file_bytes"] = file_bytes
        uploaded["filename"] = filename
        return result

    monkeypatch.setattr(main.orchestrator, "analyze_image", analyze_image)
    monkeypatch.setattr(main.store, "add", lambda analysis: analysis)
    monkeypatch.setattr(main, "upload_dir", tmp_path)

    response = client.post(
        "/api/analyze/media",
        files={"file": ("sample.png", b"image bytes", "image/png")},
        data={"modality": "Image"},
    )

    assert response.status_code == 200
    assert uploaded == {"file_bytes": b"image bytes", "filename": "sample.png"}
    assert not (tmp_path / "sample.png").exists()


def test_article_analysis(monkeypatch):
    report = FactCheckReport(
        summary='Kiểm chứng bài viết hoàn tất.',
        claims=[],
    )
    monkeypatch.setattr(main.fact_checker, 'check', lambda text: report)

    article_text = (
        "Bộ Công An phát đi cảnh báo chấn động.\n\n"
        "Người dân cần cảnh giác trước thông tin lan truyền trên mạng xã hội."
    )
    response = client.post(
        '/api/analyze/article',
        json={'text': article_text, 'title': 'Tin thử nghiệm', 'check_facts': True},
    )
    assert response.status_code == 200
    data = response.json()
    assert 'summary' in data
    assert 'paragraphs' in data
    assert len(data['paragraphs']) == 2
    assert data['summary']['overall_risk_score'] >= 0
    assert 'verdict_title' in data['summary']


def test_fact_check_is_saved_and_merged_into_text_analysis(monkeypatch):
    report = FactCheckReport(
        summary='1 luận điểm bị mâu thuẫn.',
        claims=[
            ClaimFactCheck(
                claim='Nước chanh chữa khỏi ung thư.',
                search_query='nước chanh ung thư',
                verdict='contradicted',
                confidence=0.9,
                explanation='Nguồn bác bỏ.',
                evidence=[],
            )
        ],
        factuality_risk=1.0,
        factuality_confidence=0.9,
    )
    monkeypatch.setattr(main.fact_checker, 'check', lambda text: report)

    response = client.post(
        '/api/analyze/text', json={'text': 'Nước chanh chữa khỏi ung thư.', 'check_facts': True}
    )
    data = response.json()

    assert response.status_code == 200
    assert data['factualityLabel'] == 'DANGEROUS'
    assert data['label'] == 'DANGEROUS'
    assert data['factCheckId']
    saved = client.get(f"/api/fact-checks/{data['factCheckId']}")
    assert saved.status_code == 200
    assert saved.json()['claims'][0]['verdict'] == 'contradicted'
    assert client.get(f"/api/analyses/{data['analysisId']}").json()['factCheckId'] == data['factCheckId']
    assert client.get('/api/fact-checks').json()['total'] >= 1

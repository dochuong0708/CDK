from fastapi.testclient import TestClient
import app.main as main
from app.main import app
from app.models.fact_checking import ClaimFactCheck, FactCheckReport

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

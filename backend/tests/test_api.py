from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get('/api/health').json()['status'] == 'healthy'
def test_text_analysis():
    response = client.post('/api/analyze/text', json={'text':'The committee approved the revised climate brief.'})
    assert response.status_code == 200
    assert response.json()['modality'] == 'Text'

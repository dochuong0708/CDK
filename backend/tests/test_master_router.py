import io
import json

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.api.routes.agent import close_service
    from app.main import app
    close_service()
    monkeypatch.setenv("DEEPGUARD_AGENT_DIR", str(tmp_path/"agent"))
    monkeypatch.setenv("DEEPGUARD_AUDIO_MODEL", str(tmp_path/"missing.pt"))
    with TestClient(app) as client:
        yield client


def session(client):
    return client.post("/api/agent/sessions").json()["sessionID"]


def test_multimodal_routing_trace_and_history(client):
    sid = session(client)
    prefix = f"/api/master/sessions/{sid}"
    image = io.BytesIO()
    Image.new("RGB", (16,16), "blue").save(image, format="PNG")
    audio = io.BytesIO()
    sf.write(audio, .2*np.sin(2*np.pi*440*np.arange(16000)/16000), 16000, format="WAV")
    handles = []
    for name, content, kind in [("sample.wav", audio.getvalue(), "audio"), ("sample.png", image.getvalue(), "image")]:
        response = client.post(prefix+"/files", files={"file":(name,content,"application/octet-stream")})
        assert response.status_code == 201, response.text
        assert response.json()["kind"] == kind
        handles.append(response.json()["file_id"])
    response = client.post(prefix+"/runs", json=dict(file_ids=handles, text="This is text input."))
    assert response.status_code == 200, response.text
    result = response.json()
    assert [t["tool"] for t in result["toolsExecuted"]] == ["text_heuristic_tool", "audio_features_tool", "audio_anomaly_tool", "image_heuristic_tool"]
    assert result["label"] == "SUSPICIOUS" and result["metadata"]["llmUsed"] is False
    route = f"/api/agent/sessions/{sid}/traces/{result['traced']}"
    trace = client.get(route).json()
    assert trace["status"] == "completed"
    assert len([e for e in trace["events"] if e["kind"] == "tool_completed"]) == 4
    assert client.patch(route, json={"note":"Checked"}).json()["note"] == "Checked"
    assert client.patch(route, json={"result":{}}).status_code == 422
    assert client.delete(route).status_code == 200
    assert client.get(route).status_code == 404
    assert client.get(f"/api/agent/sessions/{sid}/history").json()["runs"] == []


def test_stream_websocket_and_input_validation(client):
    sid = session(client)
    prefix = f"/api/master/sessions/{sid}"
    response = client.post(prefix+"/runs/stream", json={"text":"hello"})
    events = [line[7:] for line in response.text.splitlines() if line.startswith("event: ")]
    assert events == ["started", "routed", "tool_completed", "result"]
    with client.websocket_connect(prefix+"/ws") as ws:
        ws.send_json({"text":"another text"})
        assert [ws.receive_json()["event"] for _ in range(4)] == events
    assert client.post(prefix+"/runs", json={}).status_code == 422
    assert client.post(prefix+"/files", files={"file":("fake.png",b"not image")}).status_code == 422
    assert client.post(prefix+"/files", files={"file":("movie.mp4",b"video")}).status_code == 422


def test_session_message_crud_and_cross_session_trace(client):
    a, b = session(client), session(client)
    prefix = f"/api/agent/sessions/{a}"
    assert client.patch(prefix, json={"title":"Audio review"}).json()["title"] == "Audio review"
    assert any(s["id"] == a for s in client.get("/api/agent/sessions").json()["items"])
    message = client.post(prefix+"/messages", json={"text":"old text"}).json()
    path = prefix+f"/messages/{message['id']}"
    assert client.patch(path, json={"text":"updated text"}).json()["text"] == "updated text"
    assert client.get(path.replace(a,b)).status_code == 404
    assert client.delete(path).status_code == 200
    assert client.get(path).status_code == 404
    result = client.post(f"/api/master/sessions/{a}/runs", json={"text":"hello"}).json()
    trace = prefix+f"/traces/{result['traced']}"
    assert client.get(trace.replace(a,b)).status_code == 404
    assert client.delete(trace.replace(a,b)).status_code == 404
    assert client.delete(prefix).json()["deleted"] is True
    assert client.get(prefix+"/history").status_code == 404
    assert client.get(f"/api/agent/sessions/{b}/history").status_code == 200

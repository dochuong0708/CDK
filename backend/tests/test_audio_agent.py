import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.db.memory_store import MemoryStore
from core.tools.audio_features_tool import audio_features_tool
from core.tools.audio_sources import download_url


def recording(format="WAV"):
    buf = io.BytesIO()
    y = .2*np.sin(2*np.pi*440*np.arange(16000)/16000)
    sf.write(buf, y, 16000, format=format)
    return buf.getvalue()


@pytest.mark.parametrize("extension,format", [("wav", "WAV"), ("mp3", "MP3")])
def test_features_decode_both_formats(tmp_path, extension, format):
    path = tmp_path / f"tone.{extension}"
    path.write_bytes(recording(format))
    result = audio_features_tool(path)
    assert len(result["mfcc"]["mean"]) == 20
    assert len(result["spectrogram"]["valuesDb"]) == 80
    assert .9 < result["duration"] < 1.2
    json.dumps(result, allow_nan=False)


def test_memory_persists_and_filters_sessions(tmp_path):
    memory = MemoryStore(tmp_path)
    first, second = memory.create_session(), memory.create_session()
    memory.add_message(first, "user", "MFCC am thanh")
    memory.add_message(second, "user", "MFCC private recording")
    memory.close()
    reopened = MemoryStore(tmp_path)
    try:
        assert len(reopened.history(first)["messages"]) == 1
        hits = reopened.search(first, "MFCC")
        assert len(hits) == 1 and hits[0]["session"] == first
    finally:
        reopened.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.api.routes.agent import close_service
    from app.main import app
    close_service()
    monkeypatch.setenv("DEEPGUARD_AGENT_DIR", str(tmp_path / "agent"))
    monkeypatch.setenv("DEEPGUARD_AUDIO_MODEL", str(tmp_path / "missing.pt"))
    with TestClient(app) as client:
        yield client


def upload(client):
    session = client.post("/api/agent/sessions").json()["sessionID"]
    prefix = f"/api/agent/sessions/{session}"
    response = client.post(prefix+"/files", files={"file": ("tone.wav", recording(), "audio/wav")})
    assert response.status_code == 201, response.text
    return session, prefix, response.json()["file_id"]


def test_http_stream_history_schema_and_tool_call(client):
    session, prefix, file_id = upload(client)
    schemas = client.get("/api/agent/tools").json()["tools"]
    assert {t["name"] for t in schemas} == {"audio_features_tool", "audio_anomaly_tool"}
    response = client.post(prefix+"/tools/call", json=dict(name="audio_features_tool", arguments=dict(file_id=file_id)))
    assert response.status_code == 200 and response.json()["mfcc"]["count"] == 20
    response = client.post(prefix+"/runs/stream", json=dict(file_id=file_id))
    assert response.status_code == 200
    events = [line[7:] for line in response.text.splitlines() if line.startswith("event: ")]
    assert events == ["started", "tool_completed", "tool_completed", "result"]
    history = client.get(prefix+"/history").json()
    assert len(history["messages"]) == 2 and len(history["runs"]) == 1
    result = history["runs"][0]
    assert result["sessionID"] == session and result["riskScore"] == .5
    assert result["metadata"]["modelStatus"] == "missing"
    assert result["label"] == "SUSPICIOUS"
    assert client.get(prefix+"/memory", params={"q": "âm thanh"}).json()["items"]
    response = client.post(prefix+"/runs", json=dict(file_id=file_id))
    assert response.status_code == 200 and response.json()["traced"] != result["traced"]


def test_websocket_and_cross_session_handles(client):
    session, prefix, file_id = upload(client)
    with client.websocket_connect(prefix+"/ws") as ws:
        ws.send_json(dict(file_id=file_id))
        assert [ws.receive_json()["event"] for _ in range(4)] == ["started", "tool_completed", "tool_completed", "result"]
    other = client.post("/api/agent/sessions").json()["sessionID"]
    response = client.post(f"/api/agent/sessions/{other}/runs", json=dict(file_id=file_id))
    assert response.status_code == 404
    assert client.post(prefix+"/tools/call", json=dict(name="shell", arguments=dict(file_id=file_id))).status_code == 422
    assert client.post(prefix+"/tools/call", json=dict(name="audio_features_tool", arguments=dict(file_id=file_id, path="C:/secret"))).status_code == 422


def test_invalid_audio_upload_is_removed(client):
    session = client.post("/api/agent/sessions").json()["sessionID"]
    response = client.post(f"/api/agent/sessions/{session}/files", files={"file": ("bad.wav", b"invalid", "audio/wav")})
    assert response.status_code == 422
    from app.api.routes.agent import service
    assert not list((service().memory.root / "uploads").iterdir())


def test_mp3_cnn_xai_and_artifact_isolation(client, tmp_path, monkeypatch):
    import torch
    from app.detectors.audio.model import AudioCNN
    from app.detectors.audio.pipeline import CONFIG
    from app.api.routes.agent import close_service
    torch.set_num_threads(2)
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(architecture="audio-cnn-v1", config=CONFIG, classes=["real", "fake"],
                    threshold=.5, state_dict=AudioCNN().state_dict()), checkpoint)
    close_service()
    monkeypatch.setenv("DEEPGUARD_AUDIO_MODEL", str(checkpoint))
    session = client.post("/api/agent/sessions").json()["sessionID"]
    prefix = f"/api/agent/sessions/{session}"
    uploaded = client.post(prefix+"/files", files={"file": ("tone.mp3", recording("MP3"), "audio/mpeg")})
    assert uploaded.status_code == 201
    result = client.post(prefix+"/runs", json=dict(file_id=uploaded.json()["file_id"]))
    assert result.status_code == 200, result.text
    data = result.json()
    assert 0 <= data["riskScore"] <= 1
    anomaly = data["metadata"]["anomaly"]
    assert anomaly["status"] == "ok" and anomaly["modelStatus"] == "unvalidated"
    assert anomaly["xaiMethod"] == "Grad-CAM"
    assert "xaiPngBase64" not in anomaly
    png = client.get(anomaly["xaiUrl"])
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG\r\n\x1a\n")
    other = client.post("/api/agent/sessions").json()["sessionID"]
    assert client.get(anomaly["xaiUrl"].replace(session, other)).status_code == 404


def test_url_blocks_internal_and_unlisted_hosts(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        download_url("http://localhost/a.wav", tmp_path/"a.wav", {"localhost"})
    with pytest.raises(ValueError):
        download_url("https://unlisted.test/a.wav", tmp_path/"a.wav", set())
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError):
        download_url("https://allowed.test/a.wav", tmp_path/"a.wav", {"allowed.test"})


def test_url_intake_pinned_and_no_redirect(tmp_path, monkeypatch):
    import core.tools.audio_sources as sources
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    class Response(io.BytesIO):
        status = 200
        def getheader(self, *args):
            return "0"
    class Connection:
        def __init__(self, host, address):
            assert host == "allowed.test" and address == "93.184.216.34"
        def request(self, *a, **k): pass
        def getresponse(self): return Response(recording())
        def close(self): pass
    monkeypatch.setattr(sources, "PinnedHTTPS", Connection)
    target = tmp_path/"a.wav"
    download_url("https://allowed.test/a.wav", target, {"allowed.test"})
    assert target.read_bytes() == recording()
    Response.status = 302
    with pytest.raises(ValueError):
        download_url("https://allowed.test/a.wav", target, {"allowed.test"})
    assert not target.exists()

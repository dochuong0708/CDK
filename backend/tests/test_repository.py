import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from app.db.memory_store import MemoryStore
from app.db.repository import ConflictError
from app.models.agent_schema import RunRequest
from core.agents.audio_agent import AudioAgent


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(tmp_path/"memory")
    yield store
    store.close()


def test_message_crud_vector_consistency_and_isolation(memory):
    a, b = memory.create_session(), memory.create_session()
    repo = memory.repository
    message = repo.create_message(a, "user", "MFCC original")
    assert memory.search(a, "MFCC")[0]["text"] == "MFCC original"
    repo.update_message(a, message["id"], "spectrogram updated")
    assert memory.search(a, "spectrogram")[0]["text"] == "spectrogram updated"
    with pytest.raises(KeyError):
        repo.update_message(b, message["id"], "wrong session")
    assert memory.search(b, "spectrogram") == []
    repo.delete_message(a, message["id"])
    assert memory.search(a, "spectrogram") == []
    assert memory.history(a)["messages"] == []


def test_vector_failure_outbox_survives_restart(tmp_path, monkeypatch):
    root = tmp_path/"memory"
    memory = MemoryStore(root)
    session = memory.create_session()
    with monkeypatch.context() as patch:
        patch.setattr(memory, "_index", lambda row: (_ for _ in ()).throw(RuntimeError("offline")))
        message = memory.repository.create_message(session, "user", "pending MFCC")
        assert memory.db.execute("SELECT count(*) FROM vector_outbox").fetchone()[0] == 1
        assert memory.repository.get_message(session, message["id"])["text"] == "pending MFCC"
    memory.close()
    reopened = MemoryStore(root)
    try:
        assert reopened.repository.sync_vectors() == 0
        assert reopened.search(session, "MFCC")[0]["text"] == "pending MFCC"
        with monkeypatch.context() as patch:
            patch.setattr(reopened.vector, "delete", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
            reopened.repository.delete_message(session, message["id"])
            assert reopened.search(session, "MFCC") == []  # Stale Qdrant point is suppressed.
        assert reopened.repository.sync_vectors() == 0
    finally:
        reopened.close()


def test_trace_lifecycle_outputs_and_delete(memory):
    session = memory.create_session()
    repo = memory.repository
    trace = repo.start_trace(session, {"file_id": str(uuid4())}, ["audio_anomaly_tool"], "check audio")
    with pytest.raises(ConflictError):
        repo.delete_trace(session, trace)
    with pytest.raises(ConflictError):
        repo.delete_session(session)
    repo.append_event(session, trace, "tool_completed", dict(tool="audio_anomaly_tool", status="ok", output=dict(fakeScore=.8, modelStatus="unvalidated")))
    result = dict(sessionID=session, traced=trace, toolsExecuted=[dict(tool="audio_anomaly_tool")], finalResponse="Needs review")
    repo.finish_trace(session, trace, result)
    saved = repo.get_trace(session, trace)
    assert saved["status"] == "completed" and saved["result"] == result
    assert [x["sequence"] for x in saved["events"]] == [1,2,3]
    assert saved["events"][1]["payload"]["output"]["fakeScore"] == .8
    assert "audio_anomaly_tool" in memory.search(session, "audio_anomaly_tool")[0]["memory_text"]
    repo.annotate_trace(session, trace, "reviewed by student")
    assert repo.get_trace(session, trace)["note"] == "reviewed by student"
    with pytest.raises(ConflictError):
        repo.finish_trace(session, trace, result)
    repo.delete_trace(session, trace)
    assert memory.history(session) == dict(messages=[], runs=[])
    assert memory.search(session, "audio") == []


def test_parallel_events_have_unique_sequence(memory):
    session = memory.create_session()
    trace = memory.repository.start_trace(session, {}, [], "parallel log")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: memory.repository.append_event(session, trace, "step", {"number":n}), range(24)))
    events = memory.repository.get_trace(session, trace)["events"]
    assert [e["sequence"] for e in events] == list(range(1,26))


def test_restart_marks_running_interrupted(tmp_path):
    memory = MemoryStore(tmp_path)
    session = memory.create_session()
    trace = memory.repository.start_trace(session, {}, [], "running")
    memory.close()
    reopened = MemoryStore(tmp_path)
    try:
        result = reopened.repository.get_trace(session, trace)
        assert result["status"] == "interrupted"
        assert result["events"][-1]["kind"] == "interrupted"
    finally:
        reopened.close()


def test_migrate_week5_database_without_losing_records(tmp_path):
    session, message = str(uuid4()), str(uuid4())
    db = sqlite3.connect(tmp_path/"sessions.db")
    db.executescript("CREATE TABLE sessions(id TEXT PRIMARY KEY); CREATE TABLE messages(id TEXT PRIMARY KEY,session TEXT,role TEXT,text TEXT,created TEXT);")
    db.execute("INSERT INTO sessions VALUES (?)", (session,))
    db.execute("INSERT INTO messages VALUES (?,?,?,?,?)", (message,session,"user","old conversation","2026-10-04"))
    db.commit()
    db.close()
    memory = MemoryStore(tmp_path)
    try:
        assert memory.history(session)["messages"][0]["text"] == "old conversation"
        assert memory.search(session, "conversation")[0]["id"] == message
        memory.repository.update_session(session, "Renamed")
        assert memory.repository.list_sessions()[0]["title"] == "Renamed"
    finally:
        memory.close()


def test_failed_and_cancelled_audio_run_are_saved(memory, tmp_path, monkeypatch):
    session = memory.create_session()
    file_id = str(uuid4())
    memory.add_file(session, file_id, "a.wav", tmp_path/"a.wav")
    agent = AudioAgent(memory, str(tmp_path/"missing.pt"))
    monkeypatch.setattr(agent, "execute_tool", lambda *a, **k: (_ for _ in ()).throw(ValueError("broken tool")))
    async def fail():
        events = agent.events(session, RunRequest(file_id=file_id))
        first = await anext(events)
        with pytest.raises(ValueError):
            await anext(events)
        return first["data"]["traced"]
    trace = asyncio.run(fail())
    saved = memory.repository.get_trace(session, trace)
    assert saved["status"] == "failed"
    assert "tool_failed" in [e["kind"] for e in saved["events"]]
    async def cancel():
        events = agent.events(session, RunRequest(file_id=file_id))
        first = await anext(events)
        await events.aclose()
        return first["data"]["traced"]
    trace = asyncio.run(cancel())
    assert memory.repository.get_trace(session, trace)["status"] == "cancelled"

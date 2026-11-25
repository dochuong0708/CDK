"""Week 7: transactional conversation/trace CRUD with a durable vector outbox.

Only public execution summaries are stored; no private LLM reasoning.
SQLite is authoritative. Failed vector writes stay queued and are retried.
"""
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client import models

log = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class ConflictError(ValueError):
    pass


class ConversationRepository:
    def __init__(self, memory):
        self.memory, self.db, self.lock = memory, memory.db, memory.lock
        with self.lock, self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS agent_traces(
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, status TEXT NOT NULL,
                    input TEXT NOT NULL, selected_tools TEXT NOT NULL, result TEXT,
                    error TEXT, note TEXT NOT NULL DEFAULT '', created TEXT NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trace_events(
                    trace TEXT NOT NULL, sequence INTEGER NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, created TEXT NOT NULL, PRIMARY KEY(trace, sequence));
                CREATE TABLE IF NOT EXISTS vector_outbox(message_id TEXT PRIMARY KEY);
                CREATE INDEX IF NOT EXISTS idx_trace_session ON agent_traces(session, created);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session);
            """)
            for table, column, declaration in (
                ("sessions", "title", "TEXT NOT NULL DEFAULT ''"),
                ("messages", "trace_id", "TEXT"),
                ("messages", "memory_text", "TEXT"),
                ("files", "kind", "TEXT NOT NULL DEFAULT 'audio'"),
            ):
                if column not in {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
            # Existing messages from week 5 migrate without changing IDs or content.
            self.db.execute("INSERT OR IGNORE INTO vector_outbox SELECT id FROM messages")
            self.db.execute("PRAGMA user_version=2")
            for row in self.db.execute("SELECT id FROM agent_traces WHERE status='running'").fetchall():
                self.db.execute("UPDATE agent_traces SET status='interrupted',error=?,updated=? WHERE id=?",
                                (encode({"code": "server_restarted"}), now(), row[0]))
                self._event(row[0], "interrupted", {"code": "server_restarted"})
        self.sync_vectors()

    def sync_vectors(self):
        with self.lock:
            for row in self.db.execute("SELECT message_id FROM vector_outbox").fetchall():
                message_id = row[0]
                message = self.db.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
                try:
                    if message:
                        self.memory._index(dict(message))
                    else:
                        self.memory.vector.delete("audio_memory", points_selector=models.PointIdsList(points=[message_id]))
                except Exception:
                    log.warning("Vector sync pending; SQLite data was saved", exc_info=True)
                    continue
                with self.db:
                    self.db.execute("DELETE FROM vector_outbox WHERE message_id=?", (message_id,))
            return self.db.execute("SELECT count(*) FROM vector_outbox").fetchone()[0]

    def _message(self, session, role, text, trace_id=None, memory_text=None):
        message_id = str(uuid4())
        self.db.execute("INSERT INTO messages(id,session,role,text,trace_id,memory_text) VALUES (?,?,?,?,?,?)",
                        (message_id, session, role, text, trace_id, memory_text))
        self.db.execute("INSERT OR IGNORE INTO vector_outbox VALUES (?)", (message_id,))
        return message_id

    def create_message(self, session, role, text):
        if role not in {"user", "assistant"} or not text.strip() or len(text) > 8000:
            raise ValueError("Tin nhắn phải có role user/assistant và nội dung 1–8000 ký tự.")
        with self.lock, self.db:
            self.memory.require_session(session)
            message_id = self._message(session, role, text)
        self.sync_vectors()
        return self.get_message(session, message_id)

    def get_message(self, session, message_id):
        with self.lock:
            self.memory.require_session(session)
            row = self.db.execute("SELECT * FROM messages WHERE session=? AND id=?", (session, message_id)).fetchone()
            if row is None:
                raise KeyError("Không tìm thấy tin nhắn trong phiên.")
            return dict(row)

    def update_message(self, session, message_id, text):
        if not text.strip() or len(text) > 8000:
            raise ValueError("Nội dung phải dài 1–8000 ký tự.")
        with self.lock, self.db:
            message = self.get_message(session, message_id)
            if message["trace_id"]:
                raise ConflictError("Tin nhắn gắn với trace không được sửa; dùng ghi chú trace.")
            self.db.execute("UPDATE messages SET text=? WHERE session=? AND id=?", (text, session, message_id))
            self.db.execute("INSERT OR IGNORE INTO vector_outbox VALUES (?)", (message_id,))
        self.sync_vectors()
        return self.get_message(session, message_id)

    def delete_message(self, session, message_id):
        with self.lock, self.db:
            message = self.get_message(session, message_id)
            if message["trace_id"]:
                raise ConflictError("Xóa trace để xóa đồng bộ tin nhắn liên quan.")
            self.db.execute("DELETE FROM messages WHERE id=?", (message_id,))
            self.db.execute("INSERT OR IGNORE INTO vector_outbox VALUES (?)", (message_id,))
        self.sync_vectors()

    def list_sessions(self, limit=50, offset=0):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT id,title FROM sessions ORDER BY rowid DESC LIMIT ? OFFSET ?", (limit, offset))]

    def update_session(self, session, title):
        with self.lock, self.db:
            self.memory.require_session(session)
            self.db.execute("UPDATE sessions SET title=? WHERE id=?", (title, session))
        return dict(sessionID=session, title=title)

    def _event(self, trace, kind, payload):
        sequence = self.db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM trace_events WHERE trace=?", (trace,)).fetchone()[0]
        self.db.execute("INSERT INTO trace_events VALUES (?,?,?,?,?)", (trace, sequence, kind, encode(payload), now()))

    def start_trace(self, session, request, selected_tools, message):
        trace = str(uuid4())
        with self.lock, self.db:
            self.memory.require_session(session)
            self.db.execute("INSERT INTO agent_traces(id,session,status,input,selected_tools,created,updated) VALUES (?,?,?,?,?,?,?)",
                            (trace, session, "running", encode(request), encode(selected_tools), now(), now()))
            self._event(trace, "started", {"selectedTools": selected_tools})
            self._message(session, "user", message, trace)
        self.sync_vectors()
        return trace

    def append_event(self, session, trace, kind, payload):
        with self.lock, self.db:
            record = self.get_trace(session, trace, events=False)
            if record["status"] != "running":
                return  # A worker may finish after the streaming client disconnected.
            self._event(trace, kind, payload)
            self.db.execute("UPDATE agent_traces SET updated=? WHERE id=?", (now(), trace))

    def finish_trace(self, session, trace, result):
        if result.get("sessionID") != session or result.get("traced") != trace:
            raise ValueError("Result không khớp session/trace.")
        with self.lock, self.db:
            if self.get_trace(session, trace, events=False)["status"] != "running":
                raise ConflictError("Trace đã kết thúc.")
            self._event(trace, "result", result)
            self.db.execute("UPDATE agent_traces SET status='completed',result=?,updated=? WHERE id=?", (encode(result), now(), trace))
            self.db.execute("INSERT INTO runs VALUES (?,?,?)", (trace, session, encode(result)))
            events = [dict(kind=r[0], payload=json.loads(r[1])) for r in self.db.execute(
                "SELECT kind,payload FROM trace_events WHERE trace=? AND kind IN ('tool_completed','tool_failed') ORDER BY sequence", (trace,))]
            summaries = []
            for event in events:
                payload = event["payload"]
                output = payload.get("output", {})
                summaries.append(dict(tool=payload.get("tool"), status=payload.get("status"),
                    output={key: output[key] for key in ("fakeScore", "predictedClass", "modelStatus", "signalFlags", "riskScore", "detector", "note") if key in output}))
            memory_text = encode(dict(finalAnswer=result["finalResponse"], selectedTools=result["toolsExecuted"], toolOutputs=summaries))
            self._message(session, "assistant", result["finalResponse"], trace, memory_text)
        self.sync_vectors()

    def fail_trace(self, session, trace, code, status="failed"):
        if status not in {"failed", "cancelled"}:
            raise ValueError("Trạng thái lỗi không hợp lệ.")
        with self.lock, self.db:
            row = self.db.execute("SELECT status FROM agent_traces WHERE session=? AND id=?", (session, trace)).fetchone()
            if row and row[0] == "running":
                self._event(trace, status, {"code": code})
                self.db.execute("UPDATE agent_traces SET status=?,error=?,updated=? WHERE id=?",
                                (status, encode({"code": code}), now(), trace))
                tools = self.db.execute("SELECT selected_tools FROM agent_traces WHERE id=?", (trace,)).fetchone()[0]
                text = f"Lượt xử lý kết thúc với trạng thái {status}: {code}."
                self._message(session, "assistant", text, trace, encode(dict(status=status, selectedTools=json.loads(tools), error=code)))
        self.sync_vectors()

    def get_trace(self, session, trace, events=True):
        with self.lock:
            self.memory.require_session(session)
            row = self.db.execute("SELECT * FROM agent_traces WHERE session=? AND id=?", (session, trace)).fetchone()
            if row is None:
                raise KeyError("Không tìm thấy trace trong phiên.")
            result = dict(row)
            for key in ("input", "selected_tools", "result", "error"):
                result[key] = json.loads(result[key]) if result[key] is not None else None
            if events:
                result["events"] = [dict(sequence=r[0], kind=r[1], payload=json.loads(r[2]), created=r[3])
                    for r in self.db.execute("SELECT sequence,kind,payload,created FROM trace_events WHERE trace=? ORDER BY sequence", (trace,))]
            return result

    def list_traces(self, session, limit=50, offset=0):
        with self.lock:
            self.memory.require_session(session)
            return [dict(r) for r in self.db.execute("SELECT id,status,note,created,updated FROM agent_traces WHERE session=? ORDER BY rowid DESC LIMIT ? OFFSET ?", (session, limit, offset))]

    def annotate_trace(self, session, trace, note):
        with self.lock, self.db:
            self.get_trace(session, trace, events=False)
            self.db.execute("UPDATE agent_traces SET note=?,updated=? WHERE id=?", (note, now(), trace))
        return self.get_trace(session, trace)

    def delete_trace(self, session, trace):
        with self.lock, self.db:
            record = self.get_trace(session, trace, events=False)
            if record["status"] == "running":
                raise ConflictError("Không xóa trace đang chạy.")
            self.db.execute("INSERT OR IGNORE INTO vector_outbox SELECT id FROM messages WHERE trace_id=? AND session=?", (trace, session))
            self.db.execute("DELETE FROM messages WHERE trace_id=? AND session=?", (trace, session))
            self.db.execute("DELETE FROM trace_events WHERE trace=?", (trace,))
            self.db.execute("DELETE FROM runs WHERE id=? AND session=?", (trace, session))
            self.db.execute("DELETE FROM agent_traces WHERE id=? AND session=?", (trace, session))
        self.sync_vectors()
        return record

    def delete_session(self, session):
        with self.lock, self.db:
            self.memory.require_session(session)
            if self.db.execute("SELECT 1 FROM agent_traces WHERE session=? AND status='running'", (session,)).fetchone():
                raise ConflictError("Không xóa phiên đang chạy.")
            paths = [r[0] for r in self.db.execute("SELECT path FROM files WHERE session=?", (session,))]
            self.db.execute("INSERT OR IGNORE INTO vector_outbox SELECT id FROM messages WHERE session=?", (session,))
            self.db.execute("DELETE FROM trace_events WHERE trace IN (SELECT id FROM agent_traces WHERE session=?)", (session,))
            for table in ("agent_traces", "messages", "runs", "files"):
                self.db.execute(f"DELETE FROM {table} WHERE session=?", (session,))
            self.db.execute("DELETE FROM sessions WHERE id=?", (session,))
        self.sync_vectors()
        return paths

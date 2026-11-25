"""SQLite session history and durable local Qdrant vector memory.

The offline encoder is hashed character n-grams, NOT a pretrained semantic model.
SQLite is authoritative; vector entries are rebuilt from it when opening the store.
"""
import json
import sqlite3
from pathlib import Path
from threading import RLock
from uuid import uuid4

from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import HashingVectorizer


class MemoryStore:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.lock = RLock()
        self.db = sqlite3.connect(root / "sessions.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY, session TEXT, name TEXT, path TEXT);
            CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY, session TEXT, role TEXT, text TEXT,
                created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, session TEXT, result TEXT);
        """)
        self.encoder = HashingVectorizer(n_features=384, analyzer="char", ngram_range=(2, 4),
                                        alternate_sign=False, norm="l2")
        self.vector = QdrantClient(path=str(root / "qdrant"))
        if not self.vector.collection_exists("audio_memory"):
            self.vector.create_collection("audio_memory", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        from app.db.repository import ConversationRepository
        self.repository = ConversationRepository(self)

    def _index(self, row):
        vector = self.encoder.transform([row.get("memory_text") or row["text"]]).toarray()[0].tolist()
        self.vector.upsert("audio_memory", points=[models.PointStruct(id=row["id"], vector=vector, payload=row)])

    def create_session(self):
        session = str(uuid4())
        with self.lock, self.db:
            self.db.execute("INSERT INTO sessions(id) VALUES (?)", (session,))
        return session

    def require_session(self, session):
        with self.lock:
            if not self.db.execute("SELECT 1 FROM sessions WHERE id=?", (session,)).fetchone():
                raise KeyError("Không tìm thấy phiên.")

    def add_file(self, session, file_id, name, path, kind="audio"):
        self.require_session(session)
        with self.lock, self.db:
            self.db.execute("INSERT INTO files(id,session,name,path,kind) VALUES (?,?,?,?,?)", (file_id, session, name, str(path), kind))

    def file_kind(self, session, file_id):
        self.file_path(session, file_id)
        with self.lock:
            return self.db.execute("SELECT kind FROM files WHERE session=? AND id=?", (session, str(file_id))).fetchone()[0]

    def file_path(self, session, file_id):
        self.require_session(session)
        with self.lock:
            row = self.db.execute("SELECT path FROM files WHERE session=? AND id=?", (session, str(file_id))).fetchone()
        if row is None:
            raise KeyError("File không thuộc phiên này.")
        return Path(row["path"])

    def add_message(self, session, role, text):
        return self.repository.create_message(session, role, text)

    def save_run(self, session, result):
        with self.lock, self.db:
            self.db.execute("INSERT INTO runs VALUES (?,?,?)", (result["traced"], session, json.dumps(result, ensure_ascii=False)))

    def history(self, session):
        self.require_session(session)
        with self.lock:
            messages = [dict(r) for r in self.db.execute("SELECT * FROM messages WHERE session=? ORDER BY rowid", (session,))]
            runs = [json.loads(r[0]) for r in self.db.execute("SELECT result FROM runs WHERE session=? ORDER BY rowid", (session,))]
        return dict(messages=messages, runs=runs)

    def search(self, session, query, limit=5):
        self.require_session(session)
        with self.lock:
            self.repository.sync_vectors()
            results = self.vector.query_points("audio_memory", query=self.encoder.transform([query]).toarray()[0].tolist(),
                query_filter=models.Filter(must=[models.FieldCondition(key="session", match=models.MatchValue(value=session))]), limit=limit)
            # Never return a deleted or stale message if the vector backend was unavailable.
            valid = []
            for p in results.points:
                row = self.db.execute("SELECT * FROM messages WHERE id=? AND session=?", (str(p.id), session)).fetchone()
                if row and row["text"] == p.payload.get("text"):
                    valid.append(dict(score=p.score, **dict(row)))
            return valid

    def close(self):
        with self.lock:
            self.vector.close()
            self.db.close()

"""Conversation CRUD; shares session handles with the week 5 API."""
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from app.api.routes.agent import service, session_service
from app.models.agent_schema import MessageCreate, MessageUpdate, SessionUpdate, TraceUpdate

router = APIRouter(prefix="/api/agent", tags=["Conversation memory · week 7"])


def invoke(fn, *args):
    from app.db.repository import ConflictError
    try:
        return fn(*args)
    except KeyError as exc:
        raise HTTPException(404, exc.args[0]) from exc
    except ConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/sessions")
def sessions(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    return dict(items=service().memory.repository.list_sessions(limit, offset))


@router.patch("/sessions/{session}")
def rename_session(session: UUID, request: SessionUpdate):
    return invoke(session_service(session).memory.repository.update_session, str(session), request.title)


def cleanup(paths, root):
    removed, pending = 0, 0
    root = root.resolve()
    for raw in paths:
        path = Path(raw)
        # Resolve and check every target; never recursively delete an unverified path.
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            pending += 1
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            pending += 1
    return dict(filesRemoved=removed, filesPendingCleanup=pending)


@router.delete("/sessions/{session}")
def delete_session(session: UUID):
    agent = session_service(session)
    paths = invoke(agent.memory.repository.delete_session, str(session))
    assets = agent.memory.root / "artifacts" / str(session)
    images = list(assets.glob("*.png")) if not assets.is_symlink() else []
    files = cleanup(paths, agent.memory.root / "uploads")
    pictures = cleanup(images, agent.memory.root / "artifacts")
    return dict(deleted=True, filesRemoved=files["filesRemoved"]+pictures["filesRemoved"],
                filesPendingCleanup=files["filesPendingCleanup"]+pictures["filesPendingCleanup"],
                vectorsPending=agent.memory.repository.sync_vectors())


@router.post("/sessions/{session}/messages", status_code=201)
def create_message(session: UUID, request: MessageCreate):
    return invoke(session_service(session).memory.repository.create_message, str(session), request.role, request.text)


@router.get("/sessions/{session}/messages/{message}")
def get_message(session: UUID, message: UUID):
    return invoke(session_service(session).memory.repository.get_message, str(session), str(message))


@router.patch("/sessions/{session}/messages/{message}")
def update_message(session: UUID, message: UUID, request: MessageUpdate):
    return invoke(session_service(session).memory.repository.update_message, str(session), str(message), request.text)


@router.delete("/sessions/{session}/messages/{message}")
def delete_message(session: UUID, message: UUID):
    agent = session_service(session)
    invoke(agent.memory.repository.delete_message, str(session), str(message))
    return dict(deleted=True, vectorsPending=agent.memory.repository.sync_vectors())


@router.get("/sessions/{session}/traces")
def traces(session: UUID, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    return dict(items=invoke(session_service(session).memory.repository.list_traces, str(session), limit, offset))


@router.get("/sessions/{session}/traces/{trace}")
def trace(session: UUID, trace: UUID):
    return invoke(session_service(session).memory.repository.get_trace, str(session), str(trace))


@router.patch("/sessions/{session}/traces/{trace}")
def annotate_trace(session: UUID, trace: UUID, request: TraceUpdate):
    return invoke(session_service(session).memory.repository.annotate_trace, str(session), str(trace), request.note)


@router.delete("/sessions/{session}/traces/{trace}")
def delete_trace(session: UUID, trace: UUID):
    agent = session_service(session)
    record = invoke(agent.memory.repository.delete_trace, str(session), str(trace))
    paths = set()
    prefix = f"/api/agent/sessions/{session}/artifacts/"
    def collect(value):
        if isinstance(value, dict):
            url = value.get("xaiUrl", "")
            if isinstance(url, str) and url.startswith(prefix):
                try:
                    artifact = str(UUID(url[len(prefix):]))
                    paths.add(agent.memory.root / "artifacts" / str(session) / f"{artifact}.png")
                except ValueError:
                    pass
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(record)
    return dict(deleted=True, **cleanup(paths, agent.memory.root / "artifacts"),
                vectorsPending=agent.memory.repository.sync_vectors())


@router.post("/memory/sync")
def sync_memory():
    return dict(vectorsPending=service().memory.repository.sync_vectors())

"""Optional audio agent API. Heavy dependencies are loaded only on first use."""
import json
import logging
import os
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from app.models.agent_schema import FunctionCall, RunRequest, URLRequest

router = APIRouter(prefix="/api/agent", tags=["Audio Agent · weeks 2–5"])
_service = None
_lock = Lock()
log = logging.getLogger(__name__)


def service():
    global _service
    with _lock:
        if _service is None:
            try:
                from app.db.memory_store import MemoryStore
                from core.agents.audio_agent import AudioAgent
            except ImportError as exc:
                raise HTTPException(503, "Cài backend/requirements-agent.txt để dùng Audio Agent.") from exc
            root = Path(os.getenv("DEEPGUARD_AGENT_DIR", str(Path(__file__).resolve().parents[4] / "data/agent")))
            _service = AudioAgent(MemoryStore(root))
        return _service


def close_service():
    global _service
    with _lock:
        if _service:
            _service.memory.close()
            _service = None


def session_service(session):
    agent = service()
    try:
        agent.memory.require_session(str(session))
    except KeyError as exc:
        raise HTTPException(404, exc.args[0]) from exc
    return agent


def file_service(session, file_id):
    agent = session_service(session)
    try:
        agent.memory.file_path(str(session), file_id)
    except KeyError as exc:
        raise HTTPException(404, exc.args[0]) from exc
    return agent


@router.post("/sessions", status_code=201)
def create_session():
    return dict(sessionID=service().memory.create_session())


@router.get("/tools")
def schemas():
    return dict(tools=service().tools.schemas(), invocation="POST /api/agent/sessions/{sessionID}/tools/call")


@router.post("/sessions/{session}/files", status_code=201)
async def upload(session: UUID, file: UploadFile = File(...)):
    agent = session_service(session)
    from core.tools.audio_features_tool import EXTENSIONS, MAX_BYTES, decode_audio
    name = (file.filename or "audio.wav").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in EXTENSIONS:
        await file.close()
        raise HTTPException(422, "Chỉ nhận WAV/MP3/FLAC/OGG.")
    file_id = str(uuid4())
    directory = agent.memory.root / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (file_id + suffix)
    try:
        size = 0
        with target.open("wb") as output:
            while chunk := await file.read(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "File vượt quá 20 MB.")
                output.write(chunk)
        await run_in_threadpool(decode_audio, target)
        agent.memory.add_file(str(session), file_id, name, target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        if isinstance(exc, ValueError):
            raise HTTPException(422, str(exc)) from exc
        raise
    finally:
        await file.close()
    return dict(file_id=file_id, name=name)


@router.post("/sessions/{session}/files/url", status_code=201)
def upload_url(session: UUID, request: URLRequest):
    agent = session_service(session)
    from core.tools.audio_sources import download_url
    from core.tools.audio_features_tool import decode_audio
    file_id = str(uuid4())
    directory = agent.memory.root / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlsplit(request.url).path).suffix.lower()
    target = directory / (file_id + suffix)
    allowed = {h.strip().lower() for h in os.getenv("DEEPGUARD_AUDIO_URL_HOSTS", "").split(",") if h.strip()}
    try:
        download_url(request.url, target, allowed)
        decode_audio(target)
        agent.memory.add_file(str(session), file_id, "remote" + suffix, target)
    except (ValueError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(422, "Không nhận được âm thanh từ URL: " + str(exc)) from exc
    return dict(file_id=file_id, name="remote" + suffix)


@router.post("/sessions/{session}/tools/call")
def call_tool(session: UUID, request: FunctionCall):
    agent = file_service(session, request.arguments.file_id)
    try:
        return agent.execute_tool(str(session), request.name, request.arguments.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def safe_events(agent, session, request):
    try:
        async for event in agent.events(str(session), request):
            yield event
    except ValueError:
        yield dict(event="error", data=dict(code="invalid_audio_or_model", message="Không xử lý được audio/checkpoint; kiểm tra cấu hình và log máy chủ."))
        log.exception("Audio execution failed")
    except Exception:
        log.exception("Audio agent failed")
        yield dict(event="error", data=dict(code="execution_failed", message="Tool gặp lỗi; xem log máy chủ."))


@router.post("/sessions/{session}/runs")
async def run(session: UUID, request: RunRequest):
    agent = file_service(session, request.file_id)
    async for event in safe_events(agent, session, request):
        if event["event"] == "result":
            return event["data"]
        if event["event"] == "error":
            raise HTTPException(422, event["data"])


@router.post("/sessions/{session}/runs/stream")
async def stream(session: UUID, request: RunRequest):
    agent = file_service(session, request.file_id)
    async def body():
        async for event in safe_events(agent, session, request):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.websocket("/sessions/{session}/ws")
async def websocket(websocket: WebSocket, session: UUID):
    await websocket.accept()
    try:
        agent = session_service(session)
        request = RunRequest.model_validate(await websocket.receive_json())
        file_service(session, request.file_id)
        async for event in safe_events(agent, session, request):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    except (HTTPException, ValidationError, ValueError):
        await websocket.send_json(dict(event="error", data=dict(code="invalid_request")))
    finally:
        if websocket.client_state.name == "CONNECTED":
            await websocket.close()


@router.get("/sessions/{session}/history")
def history(session: UUID):
    return session_service(session).memory.history(str(session))


@router.get("/sessions/{session}/memory")
def memory(session: UUID, q: str = Query(min_length=1, max_length=4000)):
    return dict(encoder="hashing-char-2-4-384", items=session_service(session).memory.search(str(session), q))


@router.get("/sessions/{session}/artifacts/{artifact}")
def artifact(session: UUID, artifact: UUID):
    agent = session_service(session)
    path = agent.memory.root / "artifacts" / str(session) / f"{artifact}.png"
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy ảnh XAI.")
    return FileResponse(path, media_type="image/png")

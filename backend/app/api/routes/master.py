import json
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from app.api.routes.agent import session_service, file_service, safe_events
from app.models.agent_schema import MasterRequest

router = APIRouter(prefix="/api/master", tags=["Master router · week 7"])


def master_service(session, request=None):
    from core.agents.master_router import MasterRouter
    audio = session_service(session)
    if request:
        for file_id in request.file_ids:
            file_service(session, file_id)
    return MasterRouter(audio)


@router.post("/sessions/{session}/files", status_code=201)
async def upload_media(session: UUID, file: UploadFile = File(...)):
    agent = session_service(session)
    from core.tools.media_intake import inspect_media, IMAGE_EXTENSIONS
    from core.tools.audio_features_tool import EXTENSIONS, MAX_BYTES
    name = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in EXTENSIONS | IMAGE_EXTENSIONS:
        await file.close()
        raise HTTPException(422, "Chỉ nhận audio và PNG/JPEG/WebP; chưa hỗ trợ video.")
    directory = agent.memory.root / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid4())
    path = directory / (file_id + suffix)
    try:
        size = 0
        with path.open("wb") as destination:
            while chunk := await file.read(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "File vượt quá 20 MB.")
                destination.write(chunk)
        kind = await run_in_threadpool(inspect_media, path)
        agent.memory.add_file(str(session), file_id, name, path, kind)
    except Exception as exc:
        path.unlink(missing_ok=True)
        if isinstance(exc, ValueError):
            raise HTTPException(422, str(exc)) from exc
        raise
    finally:
        await file.close()
    return dict(file_id=file_id, name=name, kind=kind)


@router.post("/sessions/{session}/runs")
async def run(session: UUID, request: MasterRequest):
    master = master_service(session, request)
    async for event in safe_events(master, session, request):
        if event["event"] == "result":
            return event["data"]
        if event["event"] == "error":
            raise HTTPException(422, event["data"])


@router.post("/sessions/{session}/runs/stream")
async def stream(session: UUID, request: MasterRequest):
    master = master_service(session, request)
    async def body():
        async for event in safe_events(master, session, request):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.websocket("/sessions/{session}/ws")
async def websocket(ws: WebSocket, session: UUID):
    await ws.accept()
    try:
        request = MasterRequest.model_validate(await ws.receive_json())
        master = master_service(session, request)
        async for event in safe_events(master, session, request):
            await ws.send_json(event)
    except WebSocketDisconnect:
        return
    except (HTTPException, ValidationError, ValueError):
        await ws.send_json(dict(event="error", data=dict(code="invalid_request")))
    finally:
        if ws.client_state.name == "CONNECTED":
            await ws.close()

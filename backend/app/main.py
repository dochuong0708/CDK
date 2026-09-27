import os
from pathlib import Path
from uuid import uuid4
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from app.api.routes.health import router as health_router
from app.db.store import AnalysisStore
from app.models.schemas import AnalysisList, Modality, TextRequest
from app.services.orchestrator import DetectorOrchestrator
from app.detectors.audio.audio_detector import AudioUnavailable

app = FastAPI(title='DeepGuard API Phân tích', version='0.1.0')
app.include_router(health_router)

store = AnalysisStore(os.getenv('DEEPGUARD_DB_PATH', './data/deepguard.db'))
orchestrator = DetectorOrchestrator()
upload_dir = Path(os.getenv('DEEPGUARD_UPLOAD_DIR', './data/uploads'))
upload_dir.mkdir(parents=True, exist_ok=True)

@app.post('/api/analyze/text')
def analyze_text(request: TextRequest):
    result = orchestrator.analyze_text(request.text)
    return store.add(result)

@app.post('/api/analyze/media')
async def analyze_media(file: UploadFile = File(...), modality: Modality = Form(...)):
    if modality not in ('Audio', 'Image'):
        raise HTTPException(400, 'Loại media chưa được hỗ trợ trong mô-đun này')
    if not file.content_type or (not file.content_type.startswith(('audio/', 'image/', 'video/'))):
        raise HTTPException(400, 'Loại tệp không được hỗ trợ')
    name = (file.filename or 'upload').replace('\\', '/').rsplit('/', 1)[-1]
    target = upload_dir / (uuid4().hex + Path(name).suffix.lower())
    try:
        size = 0
        with target.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 20 * 1024 * 1024:
                    raise HTTPException(413, 'File vượt quá giới hạn 20 MB')
                output.write(chunk)
        detector = orchestrator.analyze_image if modality == 'Image' else orchestrator.analyze_audio
        result = await run_in_threadpool(detector, target)
        result.input = name
        return store.add(result)
    except AudioUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        target.unlink(missing_ok=True)
        await file.close()

@app.get('/api/analyses', response_model=AnalysisList)
def analyses(modality: str | None = None, risk: int | None = None):
    items = store.list(modality, risk)
    return {'items': items, 'total': len(items)}

@app.get('/api/analyses/{analysis_id}')
def analysis(analysis_id: str):
    item = store.get(analysis_id)
    if not item:
        raise HTTPException(404, 'Không tìm thấy bản phân tích')
    return item

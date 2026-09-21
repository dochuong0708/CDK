import os
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from app.api.routes.health import router as health_router
from app.db.store import AnalysisStore
from app.models.schemas import AnalysisList, Modality, TextRequest
from app.services.orchestrator import DetectorOrchestrator

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
    if not file.content_type or (not file.content_type.startswith(('audio/', 'image/', 'video/'))):
        raise HTTPException(400, 'Loại tệp không được hỗ trợ')
    target = upload_dir / file.filename
    target.write_bytes(await file.read())

    if modality == 'Image':
        result = orchestrator.analyze_image(target)
    elif modality == 'Audio':
        result = orchestrator.analyze_audio(target)
    else:
        raise HTTPException(400, 'Loại media chưa được hỗ trợ trong mô-đun này')

    return store.add(result)

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

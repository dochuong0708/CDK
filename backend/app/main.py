import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.api.routes.health import router as health_router
from app.db.store import AnalysisStore
from app.models.fact_checking import FactCheckList, FactCheckReport, FactCheckRequest
from app.models.schemas import AnalysisList, Modality, TextRequest
from app.services.article_analyzer import ArticleAnalyzer
from app.services.fact_checking import FactChecker
from app.services.orchestrator import DetectorOrchestrator

app = FastAPI(title='DeepGuard API Phân tích', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health_router)

store = AnalysisStore(os.getenv('DEEPGUARD_DB_PATH', './data/deepguard.db'))
orchestrator = DetectorOrchestrator()
fact_checker = FactChecker()
article_analyzer = ArticleAnalyzer(fact_checker=fact_checker, store=store)
upload_dir = Path(os.getenv('DEEPGUARD_UPLOAD_DIR', './data/uploads'))
upload_dir.mkdir(parents=True, exist_ok=True)


class ArticleRequest(BaseModel):
    text: str = Field(..., min_length=1)
    title: str = Field(default="")
    check_facts: bool = Field(default=True)


@app.post('/api/analyze/article')
def analyze_article(request: ArticleRequest):
    try:
        return article_analyzer.analyze_article(
            text=request.text,
            title=request.title,
            check_facts=request.check_facts,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from error


@app.post('/api/fact-check/text', response_model=FactCheckReport)
def fact_check_text(request: FactCheckRequest):
    try:
        report = fact_checker.check(request.text)
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from error
    return store.add_fact_check(report, request.text)


@app.get('/api/fact-checks', response_model=FactCheckList)
def fact_checks(limit: int = 50):
    items = store.list_fact_checks(limit)
    return {'items': items, 'total': len(items)}


@app.get('/api/fact-checks/{report_id}', response_model=FactCheckReport)
def fact_check(report_id: str):
    report = store.get_fact_check(report_id)
    if not report:
        raise HTTPException(404, 'Không tìm thấy báo cáo kiểm chứng')
    return report


@app.post('/api/analyze/text')
def analyze_text(request: TextRequest):
    report: FactCheckReport | None = None
    fact_check_error: str | None = None
    if request.check_facts:
        try:
            report = store.add_fact_check(fact_checker.check(request.text), request.text)
        except (RuntimeError, ValueError) as error:
            fact_check_error = str(error)
    result = orchestrator.analyze_text(request.text, fact_report=report)
    if fact_check_error:
        result.evidence.append(f"Không thể fact-check: {fact_check_error}")
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

frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

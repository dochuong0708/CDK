import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.routes.agent import close_service, router as agent_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.health import router as health_router
from app.api.routes.master import router as master_router
from app.db.store import AnalysisStore
from app.detectors.audio.audio_detector import AudioUnavailable
from app.models.fact_checking import FactCheckList, FactCheckReport, FactCheckRequest
from app.models.schemas import AnalysisList, Modality, TextRequest
from app.services.article_analyzer import ArticleAnalyzer
from app.services.fact_checking import FactChecker
from app.services.orchestrator import DetectorOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_service()


app = FastAPI(title='DeepGuard API Phân tích', version='0.3.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health_router)
app.include_router(agent_router)
app.include_router(conversations_router)
app.include_router(master_router)

store = AnalysisStore(os.getenv('DEEPGUARD_DB_PATH', './data/deepguard.db'))
orchestrator = DetectorOrchestrator()
fact_checker = FactChecker()
article_analyzer = ArticleAnalyzer(
    fact_checker=fact_checker,
    store=store,
    fraud_classifier=orchestrator.language_detector.fraud_classifier,
    rag_checker=orchestrator.language_detector.rag_checker,
)
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
    if modality not in ('Audio', 'Image'):
        raise HTTPException(400, 'Loại media chưa được hỗ trợ trong mô-đun này')
    if not file.content_type or (not file.content_type.startswith(('audio/', 'image/', 'video/'))):
        raise HTTPException(400, 'Loại tệp không được hỗ trợ')

    name = (file.filename or 'upload').replace('\\', '/').rsplit('/', 1)[-1]

    if modality == 'Image':
        try:
            image_bytes = await file.read()
            result = await run_in_threadpool(orchestrator.analyze_image, image_bytes, name)
            return store.add(result)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        finally:
            await file.close()

    # Xử lý modality == 'Audio'
    target = upload_dir / (uuid4().hex + Path(name).suffix.lower())
    try:
        size = 0
        with target.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 20 * 1024 * 1024:
                    raise HTTPException(413, 'File vượt quá giới hạn 20 MB')
                output.write(chunk)
        result = await run_in_threadpool(orchestrator.analyze_audio, target)
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


frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
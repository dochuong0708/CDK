from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal['Text', 'Image', 'Audio', 'Video']

class VisualXAI(BaseModel):
    heatmap_base64: str
    annotated_base64: str

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    check_facts: bool = False
class AnalysisResult(BaseModel):
    analysisId: str
    modality: Modality
    label: str
    riskScore: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]
    detector: str
    styleRisk: int | None = Field(default=None, ge=0, le=100)
    styleLabel: str | None = None
    fraudRisk: int | None = Field(default=None, ge=0, le=100)
    factualityRisk: int | None = Field(default=None, ge=0, le=100)
    factualityLabel: str | None = None
    factCheckId: str | None = None
    phobert_fraud_score: float | None = Field(default=None, ge=0, le=1)
    matched_reference_urls: list[str] = Field(default_factory=list)
    suspicious_phrases: list[dict[str, object]] = Field(default_factory=list)
    xai: VisualXAI | None = None
class Analysis(AnalysisResult):
    input: str
    createdAt: datetime
class AnalysisList(BaseModel):
    items: list[Analysis]
    total: int

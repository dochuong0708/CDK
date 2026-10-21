from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FactCheckVerdict = Literal["supported", "contradicted", "mixed", "insufficient_evidence"]
EvidenceStance = Literal["supports", "contradicts", "context", "irrelevant"]


class FactCheckRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30000)


class FactCheckEvidence(BaseModel):
    title: str
    url: str
    snippet: str
    source: str | None = None
    published_at: str | None = None
    stance: EvidenceStance
    priority_publisher: bool = False
    similarity: float | None = Field(default=None, ge=0, le=1)


class ClaimFactCheck(BaseModel):
    claim: str
    search_query: str
    verdict: FactCheckVerdict
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence: list[FactCheckEvidence]


class FactCheckReport(BaseModel):
    reportId: str | None = None
    summary: str
    claims: list[ClaimFactCheck]
    factuality_risk: float | None = Field(default=None, ge=0, le=1)
    factuality_confidence: float = Field(default=0.0, ge=0, le=1)
    retrieval_backend: str | None = None
    createdAt: datetime | None = None
    disclaimer: str = (
        "Kết quả dựa trên luận điểm được trích xuất và đoạn trích tìm kiếm; "
        "hãy mở các nguồn để kiểm tra ngữ cảnh đầy đủ."
    )


class FactCheckList(BaseModel):
    items: list[FactCheckReport]
    total: int

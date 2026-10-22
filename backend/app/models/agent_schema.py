from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: UUID = Field(description="File handle returned by this session's upload or URL intake.")


class RunRequest(AudioToolInput):
    message: str = Field(default="Phân tích âm thanh và giải thích kết quả.", max_length=4000)


class URLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(max_length=2048)


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["audio_features_tool", "audio_anomaly_tool"]
    arguments: AudioToolInput


class AgentResult(BaseModel):
    sessionID: UUID
    traced: UUID  # Literal spelling in the workbook; do not silently rename it.
    thoughtProcess: list[dict]  # Public execution events, never private model reasoning.
    toolsExecuted: list[dict]
    label: Literal["SAFE", "SUSPICIOUS", "DANGEROUS"]
    riskScore: float = Field(ge=0, le=1)
    finalResponse: str
    evidence: list[str]
    metadata: dict = Field(default_factory=dict)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MasterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(default="", max_length=5000)
    file_ids: list[UUID] = Field(default_factory=list, max_length=4)
    message: str = Field(default="Phân tích các đầu vào này.", max_length=2000)

    @model_validator(mode="after")
    def validate_inputs(self):
        if not self.text.strip() and not self.file_ids:
            raise ValueError("Cần text hoặc ít nhất một file_id.")
        if len(set(self.file_ids)) != len(self.file_ids):
            raise ValueError("file_ids không được trùng lặp.")
        return self


class SessionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(max_length=200)


class MessageUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=8000)


class MessageCreate(MessageUpdate):
    role: Literal["user", "assistant"] = "user"


class TraceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str = Field(max_length=2000)

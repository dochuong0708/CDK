from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.db.store import AnalysisStore
from app.models.fact_checking import FactCheckRequest
from app.models.schemas import Analysis, TextRequest
from app.services.fact_checking import FactChecker
from app.services.orchestrator import DetectorOrchestrator


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_text",
            "description": "Analyze text for suspicious or dangerous content.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 5000}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fact_check_text",
            "description": "Extract verifiable claims from text, search web sources, and compare the evidence.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 30000}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_media",
            "description": "Analyze a local image or audio file. Video analysis is not available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "modality": {"type": "string", "enum": ["Image", "Audio"]},
                },
                "required": ["path", "modality"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_analyses",
            "description": "List saved analyses, optionally filtered by modality and minimum risk score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "enum": ["Text", "Image", "Audio", "Video"],
                    },
                    "risk": {"type": "integer", "minimum": 0, "maximum": 100},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_analysis",
            "description": "Get a saved analysis by its analysis ID.",
            "parameters": {
                "type": "object",
                "properties": {"analysisId": {"type": "string"}},
                "required": ["analysisId"],
            },
        },
    },
]


class DeepGuardTools:
    def __init__(
        self,
        orchestrator: DetectorOrchestrator,
        store: AnalysisStore,
        fact_checker: FactChecker | None = None,
    ):
        self.orchestrator = orchestrator
        self.store = store
        self.fact_checker = fact_checker or FactChecker()

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "analyze_text":
                request = TextRequest(text=arguments.get("text", ""))
                return self._save(self.orchestrator.analyze_text(request.text))
            if name == "fact_check_text":
                request = FactCheckRequest(text=arguments.get("text", ""))
                report = self.store.add_fact_check(self.fact_checker.check(request.text), request.text)
                return report.model_dump(mode="json")
            if name == "analyze_media":
                return self._analyze_media(arguments)
            if name == "list_analyses":
                items = self.store.list(
                    modality=arguments.get("modality"), risk=arguments.get("risk")
                )
                return {"items": [self._serialize(item) for item in items], "total": len(items)}
            if name == "get_analysis":
                item = self.store.get(arguments.get("analysisId", ""))
                if item is None:
                    return {"error": "Không tìm thấy kết quả phân tích này."}
                return self._serialize(item)
            return {"error": f"Tool không tồn tại: {name}"}
        except (ValidationError, ValueError) as error:
            return {"error": str(error)}

    def _analyze_media(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = Path(arguments.get("path", "")).expanduser()
        modality = arguments.get("modality")
        if modality not in {"Image", "Audio"}:
            raise ValueError("Chỉ hỗ trợ phân tích Image hoặc Audio; Video chưa được hỗ trợ.")
        if not path.is_file():
            raise ValueError(f"Không tìm thấy tệp: {path}")

        if modality == "Image":
            result = self.orchestrator.analyze_image(path)
        else:
            result = self.orchestrator.analyze_audio(path)
        return self._save(result)

    def _save(self, item: Analysis) -> dict[str, Any]:
        return self._serialize(self.store.add(item))

    @staticmethod
    def _serialize(item: Analysis) -> dict[str, Any]:
        return item.model_dump(mode="json")
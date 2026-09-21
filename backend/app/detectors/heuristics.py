from pathlib import Path
from uuid import uuid4
from app.models.schemas import Analysis, Modality

class HeuristicDetector:
    def text(self, text: str) -> Analysis:
        score = min(95, 18 + max(0, len(text) - 120) // 8)
        return self._result('Text', text, score, 'text-heuristic-v1', ['Deterministic lexical heuristic completed', 'No production model is active in this MVP'])
    def media(self, path: Path, modality: Modality) -> Analysis:
        score = 64 if path.stat().st_size else 50
        return self._result(modality, path.name, score, f'{modality.lower()}-heuristic-v1', [f'Metadata inspected for {path.name}', 'Content heuristic completed'])
    def _result(self, modality: Modality, value: str, score: int, detector: str, evidence: list[str]) -> Analysis:
        return Analysis(analysisId=f'demo-{uuid4().hex[:8]}', modality=modality, label='Review recommended' if score >= 65 else 'Likely authentic', riskScore=score, confidence=.86 if modality == 'Text' else .63, evidence=evidence, detector=detector, input=value, createdAt=__import__('datetime').datetime.now())

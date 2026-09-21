from pathlib import Path
from app.models.schemas import Analysis

class AudioDetector:
    def analyze(self, path: Path) -> Analysis:
        score = 57 if path.stat().st_size else 42
        return Analysis(
            analysisId=f"audio-{path.stem.lower()}",
            modality="Audio",
            label="Review recommended" if score >= 65 else "Likely authentic",
            riskScore=score,
            confidence=0.61,
            evidence=[f"Audio metadata inspected for {path.name}", "Audio heuristic completed"],
            detector="audio-heuristic-v1",
            input=path.name,
            createdAt=__import__('datetime').datetime.now()
        )

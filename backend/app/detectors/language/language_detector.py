from app.models.schemas import Analysis

class LanguageDetector:
    def analyze(self, text: str) -> Analysis:
        score = min(95, 18 + max(0, len(text) - 120) // 8)
        return Analysis(
            analysisId=f"text-{len(text)}",
            modality="Text",
            label="Review recommended" if score >= 65 else "Likely authentic",
            riskScore=score,
            confidence=0.86,
            evidence=["Deterministic lexical heuristic completed", "No production model is active in this MVP"],
            detector="text-heuristic-v1",
            input=text,
            createdAt=__import__('datetime').datetime.now()
        )

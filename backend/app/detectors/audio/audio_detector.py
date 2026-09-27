import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
from app.models.schemas import Analysis


class AudioUnavailable(RuntimeError):
    pass


class AudioDetector:
    def __init__(self):
        self._model = None
        self._lock = Lock()

    def analyze(self, path: Path) -> Analysis:
        checkpoint = Path(os.getenv("DEEPGUARD_AUDIO_MODEL", "models/audio_cnn.pt"))
        if not checkpoint.is_file():
            raise AudioUnavailable("Chưa có mô hình audio đã huấn luyện. Xem docs/AUDIO.md và đặt DEEPGUARD_AUDIO_MODEL.")
        try:
            from .inference import load_checkpoint, predict
        except ImportError as exc:
            raise AudioUnavailable("Thiếu thư viện audio. Cài backend/requirements-audio.txt.") from exc
        with self._lock:
            if self._model is None:
                try:
                    self._model, self._threshold = load_checkpoint(checkpoint)
                except Exception as exc:
                    raise AudioUnavailable("Không tải được checkpoint audio. Kiểm tra phiên bản và huấn luyện lại.") from exc
            result = predict(path, self._model, self._threshold)
        score = result["fakeScore"]
        return Analysis(
            analysisId=f"audio-{uuid4().hex}", modality="Audio",
            label="Nghi giọng nói giả" if result["predictedClass"] == "fake" else "Nghi giọng nói thật",
            riskScore=round(score*100), confidence=max(score, 1-score),
            evidence=["Điểm mô hình chưa được hiệu chỉnh xác suất.",
                      "Grad-CAM giải thích lớp fake ở cửa sổ có điểm fake cao nhất; không chứng minh giả mạo.",
                      f"Ngưỡng phân loại: {self._threshold:.4f}; số cửa sổ: {len(result['windows'])}."],
            detector="audio-cnn-v1", input=path.name, createdAt=datetime.now(timezone.utc), audio=result)

from pathlib import Path
from app.detectors.image.image_detector import ImageDetector
from app.detectors.audio.audio_detector import AudioDetector
from app.detectors.language.language_detector import LanguageDetector

class DetectorOrchestrator:
    def __init__(self):
        self.image_detector = ImageDetector()
        self.audio_detector = AudioDetector()
        self.language_detector = LanguageDetector()

    def analyze_text(self, text: str):
        return self.language_detector.analyze(text)

    def analyze_image(self, path: Path):
        return self.image_detector.analyze(path)

    def analyze_audio(self, path: Path):
        return self.audio_detector.analyze(path)

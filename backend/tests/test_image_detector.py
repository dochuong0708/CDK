from pathlib import Path

import pytest
from PIL import Image

from app.db.store import AnalysisStore
from app.detectors.image.image_detector import ImageDetector


def test_analyze_returns_convnext_backend_analysis_from_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (224, 224), color=(80, 120, 160)).save(image_path)
    detector = ImageDetector()
    monkeypatch.setattr(detector, "_load_model", lambda: None)
    monkeypatch.setattr(detector, "_predict", lambda image: 0.5)

    result = detector.analyze(image_path.read_bytes(), image_path.name)

    assert result.modality == "Image"
    assert result.detector == "image-convnexttiny-v1"
    assert any("ConvNeXt-Tiny (80%):" in evidence for evidence in result.evidence)
    assert any("AI-generation risk score: 50%" in evidence for evidence in result.evidence)
    assert 0 <= result.riskScore <= 100
    assert 0 <= result.confidence <= 1
    assert result.input == image_path.name


def test_convnext_and_color_forensics_use_80_20_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from app.detectors.image import color_analysis

    detector = ImageDetector()
    detector._device = torch.device("cpu")
    detector._fake_idx = 0
    detector._eval_transform = lambda image: torch.zeros((3, 224, 224))

    class FakeModel:
        def __call__(self, image_tensor):
            return torch.tensor([[1.38629436112, 0.0]])

    detector._model = FakeModel()
    monkeypatch.setattr(
        color_analysis,
        "get_color_forensic_adjustment",
        lambda image: {"adjustment": 0.12},
    )

    probability = detector._predict(Image.new("RGB", (32, 32)))

    assert probability == pytest.approx(0.764)


def test_reanalyzing_same_image_generates_distinct_analysis_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "same-image.png"
    Image.new("RGB", (224, 224), color=(80, 120, 160)).save(image_path)
    detector = ImageDetector()
    monkeypatch.setattr(detector, "_load_model", lambda: None)
    monkeypatch.setattr(detector, "_predict", lambda image: 0.2)

    image_bytes = image_path.read_bytes()
    first = detector.analyze(image_bytes, image_path.name)
    second = detector.analyze(image_bytes, image_path.name)

    assert first.analysisId != second.analysisId
    store = AnalysisStore(str(tmp_path / "analyses.db"))
    store.add(first)
    store.add(second)
    assert len(store.list(modality="Image")) == 2


def test_analyze_rejects_corrupt_image_without_loading_model(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")
    detector = ImageDetector()

    with pytest.raises(ValueError, match="not a valid supported image"):
        detector.analyze(image_path.read_bytes(), image_path.name)

    assert detector._model is None


def test_analyze_reports_missing_checkpoint(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32)).save(image_path)
    detector = ImageDetector(model_path=tmp_path / "missing.pt")

    with pytest.raises(RuntimeError, match="checkpoint not found"):
        detector.analyze(image_path.read_bytes(), image_path.name)


def test_analyze_rejects_resnet_checkpoint_for_convnext_detector(
    tmp_path: Path,
) -> None:
    import torch

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32)).save(image_path)
    checkpoint_path = tmp_path / "resnet.pt"
    torch.save(
        {
            "state_dict": {"conv1.weight": torch.zeros((64, 3, 7, 7))},
            "fake_idx": 1,
        },
        checkpoint_path,
    )
    detector = ImageDetector(model_path=checkpoint_path)

    with pytest.raises(RuntimeError, match="compatible two-class ConvNeXt-Tiny checkpoint"):
        detector.analyze(image_path.read_bytes(), image_path.name)


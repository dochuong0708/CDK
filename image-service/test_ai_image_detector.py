from pathlib import Path

import pytest

from ai_image_detector import make_decision, validate_data_layout


def test_make_decision_handles_ai_real_and_uncertain_cases():
    assert make_decision(0.9) == "AI"
    assert make_decision(0.1) == "REAL"
    assert make_decision(0.5) == "UNCERTAIN"


def test_validate_data_layout_requires_real_and_fake_classes(tmp_path):
    train_dir = tmp_path / "train"
    (train_dir / "real").mkdir(parents=True)
    (train_dir / "fake").mkdir(parents=True)

    with pytest.raises(SystemExit, match="Không tìm thấy thư mục validation"):
        validate_data_layout(tmp_path, "val")

    val_dir = tmp_path / "val"
    (val_dir / "real").mkdir(parents=True)
    (val_dir / "fake").mkdir(parents=True)

    assert validate_data_layout(tmp_path, "val") is None


def test_validate_data_layout_rejects_missing_class_folder(tmp_path):
    train_dir = tmp_path / "train"
    (train_dir / "real").mkdir(parents=True)
    (train_dir / "fake").mkdir(parents=True)

    val_dir = tmp_path / "val"
    val_dir.mkdir()
    (val_dir / "real").mkdir()

    with pytest.raises(SystemExit, match="Cần hai thư mục lớp"):
        validate_data_layout(tmp_path, "val")

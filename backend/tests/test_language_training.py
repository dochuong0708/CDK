import argparse
import csv

import pytest

from app.detectors.language.train import (
    evaluate_saved_model,
    load_dataset,
    normalize_label,
    resolve_label_sets,
    train,
)
from app.detectors.language.model import LanguageModel


def test_normalize_label_requires_explicit_mapping_for_unknown_classes():
    with pytest.raises(ValueError, match="chưa được phân loại"):
        normalize_label("CAG", {"spam"}, {"clean"})

    assert normalize_label("CAG", {"spam", "cag"}, {"clean"}) == "risky"


def test_custom_label_mapping_overrides_sms_defaults_for_fake_news():
    risky_labels, safe_labels = resolve_label_sets(["0"], ["1"])

    assert normalize_label("0", risky_labels, safe_labels) == "risky"
    assert normalize_label("1", risky_labels, safe_labels) == "safe"


@pytest.mark.parametrize(
    ("headers", "row", "risky_overrides", "safe_overrides"),
    [
        (["target", "text"], {"target": "spam", "text": "Tin nhắn quảng cáo"}, [], []),
        (["Label", "Maintext"], {"Label": "0", "Maintext": "Bài báo tiếng Việt"}, ["0"], ["1"]),
    ],
)
def test_dataset_loader_detects_repository_column_names(
    tmp_path, headers, row, risky_overrides, safe_overrides
):
    dataset = tmp_path / "dataset.csv"
    with dataset.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)

    risky_labels, safe_labels = resolve_label_sets(risky_overrides, safe_overrides)
    texts, labels = load_dataset(dataset, None, None, risky_labels, safe_labels)

    assert len(texts) == len(labels) == 1
    assert labels[0] == "risky"


def test_training_writes_loadable_pipeline(tmp_path):
    dataset = tmp_path / "messages.csv"
    with dataset.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["text", "label"])
        writer.writeheader()
        for index in range(6):
            writer.writerow({"text": f"Cuộc họp nội bộ lịch trình công việc số {index}", "label": "safe"})
            writer.writerow({"text": f"Gửi mã OTP và chuyển tiền ngay lập tức số {index}", "label": "phishing"})

    model_path = tmp_path / "model.joblib"
    result = train(
        argparse.Namespace(
            dataset=dataset,
            test_dataset=None,
            validation_dataset=None,
            text_column="text",
            label_column="label",
            risky_label=[],
            safe_label=[],
            test_size=0.25,
            seed=42,
            max_features=1000,
            output=model_path,
        )
    )

    assert model_path.is_file()
    assert "precision" in str(result["report"])
    assert set(result["pipeline"].classes_) == {"safe", "risky"}
    assert LanguageModel(str(model_path)).predict_score("gửi mã otp chuyển tiền ngay") is not None


def test_training_uses_supplied_test_and_validation_without_training_on_them(tmp_path):
    def write_dataset(path, prefix, per_class):
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["text", "label"])
            writer.writeheader()
            for index in range(per_class):
                writer.writerow({"text": f"{prefix} nội dung an toàn cuộc họp {index}", "label": "safe"})
                writer.writerow({"text": f"{prefix} gửi mã otp và chuyển tiền {index}", "label": "phishing"})

    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    test_csv = tmp_path / "test.csv"
    write_dataset(train_csv, "huấn luyện", 6)
    write_dataset(validation_csv, "xác thực", 2)
    write_dataset(test_csv, "kiểm tra", 2)
    model_path = tmp_path / "separate-splits.joblib"

    result = train(
        argparse.Namespace(
            dataset=train_csv,
            validation_dataset=validation_csv,
            test_dataset=test_csv,
            text_column="text",
            label_column="label",
            risky_label=[],
            safe_label=[],
            test_size=0.25,
            seed=42,
            max_features=1000,
            output=model_path,
        )
    )

    assert result["train_count"] == 12
    assert result["validation_count"] == 4
    assert result["test_count"] == 4
    assert set(result["reports"]) == {"validation", "test"}
    assert model_path.is_file()

    evaluation = evaluate_saved_model(
        argparse.Namespace(
            dataset=test_csv,
            text_column="text",
            label_column="label",
            risky_label=[],
            safe_label=[],
            evaluate_model=model_path,
        )
    )
    assert evaluation["count"] == 4
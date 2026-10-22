"""Huấn luyện bộ phân loại rủi ro tiếng Việt từ CSV có cột text và label."""

import argparse
import csv
import unicodedata
from pathlib import Path

from app.detectors.language.preprocessing import preprocess_text, vectorize_text

DEFAULT_RISK_LABELS = {
    "1", "true", "scam", "phishing", "spam", "toxic", "hate", "hateful",
    "fake", "fake news", "fake_news", "dangerous", "offensive", "lừa đảo",
}
DEFAULT_SAFE_LABELS = {
    "0", "false", "ham", "safe", "normal", "clean", "real", "legitimate",
    "non-toxic", "non_toxic", "not toxic",
}


def normalize_label(label: str, risky_labels: set[str], safe_labels: set[str]) -> str:
    normalized = unicodedata.normalize("NFC", label).strip().casefold()
    if normalized in risky_labels:
        return "risky"
    if normalized in safe_labels:
        return "safe"
    raise ValueError(
        f"Nhãn '{label}' chưa được phân loại. Dùng --risky-label hoặc --safe-label để khai báo."
    )


def resolve_label_sets(
    risky_overrides: list[str], safe_overrides: list[str]
) -> tuple[set[str], set[str]]:
    risky_custom = {label.casefold() for label in risky_overrides}
    safe_custom = {label.casefold() for label in safe_overrides}
    if risky_custom & safe_custom:
        raise ValueError("Một nhãn không thể đồng thời được khai báo là risky và safe")
    risky_labels = (DEFAULT_RISK_LABELS | risky_custom) - safe_custom
    safe_labels = (DEFAULT_SAFE_LABELS | safe_custom) - risky_custom
    return risky_labels, safe_labels


def _resolve_column(
    fieldnames: list[str], requested: str | None, aliases: tuple[str, ...], role: str
) -> str:
    columns = {name.strip().casefold(): name for name in fieldnames}
    if requested:
        resolved = columns.get(requested.strip().casefold())
        if resolved:
            return resolved
        raise ValueError(
            f"Không tìm thấy cột {role} '{requested}'. Các cột hiện có: {fieldnames}"
        )

    for alias in aliases:
        if alias in columns:
            return columns[alias]
    raise ValueError(
        f"Không tự nhận diện được cột {role}. Các cột hiện có: {fieldnames}; "
        f"hãy chỉ định bằng --{'text' if role == 'văn bản' else 'label'}-column."
    )


def load_dataset(
    csv_path: Path,
    text_column: str | None,
    label_column: str | None,
    risky_labels: set[str],
    safe_labels: set[str],
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV không có dòng tiêu đề: {csv_path}")
        resolved_text_column = _resolve_column(
            reader.fieldnames,
            text_column,
            ("text", "maintext", "content", "message", "body"),
            "văn bản",
        )
        resolved_label_column = _resolve_column(
            reader.fieldnames,
            label_column,
            ("label", "target", "class", "category", "sentiment"),
            "nhãn",
        )
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(resolved_text_column) or "").strip()
            label = (row.get(resolved_label_column) or "").strip()
            if not text or not label:
                continue
            texts.append(str(preprocess_text(text)["text"]))
            try:
                labels.append(normalize_label(label, risky_labels, safe_labels))
            except ValueError as error:
                raise ValueError(f"Dòng {row_number}: {error}") from error
    return texts, labels


def train(args: argparse.Namespace) -> dict[str, object]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "Thiếu scikit-learn hoặc joblib. Cài bằng: "
            "python -m pip install -r backend/requirements.txt"
        ) from error

    risky_labels, safe_labels = resolve_label_sets(args.risky_label, args.safe_label)
    texts, labels = load_dataset(
        args.dataset, args.text_column, args.label_column, risky_labels, safe_labels
    )
    counts = {label: labels.count(label) for label in ("safe", "risky")}
    if min(counts.values()) < 5:
        raise ValueError(
            f"Cần ít nhất 5 mẫu mỗi lớp để đánh giá train/test có ý nghĩa. Số lượng: {counts}"
        )

    test_dataset = getattr(args, "test_dataset", None)
    validation_dataset = getattr(args, "validation_dataset", None)
    if test_dataset or validation_dataset:
        train_texts, train_labels = texts, labels
        test_data = None
        if test_dataset:
            test_data = load_dataset(
                test_dataset, args.text_column, args.label_column, risky_labels, safe_labels
            )
        validation_data = None
        if validation_dataset:
            validation_data = load_dataset(
                validation_dataset,
                args.text_column,
                args.label_column,
                risky_labels,
                safe_labels,
            )
    else:
        if validation_dataset:
            raise ValueError("Muốn dùng validation riêng, cần truyền thêm --test-dataset")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )
        test_data = (test_texts, test_labels)
        validation_data = None

    if test_data is not None and not test_data[0]:
        raise ValueError("Tập test không có mẫu hợp lệ")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=vectorize_text,
                    ngram_range=(1, 2),
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed),
            ),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    reports: dict[str, str] = {}
    if test_data:
        test_texts, test_labels = test_data
        predictions = pipeline.predict(test_texts)
        reports["test"] = classification_report(
            test_labels, predictions, labels=["safe", "risky"], zero_division=0
        )
    if validation_data:
        validation_texts, validation_labels = validation_data
        if not validation_texts:
            raise ValueError("Tập validation không có mẫu hợp lệ")
        validation_predictions = pipeline.predict(validation_texts)
        reports["validation"] = classification_report(
            validation_labels,
            validation_predictions,
            labels=["safe", "risky"],
            zero_division=0,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"Số mẫu: {len(texts)} (an toàn: {counts['safe']}, rủi ro: {counts['risky']})")
    print(f"Số mẫu thực sự dùng để train: {len(train_texts)}")
    if validation_data:
        print(f"Số mẫu validation: {len(validation_data[0])}")
        print("Báo cáo validation:")
        print(reports["validation"])
    if test_data:
        print(f"Số mẫu test: {len(test_data[0])}")
        print("Báo cáo test:")
        print(reports["test"])
    print(f"Đã lưu model: {args.output.resolve()}")
    return {
        "pipeline": pipeline,
        "report": reports.get("test", ""),
        "reports": reports,
        "counts": counts,
        "train_count": len(train_texts),
        "test_count": len(test_data[0]) if test_data else 0,
        "validation_count": len(validation_data[0]) if validation_data else 0,
    }


def evaluate_saved_model(args: argparse.Namespace) -> dict[str, object]:
    try:
        import joblib
        from sklearn.metrics import classification_report
    except ImportError as error:
        raise RuntimeError("Thiếu scikit-learn hoặc joblib; cài requirements của backend") from error

    risky_labels, safe_labels = resolve_label_sets(args.risky_label, args.safe_label)
    texts, labels = load_dataset(
        args.dataset, args.text_column, args.label_column, risky_labels, safe_labels
    )
    if not texts:
        raise ValueError("Tập đánh giá không có mẫu hợp lệ")
    model = joblib.load(args.evaluate_model)
    predictions = model.predict(texts)
    report = classification_report(
        labels, predictions, labels=["safe", "risky"], zero_division=0
    )
    print(f"Số mẫu đánh giá: {len(texts)}")
    print("Báo cáo đánh giá model đã lưu:")
    print(report)
    return {"report": report, "count": len(texts), "predictions": predictions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("data/language/dataset.csv"),
        help="Đường dẫn CSV, mặc định data/language/dataset.csv tính từ thư mục backend",
    )
    parser.add_argument("--text-column", help="Tên cột văn bản; mặc định tự nhận diện")
    parser.add_argument("--label-column", help="Tên cột nhãn; mặc định tự nhận diện")
    parser.add_argument("--validation-dataset", type=Path, help="CSV validation; chỉ dùng để đánh giá")
    parser.add_argument("--test-dataset", type=Path, help="CSV test riêng; chỉ dùng để đánh giá, không train")
    parser.add_argument(
        "--evaluate-model",
        type=Path,
        help="Chỉ đánh giá model joblib đã lưu trên CSV positional, không huấn luyện",
    )
    parser.add_argument("--risky-label", action="append", default=[], help="Nhãn cần gộp vào risky; có thể dùng nhiều lần")
    parser.add_argument("--safe-label", action="append", default=[], help="Nhãn cần gộp vào safe; có thể dùng nhiều lần")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/vietnamese-language-risk.joblib"),
        help="Đường dẫn model đầu ra tính từ thư mục backend",
    )
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size phải nằm trong khoảng 0.1 đến 0.5")
    if args.max_features < 1:
        parser.error("--max-features phải lớn hơn 0")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.evaluate_model:
        if arguments.validation_dataset or arguments.test_dataset:
            raise SystemExit("Chế độ --evaluate-model không dùng --validation-dataset hoặc --test-dataset")
        evaluate_saved_model(arguments)
    else:
        train(arguments)
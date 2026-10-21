import csv
import json

import pytest

from app.detectors.language import train_nli

FIELDS = ["id", "claim", "evidence", "context", "label", "dataset", "source", "topic", "url"]
EXAMPLES = [
    ("Hà Nội là thủ đô của Việt Nam", "Hà Nội là thủ đô của Việt Nam", "SUPPORTED"),
    ("Hà Nội là thủ đô của Pháp", "Hà Nội là thủ đô của Việt Nam", "REFUTED"),
    ("Hà Nội có mười triệu dân", "Hà Nội là thủ đô của Việt Nam", "NEI"),
]


def write_claims(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (claim, evidence, label, dataset) in enumerate(rows):
            writer.writerow(
                {
                    "id": str(index), "claim": claim, "evidence": evidence, "context": "",
                    "label": label, "dataset": dataset, "source": "s", "topic": "", "url": "",
                }
            )


def test_load_claims_validates_columns_and_labels(tmp_path):
    good = tmp_path / "good.csv"
    write_claims(good, [(*example, "vifactcheck") for example in EXAMPLES])
    rows = train_nli.load_claims(good)
    assert [row["label"] for row in rows] == ["SUPPORTED", "REFUTED", "NEI"]
    assert len(train_nli.load_claims(good, max_samples=2)) == 2

    bad_label = tmp_path / "bad_label.csv"
    write_claims(bad_label, [("a", "b", "TRUE", "vifactcheck")])
    with pytest.raises(ValueError, match="TRUE"):
        train_nli.load_claims(bad_label)

    missing = tmp_path / "missing.csv"
    missing.write_text("claim,label\na,NEI\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        train_nli.load_claims(missing)


def test_word_segmentation_only_for_phobert():
    assert train_nli.needs_word_segmentation("vinai/phobert-base")
    assert not train_nli.needs_word_segmentation("FacebookAI/xlm-roberta-base")
    assert train_nli.make_segmenter(True)("Hà Nội là thủ đô") == "Hà_Nội là thủ_đô"
    assert train_nli.make_segmenter(False)("Hà Nội") == "Hà Nội"


def test_classification_report_per_dataset():
    gold = ["SUPPORTED", "REFUTED", "NEI", "SUPPORTED"]
    predicted = ["SUPPORTED", "REFUTED", "SUPPORTED", "SUPPORTED"]
    report = train_nli.classification_report(gold, predicted, ["a", "a", "b", "b"])
    assert report["overall"]["n"] == 4
    assert report["overall"]["accuracy"] == 0.75
    assert report["a"]["accuracy"] == 1.0
    assert report["b"]["accuracy"] == 0.5
    assert report["b"]["f1_per_label"]["NEI"] == 0.0


def test_train_and_evaluate_tiny_model_offline(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    words = sorted({word for example in EXAMPLES for text in example[:2] for word in text.split()})
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), encoding="utf-8"
    )
    base = tmp_path / "base"
    transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False).save_pretrained(
        base
    )
    config = transformers.BertConfig(
        vocab_size=len(words) + 5, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
    )
    transformers.BertModel(config).save_pretrained(base)

    data = tmp_path / "data"
    data.mkdir()
    rows = [(*example, dataset) for example in EXAMPLES for dataset in ("vifactcheck", "viwikifc")]
    for split in ("train", "val", "test"):
        write_claims(data / f"claims_{split}.csv", rows)

    output = tmp_path / "out"
    common = ["--data-dir", str(data), "--max-length", "32", "--precision", "fp32"]
    report = train_nli.main(
        [*common, "--model", str(base), "--output", str(output), "--epochs", "1",
         "--batch-size", "2", "--grad-accum", "1"]
    )
    assert set(report["test"]) == {"overall", "vifactcheck", "viwikifc"}
    assert report["test"]["overall"]["n"] == 6

    metadata = json.loads((output / "deepguard_nli.json").read_text(encoding="utf-8"))
    assert metadata["labels"] == ["SUPPORTED", "REFUTED", "NEI"]
    assert metadata["word_segmentation"] is False
    assert "viwikifc" in metadata["training_data"]

    evaluated = train_nli.main([*common, "--evaluate-only", str(output)])
    assert evaluated["test"]["overall"]["n"] == 6

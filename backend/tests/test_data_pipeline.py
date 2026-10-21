import csv

import pytest

from app.detectors.language import data_pipeline as dp


def _news(id_, text, label="real", source="", dataset="d", origin="gold"):
    return dp.NewsRecord(
        id=id_, text=text, label=label, dataset=dataset, source=source, label_origin=origin
    )


def test_label_maps_follow_dataset_cards():
    local = list(dp.local_fake_news_records(
        [{"Label": "0", "Maintext": "Bài báo giả mạo được chia sẻ rộng rãi"}], "train"
    ))
    assert local[0].label == "fake"

    claims = list(dp.vifactcheck_records([
        {"Statement": "A", "Evidence": "B", "Context": "C", "labels": 1, "Author": "VnExpress",
         "annotation_id": 7}
    ])) + list(dp.viwikifc_records([
        {"claim": "X", "evidence": "Y", "gold_label": "Not_Enough_Information", "title": "Hà Nội",
         "pairID": "p1"}
    ]))
    assert [(c.label, c.source) for c in claims] == [
        ("REFUTED", "VnExpress"), ("NEI", "wikipedia:Hà Nội")
    ]


def test_default_sources_are_only_openly_licensed():
    assert set(dp.DEFAULT_SOURCES) == set(dp.DATASET_LICENSES)
    assert "fake_news_data" not in dp.DEFAULT_SOURCES
    with pytest.raises(SystemExit):
        dp.main(["--sources", "reintel"])


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="không nằm trong bảng quy đổi"):
        list(dp.vifactcheck_records([{"Statement": "A", "labels": 9}]))


def test_clean_text_keeps_paragraphs_and_normalizes_unicode():
    decomposed = "Tie\u0302\u0301ng   Vie\u0323\u0302t\u200b\n\n\n\nĐoạn 2"
    assert dp.clean_text(decomposed) == "Tiếng Việt\n\nĐoạn 2"


def test_near_duplicates_are_clustered_but_distinct_texts_are_not():
    base = " ".join(f"từ{i}" for i in range(60))
    texts = [
        base,
        base + " thêm",
        base.upper() + " https://bit.ly/abc",
        " ".join(f"khác{i}" for i in range(60)),
    ]
    clusters = dp.cluster_near_duplicates(texts, threshold=0.85)
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] != clusters[0]


def test_deduplicate_prefers_gold_and_drops_label_conflicts():
    article = " ".join(f"tin{i}" for i in range(40))
    rumour = " ".join(f"đồn{i}" for i in range(40))
    records = [
        _news("weak", article, origin="weak"),
        _news("gold", article + " .", origin="gold"),
        _news("r1", rumour, label="fake"),
        _news("r2", rumour, label="real"),
    ]
    kept, stats = dp.deduplicate(records, dp.news_dedup_key, 0.85, 5)
    assert [r.id for r in kept] == ["gold"]
    assert (stats.duplicates_removed, stats.conflicting_removed) == (1, 2)


def test_group_split_never_shares_a_source_between_splits():
    records = [
        _news(f"{s}-{i}", f"văn bản {s} {i}", source=f"src{s}")
        for s in range(30) for i in range(s % 5 + 1)
    ] + [_news(f"solo-{i}", f"bài lẻ {i}") for i in range(20)]
    splits = dp.split_by_dataset(records, (0.8, 0.1, 0.1), seed=1)

    assert sum(len(items) for items in splits.values()) == len(records)
    assert not dp.leaked_sources(splits)
    assert all(splits[name] for name in dp.SPLITS)
    assert len(splits["train"]) > len(splits["val"])
    assert splits == dp.split_by_dataset(records, (0.8, 0.1, 0.1), seed=1)


def test_group_split_falls_back_to_records_when_too_few_sources():
    records = [_news(f"a{i}", f"t{i}", source="one") for i in range(10)]
    splits = dp.group_split(records, (0.8, 0.1, 0.1))
    assert [len(splits[name]) for name in dp.SPLITS] == [8, 1, 1]


def test_main_writes_splits_and_license_manifest(tmp_path, monkeypatch):
    raw = tmp_path / "viwikifc.csv"
    labels = ["Supports", "Refutes", "Not_Enough_Information"]
    with raw.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["pairID", "claim", "evidence", "gold_label", "title"])
        writer.writeheader()
        for i in range(30):
            writer.writerow({
                "pairID": f"p{i}", "claim": " ".join(f"claim{i}-{j}" for j in range(20)),
                "evidence": f"bằng chứng {i}", "gold_label": labels[i % 3], "title": f"Trang {i % 10}",
            })
    monkeypatch.setattr(dp, "fetch_hf_files", lambda source, *args: [raw])
    out = tmp_path / "out"
    manifest = dp.main(["--sources", "viwikifc", "--out-dir", str(out)])

    assert manifest["claims"]["dedup"]["kept"] == 30
    assert manifest["licenses"]["viwikifc"]["license"] == "CC-BY-4.0"
    assert "news" not in manifest
    with (out / "claims_train.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows and {"claim", "evidence", "label"} <= set(rows[0])

import csv
import urllib.error

import pytest

from app.detectors.language import data_pipeline as dp
from app.detectors.language.rss_crawler import PoliteFetcher, crawl, parse_feed


def _news(id_, text, label="real", source="", dataset="d", origin="gold"):
    return dp.NewsRecord(
        id=id_, text=text, label=label, dataset=dataset, source=source, label_origin=origin
    )


def test_label_maps_follow_dataset_cards():
    reintel = list(dp.reintel_records([
        {"id": 1, "post_message": "Tin đồn thất thiệt về vaccine lan truyền", "label": 1,
         "user_name": "u1", "timestamp_post": 1588000000},
        {"id": 2, "post_message": "Bộ Y tế công bố số ca mắc mới trong ngày", "label": 0},
        {"id": 3, "post_message": "Bài trong split test không có nhãn", "label": None},
    ]))
    assert [(r.label, r.source) for r in reintel] == [("fake", "reintel-user:u1"), ("real", "")]
    assert reintel[0].published_at.startswith("2020-04-27")

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


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="không nằm trong bảng quy đổi"):
        list(dp.vifactcheck_records([{"Statement": "A", "labels": 9}]))


def test_rss_records_are_weak_real_and_require_full_text():
    rows = [
        {"url": "https://vnexpress.net/a", "publisher": "vnexpress.net", "text": "x" * 500},
        {"url": "https://vnexpress.net/b", "publisher": "vnexpress.net", "text": "tóm tắt ngắn"},
    ]
    records = list(dp.rss_records(rows, min_chars=400))
    assert len(records) == 1
    assert (records[0].label, records[0].label_origin, records[0].source) == (
        "real", "weak", "vnexpress.net"
    )


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


def test_main_writes_trainable_splits_and_manifest(tmp_path, monkeypatch):
    rss = tmp_path / "news_rss.csv"
    with rss.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["url", "publisher", "title", "text"])
        writer.writeheader()
        for i in range(12):
            body = " ".join(f"bài{i}-{j}" for j in range(80))
            writer.writerow({"url": f"https://p{i % 4}.vn/{i}", "publisher": f"p{i % 4}.vn",
                             "title": f"T{i}", "text": body})
    out = tmp_path / "out"
    manifest = dp.main(["--sources", "rss", "--rss-file", str(rss), "--out-dir", str(out)])

    assert manifest["news"]["dedup"]["kept"] == 12
    with (out / "news_train.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows and {"text", "label"} <= set(rows[0]) and rows[0]["label"] == "real"
    assert (out / "manifest.json").exists()


RSS_XML = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Tin &amp; thời sự</title><link>https://example.vn/a</link>
<description><![CDATA[<a href="x"><img src="y"></a></br>Tóm tắt <b>bài</b>]]></description>
<pubDate>Wed, 21 Oct 2026 17:24:48 +0700</pubDate></item>
<item><title>Bài 2</title><link>https://example.vn/b</link><description>Khác</description></item>
</channel></rss>"""


def test_parse_feed_strips_html():
    entries = parse_feed(RSS_XML, "example.vn")
    assert entries[0]["title"] == "Tin & thời sự"
    assert entries[0]["summary"] == "Tóm tắt bài"
    assert entries[0]["published_at"] == "2026-10-21T17:24:48+07:00"


def test_parse_feed_tolerates_unescaped_ampersand():
    broken = "<rss><channel><item><title>Omoda & Jaecoo</title>"\
        "<link>https://tingia.gov.vn/a.html</link></item></channel></rss>"
    assert parse_feed(broken, "tingia.gov.vn")[0]["title"] == "Omoda & Jaecoo"


def _fake_web(pages):
    def opener(url, timeout):
        if url not in pages:
            raise urllib.error.HTTPError(url, 404, "not found", None, None)
        return pages[url]
    return opener


def test_crawl_respects_robots_and_skips_known_urls(tmp_path):
    pages = {
        "https://example.vn/robots.txt": "User-agent: *\nDisallow: /private\n",
        "https://example.vn/rss": RSS_XML,
        "https://blocked.vn/robots.txt": "User-agent: *\nDisallow: /\n",
    }
    fetcher = PoliteFetcher(delay=0, opener=_fake_web(pages))
    out = tmp_path / "rss.csv"
    feeds = {"example.vn": "https://example.vn/rss", "blocked.vn": "https://blocked.vn/rss"}

    first = crawl(feeds, out, fetcher)
    second = crawl(feeds, out, fetcher)

    assert (first["new"], first["feeds_failed"]) == (2, 1)
    assert (second["new"], second["skipped_known"]) == (0, 2)
    with out.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["text"] == "Tin & thời sự\n\nTóm tắt bài"

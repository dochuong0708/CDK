import json
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import quote
import xml.etree.ElementTree as ET

import app.services.fact_checking as fact_checking
from app.services.fact_checking import (
    BingNewsRSSProvider,
    DDGSSearchProvider,
    FactChecker,
    FallbackNewsSearchProvider,
)


class FakeSearch:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, limit):
        self.queries.append((query, limit))
        return self.results


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(self.responses))))


def test_search_provider_bounds_requests_and_prioritizes_publishers(monkeypatch):
    import ddgs

    calls = []
    results = {
        "claim": [
            {
                "title": "General result",
                "href": "https://wikipedia.org/example",
                "body": "General context.",
            }
        ],
        "claim site:vnexpress.net": [
            {
                "title": "VnExpress report",
                "href": "https://vnexpress.net/example",
                "body": "Publisher evidence.",
            }
        ],
        "claim site:tuoitre.vn": [
            {
                "title": "Tuoi Tre report",
                "href": "https://tuoitre.vn/example",
                "body": "Independent publisher evidence.",
            }
        ],
    }

    class FakeDDGS:
        def __init__(self, timeout):
            assert timeout == 8

        def text(self, query, backend, max_results):
            assert backend == "duckduckgo,google"
            calls.append(query)
            return results[query]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)

    sources = DDGSSearchProvider().search("claim", limit=3)

    assert calls == ["claim", "claim site:vnexpress.net", "claim site:tuoitre.vn"]
    assert [source["url"] for source in sources] == [
        "https://vnexpress.net/example",
        "https://tuoitre.vn/example",
        "https://wikipedia.org/example",
    ]


def test_bing_rss_returns_publisher_url_and_snippet(monkeypatch):
    root = ET.Element("rss")
    channel = ET.SubElement(root, "channel")
    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = "Bài báo gốc"
    ET.SubElement(item, "link").text = (
        "https://www.bing.com/news/apiclick.aspx?url="
        + quote("https://vnexpress.net/bai-bao-goc", safe="")
    )
    ET.SubElement(item, "description").text = "Đoạn trích từ bài báo."
    ET.SubElement(item, "pubDate").text = "Tue, 07 Oct 2026 00:00:00 GMT"
    ET.SubElement(item, "source").text = "VnExpress"
    payload = ET.tostring(root, encoding="utf-8")

    monkeypatch.setattr(fact_checking, "urlopen", lambda request, timeout: BytesIO(payload))

    result = BingNewsRSSProvider().search("luận điểm", limit=3)

    assert result[0]["url"] == "https://vnexpress.net/bai-bao-goc"
    assert result[0]["snippet"] == "Đoạn trích từ bài báo."
    assert result[0]["source"] == "VnExpress"
    assert fact_checking._is_priority_publisher(result[0]["url"])


def test_fallback_search_uses_ddgs_when_bing_rss_has_no_results():
    class Provider:
        def __init__(self, results):
            self.results = results
            self.queries = []

        def search(self, query, limit):
            self.queries.append((query, limit))
            return self.results

    bing = Provider([])
    ddgs = Provider([{"title": "Fallback", "url": "https://example.com", "snippet": ""}])

    result = FallbackNewsSearchProvider(primary=bing, fallback=ddgs).search("claim", limit=2)

    assert result[0]["title"] == "Fallback"
    assert bing.queries == [("claim", 2)]
    assert ddgs.queries == [("claim", 2)]


def test_fact_check_extracts_claim_searches_and_returns_cited_verdict():
    search = FakeSearch(
        [
            {
                "title": "Bài báo tham khảo",
                "url": "https://vnexpress.net/example",
                "snippet": "Bài báo xác nhận sự kiện xảy ra năm 2024.",
                "source": "vnexpress.net",
                "published_at": "2024-01-01",
            }
        ]
    )
    client = FakeClient(
        [
            {"claims": [{"claim": "Sự kiện X xảy ra năm 2024", "search_query": "sự kiện X 2024"}]},
            {
                "verdict": "supported",
                "confidence": 0.82,
                "explanation": "Snippet phù hợp với luận điểm.",
                "source_findings": [{"source_index": 0, "stance": "supports"}],
            },
        ]
    )

    report = FactChecker(client=client, searcher=search).check("Sự kiện X xảy ra năm 2024.")

    assert report.claims[0].verdict == "supported"
    assert report.claims[0].evidence[0].url == "https://vnexpress.net/example"
    assert report.claims[0].evidence[0].priority_publisher is True
    assert len(client.calls) == 2
    assert search.queries == [("sự kiện X 2024", 3)]


def test_fact_check_marks_claim_insufficient_when_search_has_no_results():
    client = FakeClient([{"claims": [{"claim": "Luận điểm chưa rõ", "search_query": "luận điểm"}]}])
    report = FactChecker(client=client, searcher=FakeSearch([])).check("Có một luận điểm cần kiểm chứng.")

    assert report.claims[0].verdict == "insufficient_evidence"
    assert report.claims[0].confidence == 0
    assert report.claims[0].evidence == []
    assert len(client.calls) == 1


def test_fact_check_does_not_decide_from_non_priority_sources_alone():
    search = FakeSearch(
        [
            {
                "title": "Wikipedia article",
                "url": "https://vi.wikipedia.org/wiki/Example",
                "snippet": "A snippet that appears to support the claim.",
                "source": "vi.wikipedia.org",
                "published_at": "",
            }
        ]
    )
    client = FakeClient(
        [{"claims": [{"claim": "Claim X", "search_query": "Claim X"}]}]
    )

    report = FactChecker(client=client, searcher=search).check("Claim X.")

    assert report.claims[0].verdict == "insufficient_evidence"
    assert report.claims[0].evidence[0].stance == "context"
    assert len(client.calls) == 1


def test_context_only_sources_cannot_produce_decisive_verdict():
    search = FakeSearch(
        [
            {
                "title": "Priority source",
                "url": "https://vnexpress.net/context",
                "snippet": "Background context, no direct confirmation.",
                "source": "vnexpress.net",
                "published_at": "",
            }
        ]
    )
    client = FakeClient(
        [
            {"claims": [{"claim": "Claim X", "search_query": "Claim X"}]},
            {
                "verdict": "supported",
                "confidence": 0.9,
                "explanation": "The source supports the claim.",
                "source_findings": [{"source_index": 0, "stance": "context"}],
            },
        ]
    )

    report = FactChecker(client=client, searcher=search).check("Claim X.")

    assert report.claims[0].verdict == "insufficient_evidence"
    assert report.claims[0].confidence == 0
    assert "chỉ cung cấp ngữ cảnh" in report.claims[0].explanation


def test_irrelevant_search_results_remain_clickable_but_not_decisive():
    search = FakeSearch(
        [
            {
                "title": "Related article",
                "url": "https://vnexpress.net/related",
                "snippet": "A tangentially related excerpt.",
                "source": "vnexpress.net",
                "published_at": "",
            }
        ]
    )
    client = FakeClient(
        [
            {"claims": [{"claim": "Claim X", "search_query": "Claim X"}]},
            {
                "verdict": "supported",
                "confidence": 0.9,
                "explanation": "Model explanation.",
                "source_findings": [{"source_index": 0, "stance": "irrelevant"}],
            },
        ]
    )

    report = FactChecker(client=client, searcher=search).check("Claim X.")

    assert report.claims[0].verdict == "insufficient_evidence"
    assert report.claims[0].evidence[0].url == "https://vnexpress.net/related"
    assert report.claims[0].evidence[0].stance == "irrelevant"


def test_supporting_source_missing_claim_year_is_downgraded_to_context():
    search = FakeSearch(
        [
            {
                "title": "Related earlier article",
                "url": "https://vnexpress.net/related",
                "snippet": "A related event, without its year.",
                "source": "VnExpress",
                "published_at": "2023-06-04",
            }
        ]
    )
    client = FakeClient(
        [
            {"claims": [{"claim": "Quan hệ được nâng cấp năm 2024", "search_query": "quan hệ nâng cấp 2024"}]},
            {
                "verdict": "supported",
                "confidence": 0.9,
                "explanation": "The source supports it.",
                "source_findings": [{"source_index": 0, "stance": "supports"}],
            },
        ]
    )

    report = FactChecker(client=client, searcher=search).check("Quan hệ được nâng cấp năm 2024.")

    assert report.claims[0].verdict == "insufficient_evidence"
    assert report.claims[0].evidence[0].stance == "context"


def test_sources_omitted_by_model_remain_available_as_context_links():
    search = FakeSearch(
        [
            {
                "title": "Priority source",
                "url": "https://vnexpress.net/source",
                "snippet": "A retrieved snippet.",
                "source": "VnExpress",
                "published_at": "",
            }
        ]
    )
    client = FakeClient(
        [
            {"claims": [{"claim": "Claim X", "search_query": "Claim X"}]},
            {
                "verdict": "insufficient_evidence",
                "confidence": 0.0,
                "explanation": "Not enough evidence.",
                "source_findings": [],
            },
        ]
    )

    report = FactChecker(client=client, searcher=search).check("Claim X.")

    assert report.claims[0].verdict == "insufficient_evidence"
    assert report.claims[0].evidence[0].url == "https://vnexpress.net/source"
    assert report.claims[0].evidence[0].stance == "context"


def test_fact_check_returns_empty_report_when_no_verifiable_claims():
    client = FakeClient([{"claims": []}])
    search = FakeSearch([])

    report = FactChecker(client=client, searcher=search).check("Tôi nghĩ bài viết này rất hay.")

    assert report.claims == []
    assert "Không tìm thấy luận điểm" in report.summary
    assert search.queries == []
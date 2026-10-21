"""Thu thập tin từ RSS báo chính thống (nhãn ``real`` yếu) và trang tin giả đã bác bỏ.

Chạy định kỳ để tích lũy dữ liệu; URL đã có sẽ được bỏ qua.

    python -m app.detectors.language.rss_crawler --full-text
    python -m app.detectors.language.rss_crawler --debunks
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

from app.detectors.language.data_pipeline import DEFAULT_RAW_DIR, USER_AGENT, clean_text

# Đã kiểm tra hoạt động (RSS 2.0). Thêm nguồn bằng --feed publisher=url.
DEFAULT_FEEDS: dict[str, str] = {
    "vnexpress.net": "https://vnexpress.net/rss/tin-moi-nhat.rss",
    "tuoitre.vn": "https://tuoitre.vn/rss/tin-moi-nhat.rss",
    "thanhnien.vn": "https://thanhnien.vn/rss/home.rss",
    "dantri.com.vn": "https://dantri.com.vn/rss/home.rss",
    "vtv.vn": "https://vtv.vn/rss/home.rss",
}
# Bài của VAFC mô tả và bác bỏ tin giả: bản thân bài là tin thật, cần gán nhãn thủ công
# (trích claim sai) trước khi đưa vào huấn luyện, nên được lưu riêng.
DEBUNK_FEEDS: dict[str, str] = {"tingia.gov.vn": "https://tingia.gov.vn/rss"}

FIELDS = ("url", "publisher", "title", "summary", "text", "published_at", "crawled_at")
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def strip_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value or "")
    return clean_text(html.unescape(without_tags))


def _iso_date(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return value


def _sanitize_xml(xml_text: str) -> str:
    """Sửa lỗi phổ biến của feed: dấu & chưa escape và ký tự điều khiển không hợp lệ."""
    escaped = re.sub(r"&(?!#\d+;|#x[0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]*;)", "&amp;", xml_text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", escaped)


def parse_feed(xml_text: str, publisher: str) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        root = ET.fromstring(_sanitize_xml(xml_text))
    entries: list[dict[str, str]] = []
    for item in root.iter("item"):
        entries.append({
            "url": (item.findtext("link") or item.findtext("guid") or "").strip(),
            "publisher": publisher,
            "title": strip_html(item.findtext("title") or ""),
            "summary": strip_html(item.findtext("description") or ""),
            "published_at": _iso_date(item.findtext("pubDate") or ""),
        })
    for entry in root.iter(f"{ATOM_NS}entry"):
        link = entry.find(f"{ATOM_NS}link")
        entries.append({
            "url": (link.get("href", "") if link is not None else "").strip(),
            "publisher": publisher,
            "title": strip_html(entry.findtext(f"{ATOM_NS}title") or ""),
            "summary": strip_html(entry.findtext(f"{ATOM_NS}summary") or ""),
            "published_at": (entry.findtext(f"{ATOM_NS}updated") or "").strip(),
        })
    return [entry for entry in entries if entry["url"]]


def extract_article_text(page_html: str, url: str) -> str:
    try:
        import trafilatura
    except ImportError:
        return ""
    return clean_text(trafilatura.extract(page_html, url=url, include_comments=False) or "")


@dataclass
class PoliteFetcher:
    """Tải trang có User-Agent rõ ràng, tôn trọng robots.txt và giãn cách theo từng host."""

    delay: float = 1.0
    timeout: float = 20.0
    opener: Callable[[str, float], str] | None = None
    _robots: dict[str, urllib.robotparser.RobotFileParser | None] = field(default_factory=dict)
    _last_request: dict[str, float] = field(default_factory=dict)

    def _open(self, url: str) -> str:
        if self.opener is not None:
            return self.opener(url, self.timeout)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")

    def _throttle(self, host: str) -> None:
        wait = self._last_request.get(host, 0.0) + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            try:
                self._throttle(parsed.netloc)
                parser.parse(self._open(f"{host}/robots.txt").splitlines())
                self._robots[host] = parser
            except urllib.error.HTTPError as error:
                if error.code in (404, 410):
                    parser.parse([])
                    self._robots[host] = parser
                elif error.code in (401, 403):
                    parser.parse(["User-agent: *", "Disallow: /"])
                    self._robots[host] = parser
                else:
                    self._robots[host] = None
            except (urllib.error.URLError, TimeoutError, OSError):
                self._robots[host] = None
        robots = self._robots[host]
        return robots is not None and robots.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> str | None:
        if not self.allowed(url):
            return None
        self._throttle(urlparse(url).netloc)
        return self._open(url)


def read_known_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    csv.field_size_limit(1 << 30)
    with path.open("r", encoding="utf-8", newline="") as source:
        return {row["url"] for row in csv.DictReader(source) if row.get("url")}


def crawl(
    feeds: dict[str, str],
    out_file: Path,
    fetcher: PoliteFetcher,
    full_text: bool = False,
    max_items_per_feed: int | None = None,
) -> dict[str, int]:
    known = read_known_urls(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_file.exists()
    stats = {"feeds_ok": 0, "feeds_failed": 0, "new": 0, "skipped_known": 0, "full_text": 0}

    with out_file.open("a", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for publisher, feed_url in feeds.items():
            try:
                feed_xml = fetcher.get(feed_url)
                entries = parse_feed(feed_xml, publisher) if feed_xml else []
            except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError) as error:
                print(f"Lỗi feed {publisher}: {error}")
                stats["feeds_failed"] += 1
                continue
            if feed_xml is None:
                print(f"robots.txt không cho phép: {feed_url}")
                stats["feeds_failed"] += 1
                continue
            stats["feeds_ok"] += 1
            for entry in entries[:max_items_per_feed]:
                if entry["url"] in known:
                    stats["skipped_known"] += 1
                    continue
                text = ""
                if full_text:
                    try:
                        page = fetcher.get(entry["url"])
                        text = extract_article_text(page, entry["url"]) if page else ""
                    except (urllib.error.URLError, TimeoutError, OSError) as error:
                        print(f"Lỗi tải bài {entry['url']}: {error}")
                if text:
                    stats["full_text"] += 1
                else:
                    text = clean_text(f"{entry['title']}\n\n{entry['summary']}")
                writer.writerow({
                    **entry,
                    "text": text,
                    "crawled_at": datetime.now(tz=UTC).isoformat(),
                })
                known.add(entry["url"])
                stats["new"] += 1
            target.flush()
    return stats


def _parse_feed_overrides(values: Sequence[str]) -> dict[str, str]:
    feeds: dict[str, str] = {}
    for value in values:
        publisher, separator, url = value.partition("=")
        if not separator or not publisher or not url:
            raise SystemExit(f"--feed cần dạng publisher=url, nhận được: {value}")
        feeds[publisher.strip()] = url.strip()
    return feeds


def main(argv: Sequence[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feed", action="append", default=[], help="publisher=url (lặp lại được)")
    parser.add_argument("--debunks", action="store_true", help="Thu thập tingia.gov.vn vào file riêng")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--full-text", action="store_true", help="Tải toàn văn (cần trafilatura)")
    parser.add_argument("--delay", type=float, default=1.0, help="Giây giữa 2 request cùng host")
    parser.add_argument("--max-items-per-feed", type=int, default=None)
    args = parser.parse_args(argv)

    feeds = _parse_feed_overrides(args.feed) or (DEBUNK_FEEDS if args.debunks else DEFAULT_FEEDS)
    default_name = "debunks_tingia.csv" if args.debunks else "news_rss.csv"
    out_file = args.out or DEFAULT_RAW_DIR / "rss" / default_name
    stats = crawl(
        feeds, out_file, PoliteFetcher(delay=args.delay), args.full_text, args.max_items_per_feed
    )
    print(f"{stats} -> {out_file}")
    return stats


if __name__ == "__main__":
    main()

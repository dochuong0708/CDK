"""Phát hiện tên miền giả mạo thương hiệu (typosquatting / homoglyph)."""

from dataclasses import dataclass
from urllib.parse import urlparse

SHORTENED_HOSTS = {"bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "ow.ly", "shorturl.at"}

BRAND_DOMAINS: dict[str, set[str]] = {
    "vietcombank": {"vietcombank.com.vn"},
    "techcombank": {"techcombank.com.vn"},
    "mbbank": {"mbbank.com.vn"},
    "bidv": {"bidv.com.vn"},
    "agribank": {"agribank.com.vn"},
    "vietinbank": {"vietinbank.vn"},
    "vpbank": {"vpbank.com.vn"},
    "acb": {"acb.com.vn"},
    "sacombank": {"sacombank.com.vn"},
    "tpbank": {"tpbank.vn"},
    "shinhan": {"shinhan.com.vn"},
    "momo": {"momo.vn"},
    "zalopay": {"zalopay.vn"},
    "shopee": {"shopee.vn"},
}

# Hậu tố hai cấp phổ biến để tách phần "tên" của tên miền.
MULTI_LEVEL_SUFFIXES = {
    "com.vn", "net.vn", "org.vn", "gov.vn", "edu.vn", "info.vn", "biz.vn", "name.vn",
    "co.uk", "com.au", "co.jp", "com.sg",
}

HOMOGLYPHS = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s",
    "!": "i", "|": "l",
}
MULTI_CHAR_HOMOGLYPHS = (("rn", "m"), ("vv", "w"), ("cl", "d"))

# Thương hiệu ngắn (vd: "acb", "bidv") dễ trùng ngẫu nhiên nên không so khớp mờ.
MIN_FUZZY_BRAND_LENGTH = 5


@dataclass(frozen=True)
class DomainFinding:
    host: str
    brand: str
    reason: str
    score: float


def hostname(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.hostname or "").casefold().rstrip(".")


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalize_homoglyphs(label: str) -> str:
    normalized = "".join(HOMOGLYPHS.get(char, char) for char in label)
    for source, target in MULTI_CHAR_HOMOGLYPHS:
        normalized = normalized.replace(source, target)
    return normalized


def _is_official(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _is_official_anywhere(host: str) -> bool:
    return any(_is_official(host, domains) for domains in BRAND_DOMAINS.values())


def _name_labels(host: str) -> list[str]:
    labels = host.removeprefix("www.").split(".")
    suffix_size = 2 if ".".join(labels[-2:]) in MULTI_LEVEL_SUFFIXES else 1
    return labels[:-suffix_size] if len(labels) > suffix_size else labels


def _max_distance(brand: str) -> int:
    return 1 if len(brand) < 9 else 2


def inspect_host(host: str) -> DomainFinding | None:
    """Trả về phát hiện giả mạo nếu host không chính thức nhưng giống một thương hiệu."""
    if not host or _is_official_anywhere(host):
        return None

    labels = _name_labels(host)
    tokens = {token for label in labels for token in label.split("-") if token}

    for brand in BRAND_DOMAINS:
        contains_brand = (
            any(brand in token for token in tokens)
            if len(brand) >= MIN_FUZZY_BRAND_LENGTH
            else brand in tokens
        )
        if contains_brand:
            return DomainFinding(host, brand, f"Tên miền chứa '{brand}' nhưng không phải tên miền chính thức", 0.55)

    for brand in BRAND_DOMAINS:
        if len(brand) < MIN_FUZZY_BRAND_LENGTH:
            continue
        for token in tokens:
            normalized = normalize_homoglyphs(token)
            if normalized == brand or brand in normalized:
                return DomainFinding(host, brand, f"Tên miền dùng ký tự thay thế để giả '{brand}'", 0.6)
            distance = levenshtein(normalized, brand)
            if 0 < distance <= _max_distance(brand):
                return DomainFinding(
                    host,
                    brand,
                    f"Tên miền gần giống '{brand}' (khoảng cách Levenshtein = {distance})",
                    0.55,
                )
    return None

import re
from urllib.parse import urlparse


URGENCY_PATTERNS = (
    r"kh[oó]a\s+t[aà]i\s+kho[aả]n",
    r"x[aá]c\s+minh\s+ngay",
    r"trong\s+v[oò]ng\s+24\s*gi[oờ]",
    r"tr[uú]ng\s+th[uưở]ng",
    r"kh[oô]ng\s+th[iì]\s+(?:t[aà]i\s+kho[aả]n\s+)?s[eẽ]\s+b[iị]\s+kh[oó]a",
)
SENSITIVE_PATTERNS = (
    r"chuy[eể]n\s+ti[eề]n",
    r"m[aậ]\s*k[hẩ]u|password",
    r"(?:m[aã]\s+)?otp",
    r"s\.?\s*t\.?\s*k\.?",
    r"m[aã]\s+x[aá]c\s+th[uự]c",
)
THREAT_PATTERNS = (
    r"t[aà]i\s+kho[aả]n\s+(?:c[uủ]a\s+b[aạ]n\s+)?s[eẽ]\s+b[iị]\s+(?:kh[oó]a|h[uủ]y)",
    r"ph[aạ]t\s+ngu[yê]n|kh[oở]i\s+t[oố]",
)
SHORTENED_HOSTS = {"bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "ow.ly", "shorturl.at"}
BANK_DOMAINS = {
    "vietcombank": {"vietcombank.com.vn"},
    "techcombank": {"techcombank.com.vn"},
    "mbbank": {"mbbank.com.vn"},
    "bidv": {"bidv.com.vn"},
    "agribank": {"agribank.com.vn"},
    "vietinbank": {"vietinbank.vn"},
    "vpbank": {"vpbank.com.vn"},
    "acb": {"acb.com.vn"},
    "sacombank": {"sacombank.com.vn"},
}


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]


def _hostname(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.hostname or "").casefold().rstrip(".")


def inspect_rules(preprocessed: dict[str, object]) -> tuple[float, list[str]]:
    text = str(preprocessed["lowered"])
    evidence: list[str] = []
    score = 0.0

    if _matches(text, URGENCY_PATTERNS):
        score += 0.35
        evidence.append("Có ngôn ngữ thúc giục hoặc hứa trúng thưởng")
    if _matches(text, THREAT_PATTERNS):
        score += 0.4
        evidence.append("Có dấu hiệu đe dọa khóa tài khoản hoặc gây áp lực")
    if _matches(text, SENSITIVE_PATTERNS):
        score += 0.4
        evidence.append("Có yêu cầu chuyển tiền hoặc cung cấp thông tin xác thực")

    for url in preprocessed["urls"]:
        host = _hostname(str(url))
        if host in SHORTENED_HOSTS:
            score += 0.35
            evidence.append(f"Phát hiện URL rút gọn: {host}")
        for bank, official_domains in BANK_DOMAINS.items():
            is_official_host = any(
                host == domain or host.endswith(f".{domain}")
                for domain in official_domains
            )
            if bank in host and not is_official_host:
                score += 0.55
                evidence.append(f"Tên miền có thể giả mạo {bank}: {host}")
                break

    if preprocessed["phone_numbers"]:
        evidence.append("Văn bản chứa số điện thoại; cần tránh chia sẻ thông tin cá nhân")
    if preprocessed["bank_accounts"]:
        evidence.append("Văn bản chứa chuỗi số giống số tài khoản; cần tránh chia sẻ thông tin cá nhân")

    if not evidence:
        evidence.append("Không tìm thấy dấu hiệu rủi ro phổ biến theo bộ quy tắc hiện tại")
    return min(score, 0.95), evidence

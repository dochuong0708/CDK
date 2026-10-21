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
    if _matches(text, SENSATIONAL_PATTERNS):
        score += 0.30
        evidence.append("Sử dụng từ ngữ giật gân, câu view, thổi phồng cảm xúc")
    if _matches(text, UNVERIFIED_SOURCE_PATTERNS):
        score += 0.35
        evidence.append("Dẫn chứng nguồn tin mập mờ, nặc danh hoặc tin đồn chưa kiểm chứng")
    if _matches(text, CALL_TO_ACTION_PATTERNS):
        score += 0.25
        evidence.append("Thúc ép chia sẻ, lan truyền gấp thông tin thiếu kiểm chứng")
    if _matches(text, CONSPIRACY_PATTERNS):
        score += 0.45
        evidence.append("Chứa nội dung thuyết âm mưu hoặc khẳng định chữa bệnh siêu nhiên/thần dược")

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
        evidence.append("Không tìm thấy dấu hiệu rủi ro rõ ràng theo bộ quy tắc")
    return min(score, 0.95), evidence


SENSATIONAL_PATTERNS = (
    r"c[uự]c\s+s[oố]c",
    r"ch[aá]n\s+[đd][oộ]ng",
    r"kinh\s+ho[aà]ng",
    r"ti[eế]t\s+l[oộ]\s+[đd][oộ]ng\s+tr[oờ]i",
    r"kh[oô]ng\s+th[eể]\s+tin\s+n[oổ]i",
    r"b[aà]ng\s+ho[aà]ng",
    r"ch[eế]t\s+l[aặ]ng",
    r"ng[aã]\s+ng[uử]a",
    r"lan\s+truy[eề]n\s+ch[oó]ng\s+m[aặ]t",
    r"s[uự]\s+th[aậ]t\s+r[oợ]n\s+ng[uườ]i",
    r"kh[oô]ng\s+ng[oờ]\s+t[oớ]i",
    r"kinh\s+h[aã]i",
)

UNVERIFIED_SOURCE_PATTERNS = (
    r"ngu[oồ]n\s+tin\s+gi[aấ]u\s+t[eê]n",
    r"tin\s+[đd][oồ]n",
    r"d[aâ]n\s+m[aạ]ng\s+[đd][oồ]n",
    r"chuy[eê]n\s+gia\s+gi[aấ]u\s+t[eê]n",
    r"nh[oó]m\s+k[ií]n",
    r"nghe\s+n[oó]i\s+l[aà]",
    r"[đd][uượ]c\s+cho\s+l[aà]\s+c[oó]\s+th[aậ]t",
    r"ngu[oồ]n\s+tin\s+n[oộ]i\s+b[oộ]\s+b[ií]\s+m[aậ]t",
    r"b[aá]c\s+s[iĩ]\s+gi[aấ]u\s+t[eê]n",
)

CALL_TO_ACTION_PATTERNS = (
    r"chia\s+s[eẻ]\s+ngay\s+k[eẻ]o\s+mu[oộ]n",
    r"chia\s+s[eẻ]\s+kh[aẩ]n\s+c[aấ]p",
    r"c[aả]nh\s+b[aá]o\s+kh[aẩ]n\s+c[aấ]p",
    r"chia\s+s[eẻ]\s+g[aấ]p",
    r"lan\s+t[oả]a\s+ngay",
    r"h[aã]y\s+chia\s+s[eẻ]",
    r"h[aã]y\s+share\s+ngay",
    r"c[aả]nh\s+gi[aá]c\s+g[aấ]p",
    r"tag\s+ngay\s+ng[uườ]i\s+th[aâ]n",
)

CONSPIRACY_PATTERNS = (
    r"ch[ií]nh\s+ph[uủ]\s+che\s+gi[aấ]u",
    r"s[uự]\s+th[aậ]t\s+b[iị]\s+b[uư]ng\s+b[ií]t",
    r"th[aầ]n\s+d[uượ]c\s+ch[uữ]a\s+kh[oỏ]i",
    r"ch[uữ]a\s+d[uứt]\s+[đd]i[eể]m\s+ung\s+th[uư]",
    r"100%\s+kh[oô]ng\s+t[aái]\s+ph[aá]t",
    r"b[ií]\s+m[aậ]t\s+b[iị]\s+che\s+[đd][aậ]y",
    r"[aâ]m\s+m[uư]u\s+th[aâ]m\s+[đd][oộ]c",
    r"t[aậ]p\s+[đd]o[aà]n\s+d[uượ]c\s+che\s+gi[aấ]u",
    r"th[eế]\s+l[uự]c\s+ng[aầ]m",
)

RULE_CATEGORY_INFO = [
    (SENSATIONAL_PATTERNS, "sensational", "Từ ngữ giật gân, câu view, kích động cảm xúc"),
    (UNVERIFIED_SOURCE_PATTERNS, "unverified", "Nguồn tin nặc danh, tin đồn hoặc thiếu kiểm chứng"),
    (CALL_TO_ACTION_PATTERNS, "call_to_action", "Lời kêu gọi chia sẻ/lan truyền gấp gáp"),
    (CONSPIRACY_PATTERNS, "conspiracy", "Dấu hiệu thuyết âm mưu, thông tin y tế phản khoa học"),
    (URGENCY_PATTERNS, "urgency", "Ngôn từ giục giã, áp đặt thời hạn giả tạo"),
    (THREAT_PATTERNS, "threat", "Đe dọa, hù dọa cơ quan chức năng hoặc khóa tài khoản"),
    (SENSITIVE_PATTERNS, "sensitive", "Yêu cầu nhạy cảm liên quan đến tài khoản, chuyển tiền, OTP"),
]


def find_suspicious_phrases(text: str) -> list[dict[str, object]]:
    """Tìm các cụm từ đáng ngờ trong đoạn văn kèm vị trí và lý do cảnh báo."""
    found: list[dict[str, object]] = []
    seen_spans: set[tuple[int, int]] = set()

    for patterns, category, reason in RULE_CATEGORY_INFO:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = match.span()
                # Tránh trùng lặp hoàn toàn
                if any(s[0] <= span[0] and span[1] <= s[1] for s in seen_spans):
                    continue
                seen_spans.add(span)
                found.append({
                    "phrase": match.group(0),
                    "start": span[0],
                    "end": span[1],
                    "category": category,
                    "reason": reason,
                })
    # Sắp xếp theo thứ tự xuất hiện trong văn bản
    found.sort(key=lambda item: int(item["start"]))
    return found

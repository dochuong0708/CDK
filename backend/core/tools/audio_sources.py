"""URL intake with explicit hosts, public addresses, TLS IP pinning and no redirects."""
import http.client
import ipaddress
import socket
import ssl
from time import monotonic
from pathlib import Path
from urllib.parse import urlsplit

from core.tools.audio_features_tool import EXTENSIONS, MAX_BYTES


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address):
        super().__init__(host, timeout=15, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        raw = socket.create_connection((self.address, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


def download_url(url, target: Path, allowed_hosts: set[str]):
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or parsed.port not in (None, 443) or parsed.username or parsed.password
            or parsed.fragment or host not in allowed_hosts):
        raise ValueError("URL phải là HTTPS và thuộc DEEPGUARD_AUDIO_URL_HOSTS; không nhận credential/fragment.")
    if Path(parsed.path).suffix.lower() not in EXTENSIONS:
        raise ValueError("URL phải có đuôi WAV/MP3/FLAC/OGG.")
    addresses = {r[4][0] for r in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("URL phải trỏ đến địa chỉ Internet công khai.")
    connection = PinnedHTTPS(host, sorted(addresses)[0])
    deadline = monotonic() + 45
    try:
        connection.request("GET", parsed.path + ("?" + parsed.query if parsed.query else ""),
                           headers={"Accept": "audio/*", "Accept-Encoding": "identity"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("URL không trả HTTP 200; không tự theo redirect.")
        if int(response.getheader("Content-Length", "0")) > MAX_BYTES:
            raise ValueError("File vượt quá 20 MB.")
        size = 0
        with target.open("wb") as output:
            while chunk := response.read(65536):
                if monotonic() > deadline:
                    raise ValueError("Tải URL quá thời gian cho phép.")
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("File vượt quá 20 MB.")
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

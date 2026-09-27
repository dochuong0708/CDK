"""Download official ASVspoof 2019 LA with resumable ranges and verify its MD5.

Source metadata: Edinburgh DataShare bitstream a9f87c35-f055-4015-80e2-2fdff0d46269.
No credentials required. Uses four HTTP workers and retains partial files on failure.
"""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
SIZE = 7640952520
MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
ROOT = Path(__file__).resolve().parents[1] / "data/audio/downloads"
CHUNK = 64 * 1024 * 1024


def verify(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if path.stat().st_size != SIZE or actual != MD5:
        raise ValueError(f"Archive checksum mismatch: {actual}; expected {MD5}")
    (ROOT/"LA.verified.json").write_text(json.dumps(dict(url=URL, bytes=SIZE, md5=actual), indent=2))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT/"LA.zip"
    prefix = ROOT/"LA.zip.prefix"
    if archive.exists():
        if archive.stat().st_size == SIZE:
            verify(archive)
            print("Archive already complete and verified.", flush=True)
            return
        if prefix.exists():
            raise ValueError("Both incomplete archive and prefix exist; refusing to overwrite either.")
        archive.rename(prefix)
    prefix_size = prefix.stat().st_size if prefix.exists() else 0
    if prefix_size > SIZE:
        raise ValueError("Invalid prefix size")
    jobs = [(start, min(SIZE-1, start+CHUNK-1)) for start in range(prefix_size, SIZE, CHUNK)]

    def part_path(start, end):
        return ROOT/f"LA.{start}-{end}.part"

    def fetch(job):
        start, end = job
        path = part_path(start, end)
        for attempt in range(4):
            received = path.stat().st_size if path.exists() else 0
            if received == end-start+1:
                return
            if received > end-start+1:
                raise ValueError(f"Invalid part: {path}")
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start+received}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start+received}-{end}/{SIZE}":
                        raise ValueError("Server did not honor byte range")
                    with path.open("ab") as output:
                        while block := response.read(1024*1024):
                            output.write(block)
                if path.stat().st_size != end-start+1:
                    raise OSError("Incomplete range response")
                return
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2*(attempt+1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        while not all(f.done() for f in futures):
            size = prefix_size + sum(part_path(a, b).stat().st_size for a, b in jobs if part_path(a, b).exists())
            print(f"Downloaded {size/1024**3:.2f}/{SIZE/1024**3:.2f} GiB ({size/SIZE:.1%})", flush=True)
            time.sleep(10)
        for future in futures:
            future.result()
    assembled = ROOT/"LA.zip.assembling"
    with assembled.open("wb") as output:
        sources = ([prefix] if prefix.exists() else []) + [part_path(a, b) for a, b in jobs]
        for source in sources:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=4*1024*1024)
    verify(assembled)
    assembled.replace(archive)
    # Only files created for this download, after verifying the complete archive.
    for source in sources:
        source.unlink()
    print(f"Verified official MD5 {MD5}. Ready: {archive}", flush=True)


if __name__ == "__main__":
    main()

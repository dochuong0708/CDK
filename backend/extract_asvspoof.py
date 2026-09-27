"""Verify and safely extract the downloaded official LA archive. Supports reruns."""
import shutil
import stat
import zipfile
import zlib
from pathlib import Path

from download_asvspoof import ROOT, verify


def main():
    archive = ROOT / "LA.zip"
    verify(archive)
    destination = (ROOT.parent / "asvspoof2019").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for index, member in enumerate(members, 1):
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                checksum = 0
                with target.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024*1024), b""):
                        checksum = zlib.crc32(block, checksum)
                if target.stat().st_size != member.file_size or checksum != member.CRC:
                    raise ValueError(f"Existing extracted file differs; refusing to overwrite: {target}")
            else:
                temporary = target.with_name(target.name + ".extracting")
                with bundle.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024*1024)
                temporary.replace(target)
            if index % 5000 == 0:
                print(f"Extracted/verified {index}/{len(members)} entries", flush=True)
    print(f"Extraction complete: {destination}", flush=True)


if __name__ == "__main__":
    main()

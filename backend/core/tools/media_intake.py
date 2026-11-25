from pathlib import Path

from PIL import Image, UnidentifiedImageError
from core.tools.audio_features_tool import EXTENSIONS, MAX_BYTES, decode_audio

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def inspect_media(path: Path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("File vượt quá 20 MB.")
    if path.suffix.lower() in EXTENSIONS:
        decode_audio(path)
        return "audio"
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"} or image.width*image.height > 25_000_000:
                    raise ValueError("Chỉ nhận PNG/JPEG/WebP tối đa 25 megapixel.")
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Không đọc được ảnh.") from exc
        return "image"
    raise ValueError("Không hỗ trợ loại file này; video chưa có tool.")

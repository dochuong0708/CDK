"""
Module Phân Tích Màu & Pháp Y Hình Ảnh (Color Forensics)
Dùng để hỗ trợ mô hình Deep Learning phát hiện ảnh do AI tạo.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def _to_cv2_bgr(img_input):
    """Chuyển đổi các định dạng đầu vào (Path, PIL Image, numpy array) về BGR numpy array của OpenCV."""
    if isinstance(img_input, (str, Path)):
        return cv2.imread(str(img_input))
    elif isinstance(img_input, Image.Image):
        rgb = img_input.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        return img_input
    return None


def analyze_cfa_bayer(img_input):
    """
    1. Phân tích mảng lọc màu Bayer (CFA Demosaicing).
    Ảnh thật từ camera có vết nội suy Bayer -> Score cao.
    Ảnh AI tạo trực tiếp pixel -> Thiếu vết Bayer -> Score thấp.
    """
    img = _to_cv2_bgr(img_input)
    if img is None or img.size == 0:
        return 0.0

    g = img[:, :, 1].astype(np.float32)
    laplacian = cv2.Laplacian(g, cv2.CV_32F)
    return float(np.std(laplacian))


def analyze_ycbcr_chroma(img_input):
    """
    2. Phân tích tỷ lệ nhiễu kênh màu (Cb, Cr) so với kênh độ sáng (Y).
    Ảnh AI thường có các kênh màu bị làm mịn/loang lổ bất thường.
    """
    img = _to_cv2_bgr(img_input)
    if img is None or img.size == 0:
        return {"var_y": 0.0, "var_chroma": 0.0, "chroma_ratio": 0.0}

    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycbcr)

    var_y = float(cv2.Laplacian(y, cv2.CV_64F).var())
    var_cb = float(cv2.Laplacian(cb, cv2.CV_64F).var())
    var_cr = float(cv2.Laplacian(cr, cv2.CV_64F).var())

    chroma_var = (var_cb + var_cr) / 2.0
    chroma_ratio = float(chroma_var / (var_y + 1e-5))

    return {
        "var_y": var_y,
        "var_chroma": float(chroma_var),
        "chroma_ratio": chroma_ratio,
    }


def analyze_hsv_histogram(img_input):
    """
    3. Phân tích độ bão hòa & đỉnh phổ màu (HSV Space).
    """
    img = _to_cv2_bgr(img_input)
    if img is None or img.size == 0:
        return {"mean_saturation": 0.0, "std_saturation": 0.0, "max_hue_peak": 0.0}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_sat = float(np.mean(s))
    std_sat = float(np.std(s))

    hist_h = cv2.calcHist([h], [0], None, [180], [0, 180]).flatten()
    hist_sum = hist_h.sum()
    if hist_sum > 0:
        hist_h = hist_h / hist_sum
    max_hue_peak = float(np.max(hist_h))

    return {
        "mean_saturation": mean_sat,
        "std_saturation": std_sat,
        "max_hue_peak": max_hue_peak,
    }


def get_color_forensic_adjustment(img_input):
    """
    Tổng hợp kết quả phân tích màu và trả về độ điều chỉnh (adjustment) cho xác suất AI:
    - Nếu có dấu hiệu bất thường màu/Bayer: +0.05 đến +0.12 (Tăng nghi vấn AI)
    - Nếu có vết Bayer tự nhiên của Camera thật: -0.05 đến -0.08 (Giảm nghi vấn AI)
    """
    cfa = analyze_cfa_bayer(img_input)
    ycbcr = analyze_ycbcr_chroma(img_input)
    hsv = analyze_hsv_histogram(img_input)

    adjustment = 0.0
    chroma_ratio = ycbcr["chroma_ratio"]

    # Cảnh báo bất thường màu & nhiễu Bayer
    if cfa < 18.0 and chroma_ratio < 0.06:
        adjustment += 0.12  # Cực kỳ nghi vấn AI
    elif cfa < 25.0 and chroma_ratio < 0.1:
        adjustment += 0.06  # Nghi vấn AI nhẹ

    # Ảnh chụp thật thường có CFA cao (>35) và nhiễu màu tự nhiên
    if cfa > 35.0 and chroma_ratio > 0.15:
        adjustment -= 0.08  # Thiên về ảnh thật

    return {
        "adjustment": adjustment,
        "cfa_score": cfa,
        "chroma_ratio": chroma_ratio,
        "mean_saturation": hsv["mean_saturation"],
        "max_hue_peak": hsv["max_hue_peak"],
    }
"""
Ứng dụng web chạy trên máy bạn: kiểm tra ảnh / video / âm thanh có phải do AI tạo không.

Cách chạy (trong thư mục dự án, đã bật venv, đã có detector.pt):
    python app.py
Trình duyệt sẽ tự mở trang http://127.0.0.1:7860
Dừng ứng dụng: nhấn Ctrl+C trong Terminal.

File bạn tải lên chỉ được xử lý trên máy của bạn.
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ai_image_detector import (
    FAKE_THRESHOLD,
    REAL_THRESHOLD,
    build_model,
    get_device,
)

MODEL_PATH = Path("detector.pt")
VIDEO_FRAMES = 24  # Số khung hình lấy mẫu từ mỗi video

DISCLAIMER = (
    "**Lưu ý:** Kết quả chỉ là xác suất, không phải kết luận chắc chắn. "
    "Mô hình hiện áp dụng cơ chế **Patch-based Scanning (Top-20%)** để giảm tỷ lệ báo nhầm trên ảnh phân giải cao. "
    "Không dùng để kết luận pháp lý về nội dung của người khác."
)

device = get_device()

# Transformation CHỈ ToTensor + Normalize (KHÔNG Resize, KHÔNG Crop) để giữ nguyên pixel gốc
patch_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    if not MODEL_PATH.exists():
        return None, 0
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["fake_idx"]


MODEL, FAKE_IDX = load_model()
NO_MODEL_MSG = "⚠️ Chưa có file `detector.pt`. Hãy huấn luyện mô hình trước (lệnh `train`)."


@torch.no_grad()
def predict_image_fake_prob(img, patch_size=224, stride=112, top_k_percent=0.20, batch_size=32):
    """
    Tính xác suất AI cho một ảnh bằng cơ chế:
    1. Quét Patch 224x224 trực tiếp trên độ phân giải gốc (không Resize).
    2. Lấy trung bình Top 20% Patch có điểm cao nhất (giảm False Positive trên ảnh REAL).
    """
    if MODEL is None:
        return 0.0

    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    img_tensor = patch_transform(img_rgb).unsqueeze(0)  # [1, 3, H, W]

    # Pad nếu ảnh quá nhỏ (nhỏ hơn patch_size)
    pad_w = max(0, patch_size - w)
    pad_h = max(0, patch_size - h)
    if pad_w > 0 or pad_h > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # Tách ảnh thành các patch dạng cửa sổ trượt (unfold)
    patches = img_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)

    # Dự đoán theo batch để tránh tràn RAM/VRAM
    scores = []
    for i in range(0, patches.size(0), batch_size):
        batch = patches[i:i + batch_size].to(device)
        out = MODEL(batch)
        p = torch.softmax(out, dim=1)[:, FAKE_IDX]
        scores.extend(p.cpu().tolist())

    scores_tensor = torch.tensor(scores)

    # Nếu ảnh nhỏ chỉ tạo ra 1 patch
    if len(scores_tensor) == 1:
        return float(scores_tensor[0])

    # Top-K Average Pooling: Lấy trung bình top K% (mặc định 20%) patch có điểm FAKE cao nhất
    k = max(1, int(len(scores_tensor) * top_k_percent))
    top_k_scores, _ = torch.topk(scores_tensor, k)
    return float(top_k_scores.mean().item())


def verdict(p, noun):
    if p >= FAKE_THRESHOLD:
        return f"🔴 {noun} nhiều khả năng do AI tạo"
    if p <= REAL_THRESHOLD:
        return f"🟢 {noun} nhiều khả năng là thật"
    return "🟡 Không chắc chắn - cần người xem xét"


# ---------- Ảnh ----------
def analyze_image(img):
    if MODEL is None:
        return None, NO_MODEL_MSG
    if img is None:
        return None, "Hãy tải một ảnh lên."
    
    p = predict_image_fake_prob(img)
    return {"Do AI tạo": p, "Thật": 1 - p}, f"### {verdict(p, 'Ảnh')}\nXác suất AI: **{p:.1%}**"


# ---------- Video ----------
def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in np.linspace(0, total - 1, min(n, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

# ---------- Analyze Video ----------
def analyze_video(path):
    if MODEL is None:
        return None, NO_MODEL_MSG
    if not path:
        return None, "Hãy tải một video lên."

    frames = sample_frames(path, VIDEO_FRAMES)
    if not frames:
        return None, "Không đọc được khung hình nào từ video này."

    # Phân tích từng frame
    probs = [predict_image_fake_prob(frame) for frame in frames]
    probs_np = np.array(probs)

    mean_p = float(probs_np.mean())
    max_p = float(probs_np.max())
    flagged = float((probs_np >= FAKE_THRESHOLD).mean())

    # Tạo báo cáo danh sách các frame bị nghi vấn cao nhất
    suspicious_frames = [
        f"Frame {idx + 1}: **{p:.1%}**"
        for idx, p in enumerate(probs)
        if p >= FAKE_THRESHOLD
    ]

    detail_str = ""
    if suspicious_frames:
        detail_str = "\n\n**Các khung hình bị nghi vấn cao:**\n- " + "\n- ".join(
            suspicious_frames[:5]
        )
        if len(suspicious_frames) > 5:
            detail_str += f"\n- ...và {len(suspicious_frames) - 5} frame khác."

    text = (
        f"### {verdict(mean_p, 'Video')}\n"
        f"- Đã lấy mẫu & phân tích **{len(frames)}** khung hình (Patch-based Scanning)\n"
        f"- Xác suất AI trung bình: **{mean_p:.1%}** (Đỉnh điểm: **{max_p:.1%}**)\n"
        f"- Tỷ lệ khung hình bị đánh dấu là AI: **{flagged:.0%}**"
        f"{detail_str}\n\n"
        "_Lưu ý: Hệ thống đang quét từng khung hình bằng mô hình ảnh tĩnh để phát hiện dấu vết sinh bởi AI._"
    )

    return {"Do AI tạo": mean_p, "Thật": 1 - mean_p}, text

# ---------- Âm thanh ----------
def analyze_audio(path):
    return (
        "### ⏳ Chưa sẵn sàng\n"
        "Tab âm thanh chưa có mô hình. Cần thêm một mô hình phát hiện giọng nói giả "
        "(huấn luyện trên dữ liệu như ASVspoof, hoặc dùng mô hình có sẵn) rồi viết lại "
        "hàm `analyze_audio` trong `app.py`."
    )


# ---------- Giao diện ----------
with gr.Blocks(title="Kiểm tra nội dung do AI tạo") as demo:
    gr.Markdown("# Kiểm tra nội dung do AI tạo")
    gr.Markdown(DISCLAIMER)
    if MODEL is None:
        gr.Markdown(NO_MODEL_MSG)

    with gr.Tab("Ảnh"):
        img_in = gr.Image(type="pil", label="Tải ảnh lên")
        img_btn = gr.Button("Kiểm tra", variant="primary")
        img_label = gr.Label(label="Xác suất")
        img_text = gr.Markdown()
        img_btn.click(analyze_image, img_in, [img_label, img_text])

    with gr.Tab("Video"):
        vid_in = gr.Video(label="Tải video lên")
        vid_btn = gr.Button("Kiểm tra", variant="primary")
        vid_label = gr.Label(label="Xác suất")
        vid_text = gr.Markdown()
        vid_btn.click(analyze_video, vid_in, [vid_label, vid_text])

    with gr.Tab("Âm thanh"):
        aud_in = gr.Audio(type="filepath", label="Tải file âm thanh lên")
        aud_btn = gr.Button("Kiểm tra", variant="primary")
        aud_text = gr.Markdown()
        aud_btn.click(analyze_audio, aud_in, aud_text)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
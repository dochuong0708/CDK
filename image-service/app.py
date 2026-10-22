import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from pathlib import Path

import gradio as gr
import torch

from ai_image_detector import (
    FAKE_THRESHOLD,
    REAL_THRESHOLD,
    build_model,
    get_device,
    predict_image_prob,
)

MODEL_PATH = Path("detector.pt")
device = get_device()


def load_model():
    if not MODEL_PATH.exists():
        return None, 0
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["fake_idx"]


MODEL, FAKE_IDX = load_model()
NO_MODEL_MSG = "⚠️ Chưa có file checkpoint `detector.pt`. Hãy chạy lệnh train trước!"


def get_verdict_text(p):
    if p >= FAKE_THRESHOLD:
        return "🔴 Ảnh nhiều khả năng được tạo bởi AI"
    if p <= REAL_THRESHOLD:
        return "🟢 Ảnh nhiều khả năng là ảnh thật (REAL)"
    return "🟡 Không chắc chắn - Cần kiểm tra kỹ hơn"


def analyze_image(img):
    if MODEL is None:
        return None, NO_MODEL_MSG
    if img is None:
        return None, "Vui lòng tải lên một bức ảnh."

    p = predict_image_prob(MODEL, img, device, FAKE_IDX)
    probs = {"Do AI tạo (Fake)": p, "Ảnh thật (Real)": 1.0 - p}
    summary = f"### {get_verdict_text(p)}\n- Xác suất do AI tạo: **{p:.1%}**"

    return probs, summary


with gr.Blocks(title="AI Image Detector") as demo:
    gr.Markdown("# 🔍 Công cụ phát hiện ảnh do AI tạo (AI Image Detector)")

    if MODEL is None:
        gr.Markdown(NO_MODEL_MSG)

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Tải ảnh cần kiểm tra lên")
            btn = gr.Button("Phân tích ảnh", variant="primary")

        with gr.Column():
            lbl_out = gr.Label(label="Xác suất dự đoán")
            txt_out = gr.Markdown()

    btn.click(fn=analyze_image, inputs=img_in, outputs=[lbl_out, txt_out])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
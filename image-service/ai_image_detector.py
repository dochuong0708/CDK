"""
Phát hiện ảnh do AI tạo ra (real vs fake) bằng transfer learning với ResNet-18.

CẤU TRÚC DỮ LIỆU (tên thư mục lớp là "real" và "fake", không phân biệt hoa/thường):

    data/
      train/
        real/  *.jpg
        fake/  *.jpg
      val/
        real/
        fake/

Bộ dữ liệu công khai gợi ý: CIFAKE (Kaggle), GenImage.
(CIFAKE dùng thư mục "test" thay cho "val" -> thêm --val_split test)

CÁCH DÙNG:
    pip install -r requirements.txt

    # Huấn luyện
    python ai_image_detector.py train --data_dir data --epochs 5

    # Kiểm tra một ảnh hoặc cả thư mục
    python ai_image_detector.py predict --model detector.pt --input anh.jpg
    python ai_image_detector.py predict --model detector.pt --input thu_muc_anh/
"""

import argparse
import io
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Ngưỡng kết luận. Vùng giữa hai ngưỡng = "không chắc chắn".
FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35

def sample_video_frames(path, n=24):
    """Lấy n khung hình phân bổ đều từ video."""
    cap = cv2.VideoCapture(str(path))
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


@torch.no_grad()
def predict(args):
    device = get_device()
    ckpt = torch.load(args.model, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    fake_idx = ckpt["fake_idx"]

    target = Path(args.input)

    # 1. Xử lý nếu đầu vào là File Video
    if target.is_file() and target.suffix.lower() in VIDEO_EXTS:
        print(f"🎬 Đang phân tích video: {target.name}...")
        frames = sample_video_frames(target, n=24)
        if not frames:
            raise SystemExit("Không thể đọc khung hình từ video này.")

        probs = [
            predict_image_prob(
                model,
                frame,
                device,
                fake_idx,
                patch_size=args.patch_size,
                stride=args.stride,
                top_k_percent=args.top_k,
                batch_size=args.batch_size,
            )
            for frame in frames
        ]

        mean_p = float(np.mean(probs))
        max_p = float(np.max(probs))
        flagged_ratio = float(np.mean([p >= FAKE_THRESHOLD for p in probs]))

        verdict = make_decision(mean_p)
        print(f"\n--- KẾT QUẢ PHÂN TÍCH VIDEO ---")
        print(f"Số frame phân tích: {len(frames)}")
        print(f"Xác suất AI trung bình: {mean_p:.1%}")
        print(f"Frame có xác suất AI cao nhất: {max_p:.1%}")
        print(f"Tỷ lệ frame nghi vấn AI: {flagged_ratio:.0%}")
        print(f"Kết luận chung: {verdict}")
        return

    # 2. Xử lý nếu đầu vào là Ảnh hoặc Thư mục ảnh (Giữ nguyên)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        paths = [target]

    if not paths:
        raise SystemExit("Không tìm thấy ảnh hoặc video hợp lệ.")

    for p in paths:
        img = Image.open(p).convert("RGB")
        prob_fake = predict_image_prob(
            model,
            img,
            device,
            fake_idx,
            patch_size=args.patch_size,
            stride=args.stride,
            top_k_percent=args.top_k,
            batch_size=args.batch_size,
        )

        verdict = make_decision(prob_fake)
        label = {
            "AI": "Nhiều khả năng do AI tạo",
            "REAL": "Nhiều khả năng là ảnh thật",
            "UNCERTAIN": "Không chắc chắn - cần người xem xét",
        }[verdict]
        print(f"{p}: xác suất AI = {prob_fake:.1%} -> {label} [{verdict}]")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RandomJPEG:
    """Nén JPEG ngẫu nhiên để mô hình quen với ảnh đã qua mạng xã hội."""

    def __init__(self, p=0.3, quality=(60, 95)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def patch_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def make_decision(prob_fake):
    if prob_fake >= FAKE_THRESHOLD:
        return "AI"
    if prob_fake <= REAL_THRESHOLD:
        return "REAL"
    return "UNCERTAIN"


def validate_data_layout(data_dir, val_split):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / val_split

    if not train_dir.exists():
        raise SystemExit(f"Không tìm thấy thư mục train: {train_dir}")
    if not val_dir.exists():
        raise SystemExit(f"Không tìm thấy thư mục validation: {val_dir}")

    train_classes = sorted(p.name.lower() for p in train_dir.iterdir() if p.is_dir())
    val_classes = sorted(p.name.lower() for p in val_dir.iterdir() if p.is_dir())
    if "real" not in train_classes or "fake" not in train_classes:
        raise SystemExit(f"Cần hai thư mục lớp tên 'real' và 'fake' trong {train_dir}, hiện có: {train_classes}")
    if "real" not in val_classes or "fake" not in val_classes:
        raise SystemExit(f"Cần hai thư mục lớp tên 'real' và 'fake' trong {val_dir}, hiện có: {val_classes}")


def train_transform():
    return transforms.Compose([
        RandomJPEG(p=0.4),
        transforms.RandomCrop(224, pad_if_needed=True), # Không Resize để học nhiễu gốc
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train(args):
    device = get_device()
    print(f"Thiết bị: {device}")

    data_dir = Path(args.data_dir)
    validate_data_layout(data_dir, args.val_split)

    train_ds = datasets.ImageFolder(data_dir / "train", train_transform())
    val_ds = datasets.ImageFolder(data_dir / args.val_split, eval_transform())

    lower = [c.lower() for c in train_ds.classes]
    if "fake" not in lower or "real" not in lower:
        raise SystemExit(f"Cần hai thư mục lớp tên 'real' và 'fake', hiện có: {train_ds.classes}")
    fake_idx = lower.index("fake")
    print(f"Các lớp: {train_ds.classes} (fake_idx={fake_idx})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available())

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | loss {total_loss / len(train_ds):.4f} "
              f"| val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "fake_idx": fake_idx,
            }, args.out)
            print(f"  -> Đã lưu mô hình tốt nhất vào {args.out}")

    print(f"Xong. Val acc tốt nhất: {best_acc:.4f}")


@torch.no_grad()
def predict_image_prob(model, image, device, fake_idx, patch_size=224, stride=160,
                      top_k_percent=0.08, batch_size=32):
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    img_tensor = patch_transform()(img_rgb).unsqueeze(0)

    # Pad viền nếu ảnh nhỏ hơn 224x224 (Đồng bộ với app.py)
    pad_w = max(0, patch_size - w)
    pad_h = max(0, patch_size - h)
    if pad_w > 0 or pad_h > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    patches = img_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)

    scores = []
    for i in range(0, patches.size(0), batch_size):
        patch_batch = patches[i:i + batch_size].to(device)
        out = model(patch_batch)
        scores.extend(torch.softmax(out, dim=1)[:, fake_idx].cpu().tolist())

    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    if scores_tensor.numel() == 1:
        return float(scores_tensor[0].item())

    k = max(1, int(len(scores_tensor) * top_k_percent))
    top_k_scores = torch.topk(scores_tensor, k).values
    return float(top_k_scores.mean().item())

@torch.no_grad()
def predict(args):
    device = get_device()
    ckpt = torch.load(args.model, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    fake_idx = ckpt["fake_idx"]

    target = Path(args.input)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        paths = [target]
    if not paths:
        raise SystemExit("Không tìm thấy ảnh nào.")

    for p in paths:
        img = Image.open(p).convert("RGB")
        prob_fake = predict_image_prob(
            model,
            img,
            device,
            fake_idx,
            patch_size=args.patch_size,
            stride=args.stride,
            top_k_percent=args.top_k,
            batch_size=args.batch_size,
        )

        verdict = make_decision(prob_fake)
        label = {
            "AI": "Nhiều khả năng do AI tạo",
            "REAL": "Nhiều khả năng là ảnh thật",
            "UNCERTAIN": "Không chắc chắn - cần người xem xét",
        }[verdict]
        print(f"{p}: xác suất AI = {prob_fake:.1%} -> {label} [{verdict}]")


def main():
    parser = argparse.ArgumentParser(description="Phát hiện ảnh do AI tạo")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Huấn luyện mô hình")
    t.add_argument("--data_dir", required=True)
    t.add_argument("--val_split", default="val", help="Tên thư mục validation (mặc định: val)")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--out", default="detector.pt")
    t.set_defaults(func=train)

    p = sub.add_parser("predict", help="Dự đoán ảnh")
    p.add_argument("--model", default="detector.pt")
    p.add_argument("--input", required=True, help="Đường dẫn ảnh hoặc thư mục")
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--stride", type=int, default=112)
    p.add_argument("--top_k", type=float, default=0.20, help="Phần trăm patch có điểm AI cao nhất được dùng để trung bình")
    p.add_argument("--batch_size", type=int, default=32)
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

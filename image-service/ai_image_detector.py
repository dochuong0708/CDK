import argparse
import io
import random
from pathlib import Path
from color_analysis import get_color_forensic_adjustment

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

FAKE_THRESHOLD = 0.60
REAL_THRESHOLD = 0.40


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RandomJPEG:
    def __init__(self, p=0.5, quality=(50, 95)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def train_transform():
    return transforms.Compose([
        RandomJPEG(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def patch_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_model(pretrained=True):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    return model


def make_decision(prob_fake):
    if prob_fake >= FAKE_THRESHOLD:
        return "AI"
    if prob_fake <= REAL_THRESHOLD:
        return "REAL"
    return "UNCERTAIN"


@torch.no_grad()
def predict_image_prob(model, image, device, fake_idx, patch_size=224, stride=112, batch_size=32):
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    # 1. Điểm toàn cảnh
    global_tensor = eval_transform()(img_rgb).unsqueeze(0).to(device)
    global_out = model(global_tensor)
    global_prob = float(torch.softmax(global_out, dim=1)[0, fake_idx].cpu().item())

    if w < patch_size or h < patch_size:
        dl_score = global_prob
    else:
        # 2. Điểm chi tiết (Patch Scanning)
        img_tensor = patch_transform()(img_rgb).unsqueeze(0)
        patches = img_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)

        scores = []
        for i in range(0, patches.size(0), batch_size):
            patch_batch = patches[i:i + batch_size].to(device)
            out = model(patch_batch)
            scores.extend(torch.softmax(out, dim=1)[:, fake_idx].cpu().tolist())

        mean_patch_score = float(np.mean(scores))
        dl_score = 0.6 * global_prob + 0.4 * mean_patch_score

    # 3. Kết hợp Phân tích màu sắc (Color Forensics)
    forensic_info = get_color_forensic_adjustment(img_rgb)
    final_score = dl_score + forensic_info["adjustment"]

    # Giới hạn xác suất trong dải [0.0, 1.0]
    return float(np.clip(final_score, 0.0, 1.0))


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
    print(f"Thiết bị huấn luyện: {device}")

    data_dir = Path(args.data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", train_transform())
    val_ds = datasets.ImageFolder(data_dir / args.val_split, eval_transform())

    lower = [c.lower() for c in train_ds.classes]
    fake_idx = lower.index("fake")
    print(f"Các lớp: {train_ds.classes} (Index FAKE: {fake_idx})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available())

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

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
        print(f"Epoch {epoch}/{args.epochs} | Loss: {total_loss / len(train_ds):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "fake_idx": fake_idx,
            }, args.out)
            print(f"  -> Đã lưu mô hình tốt hơn vào {args.out}")

    print(f"\nHoàn tất train! Val Accuracy cao nhất: {best_acc:.4f}")


@torch.no_grad()
def predict(args):
    device = get_device()
    ckpt = torch.load(args.model, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    fake_idx = ckpt["fake_idx"]

    target = Path(args.input)
    paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_EXTS) if target.is_dir() else [target]

    if not paths:
        raise SystemExit("Không tìm thấy tệp ảnh hợp lệ.")

    print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
    for p in paths:
        img = Image.open(p).convert("RGB")
        prob_fake = predict_image_prob(model, img, device, fake_idx)
        verdict = make_decision(prob_fake)

        label_map = {
            "AI": "Nhiều khả năng do AI tạo",
            "REAL": "Nhiều khả năng là ảnh thật",
            "UNCERTAIN": "Chưa rõ ràng (Cần xem xét thêm)",
        }
        print(f"📸 {p.name}: Xác suất AI = {prob_fake:.1%} -> {label_map[verdict]} [{verdict}]")


def main():
    parser = argparse.ArgumentParser(description="Phát hiện ảnh AI")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Huấn luyện mô hình")
    t.add_argument("--data_dir", required=True, help="Thư mục chứa train/ và val/")
    t.add_argument("--val_split", default="val")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--out", default="detector.pt")
    t.set_defaults(func=train)

    p = sub.add_parser("predict", help="Dự đoán ảnh hoặc thư mục ảnh")
    p.add_argument("--model", default="detector.pt")
    p.add_argument("--input", required=True, help="Đường dẫn file ảnh hoặc thư mục")
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
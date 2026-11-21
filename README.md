# DeepGuard Agent & Web Dashboard

DeepGuard là hệ thống chạy local để phân tích nội dung văn bản, phát hiện tin giả (Fake News), kiểm chứng sự thật (Fact-Checking) và sàng lọc tệp đa phương tiện (ảnh, âm thanh). Hệ thống cung cấp **Giao diện Web trực quan (Web Dashboard)** và **Giao diện dòng lệnh (Agent CLI)**.

---

## 🚀 Khởi động Giao diện Web (Web Dashboard)

Giao diện Web cho phép dán một bài báo bất kỳ (không giới hạn số chữ), tự động chia nhỏ từng đoạn, quét mô hình AI & luật rủi ro, phát hiện câu từ giật gân/sai sự thật, đối chiếu với báo chí chính thống và tổng kết điểm tin cậy.

Bạn có thể chạy hệ thống theo **1 trong 2 cách** sau:

### Cách 1: Chạy 1 Terminal duy nhất (Khuyên dùng - Nhanh nhất)
Giao diện Web đã được đóng gói sẵn vào bên trong backend FastAPI. Bạn chỉ cần mở **1 cửa sổ Terminal**:

```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

👉 **Mở trình duyệt truy cập:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### Cách 2: Chạy 2 Terminal song song (Dành cho nhà phát triển React / Vite)
Dành cho trường hợp bạn muốn chỉnh sửa code giao diện và xem thay đổi ngay tức thì (Hot Reload). Mở **2 cửa sổ Terminal**:

* **Terminal 1 (Backend FastAPI):**
  ```powershell
  cd backend
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  ```

* **Terminal 2 (Frontend React/Vite):**
  ```powershell
  cd frontend
  npm run dev
  ```

👉 **Mở trình duyệt truy cập:** [http://localhost:5173](http://localhost:5173)

---

## 🛑 Cách dừng (tắt) Server

Khi không sử dụng nữa, bạn có thể dừng server bằng một trong các cách sau:

1. **Cách thông thường:**
   * Nhấp chuột vào cửa sổ Terminal đang chạy server và nhấn tổ hợp phím **`Ctrl + C`** (nhấn 1 đến 2 lần). Terminal sẽ dừng server và quay về dấu nhắc lệnh thông thường.
2. **Nếu server bị kẹt cổng hoặc chạy ngầm trong máy:**
   * Mở PowerShell và gõ lệnh sau để tắt cưỡng bức tiến trình đang chiếm cổng:
     ```powershell
     # Tắt tiến trình cổng 8000 (Backend)
     Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force

     # Tắt tiến trình cổng 5173 (Frontend)
     Stop-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess -Force
     ```

---

## 📖 Hướng dẫn sử dụng 3 Chế độ trên Giao diện Web

Trên thanh điều hướng đầu trang, bạn có thể chuyển đổi linh hoạt giữa **3 chế độ**:

### 1. 📝 Chế độ Phân tích Văn bản (Hoàn thiện đầy đủ)
* **Dán bài viết bất kỳ:** Nhập bài báo hoặc tin tức (không giới hạn số chữ).
* **Bài mẫu thử nhanh:** Có sẵn 3 mẫu (`Lừa đảo / Phishing`, `Y tế / Giật gân`, `Chính thống / Tin chuẩn`).
* **Tính năng:**
  - Tách và chấm điểm rủi ro cho từng đoạn văn (Safe / Suspicious / Dangerous).
  - Tự động bôi đỏ/vàng câu từ nghi vấn (giật gân, đe dọa, thuyết âm mưu, nguồn nặc danh).
  - Đối chiếu fact-check với báo chí chính thống (VnExpress, Tuổi Trẻ, Thanh Niên...).
  - Tổng kết điểm rủi ro toàn bài và đưa ra khuyến nghị.

### 2. 🖼️ Chế độ Phân tích Hình ảnh
* **Kéo thả hoặc tải ảnh lên:** Hỗ trợ JPG, PNG, WEBP.
* **Phân tích:** Backend kết hợp ConvNeXt-Tiny local (`backend/models/detector.pt`) với ViT Hugging Face, EXIF và FFT. Điểm cuối cùng là trung bình 50/50 giữa ConvNeXt-Tiny và điểm đa tín hiệu; điểm đa tín hiệu dùng trọng số 70% ViT, 15% EXIF và 15% FFT.
* Checkpoint local phải là ConvNeXt-Tiny đã huấn luyện cho hai lớp `real`/`fake`; checkpoint ResNet18 hiện có không tương thích và cần được thay bằng checkpoint ConvNeXt trước khi endpoint phân tích ảnh chạy được.
* Ảnh tải lên được phân tích trực tiếp từ bytes, không ghi thành tệp tạm; công cụ agent đọc bytes từ tệp local. Luồng âm thanh vẫn giữ cách lưu tệp hiện tại.
* **Cài đặt chạy local:** Chạy `python -m pip install -r requirements.txt -r requirements-image.txt` trong thư mục `backend`.
* Hugging Face tải model khoảng 300–400 MB khi phân tích ảnh lần đầu; lần tiếp theo dùng cache. Nếu model không tải/chạy được hoặc checkpoint local thiếu, API báo lỗi thay vì âm thầm bỏ qua một detector.

### 3. 🎙️ Chế độ Phân tích Âm thanh (Biểu trưng / Prototype)
* **Kéo thả hoặc tải tệp âm thanh:** Hỗ trợ MP3, WAV, M4A, FLAC.
* **Tệp mẫu mô phỏng:**
  - `Cuộc gọi Deepfake Voice Clone mượn tiền khẩn cấp`: Phát hiện tần số âm sắc nhân tạo & nhịp ngắt quãng máy móc.
  - `Ghi âm tự động Robocall đe dọa`: Phát hiện dải tần bị cắt gọt.
  - `Bản tin phát thanh thời sự thật`: Nhận diện độ rung thanh quản tự nhiên.
* **Hiển thị:** Biểu đồ sóng âm thanh mô phỏng (waveform visualizer), tỷ lệ nghi vấn giọng nói AI và nhãn detector mẫu `audio-heuristic-v1`.

---

## 🤖 Chạy Agent qua Terminal (CLI)

Nếu muốn hội thoại dạng chat trực tiếp với LLM qua Ollama:

```powershell
cd backend
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
$env:DEEPGUARD_LANGUAGE_MODEL_PATH = (Resolve-Path '.\data\models\vietnam_fake_news_risk_v3.joblib').Path
python -m app.agent.cli
```

- Gõ câu hỏi hoặc yêu cầu phân tích, ví dụ: `Hãy kiểm chứng: "Sự kiện X xảy ra tại Hà Nội năm 2024."`
- Gõ `/exit` để thoát.

---

## 🏛️ Kiến trúc Dự án

- `frontend/`: Ứng dụng Web Dashboard xây dựng bằng **React + TypeScript + Vite**, giao diện Dark Theme hiện đại.
- `backend/app/main.py`: FastAPI server cung cấp REST API và phục vụ giao diện tĩnh.
- `backend/app/services/article_analyzer.py`: Phân đoạn bài báo, điều phối mô hình NLP, luật ngôn ngữ và fact-checking.
- `backend/app/detectors/language/`:
  - `model.py`: Nạp mô hình học máy (`vietnam_fake_news_risk_v3.joblib`).
  - `heuristics.py`: Bộ quy tắc tách tín hiệu hình thức (giật gân, câu view, thuyết âm mưu) và lừa đảo.
  - `domains.py`: Phát hiện tên miền giả mạo ngân hàng/ví bằng Levenshtein + chuẩn hóa ký tự dễ nhầm.
  - `style_classifier.py`: Phân loại giật gân/lừa đảo theo ngữ cảnh bằng embedding tiếng Việt (tùy chọn, `requirements-ml.txt`).
- `backend/app/services/risk_fusion.py`: Tách và kết hợp 3 trục rủi ro: Hình thức (style), Lừa đảo (fraud), Sự thật (factuality).
- `backend/app/services/fact_checking.py`: RAG fact-check: tìm kiếm, xếp hạng snippet bằng cosine similarity, LLM đánh giá; báo cáo lưu vào SQLite (`/api/fact-checks`).
- `backend/data/models/`: Chứa các mô hình học máy đã huấn luyện.
- `backend/app/detectors/image/`: Detector ảnh ConvNeXt-Tiny và phân tích màu sắc; checkpoint tương thích với kiến trúc ConvNeXt được đặt tại `backend/models/detector.pt`.
- `backend/tests/`: Bộ kiểm thử tự động pytest.

---

## 🧪 Chạy Kiểm thử (Test)

Tại thư mục `backend/`:
```powershell
python -m pytest tests/test_api.py
```

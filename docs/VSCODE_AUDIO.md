# Chạy thử Audio trong Visual Studio Code

Đã chuẩn bị môi trường `.venv`, Python Debugger, thư viện frontend và toàn bộ dữ liệu ASVspoof 2019 LA trên máy này. Archive ở `data/audio/downloads/LA.zip`, dữ liệu giải nén ở `data/audio/asvspoof2019/LA`. Checksum MD5 đã khớp nguồn chính thức; không cần tải lại.

**Đã chạy demo thành công ngày 27/09/2026:** thư mục `backend/outputs/asvspoof_demo/20260927-213244-8b510a` có model, manifest, metrics và hai ảnh XAI. Có thể mở ảnh ngay mà không cần train lại. Demo dùng 40 mẫu train + 40 dev + 40 test, học 1 epoch; Accuracy test là **45%**, nên chỉ dùng kiểm tra pipeline, chưa đạt chất lượng phân loại. Hai ví dụ XAI có thể bị dự đoán sai; tên thư mục real/fake là nhãn gốc.

Đã kiểm tra 17 tests, khởi động frontend HTTP 200 và gọi API với file ASVspoof thật: điểm khớp CLI, PNG hợp lệ, ảnh được khôi phục từ lịch sử. Báo cáo kiểm tra API nằm trong `api-check.json` của thư mục demo. Các server kiểm thử đã dừng để không chiếm cổng khi bạn chạy F5.

## Mở đúng dự án

1. Mở VS Code → **File → Open Folder** → chọn `D:\KI1NAM3\CDK`.
2. Nếu đây là bản mã nguồn bạn tin cậy, chọn **Trust** khi VS Code hỏi.
3. Mở Extensions (`Ctrl+Shift+X`), kiểm tra **Python** và **Python Debugger** của Microsoft đã được bật. Dự án đã thêm hai extension vào danh sách đề xuất.
4. Nhấn `Ctrl+Shift+P` → **Python: Select Interpreter** → chọn `D:\KI1NAM3\CDK\.venv\Scripts\python.exe`.

## Cách 1: Chạy bằng F5

Nhấn `Ctrl+Shift+D`, chọn **Audio: ASVspoof demo (train + XAI)**, rồi nhấn **F5**.

Script tự thực hiện:

```text
ASVspoof 2019 LA đã giải nén
→ chọn 20 real + 20 fake mỗi split (tối đa 120 file)
→ kiểm tra trùng dữ liệu và giải mã file
→ huấn luyện CNN 1 epoch trên train
→ chọn ngưỡng trên dev
→ đánh giá test
→ tạo ảnh phổ/XAI cho một file real và một file fake thuộc test
```

Xem log trong **Terminal**. Khi hoàn tất sẽ có dòng `DONE. Open this folder in VS Code: ...`. Mỗi lần F5 tạo một thư mục mới, không ghi đè model của lần trước.

Mở `backend/outputs/asvspoof_demo/latest.json` để biết chính xác thư mục của lần chạy mới nhất. Trong thư mục đó:

| File | Nội dung |
|---|---|
| `model.pt` | Trọng số demo và ngưỡng phân loại |
| `manifest.csv`, `manifest.summary.json` | Các file đã chọn, nhãn và thống kê |
| `audit.json` | Kết quả kiểm tra dữ liệu |
| `model.history.json` | Loss train và chỉ số dev |
| `metrics.json`, `metrics.scores.csv` | Chỉ số và điểm dự đoán trên test nhỏ |
| `real/xai.png`, `fake/xai.png` | Ảnh waveform, phổ Mel và Grad-CAM |
| `real/prediction.json`, `fake/prediction.json` | Điểm thật/giả theo file và cửa sổ |

Nhấp vào `xai.png` trong Explorer của VS Code để xem ảnh. `real` và `fake` ở tên thư mục là **nhãn dữ liệu gốc**, không phải cam kết mô hình dự đoán đúng. Model 1 epoch với tập rất nhỏ chỉ chứng minh pipeline chạy được, không phải mô hình đủ chính xác để sử dụng thực tế.

## Cách 2: Chạy bằng Terminal

Trong VS Code chọn **Terminal → New Terminal**, dùng PowerShell:

```powershell
Set-Location D:\KI1NAM3\CDK
.\.venv\Scripts\python.exe -X utf8 backend\run_audio_demo.py
```

Nếu muốn thử nhiều mẫu hoặc nhiều epoch hơn:

```powershell
.\.venv\Scripts\python.exe -X utf8 backend\run_audio_demo.py --limit-per-class 100 --epochs 3
```

Mặc định dữ liệu nằm ở `data/audio/asvspoof2019/LA`. Nếu di chuyển dữ liệu, truyền `--root "đường_dẫn_tới_LA"`.

## Chạy giao diện web

Sau khi có ít nhất một lần demo hoàn tất:

1. Chọn cấu hình **Audio: API with latest demo model** rồi nhấn F5. API chạy tại `http://127.0.0.1:8000`. Có thể dùng Terminal thay thế:

```powershell
.\.venv\Scripts\python.exe backend\serve_audio_demo.py
```

2. Mở Terminal thứ hai ở thư mục dự án, chạy:

```powershell
.\.venv\Scripts\python.exe backend\run_audio_frontend.py
```

3. Mở [DeepGuard local](http://127.0.0.1:5173), tìm bảng **Phân loại giọng nói thật / giả**, tải file `.flac` / `.wav` / `.ogg`, rồi chọn **Phân tích giọng nói**. Có thể lấy đường dẫn file mẫu tại `examples.real.audio` hoặc `examples.fake.audio` trong `latest.json`.
4. Xem điểm mô hình, phổ/XAI và tải PNG. Nhấn `Ctrl+C` trong các terminal để dừng server; khi debug thì bấm Stop.

API dùng checkpoint của lần demo thành công gần nhất. Nếu train lại khi API đang chạy, khởi động lại API để nạp checkpoint mới. Không chọn cấu hình API cũ nếu muốn dùng tự động checkpoint demo.

## Chạy dữ liệu đầy đủ

Xem [ASVSPOOF.md](ASVSPOOF.md) bước 4–7, dùng đường dẫn dataset local:

```powershell
$dataset = 'D:\KI1NAM3\CDK\data\audio\asvspoof2019\LA'
```

Giữ nguyên train/dev/test chính thức; không lấy test để chỉnh ngưỡng. Mô hình hiện là CNN baseline, chưa pretrained. Đánh giá tập đầy đủ trước khi đưa độ chính xác vào báo cáo.

## Khi gặp lỗi

- **Không có Python interpreter:** chọn lại `.venv\Scripts\python.exe`, không chọn Python khác.
- **Không có cấu hình Audio:** mở thư mục gốc `CDK`, không chỉ mở riêng `backend`.
- **Thiếu dữ liệu:** kiểm tra đã tải và giải nén `data/audio/asvspoof2019/LA`; giữ file ZIP đến khi kiểm tra checksum thành công.
- **Port 8000 hoặc 5173 đã dùng:** dừng phiên server trước đó bằng Ctrl+C, rồi chạy lại.
- **Không có latest.json:** cần chạy `run_audio_demo.py` hoàn tất trước khi mở API.
- **Model dự đoán sai mẫu real/fake:** mô hình demo rất nhỏ; đây không phải lỗi chạy chương trình. Xem điểm và chuyển sang huấn luyện đầy đủ.

Nguồn dữ liệu: [ASVspoof 2019, Edinburgh DataShare](https://doi.org/10.7488/ds/2555). README và giấy phép gốc được giữ trong `data/audio/downloads`.

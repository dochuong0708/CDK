# Chạy DeepGuard Audio với ASVspoof trên Windows

**Dữ liệu đã sẵn sàng trên máy này:** toàn bộ LA.zip đã tải từ Edinburgh DataShare, kiểm tra MD5 và giải nén tại `D:\KI1NAM3\CDK\data\audio\asvspoof2019\LA`. Không cần tải lại. Để chạy nhanh trong VS Code, xem [VSCODE_AUDIO.md](VSCODE_AUDIO.md).

## 1. Chọn và tải dữ liệu

Pipeline này hỗ trợ **ASVspoof 2019 Logical Access (LA)**: nhãn `bonafide` → `real=0`, `spoof` → `fake=1`. Giữ nguyên train/dev/eval của ban tổ chức; `eval` được đặt tên `test` trong manifest.

Tải **LA.zip** từ [Edinburgh DataShare, nguồn chính thức ASVspoof 2019](https://doi.org/10.7488/ds/2555). Trang hiện niêm yết LA.zip khoảng **7,12 GB**; cần thêm dung lượng để giải nén. Đọc LICENSE/README đi kèm. Không cần tải PA.zip cho bài toán giọng tổng hợp/voice conversion này.

Giải nén, ví dụ vào `D:\datasets\ASVspoof2019`. Thư mục mong đợi:

```text
D:\datasets\ASVspoof2019\LA\
  ASVspoof2019_LA_cm_protocols\
    ASVspoof2019.LA.cm.train.trn.txt
    ASVspoof2019.LA.cm.dev.trl.txt
    ASVspoof2019.LA.cm.eval.trl.txt
  ASVspoof2019_LA_train\flac\*.flac
  ASVspoof2019_LA_dev\flac\*.flac
  ASVspoof2019_LA_eval\flac\*.flac
```

Dùng **CM protocol**, không dùng ASV protocol. Giữ nguyên FLAC, không cần chuyển hàng loạt sang WAV. `--root` chấp nhận thư mục `LA` hoặc thư mục cha chứa `LA`; không tự tìm sâu qua nhiều tầng thư mục.

ASVspoof 2021 không phát hành train/dev mới; ban tổ chức khuyến khích dùng train/dev 2019. Bộ 2021 DF dành cho đánh giá mở rộng, có protocol riêng và **chưa được parser này hỗ trợ**. Nguồn: [ASVspoof 2021](https://www.asvspoof.org/index2021.html).

## 2. Mở PowerShell và cài môi trường

Dùng Python 3.11 hoặc 3.12; môi trường dự án hiện đã có `.venv`. Khi cài trên máy khác mới cần chạy lệnh tạo môi trường:

```powershell
Set-Location D:\KI1NAM3\CDK
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-audio.txt
Set-Location backend
$py = (Resolve-Path ..\.venv\Scripts\python.exe).Path
$dataset = 'D:\datasets\ASVspoof2019\LA'
$env:MPLCONFIGDIR = "$PWD\outputs\mpl-cache"
& $py -X utf8 audio_cli.py --help
```

Không cần `Activate.ps1`; gọi thẳng Python của `.venv` để tránh lỗi ExecutionPolicy. **Các lệnh còn lại chạy trong `D:\KI1NAM3\CDK\backend` và cùng terminal để giữ biến `$py`, `$dataset`.** Khi mở terminal mới, chạy lại `Set-Location`, `$py`, `$dataset`.

## 3. Chạy thử nhỏ trước

Tạo tập nhỏ, tối đa **20 file mỗi lớp trong mỗi split**: tối đa 120 file, giữ nguyên split và lấy mẫu có seed cố định. Tập nhỏ chỉ để kiểm tra pipeline, không dùng số đo này làm kết quả ASVspoof chính thức.

```powershell
& $py -X utf8 audio_cli.py prepare-asvspoof2019 --root "$dataset" --limit-per-class 20 --seed 42 --output ../data/audio/asvspoof2019_quick.csv
& $py -X utf8 audio_cli.py validate-manifest --manifest ../data/audio/asvspoof2019_quick.csv --decode-audio --output outputs/asvspoof_quick/audit.json
& $py -X utf8 audio_cli.py train --manifest ../data/audio/asvspoof2019_quick.csv --epochs 1 --batch-size 8 --device cpu --output models/asvspoof_quick.pt
& $py -X utf8 audio_cli.py evaluate --manifest ../data/audio/asvspoof2019_quick.csv --model models/asvspoof_quick.pt --output outputs/asvspoof_quick/test_metrics.json
```

Sau mỗi lệnh, chỉ tiếp tục nếu không có `Error` và `$LASTEXITCODE` bằng 0. Chuẩn bị manifest kiểm tra toàn bộ đường dẫn trong protocol trước khi lấy subset: cần giải nén đủ các split đã chọn. Nếu mới có train/dev, thêm `--splits train dev` khi chuẩn bị; huấn luyện vẫn được, nhưng lệnh evaluate cần một manifest có test.

Hai lệnh prepare/train từ chối ghi đè manifest/checkpoint cũ. Khi chạy lại cùng thí nghiệm, dùng manifest đã có và đặt tên checkpoint mới, ví dụ `models/asvspoof_quick_v2.pt`. Không có chức năng resume optimizer; mỗi lệnh train bắt đầu từ đầu.

## 4. Huấn luyện đầy đủ

Sau khi quick run thành công, tạo manifest đầy đủ (không có `--limit-per-class`):

```powershell
& $py -X utf8 audio_cli.py prepare-asvspoof2019 --root "$dataset" --output ../data/audio/asvspoof2019_full.csv
& $py -X utf8 audio_cli.py validate-manifest --manifest ../data/audio/asvspoof2019_full.csv --decode-audio --output outputs/asvspoof_full/audit.json
& $py -X utf8 audio_cli.py train --manifest ../data/audio/asvspoof2019_full.csv --epochs 20 --batch-size 16 --device cpu --output models/asvspoof2019_cnn.pt
```

Mỗi split phải có cả real và fake. Validator kiểm tra đường dẫn, nội dung file trùng bằng SHA-256, speaker xuất hiện ở nhiều split; `--decode-audio` giải mã toàn bộ bằng đúng preprocessing của mô hình và ghi lỗi vào JSON. Không tự bỏ file lỗi để tránh thay đổi benchmark mà không báo cáo. Kiểm tra hash/giải mã cả bộ dữ liệu có thể mất thời gian; train/evaluate cũng kiểm tra hash trước khi chạy. Log tiến độ hiện theo batch khi train và mỗi 500 file khi chấm dev/test.

File `.summary.json` cạnh CSV chứa số mẫu, người nói, lớp và attack theo split. Hãy xem trước khi train. Chưa có kiểm tra tự động số lượng mẫu với bản phát hành chính thức; một protocol bị cắt ngắn vẫn có thể hợp lệ về cấu trúc. Pipeline giới hạn audio 0,5–60 giây, 8–192 kHz, tối đa 8 kênh và từ chối im lặng. Nếu audit phát hiện mẫu ngoài giới hạn, giữ báo cáo lỗi để xử lý có chủ đích, không bỏ mẫu rồi gọi kết quả là benchmark đầy đủ.

CPU chạy được nhưng full train có thể lâu; không có ước lượng thời gian khi chưa đo trên máy bạn. Nếu có GPU NVIDIA, cài đúng bản PyTorch CUDA theo [hướng dẫn PyTorch chính thức](https://pytorch.org/get-started/locally/) thay cho bản CPU, kiểm tra rồi đổi `--device cpu` thành `--device cuda`:

```powershell
& $py -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Dev chạy trên cùng thiết bị với train; checkpoint được lưu dạng tensor CPU để dự đoán trên CPU. Hết bộ nhớ GPU thì giảm `--batch-size` xuống 8 hoặc 4. GPU chưa được kiểm thử trong môi trường bàn giao này.

## 5. Đánh giá test sau khi chọn mô hình

```powershell
& $py -X utf8 audio_cli.py evaluate --manifest ../data/audio/asvspoof2019_full.csv --model models/asvspoof2019_cnn.pt --output outputs/asvspoof_full/test_metrics.json
Get-Content outputs/asvspoof_full/test_metrics.json
```

Checkpoint tốt nhất và ngưỡng chỉ được chọn từ **dev**. Không dùng test để chỉnh ngưỡng hoặc lựa chọn epoch. Train chỉ học trên train; việc đọc hash của test để kiểm tra trùng dữ liệu không tham gia tối ưu mô hình. Giữ manifest, summary, audit và thông số chạy cùng báo cáo.

| Kết quả | Vị trí tính từ backend |
|---|---|
| Trọng số và ngưỡng từ dev | `models/asvspoof2019_cnn.pt` |
| Loss train và metrics dev mỗi epoch | `models/asvspoof2019_cnn.history.json` |
| Accuracy, F1(fake), ROC-AUC, EER xấp xỉ, confusion matrix | `outputs/asvspoof_full/test_metrics.json` |
| Nhãn và điểm fake mỗi file test | `outputs/asvspoof_full/test_metrics.scores.csv` |

Accuracy/F1/AUC/EER ở thang 0–1; nhân 100 khi báo phần trăm. Confusion matrix có hàng là nhãn thật, cột là dự đoán, thứ tự `[real, fake]`. `eer_approx` là EER xấp xỉ theo điểm ROC gần giao nhau, không thay thế bộ chấm EER/t-DCF chính thức của ASVspoof. Score là điểm softmax chưa hiệu chỉnh.

## 6. Dự đoán một file và xuất XAI

Có thể chọn tự động một file từ test để thử lệnh, hoặc thay `$sample` bằng đường dẫn bản ghi của bạn:

```powershell
$sample = (Import-Csv ../data/audio/asvspoof2019_full.csv | Where-Object { $_.split -eq 'test' } | Select-Object -First 1).path
& $py -X utf8 audio_cli.py predict --audio "$sample" --model models/asvspoof2019_cnn.pt --output outputs/asvspoof_full/example
Start-Process "$PWD\outputs\asvspoof_full\example\xai.png"
```

Thư mục kết quả có `prediction.json` và `xai.png`. Ảnh gồm waveform, Mel-spectrogram và Grad-CAM cho lớp fake tại cửa sổ có điểm fake cao nhất. Đây là giải thích mô hình, không phải chứng cứ chắc chắn hay ground truth vị trí bị chỉnh sửa.

Nếu dùng mô hình quick, thay manifest/model/output bằng các đường dẫn quick ở bước 3. Dự đoán được không có nghĩa mô hình quick đã đủ tốt.

## 7. Mở giao diện DeepGuard

Terminal backend:

```powershell
$env:DEEPGUARD_AUDIO_MODEL = (Resolve-Path models/asvspoof2019_cnn.pt).Path
& $py -m uvicorn app.main:app --reload
```

Terminal frontend mới:

```powershell
Set-Location D:\KI1NAM3\CDK\frontend
npm install
npm run dev
```

Mở URL Vite in trong terminal. Dùng bảng “Phân loại giọng nói thật / giả”, tải file và xem/tải đồ thị XAI. Khi đổi checkpoint, khởi động lại backend.

## Lỗi thường gặp

| Lỗi | Xử lý |
|---|---|
| Không tìm thấy `ASVspoof2019_LA_cm_protocols` | Trỏ `$dataset` vào đúng thư mục LA đã giải nén; tránh tầng `LA/LA` ngoài dự kiến |
| Cần protocol 5 cột hoặc sai ID | Dùng CM protocol 2019 LA; không đưa PA, ASV hoặc keys 2021 vào parser |
| Thiếu `.flac` | Giải nén đủ split tương ứng; không đổi tên file |
| Nội dung file trùng / người nói trùng split | Kiểm tra nguồn protocol; không chia ngẫu nhiên lại dữ liệu để bỏ qua lỗi |
| Checkpoint/manifest đã tồn tại | Dùng đầu ra mới; dùng lại manifest có sẵn nếu dữ liệu không đổi |
| `ModuleNotFoundError` | Cài requirements bằng đúng `$py`, tránh dùng Python hệ thống khác |
| `CUDA không khả dụng` | Dùng CPU hoặc cài đúng PyTorch CUDA phù hợp phần cứng |
| Web trả 503 | Đặt `DEEPGUARD_AUDIO_MODEL` tới checkpoint đã train và khởi động lại backend |

Đã tải LA.zip (7.640.952.520 byte), MD5 khớp nguồn chính thức `30c98f11d8b2bc21f2c257bfd78bb5c5`, và giải nén đầy đủ. Chưa huấn luyện benchmark đầy đủ trên ASVspoof. Kiểm thử tự động dùng FLAC tổng hợp có cấu trúc protocol tương tự; kết quả kiểm thử không phải độ chính xác deepfake. Lệnh demo trong VS Code dùng một tập con của dữ liệu ASVspoof thật.

Kiểm chứng ngày 27/09/2026: `python -m pytest -q` có **17 tests passed**, gồm luồng CLI prepare → validate → train → evaluate → predict/PNG. Một cảnh báo deprecation từ Starlette/AnyIO không làm test thất bại. Chạy lại trong backend bằng `& $py -m pytest -q`.

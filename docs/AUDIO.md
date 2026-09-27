# DeepGuard Audio: phân loại thật/giả và phổ âm thanh XAI

**Bắt đầu với ASVspoof:** xem [hướng dẫn chạy từng bước trên Windows](ASVSPOOF.md). Lệnh `prepare-asvspoof2019 --root` tự tìm protocol/FLAC, hỗ trợ subset chạy thử và `validate-manifest --decode-audio` để kiểm tra dữ liệu trước huấn luyện.

## Trạng thái bàn giao

Đã triển khai mã nguồn CNN, huấn luyện, đánh giá, suy luận, Grad-CAM, API và giao diện React.
**Đã có checkpoint demo học trên tập con ASVspoof 2019 LA thật, lưu trong thư mục outputs bị Git bỏ qua. Chưa huấn luyện benchmark đầy đủ, không có cam kết Accuracy >88%.** Xem [cách chạy bằng VS Code](VSCODE_AUDIO.md). Demo hiện tại học 1 epoch trên 40 mẫu train, chọn ngưỡng trên 40 mẫu dev và đạt Accuracy 45% trên 40 mẫu test; chất lượng này chưa đủ để sử dụng thực tế.
Các kiểm thử dùng tín hiệu tổng hợp/trọng số ngẫu nhiên chỉ kiểm tra phần mềm, không đo khả năng phát hiện deepfake.
API trả HTTP 503 nếu thiếu mô hình; không âm thầm dùng heuristic hoặc trọng số ngẫu nhiên.

## Thiết kế

```text
WAV / FLAC / OGG
  → kiểm tra thời lượng, giải mã, trộn mono, resample 16 kHz, peak normalize
  → cửa sổ 4 giây (đuôi được zero-pad)
  → STFT 512, hop 160, 80 bộ lọc Mel tam giác (HTK), log-power [-80, 0] dB
  → chuẩn hóa cố định [0, 1]
  → Conv2D(16) → ReLU → MaxPool → Conv2D(32) → ReLU → MaxPool
  → Conv2D(64) → ReLU → MaxPool → Global Average Pool → Linear(2)
  → softmax: real=0, fake=1
```

Đây là **CNN baseline nhỏ**, không phải Wav2Vec2, ECAPA hay ResNet-18 pretrained. Chọn đầu vào phổ 2D để Grad-CAM dễ diễn giải và huấn luyện được trên CPU. Có thể nâng cấp sang ResNet-18 sau khi có benchmark baseline. Không nối embedding Wav2Vec2 vào ResNet 2D khi chưa định nghĩa phép biến đổi phù hợp.

Điểm toàn file là trung bình điểm fake của cửa sổ, có trọng số theo thời lượng thật. Tập dev và test cũng dùng chính cách tính này. Cách lấy trung bình có thể bỏ sót một đoạn giả rất ngắn trong file dài; bài toán hiện tại là phân loại file, chưa phải định vị đoạn giả.

Ở 16 kHz, tần số Nyquist là 8 kHz: không có dữ liệu trên 8 kHz. Năng lượng cao tần yếu cũng có thể do microphone, codec, resampling; không dùng làm luật kết luận thật/giả. Pipeline chưa có VAD: người dùng phải đưa bản ghi giọng nói, tiếng nhạc/tiếng động nằm ngoài phạm vi.

## Cài đặt trên Windows

Từ thư mục gốc `D:\KI1NAM3\CDK`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-audio.txt
cd backend
```

Nếu chỉ dùng CPU, có thể cài PyTorch từ kho CPU chính thức trước khi cài requirements:

```powershell
..\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Dữ liệu và protocol

Dùng ASVspoof 2019 **Logical Access (LA)** train/dev để học và chọn mô hình, eval làm test. ASVspoof 2021 DF phù hợp làm đánh giá mở rộng; không dùng tập đánh giá 2021 để chọn ngưỡng rồi báo lại điểm trên chính tập đó. Tải dữ liệu từ nguồn chính thức và tuân thủ điều kiện sử dụng:

- [ASVspoof 2019](https://www.asvspoof.org/index2019.html)
- [ASVspoof 2021 và protocol](https://www.asvspoof.org/index2021.html)

Chuyển protocol 2019 LA thành CSV (thay đường dẫn theo nơi giải nén):

```powershell
..\.venv\Scripts\python.exe audio_cli.py prepare-asvspoof2019 --partition train "D:/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" "D:/datasets/LA/ASVspoof2019_LA_train/flac" --partition dev "D:/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt" "D:/datasets/LA/ASVspoof2019_LA_dev/flac" --partition test "D:/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt" "D:/datasets/LA/ASVspoof2019_LA_eval/flac" --output ../data/audio/manifest.csv
```

Hoặc tự tạo CSV UTF-8 (đường dẫn tương đối tính từ vị trí CSV):

```csv
path,label,speaker,split,attack
train/real_001.wav,real,speaker_001,train,bonafide
train/fake_001.wav,fake,speaker_002,train,tts_a
dev/real_001.wav,real,speaker_003,dev,bonafide
dev/fake_001.wav,fake,speaker_004,dev,tts_b
test/real_001.wav,real,speaker_005,test,bonafide
test/fake_001.wav,fake,speaker_006,test,tts_c
```

Ví dụ trên chỉ mô tả schema, không phải tập dữ liệu đủ để huấn luyện. Mỗi split phải có cả hai lớp và nhiều người nói. Không chia ngẫu nhiên các crop từ cùng bản ghi vào nhiều split. Mã kiểm tra trùng đường dẫn, hash file và speaker; vẫn cần kiểm tra thủ công các bản re-encode, cùng nội dung, giọng clone và nhóm nguồn thu. Nếu bổ sung tiếng Việt, cần giọng thật và giả đa dạng với điều kiện thu tương đương; giữ riêng người nói và công cụ sinh giọng chưa từng gặp để test.

## Huấn luyện và đánh giá

Chạy trong `backend`:

```powershell
..\.venv\Scripts\python.exe audio_cli.py train --manifest ../data/audio/manifest.csv --epochs 20 --batch-size 16 --output models/audio_cnn.pt
..\.venv\Scripts\python.exe audio_cli.py evaluate --manifest ../data/audio/manifest.csv --model models/audio_cnn.pt --output outputs/test_metrics.json
```

Mặc định CPU; chỉ dùng `--device cuda` khi đã cài PyTorch CUDA và có GPU phù hợp. Mỗi epoch lấy random crop cho train; dev/test tính toàn file. Loss có trọng số cân bằng lớp. Seed mặc định 42; không đảm bảo bitwise reproducibility giữa thiết bị/phiên bản.

Checkpoint tốt nhất chọn theo EER xấp xỉ trên dev. Ngưỡng chọn tại điểm ROC dev gần FPR=FNR nhất, sau đó đóng băng cho test. `eer_approx` là trung bình FPR/FNR tại điểm ROC gần nhau nhất, không phải nội suy EER chính thức của challenge. Kết quả gồm Accuracy, F1 cho fake, ROC-AUC, EER xấp xỉ, confusion matrix theo thứ tự `[real, fake]`, file điểm từng bản ghi và lịch sử huấn luyện. Không triển khai t-DCF vì chưa có điểm hệ thống ASV cần thiết.

Để viết báo cáo, bổ sung bảng theo từng attack/codec/ngôn ngữ và kiểm tra chéo ASVspoof → giọng Việt. Softmax chưa hiệu chỉnh: không diễn giải điểm 90% là xác suất 90% bản ghi bị làm giả. Trường `confidence` cũ chỉ giữ tương thích contract và bằng max hai điểm softmax; UI audio gọi đúng là điểm mô hình.

## Dự đoán và xuất đồ thị

```powershell
..\.venv\Scripts\python.exe audio_cli.py predict --audio "D:/samples/voice.wav" --model models/audio_cnn.pt --output outputs/prediction
```

Sinh `prediction.json` và `xai.png` gồm waveform toàn file, phổ Mel và Grad-CAM. Cửa sổ giải thích là cửa sổ có điểm fake lớn nhất; vùng được chọn được tô trên waveform. Phần zero-padding bị loại khỏi đồ thị. Heatmap luôn hướng đến **logit lớp fake**, kể cả khi nhãn toàn file là real. Màu là mức đóng góp dương tương đối trong một cửa sổ; không so trực tiếp cường độ màu giữa file và không coi đây là ground truth đoạn giả. Heatmap toàn 0 là trường hợp hợp lệ khi không có đóng góp dương theo Grad-CAM.

Phương pháp: [Grad-CAM, Selvaraju et al.](https://arxiv.org/abs/1610.02391). Trọng số kênh là trung bình gradient của logit mục tiêu theo activation, rồi cộng có trọng số, ReLU và nội suy về kích thước phổ.

## API và giao diện

```powershell
$env:DEEPGUARD_AUDIO_MODEL = "$PWD/models/audio_cnn.pt"
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Trong terminal khác, chạy frontend theo README của dự án (`npm install`, `npm run dev` trong `frontend`). Trang chính có bảng “Phân loại giọng nói thật / giả”. Upload → điểm mô hình → bảng điểm theo cửa sổ → ảnh XAI → tải PNG. Khởi động lại backend sau khi thay checkpoint.

Endpoint `POST /api/analyze/media`, multipart `file` và `modality=Audio`. Kết quả giữ contract cũ, bổ sung `audio` chứa `fakeScore`, `threshold`, `predictedClass`, `duration`, `sampleRate`, `windows`, `explainedWindow`, `xaiMethod`, `xaiTarget`, `xaiPngBase64` và chú thích. Ảnh được lưu trong SQLite cùng chi tiết phân tích; `GET /api/analyses/{id}` trả lại ảnh, danh sách lịch sử không tải ảnh để tránh payload lớn.

HTTP 503: thiếu model/thư viện hoặc checkpoint không tương thích. HTTP 422: âm thanh lỗi, im lặng hoặc ngoài giới hạn. HTTP 413: quá 20 MB. Tệp tạm dùng UUID, không dùng tên upload làm đường dẫn, được xóa sau request. Không upload checkpoint từ người dùng; chỉ nạp checkpoint local tin cậy với `weights_only=True`.

## Kiểm thử và giới hạn

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Kiểm thử resample stereo, chia cửa sổ, phổ hữu hạn, file lỗi/im lặng, Grad-CAM phụ thuộc lớp, tạo PNG, trung bình theo thời lượng, chống trùng speaker, API khi thiếu model và khôi phục XAI từ database. Tín hiệu hình sin không phải dữ liệu benchmark.

Đã kiểm chứng trong môi trường phát triển: 9 tests passed; TypeScript `tsc -b` và Vite production build thành công; lệnh `python tests/smoke_audio_cli.py` chạy đủ train → evaluate → predict → PNG. Ảnh đã được kiểm tra trực quan. Có một cảnh báo deprecation từ Starlette/AnyIO, không làm test thất bại. Chưa kiểm thử tương tác trình duyệt. Smoke test tạo dữ liệu và checkpoint tại `backend/outputs/smoke/`; không dùng checkpoint này làm mô hình giọng nói.

Giới hạn: baseline chưa được benchmark, chưa pretrained, chưa hiệu chỉnh score, không nhận dạng người nói, chưa VAD, chưa định vị giả mạo chuẩn xác, giới hạn 60 giây/file. Trước khi dùng kết quả trong báo cáo, phải huấn luyện trên dữ liệu thật và báo điểm held-out test cùng cấu hình/dữ liệu. Không suy diễn độ chính xác từ việc test phần mềm chạy thành công.

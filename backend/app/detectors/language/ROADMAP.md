# Lộ trình nâng cấp phát hiện tin giả tiếng Việt

Mục tiêu: đạt kết quả ổn định trên dữ liệu mới (khác nguồn, khác chủ đề, khác thời điểm), không chỉ trên tập test cùng nguồn với tập train.

## Hiện trạng (đo ngày 21/10/2026)

Baseline TF-IDF (1–2 gram) + Logistic Regression, train trên `news_train.csv` do `data_pipeline.py` sinh ra, chấm trên `news_test.csv`. Dữ liệu được chia theo nguồn nên các nguồn trong test không xuất hiện trong train.

| Tập test | Số mẫu | Macro-F1 |
|---|---|---|
| Toàn bộ | 551 | 0.749 |
| `fake_news_data` | 118 | 0.805 (bản chia cũ của repo cho 0.94) |
| ReINTEL | 421 | 0.673 |
| Train chỉ trên `fake_news_data`, test trên ReINTEL | 421 | 0.167 |

RSS chỉ có 12 tin thật trong test, nhưng baseline phân loại đúng 3/12 (accuracy 0.25), tức gắn nhầm phần lớn tin báo mới là giả.

Kết luận: TF-IDF học theo từ vựng và nguồn. Sang nguồn khác hoặc thời điểm khác thì kết quả giảm mạnh. Con số 0.94 trước đây bị thổi phồng vì tập train và tập test lấy từ cùng nguồn.

## Giai đoạn 0 — Dữ liệu và cách đánh giá (PR này)

- `data_pipeline.py` làm các việc sau:
  - Tải ReINTEL, ViFactCheck và ViWikiFC, kèm dữ liệu `Fake_news_data` có sẵn và tin RSS đã thu thập.
  - Quy đổi nhãn về `fake`/`real` cho tin tức và `SUPPORTED`/`REFUTED`/`NEI` cho claim. Gặp nhãn lạ thì báo lỗi, không tự đoán.
  - Loại bài gần trùng bằng MinHash + LSH. Cụm bài trùng mà mâu thuẫn nhãn thì bỏ cả cụm.
  - Chia train/val/test theo nhóm nguồn (user, tác giả, trang Wikipedia, báo), làm riêng cho từng bộ dữ liệu. Nếu một nguồn xuất hiện ở hai tập thì script báo lỗi.
  - Ghi `manifest.json` gồm cấu hình, SHA-256 của file đầu vào và thống kê.
- `rss_crawler.py` làm các việc sau:
  - Thu thập RSS của VnExpress, Tuổi Trẻ, Thanh Niên, Dân Trí và VTV, có tùy chọn tải toàn văn.
  - Thu thập riêng tin đã bị bác bỏ trên tingia.gov.vn.
  - Tuân thủ robots.txt, giãn cách request và bỏ qua URL đã thu thập.
- Những việc tiếp theo cần làm:
  - Viết `evaluate.py` chạy theo lô, tính macro-F1, PR-AUC và ECE, tách theo bộ dữ liệu, nguồn và tháng.
  - Thêm cách chia theo thời gian: train trên dữ liệu cũ, test trên dữ liệu mới.

## Giai đoạn 1 — Mở rộng dữ liệu

1. Chạy crawler RSS hằng ngày để tích lũy tin thật có toàn văn. Nhãn `real` của tin RSS là nhãn yếu (`label_origin=weak`).
2. Thu thập tingia.gov.vn để lấy các tin đồn đã bị bác bỏ. Từ mỗi bài, trích claim sai làm mẫu `fake` hoặc `REFUTED`. Cần đọc điều khoản sử dụng của trang trước.
3. Tạo tin giả tổng hợp: dùng LLM sửa tin RSS (đổi số liệu, ngày tháng, nhân vật, đảo ý, thêm giọng giật gân). Mẫu tạo ra được gắn `label_origin=synthetic`. Chỉ dùng mẫu tổng hợp trong tập train. Tập test chỉ chứa tin giả có thật ngoài đời.
4. Dùng `vietgpt/binhvq_news_vi` (khoảng 19 triệu tin, không có nhãn) cho hai việc: train thêm để model quen văn báo chí (DAPT), và làm kho bằng chứng cho RAG. Cần kiểm tra giấy phép trước khi phân phối lại.

## Giai đoạn 2 — Model phân loại tin

- Fine-tune `vinai/phobert-base-v2` (bài báo) hoặc `uitnlp/visobert` (mạng xã hội). Nếu có GPU từ 24 GB VRAM thì có thể dùng `xlm-roberta-large`.
- Bài dài hơn 256 token: chia thành nhiều đoạn có chồng lấn rồi gộp kết quả bằng max hoặc attention pooling. Cách khác là chỉ lấy phần đầu và phần cuối bài.
- Dùng loss có trọng số lớp hoặc focal loss, vì tỉ lệ fake/real hiện khoảng 1:3.
- Xuất model sang ONNX và lượng tử hóa INT8 để chấm nhanh trên dữ liệu lớn.
- Gắn vào `model.py` qua `DEEPGUARD_LANGUAGE_MODEL_PATH`, giữ nguyên interface `predict_score`.

## Giai đoạn 3 — Fact-check bằng RAG + NLI

- Lập chỉ mục kho tin chính thống (RSS + binhvq) theo cách lai: BGE-M3 (dense) kết hợp BM25, rồi sắp xếp lại bằng `BAAI/bge-reranker-v2-m3`.
- Phân loại quan hệ claim–bằng chứng bằng model NLI, fine-tune trên `claims_train.csv` (ViFactCheck + ViWikiFC, khoảng 22.000 mẫu). Model này thay `qwen2.5:3b` ở bước kết luận, nhanh hơn và cho kết quả lặp lại được.
- Chỉ dùng LLM lớn hơn (`qwen2.5:7b`/`14b`) để viết phần giải thích.

## Giai đoạn 4 — Kết hợp và hiệu chỉnh điểm

- Học trọng số kết hợp `style_risk`, `fraud_risk` và `factuality_risk` trên tập val bằng Logistic Regression hoặc LightGBM, thay cho trọng số đặt tay.
- Hiệu chỉnh xác suất bằng isotonic hoặc Platt scaling. Chọn ngưỡng SAFE/SUSPICIOUS/DANGEROUS theo mức precision/recall mong muốn.

## Giai đoạn 5 — Độ bền vững

- Chuẩn hóa văn bản không dấu, teencode và ký tự Unicode dễ nhầm. Tạo bộ test biến thể để đo khả năng chống né tránh.
- Theo dõi hiện tượng trôi dữ liệu (drift): chấm định kỳ trên tin RSS mới. Nếu tỉ lệ dự đoán `fake` trên tin báo chính thống tăng thì cần train lại.

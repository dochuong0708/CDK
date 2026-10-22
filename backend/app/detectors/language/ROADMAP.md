# Lộ trình nâng cấp phát hiện tin giả tiếng Việt

Mục tiêu: đạt kết quả ổn định trên dữ liệu mới (khác nguồn, khác chủ đề, khác thời điểm), không chỉ trên tập test cùng nguồn với tập train.

## Nguyên tắc dữ liệu

Chỉ dùng dữ liệu có giấy phép mở rõ ràng và hợp pháp:
- Không dùng bộ không ghi giấy phép hoặc phải xin tác giả (ReINTEL, VFND, `vietgpt/binhvq_news_vi`).
- Không thu thập nội dung báo điện tử hay tingia.gov.vn. Các trang này ghi rõ bản quyền và cấm sao chép nếu chưa được chấp thuận bằng văn bản.
- Mọi bộ dữ liệu mới phải được ghi vào `DATASET_LICENSES` trong `data_pipeline.py`, kèm link và thông tin ghi nguồn.

| Bộ dữ liệu | Giấy phép | Trạng thái |
|---|---|---|
| ViFactCheck | MIT | Dùng |
| ViWikiFC | CC-BY-4.0 (bắt buộc ghi nguồn) | Dùng |
| `data/TEXT/Fake_news_data` | Chưa rõ | Tắt mặc định, chờ chủ dự án xác nhận |

## Giai đoạn 0 — Dữ liệu và cách đánh giá (PR này)

- `data_pipeline.py` làm các việc sau:
  - Tải ViFactCheck và ViWikiFC.
  - Quy đổi nhãn về `SUPPORTED`/`REFUTED`/`NEI`. Gặp nhãn lạ thì báo lỗi, không tự đoán.
  - Loại claim gần trùng bằng MinHash + LSH. Cụm claim trùng mà mâu thuẫn nhãn thì bỏ cả cụm.
  - Chia train/val/test theo nhóm nguồn (tác giả bài báo, trang Wikipedia), làm riêng cho từng bộ. Nếu một nguồn xuất hiện ở hai tập thì script báo lỗi.
  - Ghi `manifest.json` gồm cấu hình, SHA-256 file đầu vào, giấy phép và thông tin ghi nguồn.
- Những việc tiếp theo cần làm:
  - Viết `evaluate.py` chạy theo lô, tính macro-F1, PR-AUC và ECE, tách theo bộ dữ liệu và nguồn.
  - Thêm cách chia theo thời gian khi dữ liệu có ngày đăng.

## Giai đoạn 1 — Mở rộng dữ liệu một cách hợp pháp

1. Tạo tập phân loại tin từ ViFactCheck (MIT). Claim `SUPPORTED` dùng làm mẫu thật, claim `REFUTED` dùng làm mẫu sai. Ghi rõ đây là dữ liệu dẫn xuất.
2. Tạo dữ liệu tổng hợp từ Wikipedia tiếng Việt (CC-BY-SA) và ViWikiFC: dùng LLM sửa câu đúng (đổi số liệu, ngày tháng, nhân vật, đảo ý, thêm giọng giật gân) để có câu sai. Mẫu tạo ra được gắn `label_origin=synthetic` và chỉ dùng trong tập train. Dữ liệu dẫn xuất từ Wikipedia phải giữ giấy phép CC-BY-SA.
3. Nhóm tự viết và tự gán nhãn dữ liệu của mình. Mẫu cho dạng tin này dựa trên kiến thức chung, không sao chép nội dung có bản quyền.

## Giai đoạn 2 — Model phân loại tin

- Fine-tune `vinai/phobert-base-v2` (bài báo) hoặc `uitnlp/visobert` (mạng xã hội). Nếu có GPU từ 24 GB VRAM thì có thể dùng `xlm-roberta-large`.
- Bài dài hơn 256 token: chia thành nhiều đoạn có chồng lấn rồi gộp kết quả bằng max hoặc attention pooling. Cách khác là chỉ lấy phần đầu và phần cuối bài.
- Dùng loss có trọng số lớp hoặc focal loss, vì tỉ lệ fake/real hiện khoảng 1:3.
- Xuất model sang ONNX và lượng tử hóa INT8 để chấm nhanh trên dữ liệu lớn.
- Gắn vào `model.py` qua `DEEPGUARD_LANGUAGE_MODEL_PATH`, giữ nguyên interface `predict_score`.

## Giai đoạn 3 — Fact-check bằng RAG + NLI

- Lập chỉ mục kho bằng chứng có giấy phép mở (Wikipedia tiếng Việt, ngữ cảnh trong ViFactCheck/ViWikiFC) theo cách lai: BGE-M3 (dense) kết hợp BM25, rồi sắp xếp lại bằng `BAAI/bge-reranker-v2-m3`.
- Phân loại quan hệ claim–bằng chứng bằng model NLI, fine-tune trên `claims_train.csv` (ViFactCheck + ViWikiFC, khoảng 22.000 mẫu). Model này thay `qwen2.5:3b` ở bước kết luận, nhanh hơn và cho kết quả lặp lại được.
- Chỉ dùng LLM lớn hơn (`qwen2.5:7b`/`14b`) để viết phần giải thích.

## Giai đoạn 4 — Kết hợp và hiệu chỉnh điểm

- Học trọng số kết hợp `style_risk`, `fraud_risk` và `factuality_risk` trên tập val bằng Logistic Regression hoặc LightGBM, thay cho trọng số đặt tay.
- Hiệu chỉnh xác suất bằng isotonic hoặc Platt scaling. Chọn ngưỡng SAFE/SUSPICIOUS/DANGEROUS theo mức precision/recall mong muốn.

## Giai đoạn 5 — Độ bền vững

- Chuẩn hóa văn bản không dấu, teencode và ký tự Unicode dễ nhầm. Tạo bộ test biến thể để đo khả năng chống né tránh.
- Theo dõi hiện tượng trôi dữ liệu (drift): chấm định kỳ trên dữ liệu mới có giấy phép. Nếu phân phối điểm thay đổi rõ thì cần train lại.

Mô-đun phát hiện rủi ro ngôn ngữ

Luồng xử lý

1. preprocessing.py chuẩn hóa Unicode về NFC, loại bỏ ký tự điều khiển, gom khoảng trắng, nhận diện URL, số điện thoại, chuỗi giống số tài khoản và tách từ tiếng Việt bằng PyVi. Nếu tokenizer chưa cài đặt, mô-đun dùng tách từ theo khoảng trắng.
2. heuristics.py áp dụng regex và tách tín hiệu thành hai nhóm: style (giật gân, nặc danh, thuyết âm mưu, kêu gọi lan truyền) và fraud (khẩn cấp, đe dọa, đòi OTP/mật khẩu/chuyển tiền, URL lừa đảo).
3. domains.py phát hiện tên miền giả mạo thương hiệu (vietcombankk.com, v1etcombank.com, teckcombank.vn) bằng Levenshtein distance và chuẩn hóa ký tự dễ nhầm (1→i, 0→o, rn→m, vv→w). Tên miền chính thức và subdomain của chúng không bị gắn cờ.
4. style_classifier.py dùng embedding tiếng Việt (mặc định bkai-foundation-models/vietnamese-bi-encoder) so khớp câu với các câu mẫu giật gân/lừa đảo/trung tính để nhận diện theo ngữ cảnh, bổ sung cho regex. Không có sentence-transformers thì bỏ qua bước này.
5. model.py nạp mô hình joblib tùy chọn (phân loại tin giả). Không có model thì hệ thống không giả vờ đã chạy AI.
6. services/risk_fusion.py kết hợp thành các trục độc lập:
   - styleRisk = noisy-OR(luật style, embedding giật gân)
   - fraudRisk = noisy-OR(luật fraud, embedding lừa đảo)
   - factualityRisk = model tin giả kết hợp kết quả fact-check RAG (trọng số theo độ tin cậy của báo cáo); null nếu chưa có tín hiệu nào.
   - Điểm tổng: max(fraudRisk, factualityRisk, 0.35·style + 0.65·factuality). Style một mình không thể đẩy bài viết lên DANGEROUS, nên bài đúng sự thật nhưng tiêu đề giật gân nhẹ vẫn được giữ ở mức thấp, trong khi bài viết bình thường nhưng bị fact-check bác bỏ vẫn bị đánh dấu nguy hiểm.

Cấu hình embedding

- Cài: python -m pip install -r backend/requirements-ml.txt (tải model từ Hugging Face ở lần chạy đầu).
- DEEPGUARD_EMBEDDING_MODEL: tên model sentence-transformers khác, hoặc off để tắt (test dùng off).
- Khi không có model ngữ nghĩa, RAG dùng TF-IDF character n-gram để tính cosine similarity.

Fact-check (RAG)

- Trích xuất claim, tìm kiếm web (DuckDuckGo/Bing), xếp hạng snippet theo cosine similarity với claim, rồi mới đưa cho LLM đánh giá. Nguồn có độ tương đồng thấp bị hạ từ supports/contradicts xuống context, tránh LLM kết luận dựa trên kết quả không liên quan.
- Mỗi báo cáo được lưu vào bảng fact_checks (SQLite, migration 002) và trả về reportId. Xem lại qua GET /api/fact-checks và GET /api/fact-checks/{id}.
- POST /api/analyze/text với check_facts=true sẽ chạy fact-check và gộp factualityRisk vào kết quả phân tích; analyses lưu factCheckId để liên kết.

Ngưỡng nhãn

- Dưới 0.30: SAFE
- Từ 0.30 đến dưới 0.70: SUSPICIOUS
- Từ 0.70 trở lên: DANGEROUS

Cách bật mô hình

- Đặt dữ liệu CSV vào data/language/. CSV cần có cột text và label; tên cột khác có thể truyền bằng --text-column và --label-column.
- Mặc định các nhãn scam, phishing, spam, toxic, hate, fake, fake news, dangerous được gộp thành risky. Các nhãn ham, safe, normal, clean, real được gộp thành safe. Nhãn riêng của dataset phải khai báo rõ bằng --risky-label hoặc --safe-label; không tự đoán mã nhãn của UIT/VLSP.
- Với Fake_news_data trong repo này, theo xác nhận của nhóm: Label 0 là tin giả/risky, Label 1 là tin thật/safe. Vì mặc định 0/1 thường được hiểu theo chiều ngược lại ở dataset khác, luôn truyền cả --risky-label 0 --safe-label 1 cho bộ này.
- Loader tự nhận diện các tên cột văn bản phổ biến (text, Maintext, content, message) và nhãn (label, target, class); dùng --text-column/--label-column để ghi đè khi cần.
- TF-IDF dùng cùng bộ tách từ tiếng Việt ở lúc train và dự đoán. Nếu thiếu tokenizer tiếng Việt, pipeline tự chuyển về tách theo khoảng trắng.
- Cài dependency bằng python -m pip install -r backend/requirements.txt.
- Chạy từ thư mục backend: python -m app.detectors.language.train ..\data\language\dataset.csv
- Có split sẵn: truyền --validation-dataset và/hoặc --test-dataset. Khi truyền các split này, toàn bộ file positional chỉ dùng làm train; validation/test chỉ được đánh giá, không tham gia fit.
- Đánh giá độc lập model đã lưu: python -m app.detectors.language.train ..\data\language\test.csv --text-column text --label-column label --evaluate-model data\models\vietnamese-language-risk.joblib
- Ví dụ nhãn dataset: python -m app.detectors.language.train ..\data\language\dataset.csv --text-column content --label-column label --risky-label HATE --risky-label OFFENSIVE --safe-label CLEAN
- Script dùng TF-IDF unigram/bigram và Logistic Regression. Nếu không truyền split ngoài, script tự chia train/test có stratify; nếu truyền validation/test thì giữ nguyên toàn bộ train. Script in precision/recall/F1 và lưu pipeline ở backend/data/models/vietnamese-language-risk.joblib.
- Đặt biến môi trường DEEPGUARD_LANGUAGE_MODEL_PATH trỏ tới file model trước khi khởi chạy backend. Ví dụ PowerShell khi đang ở thư mục backend: $env:DEEPGUARD_LANGUAGE_MODEL_PATH = (Resolve-Path 'data/models/vietnamese-language-risk.joblib').Path
- Không commit dữ liệu cá nhân, model có điều khoản hạn chế hoặc dữ liệu huấn luyện chưa được phép chia sẻ.

Lệnh cho các file Fake_news_data hiện có

Chạy từ thư mục backend. Theo xác nhận của nhóm, nhãn 0 là fake/risky và nhãn 1 là real/safe:

python -m app.detectors.language.train ..\data\TEXT\Fake_news_data\update_train_data.csv --text-column Maintext --label-column Label --risky-label 0 --safe-label 1 --validation-dataset ..\data\TEXT\Fake_news_data\update_val_data.csv --output data\models\vietnam_fake_news_risk.joblib

Sau khi train xong, đánh giá độc lập trên tập test. Lệnh này chỉ nạp model và dự đoán, không train lại:

python -m app.detectors.language.train ..\data\TEXT\Fake_news_data\fix_test_data.csv --text-column Maintext --label-column Label --risky-label 0 --safe-label 1 --evaluate-model data\models\vietnam_fake_news_risk.joblib

Lệnh train SMS (ham an toàn, spam rủi ro; tự chia train/test có stratify):

python -m app.detectors.language.train ..\data\TEXT\SMS_data\vi_dataset.csv --output data\models\vietnamese-sms-risk.joblib

Pipeline dữ liệu lớn (ReINTEL, ViFactCheck, ViWikiFC, RSS)

Lộ trình nâng cấp chi tiết nằm trong ROADMAP.md. Dữ liệu tải về nằm ở data/language/raw và kết quả xử lý nằm ở data/language/processed. Hai thư mục này không được commit vì giấy phép và dung lượng.

python -m pip install -r requirements-data.txt
python -m app.detectors.language.rss_crawler --full-text --max-items-per-feed 50
python -m app.detectors.language.rss_crawler --debunks
python -m app.detectors.language.data_pipeline

Kết quả gồm news_{train,val,test}.csv (nhãn fake/real), claims_{train,val,test}.csv (nhãn SUPPORTED/REFUTED/NEI) và manifest.json. Có thể train ngay bằng lệnh:

python -m app.detectors.language.train ..\data\language\processed\news_train.csv --text-column text --label-column label --validation-dataset ..\data\language\processed\news_val.csv --output data\models\news_risk.joblib

Bảng quy đổi nhãn: Fake_news_data 0=fake, 1=real; ReINTEL 1=fake, 0=real (chỉ dùng split train vì split test không có nhãn); ViFactCheck 0=SUPPORTED, 1=REFUTED, 2=NEI; ViWikiFC Supports/Refutes/Not_Enough_Information. Tin RSS được gắn nhãn real yếu (label_origin=weak). Không nên đánh giá model chỉ trên tin RSS.

Nguồn dữ liệu tham khảo

- UIT-NLP: tìm UIT-ViHSD và UIT-ViCTSD trên GitHub/Hugging Face; kiểm tra nhãn và giấy phép của từng bộ trước khi dùng.
- VLSP: xem các shared task tiếng Việt về tin giả, spam hoặc phân loại văn bản; quyền truy cập và điều khoản có thể khác nhau theo năm.
- ChongLuaDao: có thể dùng danh sách tên miền/URL làm tín hiệu luật. Không tự động gán nhãn toàn bộ nội dung hoặc tên miền nếu chưa kiểm tra nguồn gốc và giấy phép.
- Hugging Face và Kaggle: tìm bộ tiếng Việt về spam, phishing, toxic hoặc fake news. Kiểm tra chất lượng nhãn, trùng lặp, dữ liệu cá nhân và license trước khi nhập.
- Không trộn nhãn của các bài toán khác nhau một cách vô điều kiện. Toxic speech không đồng nghĩa phishing; fake news không đồng nghĩa spam. Ghi lại nguồn, định nghĩa nhãn và cách ánh xạ từng dataset.
- Classification là nhãn chủ đề như chính trị, xã hội, kinh tế hoặc giáo dục; nên dùng để huấn luyện một bộ phân loại chủ đề riêng hoặc làm nhãn phụ. Không đưa nhãn chủ đề vào cột nhãn rủi ro safe/risky. Muốn dùng notebook, cần xuất dữ liệu thành CSV có cột văn bản và category; script train.py hiện tại chỉ huấn luyện phân loại nhị phân rủi ro.

Hiện repo chưa có bộ dữ liệu gán nhãn hay model đã huấn luyện. Vì vậy mặc định chỉ chạy rule-based. Không nên gọi điểm rule là xác suất đã được hiệu chuẩn. Khi có model, cần chia train/validation/test, đo precision, recall, F1 và kiểm tra false positive trước khi dùng với người dùng thật.

Lưu ý giới hạn

- Danh sách tên miền hợp lệ trong domains.py chỉ là ví dụ, có thể lỗi thời và không chứng minh một website an toàn.
- URL rút gọn không đồng nghĩa chắc chắn là lừa đảo; nó chỉ là tín hiệu rủi ro.
- Không gửi OTP, mật khẩu hoặc dữ liệu tài khoản thật để thử nghiệm.
- Kết quả là tín hiệu tham khảo, không phải kết luận pháp lý hay xác minh danh tính.

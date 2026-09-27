Mô-đun phát hiện rủi ro ngôn ngữ

Luồng xử lý

1. preprocessing.py chuẩn hóa Unicode về NFC, loại bỏ ký tự điều khiển, gom khoảng trắng, nhận diện URL, số điện thoại, chuỗi giống số tài khoản và tách từ tiếng Việt bằng PyVi. Nếu tokenizer chưa cài đặt, mô-đun dùng tách từ theo khoảng trắng.
2. heuristics.py áp dụng regex và danh sách miền rút gọn/ngân hàng để tạo điểm luật và bằng chứng.
3. model.py nạp mô hình joblib tùy chọn. Model phải hỗ trợ predict_proba hoặc decision_function. Không có model thì hệ thống chỉ dùng luật, không giả vờ rằng đã chạy AI.
4. detector.py kết hợp điểm: S_final = max(S_rule, S_model), chuyển sang riskScore 0-100 và phân loại SAFE, SUSPICIOUS, DANGEROUS.

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

Nguồn dữ liệu tham khảo

- UIT-NLP: tìm UIT-ViHSD và UIT-ViCTSD trên GitHub/Hugging Face; kiểm tra nhãn và giấy phép của từng bộ trước khi dùng.
- VLSP: xem các shared task tiếng Việt về tin giả, spam hoặc phân loại văn bản; quyền truy cập và điều khoản có thể khác nhau theo năm.
- ChongLuaDao: có thể dùng danh sách tên miền/URL làm tín hiệu luật. Không tự động gán nhãn toàn bộ nội dung hoặc tên miền nếu chưa kiểm tra nguồn gốc và giấy phép.
- Hugging Face và Kaggle: tìm bộ tiếng Việt về spam, phishing, toxic hoặc fake news. Kiểm tra chất lượng nhãn, trùng lặp, dữ liệu cá nhân và license trước khi nhập.
- Không trộn nhãn của các bài toán khác nhau một cách vô điều kiện. Toxic speech không đồng nghĩa phishing; fake news không đồng nghĩa spam. Ghi lại nguồn, định nghĩa nhãn và cách ánh xạ từng dataset.
- Classification là nhãn chủ đề như chính trị, xã hội, kinh tế hoặc giáo dục; nên dùng để huấn luyện một bộ phân loại chủ đề riêng hoặc làm nhãn phụ. Không đưa nhãn chủ đề vào cột nhãn rủi ro safe/risky. Muốn dùng notebook, cần xuất dữ liệu thành CSV có cột văn bản và category; script train.py hiện tại chỉ huấn luyện phân loại nhị phân rủi ro.

Hiện repo chưa có bộ dữ liệu gán nhãn hay model đã huấn luyện. Vì vậy mặc định chỉ chạy rule-based. Không nên gọi điểm rule là xác suất đã được hiệu chuẩn. Khi có model, cần chia train/validation/test, đo precision, recall, F1 và kiểm tra false positive trước khi dùng với người dùng thật.

Lưu ý giới hạn

- Danh sách tên miền hợp lệ trong heuristics.py chỉ là ví dụ, có thể lỗi thời và không chứng minh một website an toàn.
- URL rút gọn không đồng nghĩa chắc chắn là lừa đảo; nó chỉ là tín hiệu rủi ro.
- Không gửi OTP, mật khẩu hoặc dữ liệu tài khoản thật để thử nghiệm.
- Kết quả là tín hiệu tham khảo, không phải kết luận pháp lý hay xác minh danh tính.

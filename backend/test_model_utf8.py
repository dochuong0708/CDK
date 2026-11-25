from app.detectors.language.model import LanguageModel

model = LanguageModel()

print("=== MODEL INFORMATION ===")
print("Loaded:", model.model is not None)
print("Classes:", model.model.classes_)

samples = [
    "Ngân hàng yêu cầu khách hàng cung cấp mật khẩu và mã OTP để xác minh tài khoản ngay lập tức.",
    "Hôm nay, cơ quan khí tượng thông báo dự báo thời tiết tại Hà Nội có mưa rào vào buổi chiều.",
    "Chính phủ ban hành chính sách mới nhằm hỗ trợ phát triển kinh tế và nâng cao chất lượng giáo dục.",
]

for index, sample in enumerate(samples, start=1):
    probabilities = model.model.predict_proba([sample])[0]
    classes = model.model.classes_

    print(f"\nSample {index}: {sample}")

    for label, probability in zip(classes, probabilities):
        print(f"{label}: {probability:.6f}")

    print("Risk score:", model.predict_score(sample))

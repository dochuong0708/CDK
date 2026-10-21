"""Phân loại hình thức câu văn (giật gân / lừa đảo / trung tính) bằng embedding.

Few-shot theo prototype: so sánh câu đầu vào với các câu mẫu của từng nhóm bằng cosine
similarity trên bi-encoder tiếng Việt, nên nhận diện được cách diễn đạt mới mà regex bỏ sót.
Có thể thay bằng PhoBERT fine-tuned bằng cách truyền `embedder` khác.
"""

import math
from dataclasses import dataclass

from app.nlp.embeddings import Embedder, cosine, get_semantic_embedder

PROTOTYPES: dict[str, tuple[str, ...]] = {
    "sensational": (
        "Cực sốc! Sự thật kinh hoàng khiến cả nước bàng hoàng",
        "Không thể tin nổi điều vừa xảy ra, dân mạng chết lặng",
        "Tiết lộ động trời mà ai cũng phải ngã ngửa",
        "Chia sẻ ngay kẻo muộn, thông tin này đang bị gỡ khắp nơi",
        "Chính phủ che giấu sự thật, thần dược chữa khỏi ung thư 100%",
        "Nghe nói nguồn tin giấu tên tiết lộ bí mật bị che đậy",
    ),
    "scam": (
        "Tài khoản của bạn sẽ bị khóa, vui lòng xác minh ngay trong 24 giờ",
        "Chúc mừng bạn đã trúng thưởng, hãy chuyển phí nhận quà vào số tài khoản sau",
        "Vui lòng cung cấp mã OTP để hoàn tất xác thực giao dịch",
        "Cơ quan công an yêu cầu bạn chuyển tiền để phục vụ điều tra",
        "Nhấn vào đường link để cập nhật thông tin ngân hàng tránh bị phạt",
    ),
    "neutral": (
        "Cuộc họp sẽ diễn ra vào lúc 9 giờ sáng mai tại phòng họp số 2",
        "Theo Tổng cục Thống kê, GDP quý ba tăng 7,4% so với cùng kỳ năm trước",
        "Bộ Y tế khuyến cáo người dân tiêm vaccine đầy đủ theo lịch",
        "Ủy ban nhân dân thành phố công bố kế hoạch sửa chữa tuyến đường",
        "Đội tuyển Việt Nam thắng 2-0 trong trận giao hữu tối qua",
        "Ngân hàng thông báo lịch nghỉ lễ và giờ làm việc của các chi nhánh",
    ),
}
RISKY_CATEGORIES = ("sensational", "scam")
TEMPERATURE = 0.05


@dataclass(frozen=True)
class StylePrediction:
    sensational: float
    scam: float
    top_category: str

    @property
    def risk(self) -> float:
        return max(self.sensational, self.scam)


class EmbeddingStyleClassifier:
    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder
        self._prototype_vectors: dict[str, list[list[float]]] | None = None

    @property
    def embedder(self) -> Embedder | None:
        if self._embedder is None:
            self._embedder = get_semantic_embedder()
        return self._embedder

    @property
    def available(self) -> bool:
        return self.embedder is not None

    def _prototypes(self) -> dict[str, list[list[float]]]:
        if self._prototype_vectors is None:
            assert self.embedder is not None
            self._prototype_vectors = {
                category: self.embedder.embed(list(sentences))
                for category, sentences in PROTOTYPES.items()
            }
        return self._prototype_vectors

    def predict(self, text: str) -> StylePrediction | None:
        if not text.strip() or self.embedder is None:
            return None
        vector = self.embedder.embed([text[:2000]])[0]
        best = {
            category: max(cosine(vector, prototype) for prototype in prototypes)
            for category, prototypes in self._prototypes().items()
        }
        exps = {category: math.exp(score / TEMPERATURE) for category, score in best.items()}
        total = sum(exps.values())
        probabilities = {category: value / total for category, value in exps.items()}
        top_category = max(probabilities, key=probabilities.__getitem__)
        return StylePrediction(
            sensational=probabilities["sensational"],
            scam=probabilities["scam"],
            top_category=top_category,
        )

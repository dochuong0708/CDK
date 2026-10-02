from datetime import datetime
from types import SimpleNamespace
from pathlib import Path

from app.agent.cli import _explicit_text_analysis, respond
from app.agent.tools import DeepGuardTools
from app.models.schemas import Analysis


def make_analysis(modality: str = "Text") -> Analysis:
    return Analysis(
        analysisId="analysis-1",
        modality=modality,
        label="SUSPICIOUS",
        riskScore=55,
        confidence=0.8,
        evidence=["test evidence"],
        detector="test-detector",
        input="sample",
        createdAt=datetime.now(),
    )


class FakeOrchestrator:
    def analyze_text(self, text: str) -> Analysis:
        return make_analysis()

    def analyze_image(self, path: Path) -> Analysis:
        return make_analysis("Image")

    def analyze_audio(self, path: Path) -> Analysis:
        return make_analysis("Audio")


class FakeStore:
    def __init__(self) -> None:
        self.items: list[Analysis] = []

    def add(self, item: Analysis) -> Analysis:
        self.items.append(item)
        return item

    def list(self, modality: str | None = None, risk: int | None = None) -> list[Analysis]:
        return [
            item
            for item in self.items
            if (modality is None or item.modality == modality)
            and (risk is None or item.riskScore >= risk)
        ]

    def get(self, analysis_id: str) -> Analysis | None:
        return next((item for item in self.items if item.analysisId == analysis_id), None)


def make_tools() -> tuple[DeepGuardTools, FakeStore]:
    store = FakeStore()
    return DeepGuardTools(FakeOrchestrator(), store), store


def test_analyze_text_persists_and_returns_analysis() -> None:
    tools, store = make_tools()

    result = tools.execute("analyze_text", {"text": "Nội dung cần kiểm tra"})

    assert result["modality"] == "Text"
    assert result["riskScore"] == 55
    assert len(store.items) == 1


def test_analyze_image_uses_local_file_and_persists(tmp_path: Path) -> None:
    tools, store = make_tools()
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")

    result = tools.execute("analyze_media", {"path": str(image), "modality": "Image"})

    assert result["modality"] == "Image"
    assert len(store.items) == 1


def test_video_analysis_is_reported_as_unsupported() -> None:
    tools, _ = make_tools()

    result = tools.execute("analyze_media", {"path": "clip.mp4", "modality": "Video"})

    assert "Video chưa được hỗ trợ" in result["error"]


def test_analysis_history_and_lookup() -> None:
    tools, _ = make_tools()
    tools.execute("analyze_text", {"text": "Nội dung cần kiểm tra"})

    history = tools.execute("list_analyses", {"modality": "Text", "risk": 50})
    item = tools.execute("get_analysis", {"analysisId": "analysis-1"})

    assert history["total"] == 1
    assert item["analysisId"] == "analysis-1"


def test_agent_loop_runs_tool_then_returns_assistant_answer() -> None:
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="analyze_text",
            arguments={"text": "Nội dung cần kiểm tra"},
        )
    )
    client = FakeOllamaClient(
        [
            SimpleNamespace(content="", tool_calls=[tool_call]),
            SimpleNamespace(content="Mức rủi ro là 55/100.", tool_calls=None),
        ]
    )
    tools, _ = make_tools()
    messages: list[object] = [{"role": "user", "content": "Chào, giúp tôi với nội dung này"}]

    answer = respond(client, messages, tools, "test-model")

    assert answer == "Mức rủi ro là 55/100."
    assert messages[-2]["role"] == "tool"
    assert '"riskScore": 55' in messages[-2]["content"]


def test_explicit_analysis_uses_detector_if_model_skips_tool() -> None:
    client = FakeOllamaClient(
        [SimpleNamespace(content="Văn bản này có vẻ an toàn.", tool_calls=None)]
    )
    tools, store = make_tools()
    messages: list[object] = [
        {"role": "user", "content": 'Hãy phân tích đoạn này: "nội dung thử nghiệm"'}
    ]

    answer = respond(client, messages, tools, "test-model")

    assert "Kết quả detector: SUSPICIOUS" in answer
    assert "Rủi ro: 55/100" in answer
    assert len(store.items) == 1
    assert client.messages == [
        SimpleNamespace(content="Văn bản này có vẻ an toàn.", tool_calls=None)
    ]


def test_media_analysis_is_not_routed_to_text_detector() -> None:
    assert _explicit_text_analysis("Hãy phân tích hình ảnh này: photo.png") is None
    assert _explicit_text_analysis("Kiểm tra âm thanh tại voice.wav") is None


class FakeOllamaClient:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self.messages = messages

    def chat(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(message=self.messages.pop(0))
import json
import os
import re
from typing import Any

from app.agent.tools import DeepGuardTools, TOOL_DEFINITIONS
from app.db.store import AnalysisStore
from app.services.orchestrator import DetectorOrchestrator


SYSTEM_PROMPT = """Bạn là trợ lý DeepGuard, trả lời bằng tiếng Việt.
Khi cần phân tích, hãy gọi tool tương ứng và trình bày nhãn, điểm rủi ro, độ tin cậy
và bằng chứng đúng như detector trả về. Đây là kết quả heuristic/model hỗ trợ sàng lọc,
không phải bằng chứng chắc chắn hay kết luận chuyên môn. Không tự bịa kết quả.
Với ảnh hoặc âm thanh, cần đường dẫn tệp local và modality. Video chưa được hỗ trợ.
Nếu người dùng yêu cầu xem lịch sử hoặc một kết quả cũ, hãy dùng tool phù hợp.
Nếu thiếu văn bản, đường dẫn hoặc thông tin cần thiết, hãy hỏi lại trước khi gọi tool."""


def _explicit_text_analysis(text: str) -> str | None:
    if not re.search(r"\b(phân tích|kiểm tra|đánh giá)\b", text, re.IGNORECASE):
        return None
    if re.search(
        r"\b(?:phân tích|kiểm tra|đánh giá)(?:\s+\w+){0,3}\s+"
        r"(?:ảnh|hình ảnh|bức hình|image|audio|âm thanh|video|tệp|file)\b",
        text,
        re.IGNORECASE,
    ) or re.search(
        r"\.(?:png|jpe?g|webp|gif|bmp|wav|mp3|m4a|flac|mp4|mov|avi)\b",
        text,
        re.IGNORECASE,
    ):
        return None

    quoted = re.search(r'["“](.+?)["”]', text, re.DOTALL)
    return quoted.group(1).strip() if quoted else text.strip()


def _format_analysis(result: dict[str, Any]) -> str:
    if "error" in result:
        return f"Không thể phân tích: {result['error']}"

    evidence = "\n".join(f"- {item}" for item in result["evidence"])
    return (
        f"Kết quả detector: {result['label']}\n"
        f"Rủi ro: {result['riskScore']}/100\n"
        f"Độ tin cậy: {result['confidence']:.0%}\n"
        f"Bằng chứng:\n{evidence}\n"
        "Đây là tín hiệu sàng lọc từ detector, không phải kết luận chắc chắn."
    )


def respond(client: Any, messages: list[Any], tools: DeepGuardTools, model: str) -> str:
    latest_user_message = next(
        (message for message in reversed(messages) if isinstance(message, dict) and message.get("role") == "user"),
        None,
    )
    analysis_text = (
        _explicit_text_analysis(latest_user_message.get("content", ""))
        if latest_user_message
        else None
    )
    if analysis_text is not None:
        result = tools.execute("analyze_text", {"text": analysis_text})
        return _format_analysis(result)

    for _ in range(8):
        response = client.chat(model=model, messages=messages, tools=TOOL_DEFINITIONS)
        message = response.message
        messages.append(message)
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return message.content or ""

        for tool_call in tool_calls:
            name = tool_call.function.name
            result = tools.execute(name, tool_call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    raise RuntimeError("Agent vượt quá số lượt gọi tool cho một câu hỏi.")


def main() -> None:
    from ollama import Client

    model = os.getenv("DEEPGUARD_AGENT_MODEL", "qwen2.5:3b")
    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    store = AnalysisStore(os.getenv("DEEPGUARD_DB_PATH", "./data/deepguard.db"))
    tools = DeepGuardTools(DetectorOrchestrator(), store)
    messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"DeepGuard Agent | model: {model} | nhập /exit để thoát")
    while True:
        try:
            prompt = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát DeepGuard Agent.")
            break
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            print(f"DeepGuard: {respond(client, messages, tools, model)}")
        except Exception as error:
            print(f"Không thể gọi agent: {error}")


if __name__ == "__main__":
    main()
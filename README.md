# DeepGuard Agent

DeepGuard là agent chạy local để phân tích nội dung văn bản, ảnh và âm thanh, đồng thời tra cứu lịch sử kết quả. Agent dùng Ollama cho hội thoại và gọi trực tiếp các detector cùng SQLite hiện có; giao diện web đã được gỡ khỏi dự án.

## Khả năng hiện tại

- Phân tích văn bản qua bộ luật heuristic và model NLP đã cấu hình.
- Phân tích ảnh và âm thanh qua detector heuristic hiện tại.
- Lưu kết quả và truy vấn lịch sử theo modality hoặc mức rủi ro.
- Trả về nhãn, điểm rủi ro, độ tin cậy và bằng chứng detector.
- Video chưa được hỗ trợ.

Lưu ý: detector ảnh/âm thanh hiện chỉ là bản heuristic mẫu, không xác minh được nội dung giả mạo. Điểm số là tín hiệu sàng lọc, không phải kết luận chắc chắn.

## Chạy agent

Yêu cầu Python 3.10+ và Ollama đang chạy trên máy. Tại thư mục gốc:

```powershell
cd backend
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
python -m app.agent.cli
```

`ollama pull` chỉ tải model; `python -m app.agent.cli` mới khởi động agent. Khi terminal hiện `Bạn:`, agent đang chờ bạn nhập câu hỏi rồi nhấn Enter. Agent không tự chạy phân tích khi vừa khởi động. Gõ `/exit` để thoát. Với ảnh hoặc âm thanh, gửi đường dẫn tệp local và yêu cầu phân tích; agent sẽ hỏi lại nếu thiếu thông tin.

Có thể cấu hình bằng biến môi trường:

- `OLLAMA_HOST`: mặc định `http://localhost:11434`.
- `DEEPGUARD_AGENT_MODEL`: mặc định `qwen2.5:3b`.
- `DEEPGUARD_DB_PATH`: mặc định `./data/deepguard.db` khi chạy từ `backend/`.

Trong VS Code, chọn cấu hình `DeepGuard Agent (debug)` hoặc chạy task `deepguard-agent: run`. Dùng `backend: test` để chạy test.

## Kiến trúc

- `backend/app/agent/`: hội thoại Ollama và định nghĩa các tool cho agent.
- `backend/app/services/orchestrator.py`: điều phối detector.
- `backend/app/detectors/`: detector văn bản, ảnh và âm thanh.
- `backend/app/db/`: lưu kết quả và lịch sử trong SQLite.
- `backend/app/main.py`: FastAPI API vẫn được giữ để tích hợp với client khác; agent CLI không yêu cầu API server chạy.
- `backend/tests/`: test API, detector và agent tools.

## API tùy chọn

FastAPI vẫn có thể chạy độc lập từ thư mục `backend/`:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Các route hiện có gồm health, phân tích văn bản/media, danh sách phân tích và chi tiết một kết quả. Video chưa được hỗ trợ ở API hoặc agent.

Cấu trúc hiện tại của DeepGuard Agent

```text
project/
  README.md
  .vscode/
    launch.json
    tasks.json
  backend/
    requirements.txt
    pyproject.toml
    app/
      agent/
        cli.py
        tools.py
      api/
        routes/
      detectors/
        image/
        audio/
        language/
      db/
        store.py
        migrate.py
      models/
        schemas.py
      services/
        orchestrator.py
      main.py
    tests/
    migrations/
  data/
    uploads/
  docs/
```

Mô tả các phần

- `backend/app/agent/`: hội thoại với Ollama và tools mà agent có thể gọi.
- `backend/app/services/`: điều phối các detector theo modality.
- `backend/app/detectors/`: xử lý văn bản, ảnh và âm thanh.
- `backend/app/db/`: SQLite, migrations và truy vấn lịch sử.
- `backend/app/api/`: FastAPI API tùy chọn cho client hoặc tích hợp ngoài; agent CLI không cần server API.
- `backend/tests/`: test cho agent tools, API và detector.
- `data/`: database và tệp local theo cấu hình runtime.
- `docs/`: tài liệu kiến trúc và quy trình làm việc.

Giữ mỗi detector độc lập và dùng chung schema kết quả (`label`, `riskScore`, `confidence`, `evidence`, `detector`). Kết quả ảnh/âm thanh hiện là heuristic mẫu; video chưa được triển khai.

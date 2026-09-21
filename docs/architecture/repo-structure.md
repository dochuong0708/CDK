Cấu trúc repo chuẩn hóa cho dự án DeepGuard

project/
  README.md
  .gitignore
  .env.example

  frontend/
    package.json
    vite.config.ts
    index.html
    src/
      api/
      components/
      pages/
      styles/
      types/
      main.tsx

  backend/
    requirements.txt
    pyproject.toml
    app/
      main.py
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
    tests/
      test_api.py
      test_image_detector.py
      test_audio_detector.py
      test_language_detector.py
    migrations/
      001_init_tables.sql

  data/
    uploads/
    db/

  docs/
    workflows/
      github-workflow-3-people.md
    architecture/
      repo-structure.md

Mô tả từng phần

frontend/
- chứa toàn bộ giao diện web
- giao tiếp với backend với API HTTP

backend/app/api/
- route và controller của hệ thống
- chứa các endpoint chính

backend/app/detectors/
- nơi xử lý logic theo từng loại đầu vào
- image, audio, language là 3 module riêng

backend/app/db/
- database connection, migrations, lưu trữ

backend/app/models/
- schema dữ liệu, response model, entity model

backend/tests/
- test cho API, detector, dữ liệu đầu vào

data/
- lưu file upload, database local, dữ liệu mẫu

docs/
- tài liệu tổng hợp cho nhóm

Khuyến nghị trong team
- không để trộn logic image/audio/language vào cùng file
- mỗi module nên có tên rõ ràng để dễ kiểm soát
- API response phải đồng nhất giữa các module
- mỗi module cần có test riêng và commit rõ ràng
- merge bằng pull request, không merge trực tiếp lên main

Kết luận
Cấu trúc repo trên giúp team có thể phân chia công việc rõ ràng, tăng khả năng làm việc song song, giảm xung đột và dễ tích hợp sau khi 3 phần hoàn thiện.

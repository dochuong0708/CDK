GitHub workflow cho 3 người

Mục tiêu
- Mỗi người làm 1 luồng chuyên biệt: Image, Audio, Language
- Không làm trên nhánh chính khi chưa test xong
- Tích hợp ở nhánh develop trước
- Chỉ merge vào main khi cả 3 phần đã hoạt động tốt

Nhánh đề xuất

main
- nhánh chính ổn định
- chỉ dùng khi tất cả module đều chạy và đã kiểm thử

develop
- nhánh tích hợp trung gian
- nơi merge các feature branch sau khi test cơ bản

feature/image
- module hình ảnh

feature/audio
- module âm thanh

feature/language
- module ngôn ngữ

Luồng làm việc

1. Mỗi người checkout nhánh riêng
   - git checkout -b feature/image
   - git checkout -b feature/audio
   - git checkout -b feature/language

2. Mỗi người làm việc trên nhánh riêng
   - không merge trực tiếp vào main
   - không push code chưa test

3. Sau khi hoàn thiện phần module, push nhánh riêng lên GitHub
   - git push origin feature/image
   - git push origin feature/audio
   - git push origin feature/language

4. Tạo pull request từ feature branch vào develop
   - review code
   - test chức năng cơ bản
   - sửa lỗi nếu có

5. Khi cả 3 phần đã chạy ổn trên develop
   - merge develop vào main
   - tạo release hoặc tag nếu cần

Sơ đồ flow

main
  ^
  |
  +---- develop
          ^
          |
          +--- feature/image
          +--- feature/audio
          +--- feature/language

Ghi chú
- Nếu module nào chưa xong, không merge lên develop
- Nếu code trên feature branch có bug, fix trước khi merge
- Mỗi người chủ trì module của mình, không sửa code phần người khác nếu chưa thống nhất

Quy trình làm việc nhóm

- Người 1: image detector
- Người 2: audio detector
- Người 3: language detector
- Một người có thể hỗ trợ backend hoặc testing nếu cần

Mỗi module cần có:
- source code riêng
- test riêng
- README hoặc mô tả ngắn nếu cần
- output format rõ ràng

Để chuẩn hóa repo, nên có cấu trúc như sau:

project/
  README.md
  frontend/
    src/
    public/
    package.json
  backend/
    app/
      api/
      detectors/
        image/
        audio/
        language/
      db/
      models/
      services/
    tests/
    requirements.txt
  data/
    uploads/
  docs/
    workflows/
    architecture/
  .gitignore
  .env.example

Lý do cấu trúc này tốt cho 3 người
- dễ tách module
- dễ review code
- dễ tránh conflict
- dễ confirm module nào đang làm gì
- dễ tích hợp sau này

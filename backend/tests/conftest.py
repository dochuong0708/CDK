import os
import tempfile

# Test chạy offline: dùng fallback TF-IDF thay vì tải bi-encoder, và DB tạm thay vì DB trong repo.
os.environ.setdefault("DEEPGUARD_EMBEDDING_MODEL", "off")
os.environ.setdefault(
    "DEEPGUARD_DB_PATH", os.path.join(tempfile.mkdtemp(prefix="deepguard-test-"), "test.db")
)

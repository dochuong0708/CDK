"""Keep API tests away from the user's local history database and uploads."""
import os
import tempfile
from pathlib import Path

_test_directory = tempfile.TemporaryDirectory(prefix="deepguard-tests-")
os.environ["DEEPGUARD_DB_PATH"] = str(Path(_test_directory.name) / "history.db")
os.environ["DEEPGUARD_UPLOAD_DIR"] = str(Path(_test_directory.name) / "uploads")


def pytest_sessionfinish(session, exitstatus):
    import sys
    module = sys.modules.get("app.main")
    if module is not None:
        module.store.connection.close()
    _test_directory.cleanup()

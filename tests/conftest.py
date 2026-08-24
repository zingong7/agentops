import os
import tempfile
from pathlib import Path

# Point the app at a throwaway SQLite file before anything imports app.db.
_tmp = Path(tempfile.mkdtemp(prefix="agentops-test-")) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{_tmp}"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
os.environ["SEARCH_BACKEND"] = "local"

# Create the schema up front so tests that touch memory don't depend on the API
# tests running first.
from app.db import init_db  # noqa: E402

init_db()

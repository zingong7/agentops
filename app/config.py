import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _abs(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


class Settings:
    def __init__(self) -> None:
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql+psycopg://agentops:agentops@localhost:5432/agentops"
        )
        # Model IDs: claude-opus-5 | claude-sonnet-5 | claude-haiku-4-5
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
        self.chat_model = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5")

        # max_tokens covers thinking + visible text on the 5-series, so keep it roomy
        # for the pipeline and tight for chat (chat is on the latency budget).
        self.max_tokens = int(os.getenv("MAX_TOKENS", "8000"))
        self.chat_max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1024"))

        self.search_backend = os.getenv("SEARCH_BACKEND", "local")  # local | tavily
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        # Resolved against the repo root: uvicorn and the eval harness get started
        # from different working directories and a relative path silently finds
        # nothing.
        self.corpus_dir = _abs(os.getenv("CORPUS_DIR", "data/corpus"))

        self.results_per_query = int(os.getenv("RESULTS_PER_QUERY", "4"))
        self.max_revisions = int(os.getenv("MAX_REVISIONS", "2"))
        self.confidence_floor = float(os.getenv("CONFIDENCE_FLOOR", "0.85"))
        self.history_turns = int(os.getenv("HISTORY_TURNS", "10"))

        # Hard ceiling on spend for the process. 0 disables the cap.
        cap = float(os.getenv("MAX_SPEND_USD", "5"))
        self.max_spend_usd = cap if cap > 0 else None


@lru_cache
def get_settings() -> Settings:
    return Settings()

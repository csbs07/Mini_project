"""Configuration and settings for Literature Search & RAG Chatbot."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
FULLTEXT_CACHE_DIR: Path = DATA_DIR / "fulltext"
FULLTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Polite pool email for OpenAlex
OPENALEX_MAILTO: str = os.getenv("OPENALEX_MAILTO", "researcher@example.com")

# LLM API Configuration (Groq API)
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
LLM_CHAT_MODEL: str = os.getenv("LLM_CHAT_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))

# Cost and Rate Limits
CHAT_SESSION_COST_CAP_USD: float = float(os.getenv("CHAT_SESSION_COST_CAP_USD", "1.00"))
CHAT_MAX_UPLOADS_PER_SESSION: int = int(os.getenv("CHAT_MAX_UPLOADS_PER_SESSION", "5"))
CHAT_MAX_MESSAGE_CHARS: int = int(os.getenv("CHAT_MAX_MESSAGE_CHARS", "2000"))
MAX_TOOL_CALLS_PER_TURN: int = 5
MAX_RETRIEVAL_CHUNKS: int = 6

# Estimated pricing per 1M tokens (Groq llama-3.3-70b defaults: $0.59 input, $0.79 output)
# If llama-3.1-8b is used: $0.05 input, $0.08 output
PRICING_PER_1M = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "default": {"input": 0.59, "output": 0.79},
}

# Download & Ingestion Limits
MAX_PDF_BYTES: int = 30 * 1024 * 1024  # 30 MB
MAX_PDF_PAGES: int = 100
PDF_DOWNLOAD_TIMEOUT_SECONDS: int = 30
MAX_REDIRECTS: int = 3
HOST_ALLOWLIST_STR: str = os.getenv("HOST_ALLOWLIST", "").strip()
HOST_ALLOWLIST: list[str] | None = [h.strip().lower() for h in HOST_ALLOWLIST_STR.split(",") if h.strip()] if HOST_ALLOWLIST_STR else None

# Embedding Model
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RRF_K: int = 60

# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_RPM", "30"))

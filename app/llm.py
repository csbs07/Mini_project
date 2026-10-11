"""LLM orchestrator (Groq API): tool use loop, token streaming, cost tracking, and rate limiting."""
import time
from typing import Any

import groq

from .config import (
    CHAT_SESSION_COST_CAP_USD,
    GROQ_API_KEY,
    PRICING_PER_1M,
)

# Tool definitions for Groq
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search OpenAlex for relevant academic papers based on a research idea, keywords, or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query terms or research topic",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper",
            "description": "Get detailed metadata and abstract for a specific paper by its OpenAlex ID or label (e.g. W123, P1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The OpenAlex ID or paper label",
                    }
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_citations",
            "description": "Expand citation graph by finding referenced papers for a paper ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The OpenAlex ID or paper label",
                    }
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": "Ensure full-text ingestion of a paper. Returns status (full_text | abstract_only with reason), page count, and detected section list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Paper label (e.g. P1) or OpenAlex ID",
                    }
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_paper_text",
            "description": "Hybrid BM25 + dense search over extracted full-text chunks from ingested papers. Returns passage text, P#, page numbers, and section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific question or keywords to locate inside the full text",
                    },
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of paper labels (e.g. ['P1', 'P2']) or OpenAlex IDs to restrict search",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of chunks to return (up to 6)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


class SessionState:
    """State tracking for a user chat session: cost, uploads, paper pool, chunk index."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.total_cost_usd: float = 0.0
        self.upload_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.messages: list[dict[str, Any]] = []
        self.papers: dict[str, dict[str, Any]] = {}  # paper_id -> paper_data
        self.paper_label_to_id: dict[str, str] = {}  # "P1" -> paper_id

    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Update token counts and calculate estimated cost in USD."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        pricing = PRICING_PER_1M.get(model, PRICING_PER_1M["default"])
        turn_cost = (
            (prompt_tokens * pricing["input"] / 1_000_000.0)
            + (completion_tokens * pricing["output"] / 1_000_000.0)
        )
        self.total_cost_usd += turn_cost
        return turn_cost

    def is_cost_cap_exceeded(self) -> bool:
        return self.total_cost_usd >= CHAT_SESSION_COST_CAP_USD


class RateLimiter:
    """Sliding-window in-memory per-IP rate limiter."""

    def __init__(self, max_requests_per_minute: int = 30):
        self.max_rpm = max_requests_per_minute
        self.ip_timestamps: dict[str, list[float]] = {}

    def check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        timestamps = self.ip_timestamps.setdefault(client_ip, [])
        # Prune old timestamps
        self.ip_timestamps[client_ip] = [ts for ts in timestamps if ts > window_start]
        if len(self.ip_timestamps[client_ip]) >= self.max_rpm:
            return False
        self.ip_timestamps[client_ip].append(now)
        return True


def get_groq_client() -> groq.Groq | None:
    """Initialize Groq client if API key is configured."""
    key = GROQ_API_KEY
    if not key or not key.strip():
        return None
    return groq.Groq(api_key=key.strip())

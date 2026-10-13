"""SQLite database initialization and thread-safe data access layer for users, sessions, and chat history."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DATA_DIR

DB_PATH: Path = DATA_DIR / "app.db"


def get_db_connection() -> sqlite3.Connection:
    """Create a SQLite connection with foreign keys and row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_cursor():
    """Context manager for executing database commands safely with auto-commit."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database tables and indexes if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with db_cursor() as cur:
        # Users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User auth sessions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        """)

        # Conversations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Messages
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for fast querying
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_msgs_conv ON messages(conversation_id, created_at ASC)")


# ==============================================================================
# User Management Helpers
# ==============================================================================

def create_user(username: str, email: str, password_hash: str) -> int:
    """Insert a new user. Raises sqlite3.IntegrityError if username/email already exists."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username.strip(), email.strip().lower(), password_hash),
        )
        return cur.lastrowid


def get_user_by_username_or_email(identifier: str) -> dict[str, Any] | None:
    """Find a user by username or email case-insensitively."""
    ident = identifier.strip()
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, username, email, password_hash, created_at FROM users WHERE username = ? OR email = ?",
            (ident, ident.lower()),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Get user by primary ID."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ==============================================================================
# Auth Session Helpers
# ==============================================================================

def create_user_session(user_id: int, token: str, expires_at: datetime) -> None:
    """Store an active authentication session."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO user_sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at.isoformat()),
        )


def get_user_by_session_token(token: str) -> dict[str, Any] | None:
    """Validate a session token and return the associated user if not expired."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.username, u.email, u.created_at, s.expires_at
            FROM user_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token, now_iso),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def delete_user_session(token: str) -> None:
    """Delete an auth session on logout."""
    with db_cursor() as cur:
        cur.execute("DELETE FROM user_sessions WHERE token = ?", (token,))


def delete_expired_sessions() -> None:
    """Garbage collect expired sessions."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_cursor() as cur:
        cur.execute("DELETE FROM user_sessions WHERE expires_at <= ?", (now_iso,))


# ==============================================================================
# Conversation History Helpers
# ==============================================================================

def get_or_create_conversation(conv_id: str, user_id: int, default_title: str = "New Research Chat") -> dict[str, Any]:
    """Ensure a conversation exists for the given user, creating it if needed."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        )
        row = cur.fetchone()
        if row:
            return dict(row)

        cur.execute(
            "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
            (conv_id, user_id, default_title),
        )
        cur.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,),
        )
        return dict(cur.fetchone())


def get_conversations_for_user(user_id: int) -> list[dict[str, Any]]:
    """List all conversations belonging to the given user, ordered by most recently updated."""
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_conversation(conv_id: str, user_id: int) -> dict[str, Any] | None:
    """Retrieve a conversation ensuring it belongs to the user."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def update_conversation_title(conv_id: str, user_id: int, title: str) -> bool:
    """Rename a conversation belonging to user_id."""
    clean_title = title.strip()[:100]
    if not clean_title:
        return False
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with db_cursor() as cur:
        cur.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (clean_title, now_iso, conv_id, user_id),
        )
        return cur.rowcount > 0


def delete_conversation(conv_id: str, user_id: int) -> bool:
    """Delete a conversation and its cascaded messages."""
    with db_cursor() as cur:
        cur.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        )
        return cur.rowcount > 0


def touch_conversation(conv_id: str, user_id: int, title_update: str | None = None) -> None:
    """Update conversation updated_at timestamp, and optionally update title if initial title is default."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with db_cursor() as cur:
        if title_update:
            cur.execute(
                """
                UPDATE conversations
                SET updated_at = ?,
                    title = CASE WHEN title = 'New Research Chat' THEN ? ELSE title END
                WHERE id = ? AND user_id = ?
                """,
                (now_iso, title_update.strip()[:80], conv_id, user_id),
            )
        else:
            cur.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ? AND user_id = ?",
                (now_iso, conv_id, user_id),
            )


# ==============================================================================
# Message Helpers
# ==============================================================================

def add_message(conv_id: str, role: str, content: str) -> int:
    """Insert a user or assistant message."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conv_id, role, content),
        )
        return cur.lastrowid


def get_messages_for_conversation(conv_id: str, user_id: int) -> list[dict[str, Any]]:
    """Retrieve all chronological messages for a conversation, verifying user ownership."""
    with db_cursor() as cur:
        # First verify conversation belongs to user
        cur.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        )
        if not cur.fetchone():
            return []

        cur.execute(
            "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        )
        return [dict(r) for r in cur.fetchall()]

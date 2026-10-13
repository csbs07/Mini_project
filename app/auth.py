"""Authentication logic: bcrypt password hashing, session tokens, and FastAPI dependencies."""
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import HTTPException, Request, Response, status

from .db import (
    create_user_session,
    delete_user_session,
    get_user_by_session_token,
)

COOKIE_NAME = "litsearch_session"
SESSION_DURATION_DAYS = 7


def hash_password(password: str) -> str:
    """Hash a plaintext password with a random salt using bcrypt."""
    pw_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def validate_signup_inputs(username: str, email: str, password: str) -> str | None:
    """Validate username, email, and password length. Returns error string if invalid, else None."""
    u = username.strip()
    if len(u) < 3 or len(u) > 32:
        return "Username must be between 3 and 32 characters."
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", u):
        return "Username may only contain letters, numbers, dots, hyphens, and underscores."

    e = email.strip()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e):
        return "Please provide a valid email address."

    if len(password) < 8:
        return "Password must be at least 8 characters long."

    return None


def create_user_session_token(user_id: int) -> tuple[str, datetime]:
    """Generate a secure random session token and persist it in the database."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DURATION_DAYS)
    create_user_session(user_id=user_id, token=token, expires_at=expires_at)
    return token, expires_at


def set_auth_cookie(response: Response, token: str, expires_at: datetime) -> None:
    """Set secure HttpOnly cookie on response."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        expires=expires_at,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    """Clear session cookie on response."""
    response.delete_cookie(key=COOKIE_NAME, path="/")


def extract_token_from_request(request: Request) -> str | None:
    """Extract session token from cookie or Authorization header."""
    cookie_token = request.cookies.get(COOKIE_NAME)
    if cookie_token:
        return cookie_token

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return None


def get_optional_user(request: Request) -> dict[str, Any] | None:
    """Return user dict if request has valid session, else None."""
    token = extract_token_from_request(request)
    if not token:
        return None
    return get_user_by_session_token(token)


def get_current_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency: require an authenticated user, raising 401 if missing/invalid."""
    user = get_optional_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import APIKeyCookie
from redis import Redis

from .config import Settings, get_settings

SESSION_COOKIE = "neon_admin_session"
cookie_scheme = APIKeyCookie(name=SESSION_COOKIE, auto_error=False)


def redis_client(settings: Settings | None = None) -> Redis:
    return Redis.from_url((settings or get_settings()).redis_url, decode_responses=True)


def _session_key(token: str) -> str:
    return f"session:{hashlib.sha256(token.encode()).hexdigest()}"


def verify_admin_password(password: str, settings: Settings) -> bool:
    if not settings.admin_password_hash_b64:
        return False
    try:
        encoded = base64.b64decode(settings.admin_password_hash_b64).decode()
        return PasswordHasher().verify(encoded, password)
    except (ValueError, InvalidHashError, VerifyMismatchError):
        return False


@dataclass(frozen=True)
class AdminSession:
    token: str
    csrf: str


def create_session(response: Response, settings: Settings) -> AdminSession:
    token = secrets.token_urlsafe(48)
    csrf = secrets.token_urlsafe(32)
    redis_client(settings).setex(_session_key(token), settings.session_ttl_seconds, json.dumps({"csrf": csrf}))
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )
    return AdminSession(token=token, csrf=csrf)


def destroy_session(response: Response, token: str | None, settings: Settings) -> None:
    if token:
        redis_client(settings).delete(_session_key(token))
    response.delete_cookie(SESSION_COOKIE, path="/", secure=settings.cookie_secure, samesite="strict")


def require_admin(token: str | None = Depends(cookie_scheme)) -> AdminSession:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Administrator login required.")
    raw = redis_client().get(_session_key(token))
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Administrator session expired.")
    return AdminSession(token=token, csrf=json.loads(raw)["csrf"])


def require_csrf(request: Request, session: AdminSession = Depends(require_admin)) -> AdminSession:
    if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), session.csrf):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed.")
    return session

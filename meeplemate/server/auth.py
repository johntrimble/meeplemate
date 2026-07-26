"""Firebase Auth integration for the MeepleMate FastAPI backend.

Normal mode:
  Validates the Firebase ID token supplied as `Authorization: Bearer <token>`.
  Uses firebase_admin to verify the token signature, expiry, and project ID.

Bypass mode (MM_AUTH_BYPASS=true):
  Returns the user encoded in MM_AUTH_BYPASS_USER (JSON) without validating
  any token. No Firebase credentials are required. Intended for local dev/testing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import structlog
import firebase_admin
import firebase_admin.auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# User dataclass returned by get_current_user
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthUser:
    uid: str
    email: Optional[str]
    name: Optional[str]


# ---------------------------------------------------------------------------
# Firebase Admin SDK initialisation (lazy, once per process)
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def _bypass_enabled() -> bool:
    return os.environ.get("MM_AUTH_BYPASS", "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def _get_firebase_app() -> firebase_admin.App:
    """Initialise and return the default Firebase Admin app (once)."""
    from meeplemate.config import Config
    cfg = Config()
    fb = cfg.firebase

    if fb.emulator_host:
        os.environ.setdefault("FIREBASE_AUTH_EMULATOR_HOST", fb.emulator_host)

    credential: firebase_admin.credentials.Base = None  # type: ignore[assignment]

    if fb.service_account_path:
        credential = firebase_admin.credentials.Certificate(fb.service_account_path)
    elif fb.service_account_json:
        raw = base64.b64decode(fb.service_account_json).decode()
        info = json.loads(raw)
        credential = firebase_admin.credentials.Certificate(info)
    else:
        # Fall back to Application Default Credentials (works on GCP / Azure with
        # Workload Identity, or locally if `gcloud auth application-default login`
        # has been run).
        credential = firebase_admin.credentials.ApplicationDefault()

    return firebase_admin.initialize_app(credential, {"projectId": fb.project_id})


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthUser:
    """FastAPI dependency that returns the authenticated user.

    Raises HTTP 401 if the token is missing or invalid.
    """
    # --- Bypass mode ---
    if _bypass_enabled():
        raw = os.environ.get("MM_AUTH_BYPASS_USER", '{"uid":"bypass","email":null,"name":"Bypass User"}')
        data = json.loads(raw)
        return AuthUser(uid=data["uid"], email=data.get("email"), name=data.get("name"))

    # --- Normal mode: validate Firebase ID token ---
    if credentials is None or not credentials.credentials:
        logger.warning("auth.missing_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        # Ensure the app is initialised before verifying.
        try:
            app = firebase_admin.get_app()
        except ValueError:
            app = _get_firebase_app()

        decoded = firebase_admin.auth.verify_id_token(token, app=app)
    except firebase_admin.auth.InvalidIdTokenError as exc:
        logger.warning("auth.invalid_token", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except firebase_admin.auth.ExpiredIdTokenError as exc:
        logger.warning("auth.token_expired", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception as exc:
        logger.exception("auth.token_validation_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return AuthUser(
        uid=decoded["uid"],
        email=decoded.get("email"),
        name=decoded.get("name"),
    )


# ---------------------------------------------------------------------------
# Account deletion
# ---------------------------------------------------------------------------


async def delete_firebase_user(uid: str) -> None:
    """Delete the Firebase Auth user, so their credentials stop working.

    Idempotent: a uid that is already gone is treated as success, so the caller
    can safely retry after a partial failure. Anything else propagates.

    No-op in bypass mode, which has no Firebase project to talk to.
    """
    if _bypass_enabled():
        logger.info("auth.delete_user_skipped_bypass", uid=uid)
        return

    try:
        app = firebase_admin.get_app()
    except ValueError:
        app = _get_firebase_app()

    def _delete() -> None:
        try:
            firebase_admin.auth.delete_user(uid, app=app)
        except firebase_admin.auth.UserNotFoundError:
            logger.info("auth.delete_user_already_gone", uid=uid)

    # The Admin SDK is blocking; keep it off the event loop.
    await asyncio.to_thread(_delete)

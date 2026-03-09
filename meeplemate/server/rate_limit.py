"""Token-based rate limiting for the MeepleMate API.

Enforces per-user and app-wide token consumption limits across three rolling
windows (8H, 7D, 30D).  Follows the IETF draft ratelimit-headers spec for
response headers.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field

from meeplemate.db.datalayer import BaseDataLayer, UserRecord, WindowStats
from meeplemate.server.auth import AuthUser, get_current_user

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rolling-window definitions: (name, duration_in_seconds)
# ---------------------------------------------------------------------------

WINDOWS: list[tuple[str, int]] = [
    ("8H", int(timedelta(hours=8).total_seconds())),
    ("7D", int(timedelta(days=7).total_seconds())),
    ("30D", int(timedelta(days=30).total_seconds())),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class RateLimitConfig(BaseModel):
    """Token-based rate limiting configuration (loaded from MM_RATE_LIMIT__* env vars)."""

    app_8h: int = Field(default=10_000_000, description="App-wide 8-hour token limit")
    app_7d: int = Field(default=50_000_000, description="App-wide 7-day token limit")
    app_30d: int = Field(default=150_000_000, description="App-wide 30-day token limit")
    user_8h: int = Field(default=50_000, description="Per-user 8-hour token limit")
    user_7d: int = Field(default=200_000, description="Per-user 7-day token limit")
    user_30d: int = Field(default=500_000, description="Per-user 30-day token limit")
    estimated_tokens_per_request: int = Field(
        default=2_000,
        description="Estimated tokens per request used for pre-flight limit check",
    )
    output_token_multiplier: int = Field(
        default=4,
        description="Weight applied to output (completion) tokens relative to input tokens",
    )

    def user_limit(self, window: str) -> int:
        return getattr(self, f"user_{window.lower()}")

    def app_limit(self, window: str) -> int:
        return getattr(self, f"app_{window.lower()}")


# ---------------------------------------------------------------------------
# Token counting LangChain callback
# ---------------------------------------------------------------------------

class TokenCountingCallback(BaseCallbackHandler):
    """Accumulates LLM token usage across all calls within a single request.

    Output (completion) tokens are weighted by ``output_token_multiplier`` to
    reflect their higher cost relative to input tokens.

    Falls back to ``estimated`` if no real usage data is reported by the model.
    """

    def __init__(self, estimated: int = 0, output_token_multiplier: int = 4) -> None:
        super().__init__()
        self.total: int = 0
        self.estimated = estimated
        self.output_token_multiplier = output_token_multiplier
        self._has_real_count = False

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # OpenAI-style: usage in llm_output["token_usage"]
        usage = (response.llm_output or {}).get("token_usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        if prompt or completion:
            self._has_real_count = True
            self.total += prompt + completion * self.output_token_multiplier

    @property
    def tokens(self) -> int:
        """Actual total if the model reported usage, otherwise the estimate."""
        return self.total if self._has_real_count else self.estimated


# ---------------------------------------------------------------------------
# Rate limit state
# ---------------------------------------------------------------------------

@dataclass
class WindowState:
    name: str
    duration_seconds: int
    limit: int
    used: int
    oldest_record_at: Optional[datetime]

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def reset_at(self) -> datetime:
        if self.oldest_record_at is None:
            return datetime.now(UTC) + timedelta(seconds=self.duration_seconds)
        return self.oldest_record_at + timedelta(seconds=self.duration_seconds)


@dataclass
class RateLimitState:
    windows: list[WindowState]

    def headers(self) -> dict[str, str]:
        """Build IETF draft ratelimit response headers."""
        if not self.windows:
            return {}
        now = datetime.now(UTC)
        most_restrictive = min(self.windows, key=lambda w: w.remaining)
        reset_secs = max(0, int((most_restrictive.reset_at - now).total_seconds()))
        policy = ", ".join(f"{w.limit};w={w.duration_seconds}" for w in self.windows)
        return {
            "RateLimit-Limit": str(most_restrictive.limit),
            "RateLimit-Remaining": str(most_restrictive.remaining),
            "RateLimit-Reset": str(reset_secs),
            "RateLimit-Policy": policy,
        }


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Checks token usage against per-user and app-wide rolling-window limits."""

    def __init__(self, config: RateLimitConfig, data_layer: BaseDataLayer) -> None:
        self.config = config
        self._data_layer = data_layer

    def _user_limit(self, user: UserRecord, window: str) -> int:
        overrides = user.metadata.get("rate_limits", {})
        return int(overrides.get(window, self.config.user_limit(window)))

    async def check(self, user: UserRecord) -> RateLimitState:
        """Check limits, atomically reserve tokens for the user, and return rate limit state.

        App-wide limits are checked first (best-effort, no lock).
        Per-user limits are checked atomically via pg_advisory_xact_lock; if the user passes,
        an estimated-token reservation is inserted in the same transaction.

        Raises ``HTTPException(429)`` if any window is exceeded.
        """
        now = datetime.now(UTC)
        estimated = self.config.estimated_tokens_per_request
        window_params = [(name, secs, now - timedelta(seconds=secs)) for name, secs in WINDOWS]

        # --- App-wide check (best-effort, no lock) ---
        app_stats_list = await asyncio.gather(*[
            self._data_layer.get_app_window_stats(since)
            for _, _, since in window_params
        ])

        for i, (name, duration_secs, _) in enumerate(window_params):
            app_stats: WindowStats = app_stats_list[i]
            app_limit = self.config.app_limit(name)
            if app_stats.total_tokens + estimated > app_limit:
                oldest = app_stats.oldest_recorded_at
                reset_at = (
                    oldest + timedelta(seconds=duration_secs) if oldest
                    else now + timedelta(seconds=duration_secs)
                )
                # Build a minimal state for headers (zero user usage — we haven't fetched it yet)
                placeholder_state = RateLimitState(windows=[
                    WindowState(
                        name=n,
                        duration_seconds=s,
                        limit=self._user_limit(user, n),
                        used=0,
                        oldest_record_at=None,
                    )
                    for n, s, _ in window_params
                ])
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": (
                            f"The service has reached its token quota for the {name} window. "
                            f"Please try again at {reset_at.isoformat()}."
                        ),
                        "window": name,
                        "limit": app_limit,
                        "used": app_stats.total_tokens,
                        "resets_at": reset_at.isoformat(),
                    },
                    headers=placeholder_state.headers(),
                )

        # --- Per-user atomic check + reservation ---
        user_limits = {name: self._user_limit(user, name) for name, _, _ in window_params}
        per_window_params = [(name, since) for name, _, since in window_params]

        user_stats_list = await self._data_layer.check_and_reserve_user(
            user_id=user.uid,
            window_params=per_window_params,
            estimated=estimated,
            user_limits=user_limits,
        )

        # Build state for headers
        window_states = [
            WindowState(
                name=name,
                duration_seconds=duration_secs,
                limit=self._user_limit(user, name),
                used=user_stats.total_tokens,
                oldest_record_at=user_stats.oldest_recorded_at,
            )
            for (name, duration_secs, _), user_stats in zip(window_params, user_stats_list)
        ]
        state = RateLimitState(windows=window_states)

        # Check whether the returned stats indicate a violation
        for i, (name, duration_secs, _) in enumerate(window_params):
            user_stats = user_stats_list[i]
            user_limit = self._user_limit(user, name)
            if user_stats.total_tokens + estimated > user_limit:
                oldest = user_stats.oldest_recorded_at
                reset_at = (
                    oldest + timedelta(seconds=duration_secs) if oldest
                    else now + timedelta(seconds=duration_secs)
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": (
                            f"You have used your token quota for the {name} window. "
                            f"Your limit resets at {reset_at.isoformat()}."
                        ),
                        "window": name,
                        "limit": user_limit,
                        "used": user_stats.total_tokens,
                        "resets_at": reset_at.isoformat(),
                    },
                    headers=state.headers(),
                )

        return state


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_db_user(
    request: Request,
    auth_user: AuthUser = Depends(get_current_user),
) -> UserRecord:
    """Upsert the Firebase user into the local DB and return the UserRecord."""
    data_layer: BaseDataLayer = request.app.state.deps.data_layer
    return await data_layer.upsert_user(auth_user.uid, auth_user.email, auth_user.name)


async def check_rate_limit(
    request: Request,
    db_user: UserRecord = Depends(get_db_user),
) -> RateLimitState:
    """FastAPI dependency: raises 429 if the user (or app) is over any token limit."""
    rate_limiter: RateLimiter = request.app.state.deps.rate_limiter
    return await rate_limiter.check(db_user)

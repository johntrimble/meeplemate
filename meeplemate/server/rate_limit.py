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
from pydantic import BaseModel, Field, model_validator

from meeplemate.db.datalayer import BaseDataLayer, UserRecord, WindowStats
from meeplemate.server.deps import get_db_user

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

# Each window as a fraction of the 30-day period. A share equal to this is even
# pacing; anything above it permits bursting. Quoted in the field descriptions.
WINDOW_FRACTION: dict[str, float] = {name: secs / (30 * 24 * 3600) for name, secs in WINDOWS}


class RateLimitConfig(BaseModel):
    """Token-based rate limiting configuration (loaded from MM_RATE_LIMIT__* env vars).

    Per-user quotas are *derived*, not set by hand: give this a 30-day spend cap
    and the provider's token prices, and the three window quotas and the output
    weight all follow.

    The ``*_budget_30d_usd`` caps are the real limiters: each is a spend
    ceiling in USD, which the 30D quota restates in tokens at the input price.
    The 8H and 7D quotas exist only to control how fast that ceiling can be
    reached, so each is written as the share of
    the 30D cap that one window may spend.  A share equal to the window's own
    fraction of the month (8H = 1.1%, 7D = 23.3%) is even pacing; more permits
    bursting, and the shares must be ordered ``8H <= 7D <= 100%``.

    ``output_token_multiplier`` is what makes the dollar figure exact.  Set to
    the output/input price ratio, one weighted token (``input + multiplier *
    output``) is worth exactly ``cost_per_m_input_usd`` per million *whatever
    the input/output mix*, so a quota converts back to dollars with no error
    term.  Any other weight turns the dollar value of a quota into a range that
    moves with workload shape.

    User defaults are sized for ~12 questions in an 8-hour game session and one
    session a week, measured against the 2026-07-07 eval group.  The app-wide
    ceiling is a hard cap on spend; its shares are loose on 8H, since traffic
    bunches into US evenings and an app-wide 429 hits everyone at once, and
    tighter on 7D, which is what stops one busy weekend emptying the month.
    """

    # --- Per-user quotas are derived from these ---

    user_budget_30d_usd: float = Field(
        default=0.40, gt=0,
        description="Per-user spend cap over the rolling 30-day window, in USD",
    )
    # Qwen3.6 35b a3b prices.
    cost_per_m_input_usd: float = Field(
        default=0.14, gt=0, description="Provider price per million input tokens, in USD",
    )
    cost_per_m_output_usd: float = Field(
        default=1.00, ge=0, description="Provider price per million output tokens, in USD",
    )
    # A game session has to fit inside the 8H window, which at one session a
    # week leaves the 7D share only slightly above it.
    user_share_8h: float = Field(
        default=0.28, ge=0.0, le=1.0,
        description=(
            "Share of the 30D cap a user may spend in any 8-hour window "
            f"(even pacing would be {WINDOW_FRACTION['8H']:.1%})"
        ),
    )
    user_share_7d: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description=(
            "Share of the 30D cap a user may spend in any 7-day window "
            f"(even pacing would be {WINDOW_FRACTION['7D']:.1%}); a user drains a "
            "full month no faster than 1/user_share_7d weeks"
        ),
    )

    # --- Pre-flight estimate, measured from eval runs ---

    # Mean per generation run over the 2026-07-07 group of 150 runs:
    #   python script/count_tokens.py 2026-07-07
    # Held as raw input/output rather than one weighted figure so the estimate
    # stays correct when prices, and so the multiplier, change.
    observed_input_tokens_per_request: int = Field(
        default=35_671, ge=0, description="Mean input tokens per request, from eval runs",
    )
    observed_output_tokens_per_request: int = Field(
        default=3_736, ge=0, description="Mean output tokens per request, from eval runs",
    )

    # --- App-wide quotas are derived from these ---

    app_budget_30d_usd: float = Field(
        default=50.00, gt=0,
        description="Hard app-wide spend ceiling over the rolling 30-day window, in USD",
    )
    # The 30D ceiling fixes the worst-case bill, so these only decide how fast
    # it may be spent.
    app_share_8h: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description=(
            "Share of the 30D cap the app may spend in any 8-hour window "
            f"(even pacing would be {WINDOW_FRACTION['8H']:.1%})"
        ),
    )
    app_share_7d: float = Field(
        default=0.35, ge=0.0, le=1.0,
        description=(
            "Share of the 30D cap the app may spend in any 7-day window "
            f"(even pacing would be {WINDOW_FRACTION['7D']:.1%}); the app drains a "
            "full month no faster than 1/app_share_7d weeks"
        ),
    )

    @property
    def output_token_multiplier(self) -> float:
        """Weight applied to output tokens relative to input tokens."""
        return self.cost_per_m_output_usd / self.cost_per_m_input_usd

    # Rounded, not truncated: binary floating point puts exact ratios a hair
    # either side of the integer (0.01/0.10 * 1e6 is 99999.99999999999), and
    # truncating turns clean inputs into off-by-one quotas.

    @property
    def user_30d(self) -> int:
        """The 30-day budget, in weighted tokens."""
        return round(self.user_budget_30d_usd / self.cost_per_m_input_usd * 1_000_000)

    @property
    def user_7d(self) -> int:
        return round(self.user_30d * self.user_share_7d)

    @property
    def user_8h(self) -> int:
        return round(self.user_30d * self.user_share_8h)

    @property
    def app_30d(self) -> int:
        """The app-wide 30-day ceiling, in weighted tokens."""
        return round(self.app_budget_30d_usd / self.cost_per_m_input_usd * 1_000_000)

    @property
    def app_7d(self) -> int:
        return round(self.app_30d * self.app_share_7d)

    @property
    def app_8h(self) -> int:
        return round(self.app_30d * self.app_share_8h)

    @property
    def estimated_tokens_per_request(self) -> int:
        """Weighted tokens a request is assumed to cost, for the pre-flight check.

        Reserved up front and reconciled against real usage once the request
        finishes, so it governs the safety margin rather than the final charge:
        too high denies requests that would have fit, too low lets concurrent
        requests overshoot a window.
        """
        return round(
            self.observed_input_tokens_per_request
            + self.output_token_multiplier * self.observed_output_tokens_per_request
        )

    @model_validator(mode="after")
    def _check_quotas_nest(self) -> "RateLimitConfig":
        """Quotas that do not nest silently make the tighter limit unreachable."""
        for scope, share_8h, share_7d in (
            ("user", self.user_share_8h, self.user_share_7d),
            ("app", self.app_share_8h, self.app_share_7d),
        ):
            if share_8h > share_7d:
                raise ValueError(
                    f"{scope}_share_8h ({share_8h:.1%}) exceeds {scope}_share_7d "
                    f"({share_7d:.1%}); the 7D window would never bind"
                )
        # A per-user quota above the app-wide one for the same window is
        # redundant rather than wrong — the app limit simply binds first — and
        # it is a reasonable way to run with per-user limiting effectively off.
        # Worth surfacing, not worth refusing to start over.
        for window in ("8H", "7D", "30D"):
            if self.user_limit(window) > self.app_limit(window):
                logger.warning(
                    "Per-user %s quota (%s) exceeds the app-wide one (%s); the app-wide "
                    "limit will bind first and the per-user limit will never apply.",
                    window, f"{self.user_limit(window):,}", f"{self.app_limit(window):,}",
                )
        return self

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
    reflect their higher cost relative to input tokens, so the running total is
    in input-token-equivalents.  ``api.py`` passes
    ``RateLimitConfig.output_token_multiplier``, which is the output/input
    price ratio, so the total converts to dollars at a fixed rate whatever the
    input/output mix.

    Falls back to ``estimated`` if no real usage data is reported by the model.
    """

    # Defaults to 1.0 — unweighted — rather than shadowing the configured
    # weight, which is a price ratio and has no meaningful fixed value.
    def __init__(self, estimated: int = 0, output_token_multiplier: float = 1.0) -> None:
        super().__init__()
        self.total: float = 0.0
        self.estimated = estimated
        self.output_token_multiplier = output_token_multiplier
        self._has_real_count = False

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # Non-streaming OpenAI-style responses report usage in
        # llm_output["token_usage"] as prompt/completion tokens.
        usage = (response.llm_output or {}).get("token_usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)

        # Streaming responses leave llm_output empty; usage instead rides on each
        # message's usage_metadata (input/output tokens), populated because the
        # chat model sets stream_usage=True. Fall back to summing that.
        if not (prompt or completion):
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage_metadata = getattr(message, "usage_metadata", None) or {}
                    prompt += int(usage_metadata.get("input_tokens", 0) or 0)
                    completion += int(usage_metadata.get("output_tokens", 0) or 0)

        if prompt or completion:
            self._has_real_count = True
            self.total += prompt + completion * self.output_token_multiplier

    @property
    def tokens(self) -> int:
        """Actual total if the model reported usage, otherwise the estimate.

        Weighting output by a price ratio makes ``total`` a real number of
        input-token-equivalents rather than a token count, so it is rounded
        here — the boundary where it is persisted to an integer column.
        """
        return round(self.total) if self._has_real_count else self.estimated


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
            user_id=user.id,
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

async def check_rate_limit(
    request: Request,
    db_user: UserRecord = Depends(get_db_user),
) -> RateLimitState:
    """FastAPI dependency: raises 429 if the user (or app) is over any token limit."""
    rate_limiter: RateLimiter = request.app.state.deps.rate_limiter
    return await rate_limiter.check(db_user)

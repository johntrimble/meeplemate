# Rate Limits

MeepleMate meters **LLM token spend**, not request counts. Limits are enforced per-user and app-wide across three rolling windows — **8H**, **7D**, **30D** — and every quota is *derived* from a dollar budget rather than set by hand.

Configuration lives in [`RateLimitConfig`](../meeplemate/server/rate_limit.py); enforcement lives in `RateLimiter` in the same module.

---

## The idea in one paragraph

You give the system a **30-day spend cap in dollars** and the **provider's token prices**. From those it works out how many tokens that budget buys, and the 8H and 7D quotas are then expressed as **the share of the 30-day cap** that a single window may spend. There are two independent budgets: one per user, one for the whole app.

```
budget ($) ÷ input price     → 30D quota (tokens)
30D quota × share_8h         → 8H quota
30D quota × share_7d         → 7D quota
output price ÷ input price   → output_token_multiplier
```

### Why tokens are "weighted"

Output tokens cost more than input tokens, so usage is recorded as **weighted tokens**:

```
weighted = input_tokens + output_token_multiplier × output_tokens
```

Because the multiplier is set to the *output/input price ratio*, one weighted token is worth exactly `cost_per_m_input_usd` per million **regardless of the input/output mix**. That is what lets a token quota convert back to a dollar figure with no error term. The quantity being counted is really "input-token-equivalents" — a cost proxy that happens to be denominated in tokens.

---

## Configuration reference

All settings are nested under `rate_limit`. Ten values are settable:

### Per-user budget

| Setting | Default | Meaning |
|---|---|---|
| `user_budget_30d_usd` | `0.40` | Spend cap per user per rolling 30 days. **The real limiter.** |
| `user_share_8h` | `0.28` | Share of that cap spendable in any 8-hour window |
| `user_share_7d` | `0.30` | Share of that cap spendable in any 7-day window |

### App-wide ceiling

| Setting | Default | Meaning |
|---|---|---|
| `app_budget_30d_usd` | `50.00` | **Hard ceiling on total spend** per rolling 30 days |
| `app_share_8h` | `0.10` | Share of the ceiling spendable in any 8-hour window |
| `app_share_7d` | `0.35` | Share of the ceiling spendable in any 7-day window |

### Prices

| Setting | Default | Meaning |
|---|---|---|
| `cost_per_m_input_usd` | `0.14` | Provider price per million input tokens |
| `cost_per_m_output_usd` | `1.00` | Provider price per million output tokens |

Defaults are Qwen3.6 35b a3b prices. Changing these changes every derived quota, since the budgets are fixed in dollars.

### Pre-flight estimate

| Setting | Default | Meaning |
|---|---|---|
| `observed_input_tokens_per_request` | `35_671` | Mean input tokens per question, measured from eval runs |
| `observed_output_tokens_per_request` | `3_736` | Mean output tokens per question |

Held as raw input/output rather than one weighted number so the estimate stays correct when prices change. See [Re-deriving the estimate](#re-deriving-the-estimate).

### Derived — **not settable**

These are read-only properties. Setting them via env var or YAML is **silently ignored**:

`user_8h` · `user_7d` · `user_30d` · `app_8h` · `app_7d` · `app_30d` · `output_token_multiplier` · `estimated_tokens_per_request`

### Reading the shares

A share equal to the window's own fraction of the month is *even pacing*; anything above permits bursting:

| Window | Even pacing | Reading |
|---|---|---|
| 8H | 1.1% | `user_share_8h = 0.28` is 25× even pacing — heavy bursting allowed |
| 7D | 23.3% | `app_share_7d = 0.35` means the app drains a month no faster than ~3 weeks |

The rule `share_8h ≤ share_7d ≤ 100%` is enforced at construction.

---

## How to set them

Two routes, with **env vars winning over YAML**.

### Environment variables

Prefix `MM_`, nested with `__`, case-insensitive:

```bash
MM_RATE_LIMIT__USER_BUDGET_30D_USD=0.75
MM_RATE_LIMIT__APP_BUDGET_30D_USD=25.0
MM_RATE_LIMIT__USER_SHARE_8H=0.30
MM_RATE_LIMIT__COST_PER_M_INPUT_USD=0.14
```

### YAML

Add a `rate_limit:` block to the file `MM_CONFIG_FILE` points at (e.g. `config-dev.yaml`):

```yaml
rate_limit:
  user_budget_30d_usd: 0.75
  app_budget_30d_usd: 25.0
  user_share_8h: 0.30
```

Precedence is **init args → env vars → `.env` → YAML**. Sources deep-merge per key, so a YAML block plus a single env override works as expected. Note `.env` is read by default and beats YAML.

### Checking what you got

```bash
python -c "
from meeplemate.server.rate_limit import RateLimitConfig
c = RateLimitConfig()
usd = lambda t: t * c.cost_per_m_input_usd / 1e6
for scope in ('user', 'app'):
    for w in ('8h', '7d', '30d'):
        tok = getattr(c, f'{scope}_{w}')
        print(f'{scope} {w.upper():<4} {tok:>12,}  \${usd(tok):.2f}')
print('multiplier', round(c.output_token_multiplier, 4))
print('estimate  ', c.estimated_tokens_per_request)
"
```

---

## What the defaults mean

| | share | tokens | dollars |
|---|---:|---:|---:|
| user 8H | 28% | 800,000 | $0.11 |
| user 7D | 30% | 857,143 | $0.12 |
| user 30D | 100% | 2,857,143 | $0.40 |
| app 8H | 10% | 35,714,286 | $5.00 |
| app 7D | 35% | 125,000,000 | $17.50 |
| app 30D | 100% | 357,142,857 | $50.00 |

Measured against the `2026-07-07` eval group, where a question costs ~62,000 weighted tokens on average:

- **~12 questions per 8-hour game session**, with 10 or more in ~89% of sessions
- ~1 session per week per user, ~4.6 sessions per month
- **App spend cannot exceed $50/month**, covering ~124 users at full per-user budget or ~446 sessions
- App-wide, ~44 concurrent game sessions in any 8-hour window

The app 8H share is deliberately loose because traffic bunches into US evenings and an app-wide 429 hits *everyone* at once. The 7D share is what stops one busy weekend emptying the month.

---

## Common changes

**"I want to spend at most $25/month."**
Set `app_budget_30d_usd: 25.0`. Nothing else needs to change — the shares are relative, so the 8H and 7D quotas scale automatically.

**"Users are hitting limits mid-session."**
Raise `user_share_8h` (more of the month in one sitting) or `user_budget_30d_usd` (more overall). Raising the share alone costs nothing extra per month; it only lets the same budget be spent faster.

**"I switched models / prices changed."**
Update `cost_per_m_input_usd` and `cost_per_m_output_usd`. Every quota and the multiplier re-derive automatically, and your dollar budgets stay exactly what you set. Then [re-derive the estimate](#re-deriving-the-estimate).

**"I expect a launch-night traffic spike."**
Raise `app_share_8h` (0.10 → 0.15 buys ~66 concurrent sessions). This cannot increase your maximum monthly bill — only how fast the ceiling is approached — but it does risk the month running dry sooner.

**"One user needs a bigger allowance."**
Use a per-user override rather than changing global config — see below.

---

## Per-user overrides

Individual users can be given raw token quotas via the `metadata` JSONB column on `app_user`, bypassing the derivation entirely:

```sql
UPDATE app_user
SET metadata = '{"rate_limits": {"8H": 2000000, "7D": 3000000, "30D": 10000000}}'
WHERE user_id = '<firebase-uid>';
```

- Values are **raw weighted token counts**, not dollars or shares.
- Window keys are `"8H"`, `"7D"`, `"30D"`.
- Partial overrides work; unlisted windows fall back to the derived value.
- Overrides are **not** checked against the app-wide ceiling — a boosted user still cannot push total spend past `app_budget_30d_usd`.

Useful for giving yourself a larger allowance on a deployed instance without raising everyone's.

---

## Re-deriving the estimate

`estimated_tokens_per_request` is reserved up front and reconciled against real usage once a request finishes, so it governs the **safety margin**, not the final charge. Too high denies requests that would have fit; too low lets concurrent requests overshoot a window.

It is derived from the two `observed_*` settings, which come from eval data:

```bash
python script/count_tokens.py 2026-07-07
```

Take `in/run` and `out/run` from the `OVERALL` row and set `observed_input_tokens_per_request` and `observed_output_tokens_per_request`. See [eval-debugging-guide.md](eval-debugging-guide.md) for generating run groups.

---

## What happens at the limit

Rate limiting is applied by the `check_rate_limit` FastAPI dependency, currently on **`POST /api/chats/{chat_id}/stream`** only. Everything else is unmetered.

Order of checks:

1. **App-wide** — three window queries run concurrently, best-effort with no locking.
2. **Per-user** — checked inside a `pg_advisory_xact_lock` keyed on the user id. If all windows pass, a reservation of `estimated_tokens_per_request` is inserted in the same transaction, so concurrent requests from one user cannot both slip under the limit.
3. On completion, the **actual** weighted usage is recorded as a delta (`actual − estimated`), which may be negative and refunds an over-reservation.

Exceeding any window returns **429** with a structured body:

```json
{
  "error": "rate_limit_exceeded",
  "message": "You have used your token quota for the 8H window. ...",
  "window": "8H",
  "limit": 800000,
  "used": 794321,
  "resets_at": "2026-07-25T19:30:00+00:00"
}
```

App-wide exhaustion returns the same shape with a "The service has reached its token quota" message.

Both 429s and successful streams carry IETF draft ratelimit headers, reporting the **most restrictive** window:

```
RateLimit-Limit: 800000
RateLimit-Remaining: 5679
RateLimit-Reset: 21600
RateLimit-Policy: 800000;w=28800, 857143;w=604800, 2857143;w=2592000
```

---

## Gotchas

- **Derived values cannot be set.** Assigning `user_8h`, `app_30d`, `output_token_multiplier`, or `estimated_tokens_per_request` is silently ignored — pydantic drops unknown keys. Set the budget, prices, and shares instead.
- **Two validators run at construction.** `share_8h > share_7d` is rejected (the 7D window would never bind), as is a per-user quota exceeding the app-wide quota for the same window (the user limit could never be reached).
- **Failed requests still consume quota.** If a stream errors before usage is recorded, the up-front reservation stays charged. Conservative rather than exploitable.
- **The app-wide check is racy by design.** No lock and no reservation, so a burst of concurrent requests can collectively overshoot the app ceiling by up to N × `estimated_tokens_per_request`.
- **Usage is stored append-only** in `token_usage` and aggregated on read, so windows are true rolling windows rather than fixed calendar buckets.

"""Local HTTP backend for the candidate review page.

`mm-eval review` serves the page and persists every decision to
`data/eval_gen/<run-id>/decisions/decisions.json` in the same shape as the
page's export, so a
file written by the server and one exported from the browser are
interchangeable.

Binds all interfaces by default: this runs inside the dev container, where a
loopback socket is reachable only from inside that container and the browser is
on the host. Reaching it still requires the port to be published or forwarded.

Nothing here authenticates and the page writes into the repo, so keep that
forwarding local -- `--host 127.0.0.1` restricts it to the container itself.

A PUT replaces the whole document, so it carries a `base_updated_at`
precondition: the `updated_at` the client last read. A write whose base has
been superseded is refused with 409 and the current document, rather than
reverting whoever wrote in between. Any scripted client must therefore GET,
echo `updated_at` back as `base_updated_at`, and merge on 409 -- or edit the
file directly while the server is not running.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request

from meeplemate.eval.review_page import empty_decisions
from meeplemate.util import atomic_write_text

logger = structlog.get_logger(__name__)


def read_decisions(path: Path, run_id: str, total: int = 0) -> dict[str, Any]:
    """Load the decisions file, tolerating absence and corruption.

    A malformed file returns empty rather than raising: the alternative is a
    review tool that will not start, and the reviewer has no way to repair the
    JSON from inside the browser. The bad file is left on disk and logged.
    """
    if not path.exists():
        return empty_decisions(run_id, total)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("Unreadable decisions file, starting empty",
                       path=str(path), error=repr(exc))
        return empty_decisions(run_id, total)
    if not isinstance(doc, dict) or not isinstance(doc.get("decisions"), list):
        logger.warning("Decisions file has an unexpected shape", path=str(path))
        return empty_decisions(run_id, total)
    return doc


def write_decisions(path: Path, doc: dict[str, Any]) -> None:
    atomic_write_text(path, lambda fp: json.dump(doc, fp, indent=2, ensure_ascii=False))


def build_app(
    *,
    render_page: Callable[[], str],
    decisions_file: Path,
    run_id: str,
    total: int,
) -> FastAPI:
    """The review app.

    ``render_page`` is called per request rather than once at startup, so
    re-running ``answer-candidates`` against the same run shows up on reload
    instead of needing the server restarted.
    """
    app = FastAPI(title="Boardbarian candidate review", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(render_page())

    @app.get("/api/decisions")
    async def get_decisions() -> JSONResponse:
        return JSONResponse(read_decisions(decisions_file, run_id, total))

    @app.put("/api/decisions")
    async def put_decisions(request: Request) -> JSONResponse:
        try:
            doc = await request.json()
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("decisions"), list):
            raise HTTPException(
                status_code=422,
                detail="Body must be an object with a 'decisions' array",
            )
        # This PUT replaces the document wholesale, so a client holding state
        # from before someone else's write would silently revert it. Merely
        # closing a stale tab is enough to trigger that, because the page
        # flushes on unload. Refuse the write instead and hand back what is on
        # disk, so the client can merge its own edits onto it and retry.
        #
        # A first write to a run that has never been saved has nothing to
        # conflict with: `empty_decisions` carries no `updated_at`, so the
        # precondition is skipped rather than blocking a fresh review.
        current = read_decisions(decisions_file, run_id, total)
        base = doc.pop("base_updated_at", None)
        if current.get("updated_at") and base != current["updated_at"]:
            logger.info("Rejected a stale decisions write",
                        path=str(decisions_file), base=base,
                        current=current["updated_at"])
            return JSONResponse(status_code=409, content={
                "detail": "The decisions file has changed since this client "
                          "read it. Merge onto `current` and retry.",
                "current": current,
            })
        # The run id is the server's, not the client's: a page left open from
        # an earlier run must not overwrite this run's file under its own id.
        doc["run_id"] = run_id
        # The server owns the version too, so it cannot be skewed by a browser
        # clock that disagrees with the container's.
        doc["updated_at"] = (datetime.now(timezone.utc)
                             .isoformat().replace("+00:00", "Z"))
        write_decisions(decisions_file, doc)
        logger.info("Saved decisions", path=str(decisions_file),
                    reviewed=len(doc["decisions"]))
        return JSONResponse({"ok": True, "path": str(decisions_file),
                             "reviewed": len(doc["decisions"]),
                             "updated_at": doc["updated_at"]})

    return app


def serve(app: FastAPI, *, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")

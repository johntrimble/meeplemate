"""Operator CLI for MeepleMate.

    mm-admin purge-deleted-accounts --older-than 90d

Deliberately manual rather than scheduled: this project has no cron/scheduler
infrastructure, and whether an account is still restorable is decided by the
gate in `meeplemate.db.account_recovery`, not by this job. So a purge that runs
late only reclaims storage late — it never restores an account that should have
expired, and never expires one early.
"""
import asyncio
import re
from datetime import UTC, datetime, timedelta

import click
from sqlalchemy.ext.asyncio import create_async_engine

from meeplemate.config import PGSettings
from meeplemate.db.account_recovery import RESURRECTION_WINDOW
from meeplemate.db.repository import PostgresDataLayer

_DURATION = re.compile(r"^(\d+)([dhm])$")
_UNITS = {"d": "days", "h": "hours", "m": "minutes"}


def _parse_duration(value: str) -> timedelta:
    match = _DURATION.match(value.strip().lower())
    if not match:
        raise click.BadParameter(
            f"expected a duration like '30d', '12h' or '90m', got {value!r}"
        )
    amount, unit = match.groups()
    return timedelta(**{_UNITS[unit]: int(amount)})


@click.group()
def cli():
    """Administrative commands."""


@cli.command("purge-deleted-accounts")
@click.option(
    "--older-than",
    default=f"{RESURRECTION_WINDOW.days}d",
    show_default=True,
    help="Purge accounts soft-deleted longer ago than this (e.g. 90d, 12h).",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Actually delete. Without this the command only reports what it would do.",
)
def purge_deleted_accounts(older_than: str, yes: bool):
    """Permanently remove accounts whose grace period has expired.

    Deletes the user row and everything hanging off it: chats, messages, message
    parts and token usage.
    """
    window = _parse_duration(older_than)
    if window < RESURRECTION_WINDOW:
        click.echo(
            click.style(
                f"Refusing to purge: --older-than {older_than} is shorter than the "
                f"{RESURRECTION_WINDOW.days}-day window in which users are promised "
                "they can still restore their account.",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    cutoff = datetime.now(UTC) - window
    asyncio.run(_purge(cutoff, dry_run=not yes))


async def _purge(cutoff: datetime, *, dry_run: bool) -> None:
    engine = create_async_engine(PGSettings().pg.build_url())
    try:
        data_layer = PostgresDataLayer(engine)
        summaries = await data_layer.purge_deleted_users(cutoff, dry_run=dry_run)

        if not summaries:
            click.echo(f"No accounts soft-deleted before {cutoff:%Y-%m-%d %H:%M} UTC.")
            return

        verb = "Would purge" if dry_run else "Purged"
        click.echo(f"{verb} {len(summaries)} account(s):")
        for s in summaries:
            click.echo(
                f"  {s.id}  {s.email or '(no email)':40}  "
                f"deleted {s.deleted_at:%Y-%m-%d}  "
                f"{s.chats} chats, {s.messages} messages, {s.token_usage_rows} usage rows"
            )
        if dry_run:
            click.echo("\nDry run — nothing was deleted. Re-run with --yes to commit.")
    finally:
        await engine.dispose()


def main():
    cli()


if __name__ == "__main__":
    main()

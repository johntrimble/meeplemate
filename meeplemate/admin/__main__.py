"""Operator CLI for MeepleMate.

    mm-admin purge-deleted-accounts --older-than 90d

Deliberately manual rather than scheduled: this project has no cron/scheduler
infrastructure. Running it late has no user-visible effect — a soft-deleted
account is already locked out the moment it is flagged, so this only reclaims
storage.
"""
import asyncio
import re
from datetime import UTC, datetime, timedelta

import click
from sqlalchemy.ext.asyncio import create_async_engine

from meeplemate.config import PGSettings
from meeplemate.db.repository import PostgresDataLayer

# How long a soft-deleted account is kept before it can be purged. Purely a
# retention policy — nothing in the request path reads it — but keep it at or
# above the longest rate-limit window (30D) so an operator purging aggressively
# can't destroy chats belonging to an account whose budget is still in force.
DEFAULT_RETENTION = timedelta(days=90)

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
    default=f"{DEFAULT_RETENTION.days}d",
    show_default=True,
    help="Purge accounts soft-deleted longer ago than this (e.g. 90d, 12h).",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Actually delete. Without this the command only reports what it would do.",
)
def purge_deleted_accounts(older_than: str, yes: bool):
    """Permanently remove accounts soft-deleted longer ago than the retention window.

    Deletes the user row, their chats, messages and message parts. Token usage
    is left in place: it is keyed by email rather than by account, so removing it
    could clear a live account's budget for the same address.
    """
    window = _parse_duration(older_than)
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
                f"  {s.uid:30}  {s.email or '(no email)':32}  "
                f"deleted {s.deleted_at:%Y-%m-%d}  "
                f"{s.chats} chats, {s.messages} messages"
            )
        if dry_run:
            click.echo("\nDry run — nothing was deleted. Re-run with --yes to commit.")
    finally:
        await engine.dispose()


def main():
    cli()


if __name__ == "__main__":
    main()

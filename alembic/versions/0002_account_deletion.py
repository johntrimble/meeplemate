"""account deletion: soft-delete flag, and rate limits keyed per person

Two changes, both in service of letting a user delete their account.

``app_user.deleted_at`` marks an account deleted without removing it, so an
already-issued Firebase ID token (valid for up to an hour after the underlying
user is destroyed) stops working immediately instead of at expiry.

``token_usage.user_id`` becomes ``quota_key``, holding the lowercased email
rather than the Firebase uid. A rate-limit budget belongs to a person, and a
person outlives any one uid: deleting an account mints a new uid on the next
sign-in, so a uid-keyed ledger would hand them a fresh 30-day budget. Chat
history is deliberately *not* carried across that boundary — losing it is the
price of deleting your account — but the budget is.

Also adds the FK from ``chat`` to ``app_user`` that lets a purge cascade.

Revision ID: 0002_account_deletion
Revises: 0001_initial
Create Date: 2026-07-26

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_account_deletion"
down_revision: Union[str, Sequence[str], None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # --- Soft delete -------------------------------------------------------
    op.add_column("app_user", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    # --- Re-key token_usage onto the person, not the credential ------------
    op.alter_column("token_usage", "user_id", new_column_name="quota_key")
    # Existing rows hold a uid; map each to its owner's email so a user's
    # accumulated budget survives this change. Rows whose uid has no app_user
    # row, or whose user has no email, keep the uid as their key — still
    # unique, just not poolable across a future uid change.
    # lower(trim(...)) must match UserRecord.quota_key exactly, or a backfilled
    # row would sit under a key the application never looks up again.
    op.execute("""
        UPDATE token_usage t
        SET quota_key = lower(trim(u.email))
        FROM app_user u
        WHERE u.user_id = t.quota_key AND u.email IS NOT NULL;
    """)
    op.drop_index("ix_token_usage_user_recorded", table_name="token_usage")
    op.create_index("ix_token_usage_quota_recorded", "token_usage", ["quota_key", "recorded_at"])

    # --- FK so a purge cascades from app_user to chat -----------------------
    # `chat` predates rate limiting, and `upsert_user` only ever ran on the
    # streaming path, so some chats reference uids that were never inserted
    # into app_user. Backfill them or the constraint can't be created.
    orphans = conn.execute(sa.text("""
        INSERT INTO app_user (user_id)
        SELECT DISTINCT user_id FROM chat
        ON CONFLICT (user_id) DO NOTHING
    """)).rowcount
    if orphans:
        print(f"  backfilled {orphans} app_user row(s) for chats with no owner")

    op.create_foreign_key(
        "fk_chat_user", "chat", "app_user", ["user_id"], ["user_id"], ondelete="CASCADE"
    )


def downgrade() -> None:
    op.drop_constraint("fk_chat_user", "chat", type_="foreignkey")

    # Map emails back to uids. Lossy where several uids share an address — they
    # all collapsed onto one key on the way up and there is no record of which
    # row belonged to which — so the newest matching uid wins.
    op.execute("""
        UPDATE token_usage t
        SET quota_key = u.user_id
        FROM app_user u
        WHERE lower(trim(u.email)) = t.quota_key;
    """)
    op.drop_index("ix_token_usage_quota_recorded", table_name="token_usage")
    op.alter_column("token_usage", "quota_key", new_column_name="user_id")
    op.create_index("ix_token_usage_user_recorded", "token_usage", ["user_id", "recorded_at"])

    op.drop_column("app_user", "deleted_at")

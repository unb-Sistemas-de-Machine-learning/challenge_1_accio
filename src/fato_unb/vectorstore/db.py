"""Compatibility module: re-exports database symbols from fato_unb.storage.db."""

from fato_unb.storage.db import (
    DATABASE_URL,
    Base,
    IngestionStatus,
    RawDocumentEntity,
    async_session,
    engine,
    get_session,
    init_db,
)

__all__ = [
    "DATABASE_URL",
    "Base",
    "IngestionStatus",
    "RawDocumentEntity",
    "async_session",
    "engine",
    "get_session",
    "init_db",
]

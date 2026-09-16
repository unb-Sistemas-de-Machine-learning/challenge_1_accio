import hashlib
from datetime import UTC, datetime
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fato_unb.ingestion.models import RawDocument, SourceType
from fato_unb.storage.db import IngestionStatus, RawDocumentEntity, async_session


class StagingRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None):
        self.session_factory = session_factory or async_session

    async def save_documents(self, docs: list[RawDocument]) -> int:
        if not docs:
            return 0

        inserted_count = 0
        seen_ids: set[str] = set()
        seen_urls: set[str] = set()

        async with self.session_factory() as session:
            async with session.begin():
                for doc in docs:
                    url_str = str(doc.url)
                    doc_id = doc.doc_id or hashlib.sha256(url_str.encode("utf-8")).hexdigest()

                    # Deduplicação no lote em memória
                    if doc_id in seen_ids or url_str in seen_urls:
                        continue

                    # Verifica existência no banco por doc_id ou url para idempotência
                    query = select(RawDocumentEntity.doc_id).where(
                        (RawDocumentEntity.doc_id == doc_id) | (RawDocumentEntity.url == url_str)
                    )
                    exists = (await session.execute(query)).scalar_one_or_none()
                    if exists:
                        continue

                    seen_ids.add(doc_id)
                    seen_urls.add(url_str)

                    source_type_val = (
                        doc.source_type.value
                        if hasattr(doc.source_type, "value")
                        else str(doc.source_type)
                    )

                    entity = RawDocumentEntity(
                        doc_id=doc_id,
                        url=url_str,
                        title=doc.title,
                        content=doc.content,
                        source=doc.source,
                        source_type=source_type_val,
                        published_at=doc.published_at,
                        semester_ref=doc.semester_ref,
                        status=IngestionStatus.PENDING,
                        collected_at=datetime.now(UTC),
                    )
                    session.add(entity)
                    inserted_count += 1

        return inserted_count

    async def get_pending_documents(self, limit: int = 100) -> list[RawDocument]:
        async with self.session_factory() as session:
            stmt = (
                select(RawDocumentEntity)
                .where(RawDocumentEntity.status == IngestionStatus.PENDING)
                .order_by(RawDocumentEntity.collected_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            entities = result.scalars().all()

            docs: list[RawDocument] = []
            for e in entities:
                try:
                    src_type = SourceType(e.source_type)
                except ValueError:
                    src_type = SourceType.RSS_NEWS

                pub_at = e.published_at if e.published_at is not None else datetime.now(UTC)

                docs.append(
                    RawDocument(
                        doc_id=e.doc_id,
                        url=e.url,
                        title=e.title,
                        content=e.content,
                        source=e.source,
                        source_type=src_type,
                        published_at=pub_at,
                        semester_ref=e.semester_ref,
                    )
                )
            return docs

    async def mark_as_indexed(self, doc_id: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(RawDocumentEntity)
                    .where(RawDocumentEntity.doc_id == doc_id)
                    .values(
                        status=IngestionStatus.INDEXED,
                        indexed_at=datetime.now(UTC),
                        error_message=None,
                    )
                )
                await session.execute(stmt)

    async def mark_as_failed(self, doc_id: str, error: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(RawDocumentEntity)
                    .where(RawDocumentEntity.doc_id == doc_id)
                    .values(
                        status=IngestionStatus.FAILED,
                        error_message=error,
                    )
                )
                await session.execute(stmt)

    async def reset_failed_to_pending(self) -> int:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = (
                    update(RawDocumentEntity)
                    .where(RawDocumentEntity.status == IngestionStatus.FAILED)
                    .values(status=IngestionStatus.PENDING, error_message=None)
                )
                result = await session.execute(stmt)
                return result.rowcount

    async def get_stats(self) -> dict[IngestionStatus, int]:
        async with self.session_factory() as session:
            stmt = select(
                RawDocumentEntity.status, func.count(RawDocumentEntity.doc_id)
            ).group_by(RawDocumentEntity.status)
            result = await session.execute(stmt)
            counts: dict[IngestionStatus, int] = {status: 0 for status in IngestionStatus}
            for status, count in result.all():
                if isinstance(status, str):
                    try:
                        status = IngestionStatus(status)
                    except ValueError:
                        pass
                counts[status] = count
            return counts

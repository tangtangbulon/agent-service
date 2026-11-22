from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.models import ParentChunk


class ParentChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def delete_by_product_except(self, product_id: int, keep_ids: list[UUID]) -> None:
        await self.session.execute(
            delete(ParentChunk).where(
                ParentChunk.product_id == product_id,
                ParentChunk.id.not_in(keep_ids),
            )
        )

    def add_all(self, parent_chunks: list[ParentChunk]):
        self.session.add_all(parent_chunks)

    async def list_by_ids(self, ids: list[UUID]) -> list[ParentChunk]:
        result = await self.session.scalars(select(ParentChunk).where(ParentChunk.id.in_(ids)))
        return list(result.all())

from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.chat_thread.models import ChatThread
from sqlalchemy import func, select, update
from uuid import UUID


class ChatThreadRepository:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def add(self,chat_thread:ChatThread):
        self.session.add(chat_thread)


    async def list_by_user_id(self,user_id:int):
        threads=await self.session.scalars(
            select(ChatThread).where(ChatThread.user_id==user_id)
            .order_by(
                ChatThread.updated_at.desc(),
                ChatThread.created_at.desc())
            )

        return threads.all()

    async def find_owned(
        self,
        thread_id:UUID,
        user_id: int) -> ChatThread | None:
        return await self.session.scalar(
            select(ChatThread).where(
                ChatThread.id==thread_id,
                ChatThread.user_id==user_id
            )
        )


    async def touch(self, thread_id: UUID) -> None:
        """刷新最后活跃时间，会话列表按它排序"""
        await self.session.execute(
            update(ChatThread).where(ChatThread.id == thread_id).values(updated_at=func.now())
        )

    async def delete(self,thread: ChatThread) -> None:
        await self.session.delete(thread)
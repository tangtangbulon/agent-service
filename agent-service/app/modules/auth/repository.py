from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.models import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by_username(self, username: str) -> User | None:
        return await self.session.scalar(select(User).where(User.username == username))

    async def find_by_id(self, user_id: int) -> User | None:
        return await self.session.get(User, user_id)

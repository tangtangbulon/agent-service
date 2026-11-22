from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, verify_password
from app.modules.auth.exceptions import InvalidCredentialsError, UnauthorizedError
from app.modules.auth.models import User
from app.modules.auth.repository import UserRepository
from app.modules.auth.schemas import LoginResponse, UserResponse


class AuthService:
    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)

    async def login(self, username: str, password: str) -> LoginResponse:
        user = await self.repository.find_by_username(username)
        password_ok = verify_password(password, user.password_hash if user else None)
        if user is None or not password_ok or user.status != "active":
            raise InvalidCredentialsError()

        token, expires_in = create_access_token(user.id)
        return LoginResponse(
            access_token=token,
            expires_in=expires_in,
            user=UserResponse.model_validate(user),
        )

    async def get_active_user(self, user_id: int) -> User:
        user = await self.repository.find_by_id(user_id)
        if user is None or user.status != "active":
            raise UnauthorizedError()
        return user

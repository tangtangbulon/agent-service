from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.auth.dependencies import CurrentUserId
from app.modules.auth.schemas import LoginRequest, LoginResponse, UserResponse
from app.modules.auth.service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["认证接口"])


def get_auth_service(session: AsyncSession = Depends(get_session)) -> AuthService:
    return AuthService(session)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    service: AuthService = Depends(get_auth_service),
):
    """用户名密码登录，返回访问令牌"""
    return await service.login(request.username, request.password)


@router.get("/me", response_model=UserResponse)
async def me(
    user_id: CurrentUserId,
    service: AuthService = Depends(get_auth_service),
):
    """查询当前登录用户"""
    return await service.get_active_user(user_id)

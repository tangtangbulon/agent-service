from typing import Literal

from pydantic import Field

from app.common.schemas import BaseSchema


class LoginRequest(BaseSchema):
    """登录请求"""

    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseSchema):
    """当前用户信息"""

    id: int
    username: str
    email: str
    display_name: str | None


class LoginResponse(BaseSchema):
    """登录响应"""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="令牌有效秒数")
    user: UserResponse

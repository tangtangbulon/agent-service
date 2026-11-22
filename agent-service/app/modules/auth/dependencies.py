from typing import Annotated

import structlog
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.modules.auth.exceptions import UnauthorizedError

_bearer = HTTPBearer(auto_error=False)


async def get_current_user_id(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> int:
    """从 Authorization: Bearer <token> 中解析当前用户ID"""
    if credentials is None:
        raise UnauthorizedError()
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise UnauthorizedError()
    # 必须是 async 函数：同步依赖跑在线程池里，绑定的上下文变量带不回请求协程
    structlog.contextvars.bind_contextvars(user_id=user_id)
    request.state.user_id = user_id
    return user_id


CurrentUserId = Annotated[int, Depends(get_current_user_id)]

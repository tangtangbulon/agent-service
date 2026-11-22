from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.modules.auth.exceptions import UnauthorizedError

_bearer = HTTPBearer(auto_error=False)


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> int:
    """从 Authorization: Bearer <token> 中解析当前用户ID"""
    if credentials is None:
        raise UnauthorizedError()
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise UnauthorizedError()
    return user_id


CurrentUserId = Annotated[int, Depends(get_current_user_id)]

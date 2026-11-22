from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import PwdlibError

from app.core.config import settings

password_hash = PasswordHash.recommended()

# 用户不存在时也做一次哈希校验，让响应耗时一致，避免通过耗时枚举用户名
_DUMMY_HASH = password_hash.hash("dummy-password-for-timing")


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed: str | None) -> bool:
    """校验密码；哈希为空或格式不受支持（如旧系统遗留的占位哈希）时一律视为失败"""
    try:
        return password_hash.verify(password, hashed or _DUMMY_HASH) and hashed is not None
    except PwdlibError:
        return False


def create_access_token(user_id: int) -> tuple[str, int]:
    """签发访问令牌，返回令牌和有效秒数"""
    expires_in = settings.auth.access_token_expire_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    token = jwt.encode(payload, settings.auth.jwt_secret, algorithm=settings.auth.jwt_algorithm)
    return token, expires_in


def decode_access_token(token: str) -> int | None:
    """解析访问令牌，返回用户ID；签名错误、过期或格式不对时返回None"""
    try:
        payload = jwt.decode(
            token,
            settings.auth.jwt_secret,
            algorithms=[settings.auth.jwt_algorithm],
            options={"require": ["sub", "exp"]},
        )
        return int(payload["sub"])
    except jwt.PyJWTError, ValueError:
        return None

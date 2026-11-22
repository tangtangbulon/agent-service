from datetime import UTC, datetime, timedelta

import jwt

from app.core.config import settings
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_roundtrip():
    hashed = hash_password("correct-password")
    assert hashed.startswith("$argon2")
    assert verify_password("correct-password", hashed)
    assert not verify_password("wrong-password", hashed)


def test_unsupported_or_missing_hash_never_verifies():
    assert not verify_password("anything", "$2a$12$demo.password.hash.placeholder")
    assert not verify_password("anything", None)


def test_access_token_roundtrip():
    token, expires_in = create_access_token(42)
    assert expires_in == settings.auth.access_token_expire_minutes * 60
    assert decode_access_token(token) == 42


def test_tampered_or_garbage_token_is_rejected():
    token, _ = create_access_token(42)
    assert decode_access_token(token[:-2] + ("AA" if not token.endswith("AA") else "BB")) is None
    assert decode_access_token("not-a-jwt") is None


def test_expired_token_is_rejected():
    payload = {"sub": "42", "exp": datetime.now(UTC) - timedelta(seconds=1)}
    token = jwt.encode(payload, settings.auth.jwt_secret, algorithm=settings.auth.jwt_algorithm)
    assert decode_access_token(token) is None


def test_token_signed_with_other_secret_is_rejected():
    payload = {"sub": "42", "exp": datetime.now(UTC) + timedelta(minutes=5)}
    token = jwt.encode(payload, "another-secret-" + "y" * 32, algorithm="HS256")
    assert decode_access_token(token) is None

import uuid
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.main import create_app


@asynccontextmanager
async def no_external_services(app: FastAPI):
    # 依赖注入会在参数校验报错之前先取 agent，这里放一个占位对象
    app.state.agent = object()
    yield


@pytest.fixture
def client():
    app = create_app(no_external_services)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("意外错误")

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


AUTH = {"Authorization": f"Bearer {create_access_token(1)[0]}"}


def test_liveness_probe(client):
    assert client.get("/healthy").json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/chat-threads"),
        ("get", "/api/v1/auth/me"),
        ("get", f"/api/v1/insurance-plans/{uuid.uuid4()}"),
        ("delete", f"/api/v1/chat-threads/{uuid.uuid4()}"),
    ],
)
def test_protected_endpoints_require_token(client, method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_legacy_user_id_header_is_not_trusted(client):
    assert client.get("/api/v1/chat-threads", headers={"x-user-id": "1"}).status_code == 401


def test_invalid_token_is_rejected(client):
    response = client.get("/api/v1/chat-threads", headers={"Authorization": "Bearer invalid"})
    assert response.status_code == 401


def test_invalid_uuid_path_returns_422(client):
    assert client.get("/api/v1/chat-threads/not-a-uuid/messages", headers=AUTH).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"thread_id": str(uuid.uuid4()), "message": "x" * 2001},
        {"thread_id": str(uuid.uuid4())},
        {"thread_id": str(uuid.uuid4()), "message": "你好", "decision": {"action": "approve"}},
        {"thread_id": "not-a-uuid", "message": "你好"},
    ],
)
def test_chat_request_validation(client, body):
    assert client.post("/api/v1/chat", json=body, headers=AUTH).status_code == 422


def test_login_request_validation(client):
    assert client.post("/api/v1/auth/login", json={"username": "", "password": "x"}).status_code == 422


def test_request_id_is_echoed(client):
    response = client.get("/healthy", headers={"X-Request-ID": "req-123"})
    assert response.headers["X-Request-ID"] == "req-123"
    assert client.get("/healthy").headers["X-Request-ID"]


def test_unexpected_error_returns_json_500(client):
    response = client.get("/boom")
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"

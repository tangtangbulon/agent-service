import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import update

from app.infra.database import AsyncSessionFactory
from app.modules.auth.models import User

pytestmark = pytest.mark.integration


async def test_login_and_me(client, user):
    response = await client.post("/api/v1/auth/login", json={"username": user.username, "password": user.password})
    assert response.status_code == 200
    body = response.json()
    assert body["user"]["id"] == user.id

    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.json()["username"] == user.username


async def test_login_rejects_wrong_password_and_unknown_user(client, user):
    wrong = await client.post("/api/v1/auth/login", json={"username": user.username, "password": "wrong"})
    unknown = await client.post("/api/v1/auth/login", json={"username": "no-such-user", "password": "x"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["code"] == "invalid_credentials"


async def test_disabled_user_cannot_login(client, user):
    async with AsyncSessionFactory() as session, session.begin():
        await session.execute(update(User).where(User.id == user.id).values(status="disabled"))
    response = await client.post("/api/v1/auth/login", json={"username": user.username, "password": user.password})
    assert response.status_code == 401


async def test_thread_lifecycle(client, user, agent):
    created = await client.post("/api/v1/chat-threads", json={"title": "  咨询医疗险  "}, headers=user.headers)
    assert created.status_code == 200
    thread = created.json()
    assert thread["title"] == "咨询医疗险"

    listed = await client.get("/api/v1/chat-threads", headers=user.headers)
    assert [t["id"] for t in listed.json()] == [thread["id"]]

    renamed = await client.patch(f"/api/v1/chat-threads/{thread['id']}", json={"title": "新标题"}, headers=user.headers)
    assert renamed.json()["title"] == "新标题"

    agent.history[thread["id"]] = [HumanMessage("你好"), AIMessage("您好，我是保险顾问")]
    history = await client.get(f"/api/v1/chat-threads/{thread['id']}/messages", headers=user.headers)
    assert [m["role"] for m in history.json()["messages"]] == ["user", "assistant"]

    deleted = await client.delete(f"/api/v1/chat-threads/{thread['id']}", headers=user.headers)
    assert deleted.status_code == 204
    assert (await client.get("/api/v1/chat-threads", headers=user.headers)).json() == []


async def test_threads_are_isolated_between_users(client, user, other_user):
    thread_id = (await client.post("/api/v1/chat-threads", json={"title": "我的"}, headers=user.headers)).json()["id"]
    h = other_user.headers

    assert (await client.get("/api/v1/chat-threads", headers=h)).json() == []
    assert (await client.patch(f"/api/v1/chat-threads/{thread_id}", json={"title": "x"}, headers=h)).status_code == 404
    assert (await client.get(f"/api/v1/chat-threads/{thread_id}/messages", headers=h)).status_code == 404
    assert (await client.delete(f"/api/v1/chat-threads/{thread_id}", headers=h)).status_code == 404
    # 原主人仍然可以访问
    assert (await client.get(f"/api/v1/chat-threads/{thread_id}/messages", headers=user.headers)).status_code == 200


async def test_unknown_thread_returns_404(client, user):
    response = await client.get(f"/api/v1/chat-threads/{uuid.uuid4()}/messages", headers=user.headers)
    assert response.status_code == 404
    assert response.json()["code"] == "chat_thread_not_found"

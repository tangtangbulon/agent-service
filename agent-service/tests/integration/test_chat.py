import asyncio
import json
import uuid

import pytest
from sqlalchemy import select

from app.infra.database import AsyncSessionFactory
from app.modules.chat_thread.models import ChatThread

pytestmark = pytest.mark.integration

PENDING_PLAN = {"plan": {"plan_name": "测试方案"}, "confirm_info": "是否确认保存？", "action": ["approve", "reject"]}


def parse_sse(text: str) -> list[tuple[str, str]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        event = next((line[6:].strip() for line in lines if line.startswith("event:")), "message")
        data = "\n".join(line[5:].lstrip() for line in lines if line.startswith("data:"))
        events.append((event, data))
    return events


async def create_thread(client, user) -> str:
    response = await client.post("/api/v1/chat-threads", json={"title": "对话测试"}, headers=user.headers)
    return response.json()["id"]


async def updated_at(thread_id: str):
    async with AsyncSessionFactory() as session:
        return await session.scalar(select(ChatThread.updated_at).where(ChatThread.id == uuid.UUID(thread_id)))


async def test_chat_streams_reply_and_touches_thread(client, user, agent):
    thread_id = await create_thread(client, user)
    before = await updated_at(thread_id)

    response = await client.post("/api/v1/chat", json={"thread_id": thread_id, "message": "你好"}, headers=user.headers)

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["message", "message", "done"]
    assert "".join(json.loads(d) for e, d in events if e == "message") == "您好，这是测试回复"
    after = await updated_at(thread_id)
    assert before is not None and after is not None
    assert after > before


async def test_chat_on_foreign_or_unknown_thread_returns_404(client, user, other_user, agent):
    thread_id = await create_thread(client, user)
    for tid in (thread_id, str(uuid.uuid4())):
        response = await client.post(
            "/api/v1/chat", json={"thread_id": tid, "message": "hi"}, headers=other_user.headers
        )
        assert response.status_code == 404
    assert agent.calls == [], "校验失败时不应调用 Agent"


async def test_message_while_plan_pending_returns_409_with_plan(client, user, agent):
    thread_id = await create_thread(client, user)
    agent.pending[thread_id] = PENDING_PLAN

    response = await client.post("/api/v1/chat", json={"thread_id": thread_id, "message": "继续"}, headers=user.headers)

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "chat_pending_confirmation"
    assert body["interrupt"] == PENDING_PLAN


async def test_decision_resumes_pending_plan(client, user, agent):
    thread_id = await create_thread(client, user)
    agent.pending[thread_id] = PENDING_PLAN
    body = {"thread_id": thread_id, "decision": {"action": "reject", "reject_reason": "再想想"}}

    response = await client.post("/api/v1/chat", json=body, headers=user.headers)

    assert response.status_code == 200
    resume = agent.calls[-1].resume
    assert resume == {"action": "reject", "reject_reason": "再想想"}


async def test_decision_without_pending_plan_returns_409(client, user, agent):
    thread_id = await create_thread(client, user)
    body = {"thread_id": thread_id, "decision": {"action": "approve"}}
    response = await client.post("/api/v1/chat", json=body, headers=user.headers)
    assert response.status_code == 409
    assert response.json()["code"] == "chat_decision_not_expected"


async def test_agent_failure_sends_error_event(client, user, agent):
    thread_id = await create_thread(client, user)
    agent.fail = True

    response = await client.post("/api/v1/chat", json={"thread_id": thread_id, "message": "你好"}, headers=user.headers)

    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["error"]
    assert json.loads(events[0][1])["code"] == "chat_failed"


async def test_concurrent_request_on_same_thread_returns_409(client, user, agent):
    thread_id = await create_thread(client, user)
    agent.gate = asyncio.Event()
    payload = {"thread_id": thread_id, "message": "第一条"}

    first = asyncio.create_task(client.post("/api/v1/chat", json=payload, headers=user.headers))
    await asyncio.sleep(0.3)  # 让第一个请求拿到锁并卡在生成回复上

    second = await client.post("/api/v1/chat", json={"thread_id": thread_id, "message": "第二条"}, headers=user.headers)
    assert second.status_code == 409
    assert second.json()["code"] == "chat_thread_busy"

    agent.gate.set()
    assert (await first).status_code == 200

    third = await client.post("/api/v1/chat", json={"thread_id": thread_id, "message": "第三条"}, headers=user.headers)
    assert third.status_code == 200, "前一个请求结束后锁应已释放"

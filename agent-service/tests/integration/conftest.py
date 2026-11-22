"""集成测试夹具：真实 PostgreSQL 测试库 + 假 Agent（不调用模型、不连接 Milvus）。"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy import delete

from app.core.security import create_access_token, hash_password
from app.infra.checkpointer import close_checkpointer, init_checkpointer
from app.infra.database import AsyncSessionFactory, close_database
from app.main import create_app
from app.modules.auth.models import User
from app.modules.chat_thread.models import ChatThread
from app.modules.insurance_plan.models import InsurancePlan


class _AsyncIter:
    """可多次异步迭代的列表；gate 未打开前，迭代会一直等待"""

    def __init__(self, items: list[Any], gate: asyncio.Event | None = None):
        self._items = items
        self._gate = gate

    async def _iterate(self):
        if self._gate is not None:
            await self._gate.wait()
        for item in self._items:
            yield item

    def __aiter__(self):
        return self._iterate()


class FakeRun:
    """模拟 astream_events(version="v3") 的返回值：messages / custom / subgraphs 三个通道和 interrupts()"""

    def __init__(self, chunks: list[str], gate: asyncio.Event | None):
        message = SimpleNamespace(text=_AsyncIter(chunks))
        self.messages = _AsyncIter([message], gate)
        self.custom = _AsyncIter([])
        self.subgraphs = _AsyncIter([])

    async def interrupts(self) -> list:
        return []


@dataclass
class FakeAgent:
    checkpointer: Any
    reply_chunks: list[str] = field(default_factory=lambda: ["您好", "，这是测试回复"])
    pending: dict[str, dict] = field(default_factory=dict)
    history: dict[str, list] = field(default_factory=dict)
    fail: bool = False
    gate: asyncio.Event | None = None
    calls: list[Any] = field(default_factory=list)

    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        interrupts = (SimpleNamespace(value=self.pending[thread_id]),) if thread_id in self.pending else ()
        return SimpleNamespace(interrupts=interrupts, tasks=(), values={"messages": self.history.get(thread_id, [])})

    async def astream_events(self, graph_input, config, **kwargs):
        self.calls.append(graph_input)
        if self.fail:
            raise RuntimeError("模拟模型服务故障")
        return FakeRun(self.reply_chunks, self.gate)


@pytest.fixture(scope="session")
async def app():
    application = create_app()
    checkpointer = await init_checkpointer()
    application.state.agent = FakeAgent(checkpointer)
    yield application
    await close_checkpointer()
    await close_database()


@pytest.fixture
def agent(app) -> FakeAgent:
    """每个测试拿到一个重置过的假 Agent"""
    fake = app.state.agent
    fake.reply_chunks = ["您好", "，这是测试回复"]
    fake.pending.clear()
    fake.history.clear()
    fake.fail = False
    fake.gate = None
    fake.calls.clear()
    return fake


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@dataclass
class AuthUser:
    id: int
    username: str
    password: str

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {create_access_token(self.id)[0]}"}


async def _create_user() -> AuthUser:
    username = f"test-{uuid.uuid4().hex[:12]}"
    password = "Test-password-123"
    async with AsyncSessionFactory() as session, session.begin():
        user = User(
            username=username,
            email=f"{username}@example.com",
            password_hash=hash_password(password),
            display_name="测试用户",
        )
        session.add(user)
        await session.flush()
        user_id = user.id
    return AuthUser(id=user_id, username=username, password=password)


async def _delete_user(user_id: int) -> None:
    async with AsyncSessionFactory() as session, session.begin():
        await session.execute(delete(InsurancePlan).where(InsurancePlan.user_id == user_id))
        await session.execute(delete(ChatThread).where(ChatThread.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))


@pytest.fixture
async def user() -> AsyncIterator[AuthUser]:
    created = await _create_user()
    yield created
    await _delete_user(created.id)


@pytest.fixture
async def other_user() -> AsyncIterator[AuthUser]:
    created = await _create_user()
    yield created
    await _delete_user(created.id)

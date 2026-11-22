import json
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_thread.exceptions import ChatThreadNotFoundError
from app.modules.chat_thread.models import ChatThread
from app.modules.chat_thread.repository import ChatThreadRepository
from app.modules.chat_thread.schemas import ChatHistoryResponse, ChatMessageResponse


def _clause_sources(message: BaseMessage) -> dict[str, Any] | None:
    """从本轮 query_product_clause 的工具结果里取出 ref-001 这类引用。"""
    if not isinstance(message, ToolMessage) or message.name != "query_product_clause":
        return None
    content = message.content
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
    elif isinstance(content, dict):
        payload = content
    else:
        return None
    if not isinstance(payload, dict):
        return None
    if not any(isinstance(key, str) and key.startswith("ref-") for key in payload):
        return None
    return payload


def build_history_messages(
    raw_messages: list[BaseMessage],
    additional_infos: dict[str, Any],
) -> list[ChatMessageResponse]:
    """把工具返回的 ref-001 这类条款挂到紧随其后的 AI 回复上。"""
    messages: list[ChatMessageResponse] = []
    pending_sources: dict[str, Any] | None = None
    for message in raw_messages:
        sources = _clause_sources(message)
        if sources is not None:
            pending_sources = {**(pending_sources or {}), **sources}
            continue
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        if not message.text:
            continue
        additional_info = None
        if role == "assistant":
            message_id = message.id
            stored = additional_infos.get(message_id) if isinstance(message_id, str) else None
            additional_info = pending_sources if pending_sources is not None else stored
            pending_sources = None
        messages.append(
            ChatMessageResponse(
                role=role,
                content=message.text,
                additional_info=additional_info,
            )
        )
    return messages


class ChatThreadService:
    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.session = session
        self.repository = ChatThreadRepository(session)
        self.agent = agent

    async def add(self, user_id: int, title: str):
        # 用上下文管理，开启事务，运行完成，自动commit，出现异常，自动rollback
        async with self.session.begin():
            # 创建对象，对象 映射 到数据库 一条数据
            chat_thread = ChatThread(user_id=user_id, title=title)
            # 新增对象，就是插入一条数据
            await self.repository.add(chat_thread)
            return chat_thread

    async def list_by_user_id(self, user_id: int):
        return await self.repository.list_by_user_id(user_id)

    async def rename(self, thread_id: UUID, user_id: int, title: str) -> ChatThread:
        async with self.session.begin():
            # 1.根据thread_id和user_id查询，判断是否匹配
            chat_thread = await self.repository.find_owned(thread_id, user_id)
            if chat_thread is None:
                raise ChatThreadNotFoundError()
            # 2.查询后直接修改字段，触发数据库更新
            chat_thread.title = title
            # 3.刷新数据，重新从数据库中查出最新值
            await self.session.flush()
            await self.session.refresh(chat_thread)
        return chat_thread

    async def get_chat_history(self, thread_id: UUID, user_id: int) -> ChatHistoryResponse:
        chat_thread = await self.repository.find_owned(thread_id, user_id)
        if chat_thread is None:
            raise ChatThreadNotFoundError()
        config: RunnableConfig = {"configurable": {"thread_id": str(thread_id)}}
        state_snapshot = await self.agent.aget_state(config)
        additional_infos = state_snapshot.values.get("additional_info", {})
        messages = build_history_messages(
            state_snapshot.values.get("messages", []),
            additional_infos if isinstance(additional_infos, dict) else {},
        )

        # 5.处理interrupt
        interrupts = None
        if hasattr(state_snapshot, "interrupts") and state_snapshot.interrupts:
            interrupts = state_snapshot.interrupts
        elif hasattr(state_snapshot, "tasks") and state_snapshot.tasks:
            for task in state_snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupts = task.interrupts
                    break
        return ChatHistoryResponse(
            thread_id=thread_id,
            messages=messages,
            interrupt=interrupts[0].value if interrupts else None,
        )

    async def delete(self, thread_id: UUID, user_id: int) -> None:
        async with self.session.begin():
            thread = await self.repository.find_owned(thread_id, user_id)
            if thread is None:
                raise ChatThreadNotFoundError
            checkpointer = self.agent.checkpointer
            if not isinstance(checkpointer, BaseCheckpointSaver):
                raise RuntimeError("会话记忆未初始化")
            await checkpointer.adelete_thread(str(thread_id))
            await self.repository.delete(thread)

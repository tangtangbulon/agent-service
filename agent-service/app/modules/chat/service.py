from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from aiostream import stream as astream
from fastapi.sse import ServerSentEvent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession
from langgraph.stream import CustomTransformer
from app.agents.interrupts import pending_interrupt_value
from app.agents.schemas import InsuranceAgentContext
from app.modules.chat.exceptions import (
    ChatDecisionNotExpectedError,
    ChatPendingConfirmationError,
)
from app.modules.chat.schemas import ChatRequest
from app.modules.chat_thread.exceptions import ChatThreadNotFoundError
from app.modules.chat_thread.repository import ChatThreadRepository


def _additional_info_event(event: Any) -> ServerSentEvent | None:
    if isinstance(event, dict) and event.get("type") == "additional_info":
        return ServerSentEvent(data=event.get("data"), event="additional_info")
    return None


async def iter_run_events(run: Any) -> AsyncGenerator[ServerSentEvent]:
    """读取当前图以及子图里的正文和条款引用。

    保险顾问跑在子图里，query_product_clause 的 custom 事件不会出现在父图的
    custom 通道上。StreamChannel 没有订阅者时会丢弃事件，所以每一层都要先订阅
    custom，再消费 messages。
    """
    message_stream = run.messages.__aiter__()
    custom_stream = run.custom.__aiter__()
    subgraph_stream = run.subgraphs.__aiter__()

    async def stream_messages() -> AsyncGenerator[ServerSentEvent]:
        async for message in message_stream:
            async for text in message.text:
                yield ServerSentEvent(data=text, event="message")

    async def stream_custom() -> AsyncGenerator[ServerSentEvent]:
        async for event in custom_stream:
            sse = _additional_info_event(event)
            if sse is not None:
                yield sse

    async def stream_subgraphs() -> AsyncGenerator[ServerSentEvent]:
        async for sub_stream in subgraph_stream:
            async for event in iter_run_events(sub_stream):
                yield event

    merged = astream.merge(stream_messages(), stream_custom(), stream_subgraphs())
    async with merged.stream() as streamer:
        async for event in streamer:
            yield event


@dataclass(frozen=True)
class PreparedChat:
    """已通过校验、可以直接交给 Agent 执行的一轮对话"""

    graph_input: dict[str, Any] | Command
    config: RunnableConfig
    context: InsuranceAgentContext


class ChatService:
    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.session = session
        self.repository = ChatThreadRepository(session)
        self.agent = agent

    async def prepare(self, user_id: int, request: ChatRequest) -> PreparedChat:
        """校验会话归属和确认状态，必须在开始流式输出前调用，这样错误才能以 HTTP 状态码返回"""
        thread = await self.repository.find_owned(
            request.thread_id,
            user_id,
        )
        # session 要等整个 SSE 响应结束才会被依赖注入关闭，必须在流式输出前主动归还连接
        await self.session.close()
        if thread is None:
            raise ChatThreadNotFoundError

        config: RunnableConfig = {
            "configurable": {"thread_id": str(request.thread_id)}
        }
        snapshot = await self.agent.aget_state(config)
        pending = pending_interrupt_value(snapshot)

        if request.decision is not None:
            if pending is None:
                raise ChatDecisionNotExpectedError
            graph_input: dict[str, Any] | Command = Command(
                resume=request.decision.model_dump(exclude_none=True)
            )
        else:
            if pending is not None:
                raise ChatPendingConfirmationError(pending)
            if request.message is None:
                raise ChatDecisionNotExpectedError
            graph_input = {"messages": [HumanMessage(content=request.message)]}

        return PreparedChat(
            graph_input=graph_input,
            config=config,
            context=InsuranceAgentContext(user_id=user_id),
        )

    async def chat_stream(self, prepared: PreparedChat) -> AsyncGenerator[ServerSentEvent]:
        stream = await self.agent.astream_events(
            prepared.graph_input,
            prepared.config,
            context=prepared.context,
            version="v3",
            transformers=[CustomTransformer],
        )

        async for event in iter_run_events(stream):
            yield event

        # 流结束后再读取中断，最后返回结束标识
        for interrupt in await stream.interrupts():
            payload = getattr(interrupt, "value", interrupt)
            if isinstance(payload, dict):
                yield ServerSentEvent(data=payload, event="interrupt")
        yield ServerSentEvent(data="[DONE]", event="done")

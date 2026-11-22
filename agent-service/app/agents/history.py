from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.core.config import settings


def trim_for_model(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """只保留最近的历史，从用户消息开始截取，保证不会把工具调用和工具结果拆开"""
    trimmed = trim_messages(
        list(messages),
        max_tokens=settings.llm.max_history_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    if trimmed:
        return trimmed
    # 最近一轮本身就超过上限时，至少保留从最后一条用户消息开始的内容
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return list(messages[index:])
    return list(messages)


class TrimHistoryMiddleware(AgentMiddleware[Any, Any]):
    """调用模型前裁剪历史消息；只影响本次请求，不修改保存在 checkpointer 里的会话历史"""

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        messages = cast("list[AnyMessage]", trim_for_model(request.messages))
        return await handler(request.override(messages=messages))

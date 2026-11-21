from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

from app.agents.schemas import InsuranceAgentContext


def _tool_call_ids(message: BaseMessage) -> list[str]:
    tool_calls = getattr(message, "tool_calls", None) or []
    return [call["id"] for call in tool_calls if call.get("id")]


def _tool_calls_are_closed(messages: list[BaseMessage]) -> bool:
    """每条带 tool_calls 的 assistant 消息后面，是否紧跟齐对应的 tool 消息。"""
    index = 0
    while index < len(messages):
        call_ids = _tool_call_ids(messages[index])
        if not call_ids:
            index += 1
            continue
        expected = messages[index + 1 : index + 1 + len(call_ids)]
        if len(expected) != len(call_ids):
            return False
        for call_id, message in zip(call_ids, expected, strict=True):
            if not isinstance(message, ToolMessage) or message.tool_call_id != call_id:
                return False
        index += 1 + len(call_ids)
    return True


def _copy_message(message: BaseMessage) -> BaseMessage:
    return message.model_copy(update={"id": str(uuid4())})


def _rebuild_tail(anchor: AIMessage, tail: list[BaseMessage]) -> list[BaseMessage]:
    """把错位或缺失的 tool 消息挪回对应的 assistant 消息后面。"""
    source: list[BaseMessage] = [anchor, *tail]
    consumed: set[int] = set()
    rebuilt: list[BaseMessage] = []

    for index, message in enumerate(source):
        if index in consumed:
            continue
        call_ids = _tool_call_ids(message)
        if index > 0:
            rebuilt.append(_copy_message(message))
        if not call_ids:
            continue

        found: dict[str, ToolMessage] = {}
        for later_index in range(index + 1, len(source)):
            if later_index in consumed:
                continue
            candidate = source[later_index]
            if (
                isinstance(candidate, ToolMessage)
                and candidate.tool_call_id in call_ids
                and candidate.tool_call_id not in found
            ):
                found[candidate.tool_call_id] = candidate
                consumed.add(later_index)
            if len(found) == len(call_ids):
                break

        for call_id in call_ids:
            existing = found.get(call_id)
            if existing is not None:
                rebuilt.append(_copy_message(existing))
            else:
                rebuilt.append(
                    ToolMessage(
                        content="工具调用未完成，已取消。",
                        tool_call_id=call_id,
                        status="error",
                    )
                )
    return rebuilt


def repair_tool_call_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage] | None:
    """返回可交给 add_messages 的更新；历史合法时返回 None。

    DeepSeek 要求 assistant 的 tool_calls 后面立刻跟上每个 tool_call_id 的 tool 消息。
    interrupt 之后如果又写入了用户消息，tool 结果会被追加到用户消息后面，需要先重排。
    """
    history = list(messages)
    if _tool_calls_are_closed(history):
        return None

    for index, message in enumerate(history):
        if not isinstance(message, AIMessage) or not _tool_call_ids(message):
            continue
        tail = history[index + 1 :]
        message_ids = [item.id for item in tail if isinstance(item.id, str)]
        if len(message_ids) != len(tail):
            return None
        return [
            *(RemoveMessage(id=message_id) for message_id in message_ids),
            *_rebuild_tail(message, tail),
        ]
    return None


class ToolCallHistoryMiddleware(AgentMiddleware[AgentState[Any], InsuranceAgentContext]):
    """在调用模型前修正常被打断的工具消息顺序。"""

    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[InsuranceAgentContext],
    ) -> dict[str, Any] | None:
        updates = repair_tool_call_messages(state["messages"])
        if not updates:
            return None
        return {"messages": updates}

    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[InsuranceAgentContext],
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

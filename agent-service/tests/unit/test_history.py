from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.agents.history import trim_for_model
from app.core.config import settings


def _long_conversation(turns: int) -> list:
    messages = []
    for i in range(turns):
        messages += [
            HumanMessage(f"问题{i} " + "内容" * 200),
            AIMessage("", tool_calls=[{"id": f"call-{i}", "name": "query", "args": {}}]),
            ToolMessage("结果" * 200, tool_call_id=f"call-{i}"),
            AIMessage(f"回答{i} " + "内容" * 200),
        ]
    return messages


def test_short_history_is_kept_as_is():
    messages = [HumanMessage("你好"), AIMessage("您好")]
    assert trim_for_model(messages) == messages


def test_long_history_is_trimmed_within_budget():
    messages = _long_conversation(60)
    trimmed = trim_for_model(messages)

    assert len(trimmed) < len(messages)
    assert count_tokens_approximately(trimmed) <= settings.llm.max_history_tokens
    assert trimmed[-1] is messages[-1]


def test_trimmed_history_starts_with_user_and_keeps_tool_pairs():
    trimmed = trim_for_model(_long_conversation(60))

    assert isinstance(trimmed[0], HumanMessage)
    tool_results = {m.tool_call_id for m in trimmed if isinstance(m, ToolMessage)}
    tool_calls = {c["id"] for m in trimmed if isinstance(m, AIMessage) for c in m.tool_calls}
    assert tool_results == tool_calls


def test_oversized_latest_message_is_still_sent():
    huge = [AIMessage("之前的回答"), HumanMessage("超长" * 50000)]
    assert trim_for_model(huge) == huge[1:]

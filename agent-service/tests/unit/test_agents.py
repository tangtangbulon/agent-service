import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from app.agents.intent_router import match_social_intent
from app.agents.message_history import repair_tool_call_messages
from app.agents.orchestrator import find_previous_ai_message
from app.modules.chat_thread.service import build_history_messages


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("你好", "greeting"),
        ("您好呀！", "greeting"),
        ("Hello", "greeting"),
        ("谢谢您", "thanks"),
        ("拜拜~", "goodbye"),
        ("你好，我想买医疗险", None),
        ("理赔需要什么材料", None),
    ],
)
def test_social_intent_rules(message, expected):
    assert match_social_intent(message) == expected


def test_chat_model_ignores_system_proxy():
    from langchain_deepseek import ChatDeepSeek

    from app.infra.llm import create_chat_model

    model = create_chat_model()
    assert isinstance(model, ChatDeepSeek)
    assert model.http_async_client is not None and model.http_async_client.trust_env is False
    assert model.http_client is not None and model.http_client.trust_env is False


async def test_route_does_not_hide_model_outage():
    from app.agents.intent_router import IntentRouter

    class UnavailableModel:
        async def ainvoke(self, messages):
            raise RuntimeError("connection down")

    router = IntentRouter()
    router.model = UnavailableModel()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="connection down"):
        await router.route("40岁，想买医疗险，预算5000", "")


class TestRepairToolCallMessages:
    def test_closed_history_needs_no_repair(self):
        messages = [
            HumanMessage("查一下", id="h1"),
            AIMessage("", tool_calls=[{"id": "c1", "name": "t", "args": {}}], id="a1"),
            ToolMessage("结果", tool_call_id="c1", id="t1"),
            AIMessage("好的", id="a2"),
        ]
        assert repair_tool_call_messages(messages) is None

    def test_tool_result_after_user_message_is_moved_back(self):
        # interrupt 恢复后，工具结果被追加到了新的用户消息后面
        messages = [
            AIMessage("", tool_calls=[{"id": "c1", "name": "t", "args": {}}], id="a1"),
            HumanMessage("继续", id="h1"),
            ToolMessage("结果", tool_call_id="c1", id="t1"),
        ]
        updates = repair_tool_call_messages(messages)
        assert updates is not None
        removed = {m.id for m in updates if isinstance(m, RemoveMessage)}
        rebuilt = [m for m in updates if not isinstance(m, RemoveMessage)]
        assert removed == {"h1", "t1"}
        assert isinstance(rebuilt[0], ToolMessage) and rebuilt[0].tool_call_id == "c1"
        assert isinstance(rebuilt[1], HumanMessage)

    def test_missing_tool_result_is_filled_with_error(self):
        messages = [
            AIMessage("", tool_calls=[{"id": "c1", "name": "t", "args": {}}], id="a1"),
            HumanMessage("算了", id="h1"),
        ]
        updates = repair_tool_call_messages(messages)
        assert updates is not None
        filler = next(m for m in updates if isinstance(m, ToolMessage))
        assert filler.tool_call_id == "c1" and filler.status == "error"


def test_find_previous_ai_message_skips_tool_calls():
    messages = [
        AIMessage("上一轮的回答"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
        HumanMessage("本轮问题"),
    ]
    assert find_previous_ai_message(messages) == "上一轮的回答"
    assert find_previous_ai_message([HumanMessage("第一句")]) == ""


def test_history_attaches_clause_sources_to_following_answer():
    sources = {"ref-001": {"product_id": 6, "content": "等待期条款"}}
    raw = [
        HumanMessage("等待期多久"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "query_product_clause", "args": {}}]),
        ToolMessage(json.dumps(sources, ensure_ascii=False), tool_call_id="c1", name="query_product_clause"),
        AIMessage("等待期由投保时约定[ref-001]", id="a1"),
        HumanMessage("谢谢"),
        AIMessage("不客气", id="a2"),
    ]
    history = build_history_messages(raw, additional_infos={})

    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[1].additional_info == sources
    assert history[3].additional_info is None

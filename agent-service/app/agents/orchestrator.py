from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.chitchat import init_chitchat_agent
from app.agents.insurance_advisor import init_insurance_advisor
from app.agents.intent_router import IntentRouter
from app.agents.schemas import InsuranceAgentContext, InsuranceAgentState, Intent


def create_route_node(router: IntentRouter):
    """创建意图识别节点：寒暄直接回复并结束，其余按识别出的意图进入对应工作流"""

    async def route_node(state: InsuranceAgentState) -> Command[Intent | Literal["__end__"]]:
        # 上一轮激活的工作流和最近一条AI回复，作为意图识别的上下文
        previous_workflow = state.get("active_workflow", "")
        previous_ai_message = find_previous_ai_message(list(state["messages"]))
        context = f"上轮对话工作流:{previous_workflow}\n上轮对话AI回复:{previous_ai_message}"

        result = await router.route(str(state["messages"][-1].content), context)

        # 返回字符串说明是寒暄，直接给固定回复
        if isinstance(result, str):
            return Command(
                update={
                    "messages": [AIMessage(content=result)],
                    "previous_workflow": previous_workflow,
                    "active_workflow": "chitchat",
                },
                goto="__end__",
            )

        return Command(
            update={
                "previous_workflow": previous_workflow,
                "active_workflow": result.intent,
            },
            goto=result.intent,
        )

    return route_node


def find_previous_ai_message(messages: list[BaseMessage]) -> str:
    """查找本轮用户消息之前最近一条可展示的AI回复"""
    for message in reversed(messages[:-1]):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        if message.text:
            return message.text
    return ""


async def claim_node(state: InsuranceAgentState) -> dict:
    return {"messages": [AIMessage("理赔工作流暂未接入。")]}


def human_handoff_node(state: InsuranceAgentState) -> dict:
    return {"messages": [AIMessage("已记录转人工请求，后续可接入人工客服系统。")]}


def fallback_node(state: InsuranceAgentState) -> dict:
    return {"messages": [AIMessage("我还不能确定你的需求，请说明是想咨询保险、推荐方案还是办理理赔。")]}


async def init_insurance_orchestrator(
    checkpointer: BaseCheckpointSaver,
    router: IntentRouter | None = None,
):
    """初始化保险顾问主Graph"""
    builder = StateGraph(InsuranceAgentState, context_schema=InsuranceAgentContext)
    builder.add_node("route", create_route_node(router or IntentRouter()))
    builder.add_node("chitchat", init_chitchat_agent())
    builder.add_node("recommendation_plan", await init_insurance_advisor())
    builder.add_node("claim", claim_node)
    builder.add_node("human_handoff", human_handoff_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "route")
    for node in ("chitchat", "recommendation_plan", "claim", "human_handoff", "fallback"):
        builder.add_edge(node, END)

    return builder.compile(checkpointer=checkpointer)

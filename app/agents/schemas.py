from dataclasses import dataclass
from dataclasses import dataclass
from typing import Literal
from typing import Annotated, Any
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

AdditionalInfo = dict[str, Any]

def merge_additional_info(
    left: dict[str, AdditionalInfo],
    right: dict[str, AdditionalInfo],
) -> dict[str, AdditionalInfo]:
    """按照AI消息id合并引用信息"""
    if not left:
        return right
    return {**left, **right}

class InsuranceAgentState(MessagesState):
    """保险顾问主Graph状态"""

    active_workflow: Intent     # 本轮对话激活的工作流
    previous_workflow: Intent   # 上一轮对话激活的工作流
    additional_info: Annotated[ # 记录AI回答中的引用来源信息
        dict[str, AdditionalInfo],
        merge_additional_info
    ]

@dataclass(frozen=True)
class InsuranceAgentContext:
    """保险顾问Agent运行时上下文"""

    user_id: int




# 路由意图常量
Intent=Literal[
    "chitchat",
    "recommendation_plan",
    "claim",
    "human_handoff",
    "fallback",
]


class RouteResult(BaseModel):
    """意图识别结果"""
    intent: Intent = Field(description="本轮消息应该进入的工作流")
    reason: str = Field(description="判断该意图的理由")
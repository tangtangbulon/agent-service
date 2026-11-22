from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class PlanDecision(BaseModel):
    """用户对保险方案保存的确认结果。"""

    action: Literal["approve", "reject"]
    reject_reason: str | None = None


class ChatRequest(BaseModel):
    """基础对话请求。message 与 decision 互斥。"""

    thread_id: UUID
    message: str | None = Field(default=None, min_length=1, max_length=2000)
    decision: PlanDecision | None = None

    @model_validator(mode="after")
    def require_message_or_decision(self) -> Self:
        if (self.message is None) == (self.decision is None):
            raise ValueError("message 和 decision 必须且只能提供一个")
        return self

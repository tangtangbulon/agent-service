
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import  field_validator, Field
from app.common.schemas import BaseSchema

class ChatThreadCreate(BaseSchema):
    """创建会话请求"""

    title: str = Field(default="新会话", min_length=1, max_length=200)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return value.strip()


class ChatThreadResponse(BaseSchema):
    """会话响应"""

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseSchema):
    """会话中的一条消息"""

    role: Literal["user", "assistant"]
    content: str
    additional_info: dict[str, Any] | None = None


class ChatHistoryResponse(BaseSchema):
    """会话历史"""

    thread_id: UUID
    messages: list[ChatMessageResponse]
    interrupt: dict[str, Any] | None = None




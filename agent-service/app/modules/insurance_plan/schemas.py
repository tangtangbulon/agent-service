from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.common.schemas import BaseSchema


class InsurancePlanItemCreate(BaseModel):
    """保险方案项创建"""
    product_id:int
    product_name: str = Field(description="产品的名称") # 用于方案展示
    category: str = Field(description="产品的险种类型") # 用于方案展示
    priority:int = Field(default=1, ge=1)
    recommendation_reason: str | None = None
    annual_premium_budget: Decimal | None = Field(default=None, ge=0, description="该产品的最低保费")


class InsurancePlanCreate(BaseModel):
    """保险方案创建"""
    plan_name: str = Field(min_length=1, max_length=120)
    summary: str | None = None
    insured_profile: dict[str, Any] = Field(default_factory=dict)
    items: list[InsurancePlanItemCreate] = Field(min_length=1)


class InsurancePlanItemResponse(BaseSchema):
    """保险方案项响应"""
    id: UUID
    plan_id: UUID
    product_id: int
    priority: int
    recommendation_reason: str | None
    annual_premium_budget: Decimal | None
    status: str


class InsurancePlanResponse(BaseSchema):
    """保险方案响应"""
    id: UUID
    plan_name: str
    summary: str | None
    insured_profile: dict[str, Any]
    annual_premium_budget: Decimal | None
    status: str
    created_at: datetime
    updated_at: datetime
    items: list[InsurancePlanItemResponse]

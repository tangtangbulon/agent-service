from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.auth.dependencies import CurrentUserId
from app.modules.insurance_plan.schemas import InsurancePlanResponse
from app.modules.insurance_plan.service import InsurancePlanService

router = APIRouter(prefix="/api/v1/insurance-plans", tags=["保险方案接口"])


def get_plan_service(session: AsyncSession = Depends(get_session)) -> InsurancePlanService:
    return InsurancePlanService(session)


@router.get("/{plan_id}", response_model=InsurancePlanResponse)
async def get_plan(
    plan_id: UUID,
    user_id: CurrentUserId,
    service: InsurancePlanService = Depends(get_plan_service),
):
    """查询当前用户的保险方案及方案项"""
    return await service.get_plan(plan_id, user_id)


@router.patch("/{plan_id}/items/{item_id}/insured", response_model=InsurancePlanResponse)
async def mark_item_insured(
    plan_id: UUID,
    item_id: UUID,
    user_id: CurrentUserId,
    service: InsurancePlanService = Depends(get_plan_service),
):
    """把方案项标记为已投保，所有方案项都投保后方案状态变为已投保"""
    return await service.mark_item_insured(plan_id, item_id, user_id)

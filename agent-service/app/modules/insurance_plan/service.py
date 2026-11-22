from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.exceptions import (
    InsurancePlanItemNotFoundError,
    InsurancePlanNotFoundError,
)
from app.modules.insurance_plan.models import InsurancePlan, InsurancePlanItem
from app.modules.insurance_plan.repository import InsurancePlanRepository
from app.modules.insurance_plan.schemas import (
    InsurancePlanCreate,
    InsurancePlanItemResponse,
    InsurancePlanResponse,
)


def _to_response(plan: InsurancePlan, items: list[InsurancePlanItem]) -> InsurancePlanResponse:
    return InsurancePlanResponse(
        id=plan.id,
        plan_name=plan.plan_name,
        summary=plan.summary,
        insured_profile=plan.insured_profile,
        annual_premium_budget=plan.annual_premium_budget,
        status=plan.status,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
        items=[InsurancePlanItemResponse.model_validate(item) for item in items],
    )


class InsurancePlanService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = InsurancePlanRepository(session)

    async def create_plan(
        self, 
        user_id: int, 
        data: InsurancePlanCreate
    ) -> UUID:
        try:
            plan=await self.repository.create(
                user_id=user_id, 
                data=data,
                )
            await self.session.commit()
            return plan.id
        except Exception as e:
            await self.session.rollback()
            raise e

    async def get_plan(self, plan_id: UUID, user_id: int) -> InsurancePlanResponse:
        plan = await self.repository.find_owned(plan_id, user_id)
        if plan is None:
            raise InsurancePlanNotFoundError()
        items = await self.repository.list_items(plan_id)
        return _to_response(plan, items)

    async def mark_item_insured(
        self,
        plan_id: UUID,
        item_id: UUID,
        user_id: int,
    ) -> InsurancePlanResponse:
        async with self.session.begin():
            # 锁住方案行，避免并发标记时方案状态计算错误
            plan = await self.repository.find_owned(plan_id, user_id, for_update=True)
            if plan is None:
                raise InsurancePlanNotFoundError()
            items = await self.repository.list_items(plan_id)
            item = next((item for item in items if item.id == item_id), None)
            if item is None:
                raise InsurancePlanItemNotFoundError()

            item.status = "insured"
            plan.status = "insured" if all(i.status == "insured" for i in items) else "applying"
            await self.session.flush()
            # updated_at 由数据库生成，flush 后会过期，异步模式下必须显式刷新
            await self.session.refresh(plan)
            await self.session.refresh(item)
        return _to_response(plan, items)

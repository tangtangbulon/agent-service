from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.exceptions import (
    InsurancePlanItemNotFoundError,
    InsurancePlanNotFoundError,
    InvalidPlanProductError,
)
from app.modules.products.service import ProductService
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

    async def normalize_plan(self, data: InsurancePlanCreate) -> InsurancePlanCreate:
        """校验方案里的产品都存在且在售、没有重复，并用数据库里的产品名称和险种覆盖模型生成的值"""
        product_ids = [item.product_id for item in data.items]
        duplicated = sorted({pid for pid in product_ids if product_ids.count(pid) > 1})
        if duplicated:
            raise InvalidPlanProductError(f"方案中存在重复的产品ID：{duplicated}")

        products = await ProductService(self.session).get_active_by_ids(product_ids)
        invalid = [pid for pid in product_ids if pid not in products]
        if invalid:
            raise InvalidPlanProductError(f"以下产品ID不存在或已下架：{invalid}")

        items = [
            item.model_copy(update={
                "product_name": products[item.product_id].name,
                "category": products[item.product_id].category,
            })
            for item in data.items
        ]
        return data.model_copy(update={"items": items})

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

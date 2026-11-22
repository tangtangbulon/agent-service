import uuid
from decimal import Decimal

import pytest

from app.agents.tools import query_product_clause
from app.infra.database import AsyncSessionFactory
from app.modules.insurance_plan.exceptions import InvalidPlanProductError
from app.modules.insurance_plan.schemas import InsurancePlanCreate, InsurancePlanItemCreate
from app.modules.insurance_plan.service import InsurancePlanService

pytestmark = pytest.mark.integration

# 初始化脚本里的在售产品
MEDICAL_A, MEDICAL_B = 4, 6


def plan_payload(product_ids: list[int]) -> InsurancePlanCreate:
    return InsurancePlanCreate(
        plan_name="测试方案",
        items=[
            InsurancePlanItemCreate(
                product_id=pid,
                product_name="模型写的名字",
                category="unknown",
                priority=index + 1,
                annual_premium_budget=Decimal("100"),
            )
            for index, pid in enumerate(product_ids)
        ],
    )


async def create_plan(user_id: int, product_ids: list[int]) -> uuid.UUID:
    async with AsyncSessionFactory() as session:
        service = InsurancePlanService(session)
        return await service.create_plan(user_id, await service.normalize_plan(plan_payload(product_ids)))


async def test_normalize_plan_uses_database_product_info():
    async with AsyncSessionFactory() as session:
        plan = await InsurancePlanService(session).normalize_plan(plan_payload([MEDICAL_A, MEDICAL_B]))
    assert all(item.product_name != "模型写的名字" for item in plan.items)
    assert {item.category for item in plan.items} == {"medical"}


@pytest.mark.parametrize(("product_ids", "reason"), [([MEDICAL_A, 999999], "不存在"), ([MEDICAL_A, MEDICAL_A], "重复")])
async def test_normalize_plan_rejects_invalid_products(product_ids, reason):
    async with AsyncSessionFactory() as session:
        with pytest.raises(InvalidPlanProductError, match=reason):
            await InsurancePlanService(session).normalize_plan(plan_payload(product_ids))


async def test_get_plan_is_scoped_to_owner(client, user, other_user):
    plan_id = await create_plan(user.id, [MEDICAL_A, MEDICAL_B])

    own = await client.get(f"/api/v1/insurance-plans/{plan_id}", headers=user.headers)
    assert own.status_code == 200
    assert [item["priority"] for item in own.json()["items"]] == [1, 2]
    assert own.json()["annual_premium_budget"] == "200.00"

    assert (await client.get(f"/api/v1/insurance-plans/{plan_id}", headers=other_user.headers)).status_code == 404


async def test_mark_items_insured_updates_plan_status(client, user, other_user):
    plan_id = await create_plan(user.id, [MEDICAL_A, MEDICAL_B])
    items = (await client.get(f"/api/v1/insurance-plans/{plan_id}", headers=user.headers)).json()["items"]
    url = f"/api/v1/insurance-plans/{plan_id}/items/{{}}/insured"

    assert (await client.patch(url.format(items[0]["id"]), headers=other_user.headers)).status_code == 404

    first = await client.patch(url.format(items[0]["id"]), headers=user.headers)
    assert first.json()["status"] == "applying"

    second = await client.patch(url.format(items[1]["id"]), headers=user.headers)
    assert second.json()["status"] == "insured"
    assert {item["status"] for item in second.json()["items"]} == {"insured"}

    missing = await client.patch(url.format(uuid.uuid4()), headers=user.headers)
    assert missing.json()["code"] == "insurance_plan_item_not_found"


async def test_clause_tool_rejects_unknown_product():
    result = await query_product_clause.ainvoke({"query": "等待期", "k": 5, "product_id": 999999})
    assert "不存在或已下架" in result

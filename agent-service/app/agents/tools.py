import json
from decimal import Decimal

from fastapi.encoders import jsonable_encoder
from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.agents.schemas import InsuranceAgentContext
from app.infra.database import AsyncSessionFactory
from app.modules.insurance_plan.exceptions import InvalidPlanProductError
from app.modules.insurance_plan.schemas import InsurancePlanCreate
from app.modules.insurance_plan.service import InsurancePlanService
from app.modules.products.models import Product
from app.modules.products.service import ProductService
from app.rag.runtime import get_retriever


@tool
async def query_candidate_products(
    categories: list[str], premium_min: Decimal | None = None, limit_per_category: int = 5
):
    """
    根据险种和保费条件查询可用于推荐的候选保险产品。当用户咨询具体保险产品或需要保险产品推荐时使用。

    Args:
        categories: 产品分类列表，可选值为 medical、critical_illness、life、accident。
        premium_min: 最低保费上限，可选参数，只返回最低保费小于该值的产品。
        limit_per_category: 每个险种最多返回的产品数量，可选参数，默认5。
    """

    async with AsyncSessionFactory() as session:
        # 1.初始化Service
        service = ProductService(session)
        # 2.调用Service，得到候选产品
        products: list[Product] = await service.list_candidates(categories, premium_min, limit_per_category)
        # 3.返回给AI，最好把Product处理成json返回
        return jsonable_encoder(products)


@tool
async def save_insurance_plan(
    plan: InsurancePlanCreate,
    runtime: ToolRuntime[InsuranceAgentContext],
) -> dict[str, str]:
    """
    保存当前用户的保险推荐方案。
    当你查询完候选产品后，根据用户画像、候选产品生成保险组合方案，保存到数据库
    """
    # 先校验产品再请用户确认，避免用户确认了一份包含无效产品的方案
    async with AsyncSessionFactory() as session:
        try:
            plan = await InsurancePlanService(session).normalize_plan(plan)
        except InvalidPlanProductError as exc:
            return {
                "message": f"方案未保存：{exc.message}。请先用 query_candidate_products 查询有效产品后重新生成方案。"
            }

    decision = interrupt(  # 中断执行，把方案发给用户看看
        {
            "plan": plan.model_dump(mode="json"),
            "confirm_info": "是否确认保存这份保险推荐方案？",
            "action": ["approve", "reject"],
        }
    )
    # 如果用户拒绝，则拒绝向下执行
    if decision.get("action") != "approve":
        return {"message": f"用户取消保存保险方案, 原因:{decision.get('reject_reason')}"}

    async with AsyncSessionFactory() as session:
        service = InsurancePlanService(session)
        plan_id = await service.create_plan(
            user_id=runtime.context.user_id,
            data=plan,
        )
    return {
        "plan_id": str(plan_id),
        "message": "保险方案保存成功",
    }


@tool
async def query_product_clause(
    query: str,
    k: int,
    product_id: int,
) -> str:
    """查询指定保险产品的条款内容。

    Args:
        query: 需要从保险条款中查询的问题。
        k: 返回的文档数量，1到8之间，一般用5。
        product_id: 保险产品ID，必须来自 query_candidate_products 的查询结果，不要猜测。
    """
    async with AsyncSessionFactory() as session:
        products = await ProductService(session).get_active_by_ids([product_id])
    product = products.get(product_id)
    if product is None:
        return f"产品ID {product_id} 不存在或已下架，请先用 query_candidate_products 查询正确的产品ID。"

    chunks = await get_retriever().retrieve(query, k, product_id)
    if not chunks:
        return f"没有在《{product.name}》中查询到相关保险条款。"

    # 把model转换字典数据
    sources = {
        # 给文档编号，最终格式 {"ref-001": {"content": ""}}
        f"ref-{i:03d}": {
            "product_id": chunk.product_id,
            "product_name": product.name,
            "clause_name": chunk.clause_name,
            "section_path": chunk.section_path,
            "content": chunk.content,
        }
        for i, chunk in enumerate(chunks, start=1)
    }

    # 把sources写到自定义事件中
    writer = get_stream_writer()
    writer({"type": "additional_info", "data": sources})

    return json.dumps(sources, ensure_ascii=False)

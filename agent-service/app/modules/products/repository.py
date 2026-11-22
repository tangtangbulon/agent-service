from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal
from app.modules.products.models import Product


class ProductRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by_category(
        self,
        category: str | None = None,
    ) -> list[Product]:
        # 1.组装查询条件
        # 1.1.默认只查询active状态的产品
        conditions = [Product.status == "active"]
        # 1.2.如果有category，则添加category过滤
        if category:
            conditions.append(Product.category == category)

        # 2.查询产品列表
        products = await self.session.scalars(
            select(Product)
            .where(*conditions)
            .order_by(Product.id.asc())
        )
        return list(products.all())


    async def find_by_ids(self, ids: list[int]) -> list[Product]:
        products = await self.session.scalars(select(Product).where(Product.id.in_(ids)))
        return list(products.all())

    async def find_limited_by_category(
        self,
        category: str,
        premium_min:Decimal | None,
        limit:int,
    ) -> list[Product]:
        conditions=[
            Product.status == "active",
            Product.category == category,
            ]
        if premium_min is not None:
            conditions.append(Product.min_premium < premium_min)

        products = await self.session.scalars(
            select(Product)
            .where(*conditions)
            .order_by(
                Product.min_premium.asc().nullslast(),
                Product.id.asc()
                )
            .limit(limit)
        )
        return list(products.all())
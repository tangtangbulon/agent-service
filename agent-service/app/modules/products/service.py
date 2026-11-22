from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.products.repository import ProductRepository
from app.modules.products.models import Product
from decimal import Decimal




class ProductService:

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = ProductRepository(session)


    async def list_products(self, category: str | None = None) -> list[Product]:
        return await self.repository.find_by_category(category)

    async def get_active_by_ids(self, ids: list[int]) -> dict[int, Product]:
        """按ID批量查询在售产品，不存在或已下架的产品不会出现在结果里"""
        products = await self.repository.find_by_ids(ids)
        return {product.id: product for product in products if product.status == "active"}

    
    async def list_candidates(
        self, 
        categories: list[str], 
        premium_min: Decimal | None, 
        limit_per_category: int
        ) -> list[Product]:

        candidate_products:list[Product] = []
        for category in dict.fromkeys(categories):
            products = await self.repository.find_limited_by_category(
                category=category,
                premium_min=premium_min,
                limit=limit_per_category
            )
            candidate_products.extend(products)

        return candidate_products
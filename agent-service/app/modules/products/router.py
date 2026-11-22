from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.products.schemas import ProductResponse
from app.modules.products.service import ProductService

router = APIRouter(prefix="/api/v1/products", tags=["products"])


def get_product_service(session: AsyncSession = Depends(get_session)):
    service = ProductService(session)
    yield service


@router.get("", response_model=list[ProductResponse])
async def list_products(
    category: str | None = None,
    service: ProductService = Depends(get_product_service),
) -> list[ProductResponse]:
    products = await service.list_products(category)
    return [ProductResponse.model_validate(product) for product in products]


@router.get("/candidates", response_model=list[ProductResponse])
async def list_candidates(
    categories: list[str] = Query(min_length=1),
    premium_min: Decimal | None = None,
    limit_per_category: int = 5,
    service: ProductService = Depends(get_product_service),
) -> list[ProductResponse]:
    candidates = await service.list_candidates(
        categories=categories, premium_min=premium_min, limit_per_category=limit_per_category
    )
    return [ProductResponse.model_validate(candidate) for candidate in candidates]

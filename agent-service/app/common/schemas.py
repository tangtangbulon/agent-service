from pydantic import BaseModel,ConfigDict
from decimal import Decimal
from typing import Generic, TypeVar


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    

T = TypeVar("T")

class PageResult(BaseSchema,Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
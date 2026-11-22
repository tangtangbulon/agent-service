from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PageResult[T](BaseSchema):
    items: list[T]
    total: int
    page: int
    page_size: int

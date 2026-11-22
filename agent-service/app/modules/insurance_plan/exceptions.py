from app.core.exceptions import ApplicationError


class InsurancePlanNotFoundError(ApplicationError):
    """方案不存在或不属于当前用户"""

    status_code = 404
    code = "insurance_plan_not_found"
    message = "保险方案不存在或不属于当前用户"


class InvalidPlanProductError(ApplicationError):
    """方案里的产品不存在、已下架或重复"""

    status_code = 422
    code = "invalid_plan_product"
    message = "保险方案中的产品无效"


class InsurancePlanItemNotFoundError(ApplicationError):
    """方案项不存在或不属于该方案"""

    status_code = 404
    code = "insurance_plan_item_not_found"
    message = "保险方案项不存在"

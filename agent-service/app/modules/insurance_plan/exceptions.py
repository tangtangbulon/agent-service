from app.core.exceptions import ApplicationError


class InsurancePlanNotFoundError(ApplicationError):
    """方案不存在或不属于当前用户"""

    status_code = 404
    code = "insurance_plan_not_found"
    message = "保险方案不存在或不属于当前用户"


class InsurancePlanItemNotFoundError(ApplicationError):
    """方案项不存在或不属于该方案"""

    status_code = 404
    code = "insurance_plan_item_not_found"
    message = "保险方案项不存在"

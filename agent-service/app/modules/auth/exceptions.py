from app.core.exceptions import ApplicationError


class InvalidCredentialsError(ApplicationError):
    """用户名或密码错误，或账号已停用"""

    status_code = 401
    code = "invalid_credentials"
    message = "用户名或密码错误"


class UnauthorizedError(ApplicationError):
    """缺少令牌，或令牌无效、已过期"""

    status_code = 401
    code = "unauthorized"
    message = "登录已失效，请重新登录"

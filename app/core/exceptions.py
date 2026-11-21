class ApplicationError(Exception):
    """业务异常基类，由全局异常处理器转成统一 JSON。"""

    status_code: int = 500
    code: str = "application_error"
    message: str = "服务器内部错误"

    def __init__(self, message: str | None = None) -> None:
        if message is not None:
            self.message = message
        super().__init__(self.message)

from app.core.exceptions import ApplicationError


class ChatThreadNotFoundError(ApplicationError):
    """会话不存在或不属于当前用户"""

    status_code = 404
    code = "chat_thread_not_found"
    message = "会话不存在或不属于当前用户"
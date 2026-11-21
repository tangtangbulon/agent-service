from app.core.exceptions import ApplicationError


class ChatPendingConfirmationError(ApplicationError):
    """会话正停在方案确认，不能再追加用户消息。"""

    status_code = 409
    code = "chat_pending_confirmation"
    message = "请先确认或取消当前保险方案，再继续对话"

    def __init__(self, interrupt: dict) -> None:
        self.interrupt = interrupt
        super().__init__()


class ChatDecisionNotExpectedError(ApplicationError):
    """当前没有等待确认的方案。"""

    status_code = 409
    code = "chat_decision_not_expected"
    message = "当前会话没有待确认的保险方案"

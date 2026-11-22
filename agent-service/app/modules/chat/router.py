from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.auth.dependencies import CurrentUserId

from .exceptions import ChatThreadBusyError
from .run_lock import thread_run_lock
from .schemas import ChatRequest
from .service import ChatService, PreparedChat

router = APIRouter(prefix="/api/v1/chat", tags=["与AI对话接口"])


def get_chat_service(
    http_request: Request,
    session: AsyncSession = Depends(get_session),
):
    agent = http_request.app.state.agent
    return ChatService(session, agent)


async def prepare_chat(
    request: ChatRequest,
    user_id: CurrentUserId,
    service: ChatService = Depends(get_chat_service),
) -> AsyncGenerator[PreparedChat]:
    # yield 依赖在整个流式响应结束（包括客户端提前断开）后才会执行 finally，锁一定会被释放
    if not thread_run_lock.try_acquire(request.thread_id):
        raise ChatThreadBusyError()
    try:
        yield await service.prepare(user_id, request)
    finally:
        thread_run_lock.release(request.thread_id)


@router.post("", response_class=EventSourceResponse)
async def chat(
    prepared: PreparedChat = Depends(prepare_chat),
    service: ChatService = Depends(get_chat_service),
):
    async for sse in service.chat_stream(prepared):
        yield sse

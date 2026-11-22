from fastapi.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends, Request
from app.infra.database import get_session
from app.modules.auth.dependencies import CurrentUserId
from .schemas import ChatRequest
from .service import ChatService, PreparedChat




router = APIRouter(prefix="/api/v1/chat", tags=["与AI对话接口"])


def get_chat_service(
    http_request: Request,
    session: AsyncSession = Depends(get_session),
):
    agent=http_request.app.state.agent
    return ChatService(session,agent)


async def prepare_chat(
    request: ChatRequest,
    user_id: CurrentUserId,
    service: ChatService = Depends(get_chat_service),
) -> PreparedChat:
    return await service.prepare(user_id, request)


@router.post("",response_class=EventSourceResponse)
async def chat(
    prepared: PreparedChat = Depends(prepare_chat),
    service: ChatService = Depends(get_chat_service),
):
    async for sse in service.chat_stream(prepared):
        yield sse

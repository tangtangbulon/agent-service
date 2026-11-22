from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from app.modules.auth.dependencies import CurrentUserId
from app.modules.chat_thread.service import ChatThreadService
from app.modules.chat_thread.schemas import ChatThreadCreate, ChatThreadResponse,ChatHistoryResponse
from app.infra.database import get_session
from uuid import UUID
from fastapi import Response, status,Request

router=APIRouter(
    prefix="/api/v1/chat-threads",
    tags=["会话管理的接口"],
)


def get_service(
    request: Request,
    session:AsyncSession=Depends(get_session)) -> ChatThreadService:
    return ChatThreadService(
        session=session,
        agent=request.app.state.agent,
    )


@router.post("",response_model=ChatThreadResponse)
async def create_chat_thread(
    request:ChatThreadCreate,
    user_id:CurrentUserId,
    service:ChatThreadService=Depends(get_service)):
    """
    创建会话
    """
    return await service.add(user_id,request.title)




@router.get("",response_model=list[ChatThreadResponse])
async def list_by_user_id(
    user_id:CurrentUserId,
    service:ChatThreadService=Depends(get_service)):
    """
    查询会话列表
    """
    return await service.list_by_user_id(user_id)


@router.patch("/{thread_id}",response_model=ChatThreadResponse)
async def rename(
    thread_id:UUID,
    request:ChatThreadCreate,
    user_id:CurrentUserId,
    service:ChatThreadService=Depends(get_service)):
    """
    重命名会话
    """
    return await service.rename(thread_id,user_id,request.title)


@router.get("/{thread_id}/messages", response_model=ChatHistoryResponse)
async def get_chat_messages(
    thread_id:UUID,
    user_id:CurrentUserId,
    service:ChatThreadService=Depends(get_service)):
    """
    查询会话历史
    """
    return await service.get_chat_history(thread_id,user_id)


@router.delete(
    "/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_thread(
    thread_id: UUID,
    user_id: CurrentUserId,
    service: ChatThreadService = Depends(get_service),
) -> Response:
    """删除指定会话"""

    await service.delete(thread_id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

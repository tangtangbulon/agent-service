from app.core.logging import configure_logging,get_logger
from app.core.config import settings
from fastapi import FastAPI,Request
import uvicorn
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import os
from app.agents.orchestrator import init_insurance_orchestrator


from app.core.exceptions import ApplicationError

# 配置日志
configure_logging(settings.logging.level)
# 获取日志器
logger = get_logger(__name__)


# 初始化数据库
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("数据库连接池初始化成功~✅️")
    from app.infra.database import check_database,close_database
    from app.infra.checkpointer import init_checkpointer,close_checkpointer
    from app.agents.insurance_advisor import init_insurance_advisor
    from app.rag import close_rag
    try:
        # 1.检查业务数据库连接
        await check_database()
        # 2.初始化Checkpointer
        checkpointer = await init_checkpointer()
        # 3.初始化保险顾问Agent
        app.state.agent =  await init_insurance_orchestrator(checkpointer)  

        yield
    finally:
        logger.info("应用关闭中... ")
        # 4.关闭RAG资源
        close_rag()
        # 5.关闭Checkpointer连接池
        await close_checkpointer()
        # 6.关闭SQLAlchemy连接池
        await close_database()

app=FastAPI(
    title=settings.app.name,
    debug=settings.app.debug,
    lifespan=lifespan
)

# 注册路由
from app.modules.auth.router import router as auth_router
from app.modules.products.router import router as products_router
from app.modules.chat_thread.router import router as chat_thread_router
from app.modules.chat.router import router as chat_router
from app.modules.insurance_plan.router import router as insurance_plan_router

app.include_router(auth_router)
app.include_router(products_router)
app.include_router(chat_thread_router)
app.include_router(chat_router)
app.include_router(insurance_plan_router)
#测试请求
@app.get("/healthy",summary="健康检查")
async def healthy() -> dict[str,str]:
    logger.info("健康检测接口被调用")
    return {"status": "ok"}

# 统一处理异常信息，发现是ChatThreadNotFoundError，返回对应的错误状态码和错误信息
@app.exception_handler(ApplicationError)
def handle_exception(request: Request, exc: ApplicationError):
    logger.error(f"处理异常, 请求路径: {request.url}, 异常信息: {exc}")
    content = {
        "code": exc.code,
        "message": exc.message,
    }
    interrupt = getattr(exc, "interrupt", None)
    if interrupt is not None:
        content["interrupt"] = interrupt
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host, 
        port=settings.app.port,
        reload=False
    )

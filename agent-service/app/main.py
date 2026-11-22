import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

# 必须在导入业务模块之前配置日志：导入 agents 时会连带初始化 RAG 组件并输出日志
configure_logging(settings.logging.level, settings.logging.format)
logger = get_logger(__name__)

from app.agents.orchestrator import init_insurance_orchestrator  # noqa: E402
from app.core.exceptions import ApplicationError  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.infra.database import check_database,close_database
    from app.infra.checkpointer import init_checkpointer,close_checkpointer
    from app.rag import close_rag
    try:
        # 1.检查业务数据库连接
        await check_database()
        logger.info("数据库连接池初始化成功~✅️")
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


@app.middleware("http")
async def request_context(request: Request, call_next):
    """为每个请求绑定 request_id 并记录访问日志，同一请求内的所有日志都带上 request_id"""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        if request.url.path not in ("/healthy", "/ready"):
            logger.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
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


@app.get("/healthy",summary="存活探针")
async def healthy() -> dict[str,str]:
    return {"status": "ok"}


@app.get("/ready", summary="就绪探针：检查数据库、会话记忆和向量库")
async def ready() -> JSONResponse:
    from app.infra.checkpointer import check_checkpointer
    from app.infra.database import check_database
    from app.rag import check_vector_store

    checks = {
        "database": check_database(),
        "checkpointer": check_checkpointer(),
        "vector_store": asyncio.to_thread(check_vector_store),
    }
    results: dict[str, str] = {}
    for name, check in checks.items():
        try:
            await asyncio.wait_for(check, timeout=5)
            results[name] = "ok"
        except Exception as exc:
            logger.warning("就绪检查失败", component=name, error=repr(exc))
            results[name] = "error"
    ok = all(value == "ok" for value in results.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "unavailable", "checks": results},
    )


@app.exception_handler(ApplicationError)
def handle_application_error(request: Request, exc: ApplicationError):
    log = logger.warning if exc.status_code < 500 else logger.error
    log("业务异常", path=request.url.path, code=exc.code, status=exc.status_code, detail=exc.message)
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


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("未处理的异常", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": "服务器内部错误，请稍后重试"},
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host, 
        port=settings.app.port,
        reload=False
    )

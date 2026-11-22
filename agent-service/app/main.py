import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import structlog
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.orchestrator import init_insurance_orchestrator
from app.core.config import settings
from app.core.exceptions import ApplicationError
from app.core.logging import configure_logging, get_logger
from app.infra.checkpointer import check_checkpointer, close_checkpointer, init_checkpointer
from app.infra.database import check_database, close_database
from app.modules.auth.router import router as auth_router
from app.modules.chat.router import router as chat_router
from app.modules.chat_thread.router import router as chat_thread_router
from app.modules.insurance_plan.router import router as insurance_plan_router
from app.modules.products.router import router as products_router
from app.rag.runtime import check_vector_store, close_rag, init_rag

configure_logging(settings.logging.level, settings.logging.format)
logger = get_logger(__name__)

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        # 1.检查业务数据库连接
        await check_database()
        logger.info("数据库连接池初始化成功~✅️")
        # 2.连接向量库
        init_rag()
        # 3.初始化Checkpointer和保险顾问Agent
        checkpointer = await init_checkpointer()
        app.state.agent = await init_insurance_orchestrator(checkpointer)
        yield
    finally:
        logger.info("应用关闭中... ")
        close_rag()
        await close_checkpointer()
        await close_database()


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
                # 路由里绑定的上下文变量传不回中间件，用户ID通过 request.state 传递
                user_id=getattr(request.state, "user_id", None),
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )


system_router = APIRouter(tags=["系统"])


@system_router.get("/healthy", summary="存活探针")
async def healthy() -> dict[str, str]:
    return {"status": "ok"}


@system_router.get("/ready", summary="就绪探针：检查数据库、会话记忆和向量库")
async def ready() -> JSONResponse:
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


def handle_application_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApplicationError)
    log = logger.warning if exc.status_code < 500 else logger.error
    log("业务异常", path=request.url.path, code=exc.code, status=exc.status_code, detail=exc.message)
    content = {
        "code": exc.code,
        "message": exc.message,
    }
    interrupt = getattr(exc, "interrupt", None)
    if interrupt is not None:
        content["interrupt"] = interrupt
    return JSONResponse(status_code=exc.status_code, content=content)


def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理的异常", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": "服务器内部错误，请稍后重试"},
    )


def create_app(app_lifespan: Lifespan = lifespan) -> FastAPI:
    """创建应用；测试时可以传入不连接外部服务的 lifespan"""
    app = FastAPI(title=settings.app.name, debug=settings.app.debug, lifespan=app_lifespan)
    app.middleware("http")(request_context)
    for router in (system_router, auth_router, products_router, chat_thread_router, chat_router, insurance_plan_router):
        app.include_router(router)
    app.add_exception_handler(ApplicationError, handle_application_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=False,
    )

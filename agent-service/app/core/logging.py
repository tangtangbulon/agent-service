from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

import structlog


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """配置日志：console 为彩色控制台输出，json 为每行一个 JSON 对象，便于日志系统采集。

    structlog 和标准库 logging（uvicorn、httpx 等第三方库）的日志都经过同一个格式化器，
    保证 json 模式下每一行都是合法 JSON。
    """

    log_level = _log_level(level)

    shared_processors: list[Any] = [
        # 合并 request_id、user_id 等上下文变量
        structlog.contextvars.merge_contextvars,
        # 添加 Logger 名称
        structlog.stdlib.add_logger_name,
        # 添加日志级别
        structlog.stdlib.add_log_level,
        # 添加时间
        structlog.processors.TimeStamper(fmt="iso" if fmt == "json" else "%Y-%m-%d %H:%M:%S", utc=False),
        # 渲染 stack_info=True
        structlog.processors.StackInfoRenderer(),
    ]

    # ConsoleRenderer 自带异常渲染，再加 format_exc_info 会把堆栈打印两遍
    renderer: list[Any] = (
        [
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ]
        if fmt == "json"
        else [structlog.dev.ConsoleRenderer(colors=True)]
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *renderer],
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    structlog.configure(
        processors=[
            *shared_processors,
            # 允许添加占位符参数
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # uvicorn 启动时会给自己的 logger 挂上独立的处理器，改为交给根 logger 统一格式化
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # 访问日志由应用中间件以结构化格式记录，uvicorn 自带的访问日志不再输出
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False

    # Python warnings 也走日志；v3 流式接口是有意选用的，不再重复提示实验性警告。
    # langchain_core 导入时会注册自己的警告过滤规则，必须先导入它，再插入我们的规则才能生效
    from langchain_core._api.beta_decorator import LangChainBetaWarning

    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", message=".*v3 streaming protocol.*", category=LangChainBetaWarning)

    # 控制第三方库日志数量
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO if log_level == logging.DEBUG else logging.WARNING)

    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """获取 structlog 日志器。"""
    return structlog.get_logger(name)


def _log_level(level: str) -> int:
    """将字符串日志级别转换为 logging 常量。"""
    return getattr(logging, level.strip().upper(), logging.INFO)

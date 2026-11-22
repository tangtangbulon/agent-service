"""测试公共配置。

必须在导入任何 app 模块之前设置环境变量：配置和数据库引擎在导入时就会根据环境变量创建。
- 单元测试不连接任何外部服务，必填配置用占位值即可。
- 集成测试通过 TEST_DATABASE_URL 指定 PostgreSQL 测试库，库名必须包含 test，防止误连开发库。
"""

import os
from urllib.parse import urlparse

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

os.environ.setdefault("APP_NAME", "安心保-测试")
os.environ.setdefault("AUTH_JWT_SECRET", "test-secret-" + "x" * 32)
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-placeholder")
os.environ.setdefault("MINERU_TOKEN", "test-placeholder")
os.environ.setdefault("LOGGING_LEVEL", "WARNING")
# 与生产保持一致：DEBUG 模式下未处理异常会直接返回堆栈，不经过统一异常处理器
os.environ["APP_DEBUG"] = "false"

if TEST_DATABASE_URL:
    parsed = urlparse(TEST_DATABASE_URL)
    db_name = parsed.path.lstrip("/")
    if "test" not in db_name:
        raise RuntimeError(f"TEST_DATABASE_URL 的库名必须包含 test，当前为 {db_name!r}，拒绝在非测试库上运行")
    os.environ.update(
        {
            "DB_HOST": parsed.hostname or "localhost",
            "DB_PORT": str(parsed.port or 5432),
            "DB_NAME": db_name,
            "DB_USER": parsed.username or "",
            "DB_PASSWORD": parsed.password or "",
        }
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if TEST_DATABASE_URL:
        return
    skip = pytest.mark.skip(reason="未设置 TEST_DATABASE_URL，跳过需要 PostgreSQL 的集成测试")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

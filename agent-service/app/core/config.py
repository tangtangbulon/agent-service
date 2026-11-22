from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, computed_field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


# 以当前文件为基准，找向上2级的父目录，也就是项目的根目录
APP_ROOT = Path(__file__).resolve().parents[2]

class EnvSettings(BaseSettings):
    """公共settings，配置.env文件位置"""
    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",  # settings中未定义的字段被忽略
    )

class AppSettings(EnvSettings):
    """应用相关设置"""
    name: str = Field(alias="APP_NAME")
    host: str = Field(alias="APP_HOST", default="127.0.0.1")
    port: int = Field(alias="APP_PORT", default=8001)
    debug: bool = Field(alias="APP_DEBUG", default=False, description="是否开启FastAPI的DEBUG模式")

class DatabaseSettings(EnvSettings):
    """数据库相关设置"""
    host: str = Field(alias="DB_HOST", default="localhost")
    port: int = Field(alias="DB_PORT", default=5432)
    name: str = Field(alias="DB_NAME", default="insurance")
    user: str = Field(alias="DB_USER", default="insurance")
    password: str = Field(alias="DB_PASSWORD", default="insurance123")

    echo: bool = Field(alias="DB_ECHO", default=False)
    pool_size: int = Field(alias="DB_POOL_SIZE", default=5)
    max_overflow: int = Field(alias="DB_MAX_OVERFLOW", default=10)
    pool_recycle: int = Field(alias="DB_POOL_RECYCLE", default=1800)

    def build_url(self, scheme: str) -> str:
        # 自动拼接URL路径
        return PostgresDsn.build(
            scheme=scheme,
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password
        ).encoded_string()

    @computed_field
    @property
    def url(self) -> str:
        return self.build_url("postgresql+asyncpg")

    @computed_field
    @property
    def checkpoint_url(self) -> str:
        return self.build_url("postgresql")


class LLMSettings(EnvSettings):
    """模型相关配置"""
    api_key: str = Field(alias="DEEPSEEK_API_KEY")
    chat_model: str = Field(alias="CHAT_MODEL", default="deepseek-v4-flash")
    timeout_seconds: float = Field(alias="LLM_TIMEOUT_SECONDS", default=60, gt=0, description="单次模型请求超时")
    max_retries: int = Field(alias="LLM_MAX_RETRIES", default=2, ge=0, description="模型请求失败后的重试次数")
    max_history_tokens: int = Field(
        alias="LLM_MAX_HISTORY_TOKENS", default=8000, gt=0,
        description="每次发给模型的历史消息估算token上限，超出部分从最早的消息开始丢弃",
    )

class AuthSettings(EnvSettings):
    """认证相关配置"""
    jwt_secret: str = Field(alias="AUTH_JWT_SECRET", min_length=32, description="JWT签名密钥，至少32个字符")
    jwt_algorithm: str = Field(alias="AUTH_JWT_ALGORITHM", default="HS256")
    access_token_expire_minutes: int = Field(alias="AUTH_ACCESS_TOKEN_EXPIRE_MINUTES", default=120, gt=0)

class LoggingSettings(EnvSettings):
    """日志相关配置"""
    level: str = Field(alias="LOGGING_LEVEL", default="INFO") # 可选值：DEBUG、INFO、WARNING、ERROR、CRITICAL
    format: Literal["console", "json"] = Field(alias="LOGGING_FORMAT", default="console", description="生产环境使用json，便于日志系统采集")

def _env_settings[T: EnvSettings](settings_cls: type[T]) -> Callable[[], T]:
    # 运行时可以无参构造，字段从环境变量读取；类型检查仍会把必填别名当成构造参数
    return cast(Callable[[], T], settings_cls)

class RAGSettings(EnvSettings):
    """RAG相关配置"""
    mineru_token: str = Field(alias="MINERU_TOKEN")
    milvus_host: str = Field(alias="MILVUS_HOST", default="127.0.0.1")
    milvus_port: int = Field(alias="MILVUS_PORT", default=19530)
    ollama_base_url: str = Field(alias="OLLAMA_BASE_URL", default="http://127.0.0.1:11434")
    embedding_model: str = Field(alias="EMBEDDING_MODEL", default="qwen3-embedding:0.6b")
    embedding_dimensions: int = Field(alias="EMBEDDING_DIMENSIONS", default=1024)
    @computed_field
    @property
    def milvus_url(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=_env_settings(AppSettings))
    db: DatabaseSettings = Field(default_factory=_env_settings(DatabaseSettings))
    llm: LLMSettings = Field(default_factory=_env_settings(LLMSettings))
    auth: AuthSettings = Field(default_factory=_env_settings(AuthSettings))
    logging: LoggingSettings = Field(default_factory=_env_settings(LoggingSettings))
    rag: RAGSettings = Field(default_factory=_env_settings(RAGSettings))

# 项目启动直接初始化
settings = Settings()





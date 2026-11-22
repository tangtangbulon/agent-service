from typing import Any

import httpx2
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings


def create_chat_model(**overrides: Any) -> BaseChatModel:
    """创建对话模型，统一超时、重试和关闭思考模式的配置"""
    options: dict[str, Any] = {
        "model": settings.llm.chat_model,
        "model_provider": "deepseek",
        "api_key": settings.llm.api_key,
        "timeout": settings.llm.timeout_seconds,
        "max_retries": settings.llm.max_retries,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    options.update(overrides)
    # httpx 会读取 macOS 系统代理。代理进程没在听时，连 DeepSeek 会直接失败，
    # 即使本机可以直连。对话模型与本机 Ollama 一样，不走系统代理。
    timeout = options["timeout"]
    options.setdefault("http_client", httpx2.Client(trust_env=False, timeout=timeout))
    options.setdefault("http_async_client", httpx2.AsyncClient(trust_env=False, timeout=timeout))
    return init_chat_model(**options)


def create_embeddings() -> OllamaEmbeddings:
    """创建向量模型；服务端和入库脚本必须使用同一套配置，否则向量不可比"""
    return OllamaEmbeddings(
        model=settings.rag.embedding_model,
        base_url=settings.rag.ollama_base_url,
        dimensions=settings.rag.embedding_dimensions,
        # Ollama 是内网服务，不应经过开发机上配置的系统代理
        client_kwargs={"trust_env": False},
    )

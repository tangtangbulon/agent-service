from langchain_milvus import Milvus

from app.core.logging import get_logger
from app.rag.retriever import ParentChunkRetriever
from app.rag.vector_store import build_vector_store

logger = get_logger(__name__)

_vector_store: Milvus | None = None
_retriever: ParentChunkRetriever | None = None


def init_rag() -> None:
    """应用启动时连接向量库并创建检索器；连接失败会让启动直接失败，而不是等到第一次检索才报错"""
    global _vector_store, _retriever
    _vector_store = build_vector_store()
    _retriever = ParentChunkRetriever(_vector_store)
    logger.info("RAG 检索组件初始化完成✅️")


def get_retriever() -> ParentChunkRetriever:
    if _retriever is None:
        raise RuntimeError("RAG 检索组件未初始化，请在应用启动时调用 init_rag()")
    return _retriever


def check_vector_store() -> None:
    """就绪检查：确认 Milvus 可连接且知识库 Collection 存在"""
    if _vector_store is None:
        raise RuntimeError("RAG 检索组件未初始化")
    if not _vector_store.client.has_collection(_vector_store.collection_name, timeout=3):
        raise RuntimeError(f"Milvus collection {_vector_store.collection_name} 不存在，请先执行入库脚本")


def close_rag() -> None:
    global _vector_store, _retriever
    if _vector_store is not None:
        _vector_store.client.close()
    _vector_store = None
    _retriever = None

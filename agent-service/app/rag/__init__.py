from langchain_milvus import BM25BuiltInFunction, Milvus

from app.core.config import settings
from app.infra.llm import create_embeddings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import ParentChunkRetriever


# 1.初始化向量模型
_embeddings = create_embeddings()

# 2.初始化VectorStore
_vector_store = Milvus(
    embedding_function=_embeddings,
    collection_name="insurance_collection",
    builtin_function=BM25BuiltInFunction(
        analyzer_params={"type": "chinese"}
    ),
    vector_field=["dense", "sparse"],
    connection_args={"uri": settings.rag.milvus_url},
    auto_id=True,
)

# 3.创建模块级Pipeline和Retriever
pipeline = RAGPipeline(_vector_store)
retriever = ParentChunkRetriever(_vector_store)


def check_vector_store() -> None:
    """就绪检查：确认 Milvus 可连接且知识库 Collection 存在"""
    if not _vector_store.client.has_collection(_vector_store.collection_name, timeout=3):
        raise RuntimeError(f"Milvus collection {_vector_store.collection_name} 不存在，请先执行入库脚本")


def close_rag() -> None:
    """关闭RAG组件"""
    pipeline.close()
    _vector_store.client.close()

__all__ = ['pipeline', 'retriever', 'check_vector_store', 'close_rag']

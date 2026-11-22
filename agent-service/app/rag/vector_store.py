from langchain_milvus import BM25BuiltInFunction, Milvus

from app.core.config import settings
from app.infra.llm import create_embeddings

COLLECTION_NAME = "insurance_collection"


def build_vector_store() -> Milvus:
    """创建 Milvus 向量库连接：稠密向量 + BM25 稀疏向量混合检索，首次写入时自动建 Collection"""
    return Milvus(
        embedding_function=create_embeddings(),
        collection_name=COLLECTION_NAME,
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.rag.milvus_url},
        auto_id=True,
    )

import asyncio

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.products.models import Product
from app.modules.products.service import ProductService
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


def build_embedding_model() -> OllamaEmbeddings:
    """初始化向量模型，参数与 notebooks/09/rag.ipynb 一致。"""
    return OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        dimensions=1024,
    )


def build_vector_store(embedding_model: OllamaEmbeddings) -> Milvus:
    """初始化 Milvus VectorStore，首次使用时会自动创建 Collection。"""
    return Milvus(
        embedding_function=embedding_model,
        collection_name="insurance_collection",
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.rag.milvus_url},
        auto_id=True,
    )


async def load_products() -> list[Product]:
    """查询全部在售产品，脱离 Session 后再交给 Pipeline。"""
    async with AsyncSessionFactory() as session:
        products = await ProductService(session).list_products()
        session.expunge_all()
        return products


async def main() -> None:
    configure_logging(settings.logging.level)

    embedding_model = build_embedding_model()
    vector_store = build_vector_store(embedding_model)
    pipeline = RAGPipeline(vector_store)

    try:
        products = await load_products()
        logger.info("开始构建 RAG 知识库", product_count=len(products))

        for product in products:
            logger.info(
                "入库产品条款",
                product_id=product.id,
                name=product.name,
                clause_name=product.clause_name,
            )
            await pipeline.ingest(product)

        logger.info("RAG 知识库构建完成", product_count=len(products))
    finally:
        pipeline.close()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())

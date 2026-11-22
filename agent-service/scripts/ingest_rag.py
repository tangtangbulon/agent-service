import argparse
import asyncio

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.infra.llm import create_embeddings
from app.modules.products.models import Product
from app.modules.products.service import ProductService
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


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


async def load_products(product_ids: list[int] | None) -> list[Product]:
    """查询在售产品（可只取指定ID），脱离 Session 后再交给 Pipeline。"""
    async with AsyncSessionFactory() as session:
        products = await ProductService(session).list_products()
        session.expunge_all()
    if product_ids:
        products = [product for product in products if product.id in product_ids]
        missing = set(product_ids) - {product.id for product in products}
        if missing:
            raise SystemExit(f"这些产品ID不存在或未在售：{sorted(missing)}")
    return products


async def main() -> None:
    parser = argparse.ArgumentParser(description="解析保险条款并写入 RAG 知识库")
    parser.add_argument(
        "--product-id", type=int, action="append", dest="product_ids",
        help="只入库指定产品，可重复传入；不传则入库全部在售产品",
    )
    args = parser.parse_args()

    configure_logging(settings.logging.level, settings.logging.format)

    embedding_model = create_embeddings()
    vector_store = build_vector_store(embedding_model)
    pipeline = RAGPipeline(vector_store)

    try:
        products = await load_products(args.product_ids)
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

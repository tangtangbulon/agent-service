import re
from pathlib import Path
from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from mineru import MinerU

from app.core.config import APP_ROOT, settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.infra.llm import create_chat_model
from app.modules.products.models import Product
from app.rag.models import ParentChunk
from app.rag.repository import ParentChunkRepository

logger = get_logger(__name__)


MARKDOWN_OPTIMIZATION_PROMPT = """
下面是从保险条款 PDF 解析得来的 Markdown 片段，由于 PDF 中的各个小节是以表格形式存在，所以解析时出现错乱。你分析内容，帮我转为格式正确的 Markdown，特别是标题编号要正确。
- 标题等级严格按以下规则，所有片段必须一致：
  - 1 级：形如“第一部分 总则”“第七部分 释义”的部分标题；
  - 2 级：形如“第六条 保险责任”的条款标题，以及释义部分中形如“四、等待期”的释义条目；
  - 3 级：形如“（一）一般医疗保险金”的编号标题；
  - 4 级：形如“1.恶性肿瘤”的编号标题。
- 只转换原文中真实存在的标题，不要新增、补写或编造原文没有的标题和编号；原文标题没有编号的，保持原样。
- 目录和文档名不计入标题，直接去掉。
- 输出结果只包含 Markdown 正文，不要解释或代码围栏。
- 不要修改、删减或补充保险条款的内容。
{segment_hint}
""".strip()

FIRST_SEGMENT_HINT = "- 这是文档的开头片段，正文开始之前的封面、目录等内容不要输出。"
MIDDLE_SEGMENT_HINT = (
    "- 这是文档中间的一个片段，会直接拼接在上一个片段之后。"
    "上一个片段结束时所在的标题路径是：{heading_path}。"
    "如果本片段开头的内容仍属于这些标题之下，直接输出正文，不要重复输出这些标题；"
    "遇到原文中新的标题时，按上面的等级规则输出。"
)


def last_heading_path(markdown: str) -> list[str]:
    """取出 Markdown 末尾所在位置的标题路径，例如 ['# 第七部分 释义', '## 四、等待期']"""
    path: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        level = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= level <= 6 and stripped[level : level + 1] == " ":
            path = [*path[: level - 1], stripped]
    return path


# 单段输入字符数上限；输出 token 上限要足够容纳整理后的单段内容
SEGMENT_MAX_CHARS = 6000
SEGMENT_MAX_OUTPUT_TOKENS = 8192


class IngestTruncatedError(RuntimeError):
    """模型输出达到长度上限被截断，继续入库会导致知识库内容缺失"""


class IngestQualityError(RuntimeError):
    """整理结果缺少标题结构，切块后整篇条款会变成一两个巨大的块，检索效果极差"""


# 整理后的文档至少要有这么多个标题才算结构正常
MIN_HEADINGS = 3
# 单段整理结果短于输入的这个比例，视为模型省略了内容（去掉目录、表格符号后通常在 0.8 以上）
SEGMENT_MIN_OUTPUT_RATIO = 0.5
FIRST_SEGMENT_MIN_OUTPUT_RATIO = 0.3
SEGMENT_MAX_ATTEMPTS = 3


def count_headings(markdown: str) -> int:
    return sum(1 for line in markdown.splitlines() if re.match(r"^#{1,6} ", line.strip()))


def split_markdown_segments(markdown: str, max_chars: int = SEGMENT_MAX_CHARS) -> list[str]:
    """按空行把长文档切成不超过 max_chars 的片段；超长段落再按行切"""
    segments: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal size
        # 片段末尾的标题挪到下一个片段开头，保证标题和它的正文在同一段里整理
        carry: list[str] = []
        while len(current) > 1 and count_headings(current[-1]) and len(current[-1]) < 200:
            carry.insert(0, current.pop())
        if current:
            segments.append("\n\n".join(current))
        current[:] = carry
        size = sum(len(p) + 2 for p in carry)

    for paragraph in markdown.split("\n\n"):
        pieces = [paragraph]
        if len(paragraph) > max_chars:
            pieces, buf = [], ""
            for line in paragraph.splitlines():
                if buf and len(buf) + len(line) + 1 > max_chars:
                    pieces.append(buf)
                    buf = ""
                buf = f"{buf}\n{line}" if buf else line
            if buf:
                pieces.append(buf)
        for piece in pieces:
            if size and size + len(piece) + 2 > max_chars:
                flush()
            current.append(piece)
            size += len(piece) + 2
    if current:
        segments.append("\n\n".join(current))
    return segments


class RAGPipeline:
    def __init__(self, vector_store: Milvus):
        self._vector_store = vector_store
        # 1.初始化文档解析工具
        self._mineru = MinerU(settings.rag.mineru_token)

        self._model = create_chat_model(
            max_tokens=SEGMENT_MAX_OUTPUT_TOKENS,
            timeout=max(settings.llm.timeout_seconds, 300),
        )

        # 2.初始化文档切分工具
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=True,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", ";", "，", ","],
            keep_separator="end",
        )
        logger.info("RAG Pipeline 初始化完成✅️")

    async def ingest(self, product: Product) -> None:
        # 1.解析并优化保险条款
        markdown = self._parse_and_optimize(product.clause_name)
        # 2.切分父块和子块
        parent_chunks, child_chunks = self._chunk(markdown, product)
        # 3.先写入新父块并保留旧父块，再替换子块，最后删除旧父块；任一步失败都不会出现子块指向不存在的父块
        await self._add_parent_chunks(parent_chunks)
        self._save_child_chunks(product.id, child_chunks)
        await self._delete_stale_parent_chunks(product.id, [parent.id for parent in parent_chunks])

    def _parse_and_optimize(self, clause_name: str) -> str:
        # 1.使用MinerU解析PDF，解析结果缓存到 data/processed，重复入库时不再重复解析
        raw_markdown = self._parse_pdf(clause_name)

        # 2.分段交给大模型整理，避免长文档输出超过 token 上限被截断
        segments = split_markdown_segments(raw_markdown)
        optimized: list[str] = []
        heading_path: list[str] = []
        for index, segment in enumerate(segments):
            if index == 0:
                hint = FIRST_SEGMENT_HINT
            else:
                hint = MIDDLE_SEGMENT_HINT.format(heading_path=" > ".join(heading_path) or "（无）")
            label = f"{clause_name} 第 {index + 1}/{len(segments)} 段"
            # 开头片段会去掉封面和目录，允许输出明显短于输入
            min_ratio = FIRST_SEGMENT_MIN_OUTPUT_RATIO if index == 0 else SEGMENT_MIN_OUTPUT_RATIO
            optimized.append(self._optimize_segment(segment, hint, label, min_ratio))
            heading_path = last_heading_path("\n\n".join(optimized))
        markdown = "\n\n".join(optimized)
        headings = count_headings(markdown)
        logger.info(
            "条款整理完成",
            clause_name=clause_name,
            segments=len(segments),
            headings=headings,
            raw_chars=len(raw_markdown),
            optimized_chars=len(markdown),
        )
        if headings < MIN_HEADINGS:
            raise IngestQualityError(
                f"{clause_name} 整理结果只有 {headings} 个标题，模型可能没有输出标题结构，请重新入库"
            )
        return markdown

    def _optimize_segment(self, segment: str, hint: str, label: str, min_ratio: float) -> str:
        """整理单段；模型偶尔会只返回一小部分或丢掉全部标题，这种结果重试，重试仍失败则中止入库"""
        problem = ""
        for attempt in range(1, SEGMENT_MAX_ATTEMPTS + 1):
            response = self._model.invoke(
                [
                    {"role": "system", "content": MARKDOWN_OPTIMIZATION_PROMPT.format(segment_hint=hint)},
                    {"role": "user", "content": segment},
                ]
            )
            if response.response_metadata.get("finish_reason") == "length":
                raise IngestTruncatedError(f"{label}整理结果被截断，请调小 SEGMENT_MAX_CHARS")
            text = response.text.strip()
            if len(text) < len(segment) * min_ratio:
                problem = f"输出 {len(text)} 字，输入 {len(segment)} 字，内容可能被省略"
            elif count_headings(segment) and not count_headings(text):
                problem = "输入含标题但输出没有任何标题"
            else:
                return text
            logger.warning("段落整理结果异常，重试", segment=label, attempt=attempt, problem=problem)
        raise IngestQualityError(f"{label}重试 {SEGMENT_MAX_ATTEMPTS} 次仍异常：{problem}")

    def _parse_pdf(self, clause_name: str) -> str:
        cache_path = APP_ROOT / "data" / "processed" / f"{Path(clause_name).stem}.mineru.md"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        file_path = Path(APP_ROOT / "data" / "raw" / "kb" / clause_name).as_posix()
        markdown = self._mineru.extract(file_path).markdown
        if not markdown:
            raise RuntimeError(f"MinerU 没有解析出 {clause_name} 的内容")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(markdown, encoding="utf-8")
        return markdown

    def _chunk(self, markdown: str, product: Product) -> tuple[list[ParentChunk], list[Document]]:
        # 1.按照Markdown标题切分父块
        sections = self._markdown_splitter.split_text(markdown)
        parent_chunks: list[ParentChunk] = []
        child_chunks: list[Document] = []

        # 2.遍历父块，继续切分子块
        for section in sections:
            section_path = list(section.metadata.values())
            parent = ParentChunk(
                id=uuid4(),
                product_id=product.id,
                clause_name=product.clause_name,
                section_path=section_path,
                content=section.page_content,
            )
            parent_chunks.append(parent)

            # 子块通过parent_id关联父块
            section.metadata = {
                "product_id": product.id,
                "parent_id": str(parent.id),
            }
            children = self._child_splitter.split_documents([section])
            # 把章节路径拼接到子块正文中，补充语义上下文
            section_header = "\n".join(section_path) + "\n"
            for child in children:
                child.page_content = section_header + child.page_content
            child_chunks.extend(children)
        return parent_chunks, child_chunks

    def _save_child_chunks(
        self,
        product_id: int,
        child_chunks: list[Document],
    ) -> None:
        # Collection存在时，先删除该产品的旧子块

        if self._vector_store.client.has_collection(self._vector_store.collection_name):
            self._vector_store.delete(expr=f"product_id == {product_id}")
        # 再新增子块
        self._vector_store.add_documents(child_chunks)

    async def _add_parent_chunks(self, parent_chunks: list[ParentChunk]) -> None:
        async with AsyncSessionFactory() as session, session.begin():
            ParentChunkRepository(session).add_all(parent_chunks)

    async def _delete_stale_parent_chunks(self, product_id: int, keep_ids: list[UUID]) -> None:
        async with AsyncSessionFactory() as session, session.begin():
            await ParentChunkRepository(session).delete_by_product_except(product_id, keep_ids)

    def close(self):
        """关闭mineru连接"""

        self._mineru.close()

import uuid
from types import SimpleNamespace

from langchain_core.documents import Document

from app.modules.chat.run_lock import ThreadRunLock
from app.rag import retriever as retriever_module
from app.rag.retriever import MAX_TOP_K, ParentChunkRetriever


def test_thread_run_lock():
    lock = ThreadRunLock()
    thread_id = uuid.uuid4()
    assert lock.try_acquire(thread_id)
    assert not lock.try_acquire(thread_id), "同一会话不能同时执行两个请求"
    assert lock.try_acquire(uuid.uuid4()), "不同会话互不影响"
    lock.release(thread_id)
    assert lock.try_acquire(thread_id)


class FakeVectorStore:
    def __init__(self, parent_ids: list[str]):
        self.parent_ids = parent_ids
        self.calls: list[dict] = []

    async def asimilarity_search(self, **kwargs):
        self.calls.append(kwargs)
        return [Document(page_content="子块", metadata={"parent_id": pid}) for pid in self.parent_ids]


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def test_retriever_skips_missing_parents_and_keeps_order(monkeypatch):
    p1, p2, missing = (str(uuid.uuid4()) for _ in range(3))
    stored = {p1: SimpleNamespace(id=p1, content="父块1"), p2: SimpleNamespace(id=p2, content="父块2")}

    class FakeRepository:
        def __init__(self, session):
            pass

        async def list_by_ids(self, ids):
            return [stored[i] for i in ids if i in stored]

    monkeypatch.setattr(retriever_module, "AsyncSessionFactory", FakeSession)
    monkeypatch.setattr(retriever_module, "ParentChunkRepository", FakeRepository)
    monkeypatch.setattr(retriever_module, "create_reranker", lambda query: None)

    vector_store = FakeVectorStore([p2, missing, p2, p1])
    chunks = await ParentChunkRetriever(vector_store).retrieve("等待期", top_k=5, product_id=6)  # type: ignore[arg-type]

    assert [c.content for c in chunks] == ["父块2", "父块1"], "按召回顺序去重，缺失的父块被跳过"
    assert vector_store.calls[0]["expr"] == "product_id == 6"


async def test_retriever_clamps_top_k(monkeypatch):
    monkeypatch.setattr(retriever_module, "create_reranker", lambda query: None)
    vector_store = FakeVectorStore([])
    retriever = ParentChunkRetriever(vector_store)  # type: ignore[arg-type]

    await retriever.retrieve("q", top_k=100, product_id=1)
    await retriever.retrieve("q", top_k=0, product_id=1)

    assert [c["k"] for c in vector_store.calls] == [MAX_TOP_K, 1]
    assert all(c["fetch_k"] >= c["k"] for c in vector_store.calls)

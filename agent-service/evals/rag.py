"""条款问答评估。

1. 检索：每个问题在指定产品内检索，看期望章节是否出现在结果里（hit@1、hit@k、MRR）。
2. 回答（--with-answers）：跑完整的意图路由和保险顾问，检查
   - 是否调用了条款查询且产品ID正确；
   - 回答里的 [ref-xxx] 是否都来自本轮检索结果、是否至少有一处引用。

需要连接向量库、数据库和模型，按需手动运行：
    uv run python -m evals.rag [--k 5] [--with-answers]
"""

import argparse
import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.orchestrator import init_insurance_orchestrator
from app.agents.schemas import InsuranceAgentContext, InsuranceAgentState
from app.core.config import settings
from app.core.logging import configure_logging
from app.infra.database import close_database
from app.rag.runtime import close_rag, get_retriever, init_rag

DATASET = Path(__file__).parent / "data" / "rag_retrieval.jsonl"
CITATION = re.compile(r"\[(ref-\d{3})\]")


def load_cases() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


async def evaluate_retrieval(cases: list[dict], k: int) -> None:
    retriever = get_retriever()
    hit1 = hitk = 0
    reciprocal_ranks = []
    misses = []
    for case in cases:
        chunks = await retriever.retrieve(case["question"], k, case["product_id"])
        paths = ["/".join(chunk.section_path) for chunk in chunks]
        rank = next((i for i, path in enumerate(paths, start=1) if case["expected_section"] in path), None)
        hit1 += rank == 1
        hitk += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0)
        if rank is None:
            misses.append((case["question"], case["expected_section"], paths[:3]))

    n = len(cases)
    print(f"检索评估（k={k}，{n} 个问题）")
    print(f"  hit@1 = {hit1 / n:.1%}   hit@{k} = {hitk / n:.1%}   MRR = {sum(reciprocal_ranks) / n:.3f}")
    for question, expected, paths in misses:
        print(f"  未命中：{question}  期望章节含“{expected}”，前3个结果：{paths}")


def inspect_answer(messages: list, expected_product_id: int) -> dict:
    """从本轮对话消息里取出条款查询的产品ID、检索到的引用编号和最终回答里的引用"""
    product_ids = [
        call["args"].get("product_id")
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call["name"] == "query_product_clause"
    ]
    sources: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "query_product_clause" and isinstance(m.content, str):
            try:
                sources |= set(json.loads(m.content))
            except json.JSONDecodeError:
                pass
    answer = next((m.text for m in reversed(messages) if isinstance(m, AIMessage) and m.text), "")
    cited = set(CITATION.findall(answer))
    return {
        "tool_called": bool(product_ids),
        "product_ok": expected_product_id in product_ids,
        "has_citation": bool(cited),
        "citations_valid": cited <= sources,
        "invalid": sorted(cited - sources),
        "answer": answer,
    }


async def evaluate_answers(cases: list[dict]) -> None:
    graph = await init_insurance_orchestrator(InMemorySaver())
    results = []
    for case in cases:
        graph_input = cast(InsuranceAgentState, {"messages": [HumanMessage(case["question"])]})
        state = await graph.ainvoke(
            graph_input,
            {"configurable": {"thread_id": str(uuid.uuid4())}},
            context=InsuranceAgentContext(user_id=1),
        )
        results.append((case, inspect_answer(state["messages"], case["product_id"])))

    n = len(results)

    def rate(key: str) -> str:
        return f"{sum(r[key] for _, r in results) / n:.1%}"

    print(f"\n回答评估（{n} 个问题）")
    print(f"  调用条款查询 {rate('tool_called')}   产品ID正确 {rate('product_ok')}   ", end="")
    print(f"带引用 {rate('has_citation')}   引用全部有效 {rate('citations_valid')}")
    for case, r in results:
        if not (r["product_ok"] and r["has_citation"] and r["citations_valid"]):
            print(f"  问题：{case['question']}")
            print(f"    产品ID正确={r['product_ok']} 带引用={r['has_citation']} 无效引用={r['invalid']}")
            print(f"    回答：{r['answer'][:120]}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="评估条款检索与回答引用")
    parser.add_argument("--k", type=int, default=5, help="每个问题检索的结果数")
    parser.add_argument("--with-answers", action="store_true", help="同时跑完整对话并校验引用（会调用模型）")
    args = parser.parse_args()
    configure_logging("WARNING", settings.logging.format)

    cases = load_cases()
    init_rag()
    try:
        await evaluate_retrieval(cases, args.k)
        if args.with_answers:
            await evaluate_answers(cases)
    finally:
        close_rag()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())

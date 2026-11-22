"""意图路由评估：用标注好的用户消息测量 IntentRouter 的准确率。

会真实调用模型，按需手动运行：
    uv run python -m evals.intent_router [--min-accuracy 0.9]
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.agents.intent_router import IntentRouter
from app.core.config import settings
from app.core.logging import configure_logging

DATASET = Path(__file__).parent / "data" / "intent_router.jsonl"


async def classify(router: IntentRouter, message: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        result = await router.route(message, context="上轮对话工作流:\n上轮对话AI回复:")
    # 规则匹配的寒暄返回固定回复字符串，最终会作为闲聊处理
    return "chitchat" if isinstance(result, str) else result.intent


async def main() -> None:
    parser = argparse.ArgumentParser(description="评估意图路由准确率")
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="准确率低于该值时以非零状态退出")
    args = parser.parse_args()
    configure_logging("WARNING", settings.logging.format)

    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    router = IntentRouter()
    semaphore = asyncio.Semaphore(4)
    predictions = await asyncio.gather(*(classify(router, case["message"], semaphore) for case in cases))

    per_intent: dict[str, Counter] = defaultdict(Counter)
    errors = []
    for case, predicted in zip(cases, predictions, strict=True):
        primary = case["expected"][0]
        ok = predicted in case["expected"]
        per_intent[primary]["total"] += 1
        per_intent[primary]["correct"] += ok
        if not ok:
            errors.append((case["message"], case["expected"], predicted))

    correct = sum(c["correct"] for c in per_intent.values())
    accuracy = correct / len(cases)
    print(f"意图路由准确率：{correct}/{len(cases)} = {accuracy:.1%}\n")
    print(f"{'意图':<22}{'正确/总数':>10}")
    for intent, counter in sorted(per_intent.items()):
        print(f"{intent:<22}{counter['correct']:>4}/{counter['total']:<5}")
    if errors:
        print("\n错误样例：")
        for message, expected, predicted in errors:
            print(f"  {message!r}  期望 {'/'.join(expected)}，实际 {predicted}")

    if accuracy < args.min_accuracy:
        raise SystemExit(f"准确率 {accuracy:.1%} 低于阈值 {args.min_accuracy:.0%}")


if __name__ == "__main__":
    asyncio.run(main())

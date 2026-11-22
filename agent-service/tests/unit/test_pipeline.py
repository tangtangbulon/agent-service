from types import SimpleNamespace

import pytest

from app.rag.pipeline import (
    IngestQualityError,
    IngestTruncatedError,
    RAGPipeline,
    count_headings,
    last_heading_path,
    split_markdown_segments,
)


class FakeModel:
    """按顺序返回预设的整理结果，记录收到的系统提示词"""

    def __init__(self, outputs: list[tuple[str, str]]):
        self._outputs = iter(outputs)
        self.system_prompts: list[str] = []

    def invoke(self, messages):
        self.system_prompts.append(messages[0]["content"])
        text, finish_reason = next(self._outputs)
        return SimpleNamespace(text=text, response_metadata={"finish_reason": finish_reason})


def make_pipeline(raw_markdown: str, outputs: list[tuple[str, str]]) -> tuple[RAGPipeline, FakeModel]:
    pipeline = RAGPipeline.__new__(RAGPipeline)  # 跳过构造函数，避免连接 MinerU 和模型服务
    model = FakeModel(outputs)
    pipeline._model = model  # type: ignore[assignment]
    pipeline._parse_pdf = lambda clause_name: raw_markdown  # type: ignore[method-assign]
    return pipeline, model


RAW = "# 一\n\n## 二\n\n" + "内容" * 2900 + "\n\n## 第五条\n\n" + "正文" * 2000
GOOD_FIRST = "# 一\n\n## 二\n\n" + "内容" * 2900
GOOD_SECOND = "## 第五条\n\n" + "正文" * 2000


class TestSplitMarkdownSegments:
    def test_segments_are_bounded_and_lossless(self):
        doc = "\n\n".join(f"## 第{i}条\n\n" + "条款内容" * 300 for i in range(1, 21))
        segments = split_markdown_segments(doc, max_chars=6000)
        assert len(segments) > 1
        assert all(len(s) <= 6000 for s in segments)
        assert "\n\n".join(segments) == doc

    def test_trailing_heading_moves_to_next_segment(self):
        segments = split_markdown_segments(RAW)
        assert "\n\n".join(segments) == RAW
        assert segments[1].startswith("## 第五条")
        assert not segments[0].rstrip().endswith("## 第五条")

    def test_oversized_paragraph_is_split_by_lines(self):
        paragraph = "\n".join("很长的一行" * 50 for _ in range(100))
        segments = split_markdown_segments(paragraph, max_chars=2000)
        assert all(len(s) <= 2000 for s in segments)


def test_heading_helpers():
    markdown = "# 第二部分\n\n## 第六条\n\n### （一）\n\n正文\n\n# 第七部分 释义\n\n## 四、等待期\n\n指……"
    assert count_headings(markdown) == 5
    assert last_heading_path(markdown) == ["# 第七部分 释义", "## 四、等待期"]
    assert last_heading_path("#不是标题\n正文") == []


class TestParseAndOptimize:
    def test_heading_path_is_passed_to_next_segment(self):
        pipeline, model = make_pipeline(RAW, [(GOOD_FIRST, "stop"), (GOOD_SECOND, "stop")])
        result = pipeline._parse_and_optimize("x.pdf")
        assert count_headings(result) == 3
        assert "# 一 > ## 二" in model.system_prompts[1]

    def test_truncated_output_aborts(self):
        pipeline, _ = make_pipeline(RAW, [(GOOD_FIRST, "length")])
        with pytest.raises(IngestTruncatedError):
            pipeline._parse_and_optimize("x.pdf")

    def test_bad_segment_is_retried(self):
        outputs = [(GOOD_FIRST, "stop"), ("太短", "stop"), ("无标题" + "正文" * 2000, "stop"), (GOOD_SECOND, "stop")]
        pipeline, _ = make_pipeline(RAW, outputs)
        assert count_headings(pipeline._parse_and_optimize("x.pdf")) == 3

    def test_segment_failing_every_retry_aborts(self):
        pipeline, _ = make_pipeline(RAW, [(GOOD_FIRST, "stop")] + [("短", "stop")] * 3)
        with pytest.raises(IngestQualityError, match="重试 3 次"):
            pipeline._parse_and_optimize("x.pdf")

    def test_document_without_headings_aborts(self):
        raw = "正文" * 100
        pipeline, _ = make_pipeline(raw, [("整理后的正文" * 40, "stop")])
        with pytest.raises(IngestQualityError, match="标题"):
            pipeline._parse_and_optimize("x.pdf")

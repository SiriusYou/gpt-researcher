"""Tests for the alignment scoring system."""

import json
import time
import pytest

from gpt_researcher.alignment.models import (
    AlignmentResult,
    AlignmentScore,
    RetryDecision,
)
from gpt_researcher.alignment.scorer import AlignmentScorer, _truncate_report
from gpt_researcher.alignment.orchestrator import RetryOrchestrator


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestAlignmentScore:
    def test_scored_creation(self):
        score = AlignmentScore(score=8.5, reasoning="Good", suggestions=["minor fix"], cost=0.01, status="scored")
        assert score.score == 8.5
        assert score.status == "scored"

    def test_llm_unavailable(self):
        score = AlignmentScore(score=None, reasoning="Connection failed", status="llm_unavailable")
        assert score.score is None
        assert score.status == "llm_unavailable"

    def test_parse_error(self):
        score = AlignmentScore(score=0.0, reasoning="Bad JSON", status="parse_error")
        assert score.score == 0.0
        assert score.status == "parse_error"

    def test_frozen(self):
        score = AlignmentScore(score=7.0, reasoning="OK")
        with pytest.raises(AttributeError):
            score.score = 8.0


class TestRetryDecision:
    def test_creation(self):
        d = RetryDecision(should_retry=True, reason="below_threshold")
        assert d.should_retry is True
        assert d.reason == "below_threshold"


class TestAlignmentResult:
    def test_creation(self):
        result = AlignmentResult(
            final_report="report",
            final_score=7.5,
            retry_count=1,
            score_history=[],
            total_cost=0.05,
            passed=True,
            termination_reason="passed",
        )
        assert result.passed is True
        assert result.retry_count == 1


# ---------------------------------------------------------------------------
# Truncation tests
# ---------------------------------------------------------------------------

class TestTruncateReport:
    def test_short_report_unchanged(self):
        short = "# Title\n\nShort report content."
        assert _truncate_report(short) == short

    def test_long_report_truncated(self):
        long_report = "# Title\n\n" + "word " * 10000 + "\n\n## Conclusion\n\nFinal thoughts."
        result = _truncate_report(long_report, max_tokens=500)
        assert len(result) < len(long_report)
        assert "# Title" in result
        assert "Conclusion" in result


# ---------------------------------------------------------------------------
# Scorer parse tests
# ---------------------------------------------------------------------------

class TestScorerParsing:
    def test_valid_json(self):
        response = json.dumps({
            "score": 8.2,
            "reasoning": "Good alignment",
            "suggestions": ["Add more detail"]
        })
        result = AlignmentScorer._parse_response(response)
        assert result.score == 8.2
        assert result.status == "scored"
        assert result.suggestions == ["Add more detail"]

    def test_json_in_markdown_fences(self):
        response = '```json\n{"score": 7.0, "reasoning": "OK", "suggestions": []}\n```'
        result = AlignmentScorer._parse_response(response)
        assert result.score == 7.0
        assert result.status == "scored"

    def test_invalid_json(self):
        result = AlignmentScorer._parse_response("not json at all")
        assert result.score == 0.0
        assert result.status == "parse_error"

    def test_missing_score_field(self):
        response = json.dumps({"reasoning": "No score here"})
        result = AlignmentScorer._parse_response(response)
        assert result.score == 0.0
        assert result.status == "parse_error"

    def test_score_clamped_to_range(self):
        response = json.dumps({"score": 15.0, "reasoning": "Over", "suggestions": []})
        result = AlignmentScorer._parse_response(response)
        assert result.score == 10.0

    def test_negative_score_clamped(self):
        response = json.dumps({"score": -2.0, "reasoning": "Under", "suggestions": []})
        result = AlignmentScorer._parse_response(response)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Orchestrator should_retry tests
# ---------------------------------------------------------------------------

class TestShouldRetry:
    def _make_orchestrator(self, **kwargs):
        defaults = dict(
            scorer=None,
            threshold=7.0,
            max_retries=2,
            stagnation_delta=0.5,
            auto_retry=True,
        )
        defaults.update(kwargs)
        return RetryOrchestrator(**defaults)

    def test_passed_score(self):
        orch = self._make_orchestrator()
        score = AlignmentScore(score=8.0, reasoning="Good")
        decision = orch.should_retry(score, [score])
        assert decision.should_retry is False
        assert decision.reason == "passed"

    def test_below_threshold_retries(self):
        orch = self._make_orchestrator()
        score = AlignmentScore(score=5.0, reasoning="Low")
        decision = orch.should_retry(score, [score])
        assert decision.should_retry is True
        assert decision.reason == "below_threshold"

    def test_llm_unavailable_no_retry(self):
        orch = self._make_orchestrator()
        score = AlignmentScore(score=None, reasoning="Failed", status="llm_unavailable")
        decision = orch.should_retry(score, [score])
        assert decision.should_retry is False
        assert decision.reason == "llm_unavailable"

    def test_max_retries_reached(self):
        orch = self._make_orchestrator(max_retries=1)
        s1 = AlignmentScore(score=4.0, reasoning="Low")
        s2 = AlignmentScore(score=5.0, reasoning="Still low")
        decision = orch.should_retry(s2, [s1, s2])
        assert decision.should_retry is False
        assert decision.reason == "max_retries"

    def test_stagnation_detected(self):
        orch = self._make_orchestrator(stagnation_delta=0.5)
        s1 = AlignmentScore(score=5.0, reasoning="Low")
        s2 = AlignmentScore(score=5.2, reasoning="Barely better")
        decision = orch.should_retry(s2, [s1, s2])
        assert decision.should_retry is False
        assert decision.reason == "stagnation"

    def test_auto_retry_disabled(self):
        orch = self._make_orchestrator(auto_retry=False)
        score = AlignmentScore(score=3.0, reasoning="Very low")
        decision = orch.should_retry(score, [score])
        assert decision.should_retry is False

    def test_none_scores_skipped_in_stagnation(self):
        orch = self._make_orchestrator(max_retries=5)
        s1 = AlignmentScore(score=5.0, reasoning="OK")
        s2 = AlignmentScore(score=None, reasoning="Error", status="parse_error")
        s3 = AlignmentScore(score=6.0, reasoning="Better")
        # Only s1 and s3 are valid, improvement = 1.0 > 0.5 delta
        # retry_count = len(history)-1 = 2, max_retries = 5, so not maxed out
        decision = orch.should_retry(s3, [s1, s2, s3])
        assert decision.should_retry is True


# ---------------------------------------------------------------------------
# Integration test with mock LLM
# ---------------------------------------------------------------------------

class MockConfig:
    smart_llm_model = "openai:gpt-4o-mini"
    smart_llm_provider = "openai"
    smart_token_limit = 4000
    llm_kwargs = {}
    alignment_enabled = True
    alignment_score_threshold = 7.0
    alignment_max_retries = 2
    alignment_auto_retry = True
    alignment_stagnation_delta = 0.5


class MockResearcher:
    """Minimal mock that simulates GPTResearcher for orchestrator tests."""

    def __init__(self, reports, cfg=None):
        self.reports = list(reports)
        self.query = "test query"
        self.report_type = "research_report"
        self.context = []
        self.cfg = cfg or MockConfig()
        self._costs = 0.0
        self._attempt = 0

    async def conduct_research(self):
        self.context = ["mock context"]
        return self.context

    async def write_report(self, custom_prompt="", **kwargs):
        report = self.reports[min(self._attempt, len(self.reports) - 1)]
        self._attempt += 1
        return report

    def get_costs(self):
        return self._costs

    def add_costs(self, cost):
        self._costs += cost


class MockScorer(AlignmentScorer):
    """Scorer that returns predetermined scores."""

    def __init__(self, scores):
        self._scores = list(scores)
        self._idx = 0

    async def evaluate_alignment(self, query, report, report_type="research_report"):
        score = self._scores[min(self._idx, len(self._scores) - 1)]
        self._idx += 1
        return score


@pytest.mark.asyncio
async def test_orchestrator_passes_first_try():
    scores = [AlignmentScore(score=8.0, reasoning="Good")]
    scorer = MockScorer(scores)
    orch = RetryOrchestrator(scorer=scorer, threshold=7.0)
    researcher = MockResearcher(reports=["great report"])

    result = await orch.run(researcher)

    assert result.passed is True
    assert result.retry_count == 0
    assert result.termination_reason == "passed"
    assert result.final_score == 8.0


@pytest.mark.asyncio
async def test_orchestrator_retries_then_passes():
    scores = [
        AlignmentScore(score=5.0, reasoning="Low", suggestions=["Add more"]),
        AlignmentScore(score=7.5, reasoning="Better"),
    ]
    scorer = MockScorer(scores)
    orch = RetryOrchestrator(scorer=scorer, threshold=7.0, max_retries=2)
    researcher = MockResearcher(reports=["bad report", "good report"])

    result = await orch.run(researcher)

    assert result.passed is True
    assert result.retry_count == 1
    assert len(result.score_history) == 2


@pytest.mark.asyncio
async def test_orchestrator_max_retries_exhausted():
    scores = [
        AlignmentScore(score=3.0, reasoning="Bad"),
        AlignmentScore(score=4.0, reasoning="Still bad"),
        AlignmentScore(score=5.0, reasoning="Improving"),
    ]
    scorer = MockScorer(scores)
    orch = RetryOrchestrator(scorer=scorer, threshold=7.0, max_retries=2)
    researcher = MockResearcher(reports=["r1", "r2", "r3"])

    result = await orch.run(researcher)

    assert result.passed is False
    assert result.termination_reason == "max_retries"
    assert result.final_score == 5.0


@pytest.mark.asyncio
async def test_orchestrator_llm_unavailable_no_retry():
    scores = [AlignmentScore(score=None, reasoning="Failed", status="llm_unavailable")]
    scorer = MockScorer(scores)
    orch = RetryOrchestrator(scorer=scorer, threshold=7.0)
    researcher = MockResearcher(reports=["report"])

    result = await orch.run(researcher)

    assert result.passed is False
    assert result.termination_reason == "llm_unavailable"
    assert result.retry_count == 0


@pytest.mark.asyncio
async def test_orchestrator_stagnation_stops_retry():
    scores = [
        AlignmentScore(score=5.0, reasoning="Low"),
        AlignmentScore(score=5.1, reasoning="Barely improved"),
    ]
    scorer = MockScorer(scores)
    orch = RetryOrchestrator(scorer=scorer, threshold=7.0, max_retries=3, stagnation_delta=0.5)
    researcher = MockResearcher(reports=["r1", "r2"])

    result = await orch.run(researcher)

    assert result.termination_reason == "stagnation"
    assert result.retry_count == 1


@pytest.mark.asyncio
async def test_orchestrator_timeout_terminates():
    """Timeout fires when elapsed > 3 * max(first_cycle, 60s).

    We patch the floor to 0 so instant mocks can trigger timeout.
    """
    scores = [
        AlignmentScore(score=4.0, reasoning="Low"),
        AlignmentScore(score=5.0, reasoning="Still low"),
    ]
    scorer = MockScorer(scores)
    orch = RetryOrchestrator(scorer=scorer, threshold=7.0, max_retries=3)

    class SlowResearcher(MockResearcher):
        """Simulates a researcher where first cycle takes measurable time."""
        async def conduct_research(self):
            self.context = ["mock"]
            return self.context

    researcher = SlowResearcher(reports=["r1", "r2", "r3"])

    # Monkey-patch the orchestrator to use a tiny floor for testing
    import gpt_researcher.alignment.orchestrator as orch_mod
    original_run = orch.run

    async def patched_run(researcher, websocket_callback=None):
        # Temporarily override the timeout floor by patching time
        import unittest.mock
        original_monotonic = time.monotonic
        call_count = [0]
        base_time = original_monotonic()

        def fake_monotonic():
            call_count[0] += 1
            # After first cycle measurement, jump ahead to trigger timeout
            if call_count[0] > 4:
                return base_time + 300  # 5 minutes elapsed
            return original_monotonic()

        with unittest.mock.patch("time.monotonic", side_effect=fake_monotonic):
            return await original_run(researcher, websocket_callback)

    result = await patched_run(researcher)
    assert result.termination_reason == "timeout"


@pytest.mark.asyncio
async def test_orchestrator_advisory_mode():
    scores = [AlignmentScore(score=4.0, reasoning="Low")]
    scorer = MockScorer(scores)
    orch = RetryOrchestrator(scorer=scorer, threshold=7.0, auto_retry=False)
    researcher = MockResearcher(reports=["report"])

    result = await orch.run(researcher)

    assert result.retry_count == 0
    assert result.final_score == 4.0
    # Advisory mode: scored but no retry
    assert len(result.score_history) == 1

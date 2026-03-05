"""Retry orchestrator: decides whether to retry based on alignment scores."""

from __future__ import annotations

import logging
import time

from .models import AlignmentResult, AlignmentScore, RetryDecision
from .scorer import AlignmentScorer

logger = logging.getLogger(__name__)


class RetryOrchestrator:
    """Manages the alignment scoring and retry loop."""

    def __init__(
        self,
        scorer: AlignmentScorer,
        threshold: float = 7.0,
        max_retries: int = 2,
        stagnation_delta: float = 0.5,
        auto_retry: bool = True,
    ):
        self.scorer = scorer
        self.threshold = threshold
        self.max_retries = max_retries
        self.stagnation_delta = stagnation_delta
        self.auto_retry = auto_retry

    def should_retry(
        self, current_score: AlignmentScore, score_history: list[AlignmentScore]
    ) -> RetryDecision:
        """Decide whether to retry based on current score and history."""
        if current_score.status == "llm_unavailable":
            return RetryDecision(should_retry=False, reason="llm_unavailable")

        if current_score.score is not None and current_score.score >= self.threshold:
            return RetryDecision(should_retry=False, reason="passed")

        if not self.auto_retry:
            return RetryDecision(should_retry=False, reason="passed")

        # Count actual retries (score_history includes all attempts)
        retry_count = len(score_history) - 1  # first attempt is not a retry
        if retry_count >= self.max_retries:
            return RetryDecision(should_retry=False, reason="max_retries")

        # Stagnation detection: compare valid scores only
        valid_scores = [
            s.score for s in score_history
            if s.score is not None
        ]
        if len(valid_scores) >= 2:
            improvement = valid_scores[-1] - valid_scores[-2]
            if improvement < self.stagnation_delta:
                return RetryDecision(should_retry=False, reason="stagnation")

        return RetryDecision(should_retry=True, reason="below_threshold")

    async def run(
        self,
        researcher,
        websocket_callback=None,
    ) -> AlignmentResult:
        """Execute the full research + alignment + retry loop.

        Args:
            researcher: GPTResearcher instance (already has query, config, etc.)
            websocket_callback: Optional async callable for status updates.
        """
        score_history: list[AlignmentScore] = []
        best_report = ""
        best_score: float | None = None
        research_start = time.monotonic()

        # Time budget: 3x the first research cycle
        first_cycle_duration: float | None = None

        for attempt in range(1 + self.max_retries):
            # Notify: evaluating
            if websocket_callback:
                await websocket_callback({
                    "type": "alignment",
                    "status": "evaluating",
                    "message": "Evaluating alignment...",
                })

            # Conduct research + write report
            try:
                if attempt == 0:
                    cycle_start = time.monotonic()
                    await researcher.conduct_research()
                    report = await researcher.write_report()
                    first_cycle_duration = time.monotonic() - cycle_start
                else:
                    # Build enhanced prompt from suggestions
                    suggestions_text = "; ".join(
                        score_history[-1].suggestions
                    ) if score_history[-1].suggestions else ""
                    custom_prompt = (
                        f"Previous attempt scored {score_history[-1].score}/10. "
                        f"Improvements needed: {suggestions_text}"
                    ) if suggestions_text else ""

                    # Reset context for fresh research
                    researcher.context = []
                    await researcher.conduct_research()
                    report = await researcher.write_report(
                        custom_prompt=custom_prompt
                    )
            except Exception as e:
                logger.warning("Research failed on attempt %d: %s", attempt, e)
                if best_report:
                    return AlignmentResult(
                        final_report=best_report,
                        final_score=best_score,
                        retry_count=max(0, attempt - 1),
                        score_history=score_history,
                        total_cost=researcher.get_costs(),
                        passed=best_score is not None and best_score >= self.threshold,
                        termination_reason="timeout",
                    )
                raise

            # Evaluate alignment
            alignment_score = await self.scorer.evaluate_alignment(
                query=researcher.query,
                report=report,
                report_type=researcher.report_type,
            )
            score_history.append(alignment_score)

            # Track best
            if alignment_score.score is not None:
                if best_score is None or alignment_score.score > best_score:
                    best_report = report
                    best_score = alignment_score.score

            # Notify: score
            if websocket_callback and alignment_score.status == "scored":
                await websocket_callback({
                    "type": "alignment",
                    "status": "score",
                    "score": alignment_score.score,
                    "threshold": self.threshold,
                })
            elif websocket_callback and alignment_score.status == "llm_unavailable":
                await websocket_callback({
                    "type": "alignment",
                    "status": "skipped",
                    "reason": "llm_unavailable",
                })

            # Decide retry
            decision = self.should_retry(alignment_score, score_history)

            if not decision.should_retry:
                # Use current report if it's the best (or only)
                final_report = best_report or report
                final_score = best_score if best_score is not None else alignment_score.score
                passed = final_score is not None and final_score >= self.threshold

                if websocket_callback:
                    await websocket_callback({
                        "type": "alignment",
                        "status": "complete",
                        "final_score": final_score,
                        "retries": attempt,
                    })

                return AlignmentResult(
                    final_report=final_report,
                    final_score=final_score,
                    retry_count=attempt,
                    score_history=score_history,
                    total_cost=researcher.get_costs(),
                    passed=passed,
                    termination_reason=decision.reason,
                )

            # Check timeout before retrying (floor at 60s to avoid false triggers)
            if first_cycle_duration is not None:
                elapsed = time.monotonic() - research_start
                timeout_budget = 3 * max(first_cycle_duration, 60.0)
                if elapsed > timeout_budget:
                    if websocket_callback:
                        await websocket_callback({
                            "type": "alignment",
                            "status": "complete",
                            "final_score": best_score,
                            "retries": attempt,
                        })
                    return AlignmentResult(
                        final_report=best_report or report,
                        final_score=best_score,
                        retry_count=attempt,
                        score_history=score_history,
                        total_cost=researcher.get_costs(),
                        passed=best_score is not None and best_score >= self.threshold,
                        termination_reason="timeout",
                    )

            # Notify: retrying
            if websocket_callback:
                await websocket_callback({
                    "type": "alignment",
                    "status": "retrying",
                    "attempt": attempt + 1,
                    "max": self.max_retries,
                    "reason": decision.reason,
                })

        # Should not reach here, but safety fallback
        return AlignmentResult(
            final_report=best_report,
            final_score=best_score,
            retry_count=self.max_retries,
            score_history=score_history,
            total_cost=researcher.get_costs(),
            passed=best_score is not None and best_score >= self.threshold,
            termination_reason="max_retries",
        )

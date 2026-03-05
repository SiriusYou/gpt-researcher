"""Data models for the alignment scoring system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlignmentScore:
    """Result of a single alignment evaluation."""

    score: float | None  # 0.0-10.0, None if evaluation unavailable
    reasoning: str
    suggestions: list[str] = field(default_factory=list)
    cost: float = 0.0
    status: str = "scored"  # "scored" | "llm_unavailable" | "parse_error"


@dataclass(frozen=True)
class RetryDecision:
    """Decision on whether to retry research."""

    should_retry: bool
    reason: str  # "below_threshold" | "stagnation" | "max_retries" | "passed" | "llm_unavailable"


@dataclass
class AlignmentResult:
    """Aggregated result of a research + alignment + retry cycle."""

    final_report: str
    final_score: float | None
    retry_count: int
    score_history: list[AlignmentScore]
    total_cost: float
    passed: bool | None  # None if unscored
    termination_reason: str  # "passed" | "max_retries" | "stagnation" | "timeout" | "llm_unavailable"

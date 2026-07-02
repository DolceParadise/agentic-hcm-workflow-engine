from __future__ import annotations

from dataclasses import dataclass

from memory import MemoryMatch
from models import Anomaly, RecommendedAction


@dataclass(frozen=True)
class LearnedProposal:
    action: RecommendedAction
    confidence_adjustment: float
    sample_count: int
    memory_matches: tuple[MemoryMatch, ...] = ()


class FeedbackLearner:
    """Reward-weighted contextual bandit with semantic-memory warm starts."""

    def __init__(self, connection_factory) -> None:
        self._connect = connection_factory

    def propose(self, anomaly: Anomaly, matches: list[MemoryMatch]) -> LearnedProposal:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT action, reward FROM rl_experiences
                WHERE category = ?""",
                (anomaly.category,),
            ).fetchall()
        rewards: dict[str, list[float]] = {}
        for row in rows:
            rewards.setdefault(row["action"], []).append(float(row["reward"]))
        for match in matches:
            if match.similarity >= 0.55:
                rewards.setdefault(match.action, []).append(match.reward * match.similarity)
        action: RecommendedAction = anomaly.recommended_action
        adjustment = 0.0
        if rewards:
            scores = {
                candidate: sum(values) / (len(values) + 1) for candidate, values in rewards.items()
            }
            positive = {candidate: score for candidate, score in scores.items() if score > 0}
            if positive:
                action = max(positive, key=positive.get)  # type: ignore[assignment]
                adjustment = min(0.12, positive[action] * 0.12)
            elif scores.get(action, 0) < 0:
                adjustment = max(-0.12, scores[action] * 0.12)
        return LearnedProposal(action, adjustment, len(rows), tuple(matches))

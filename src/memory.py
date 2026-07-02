from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from models import Anomaly, RecommendedAction


@dataclass(frozen=True)
class MemoryMatch:
    anomaly_id: str
    action: RecommendedAction
    outcome: str
    reward: float
    similarity: float


class EpisodicVectorMemory:
    """Small, dependency-free vector store backed by SQLite.

    Feature hashing provides stable semantic-ish sparse vectors suitable for this bounded
    incident vocabulary. The storage contract can be replaced by Chroma or Qdrant later.
    """

    DIMENSIONS = 256

    def __init__(self, connection_factory) -> None:
        self._connect = connection_factory

    @classmethod
    def context(cls, anomaly: Anomaly) -> str:
        evidence = " ".join(f"{key} {value}" for key, value in sorted(anomaly.evidence.items()))
        return f"{anomaly.category} {anomaly.risk} {anomaly.description} {evidence}".lower()

    @classmethod
    def embed(cls, text: str) -> list[float]:
        vector = [0.0] * cls.DIMENSIONS
        tokens = re.findall(r"[a-z0-9_-]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            index = int.from_bytes(digest, "big") % cls.DIMENSIONS
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def add(
        self,
        anomaly: Anomaly,
        action: RecommendedAction,
        outcome: str,
        reward: float,
    ) -> None:
        context = self.context(anomaly)
        vector = self.embed(context)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO episodic_memory
                (anomaly_id, context, vector_json, action, outcome, reward)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (anomaly.anomaly_id, context, json.dumps(vector), action, outcome, reward),
            )

    def search(self, anomaly: Anomaly, limit: int = 3) -> list[MemoryMatch]:
        query = self.embed(self.context(anomaly))
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT anomaly_id, vector_json, action, outcome, reward
                FROM episodic_memory ORDER BY id DESC LIMIT 500"""
            ).fetchall()
        matches = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            similarity = sum(left * right for left, right in zip(query, vector, strict=True))
            matches.append(
                MemoryMatch(
                    row["anomaly_id"],
                    row["action"],
                    row["outcome"],
                    float(row["reward"]),
                    round(similarity, 4),
                )
            )
        return sorted(matches, key=lambda item: item.similarity, reverse=True)[:limit]

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from hcm_engine.models import TraceStep


@dataclass
class TraceCollector:
    steps: list[TraceStep] = field(default_factory=list)

    @contextmanager
    def span(
        self, name: str, kind: str, inputs: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        result: dict[str, Any] = {}
        status = "ok"
        try:
            yield result
        except Exception as exc:
            status = "error"
            result["error"] = str(exc)
            raise
        finally:
            self.steps.append(
                TraceStep(
                    name=name,
                    kind=kind,  # type: ignore[arg-type]
                    input=inputs,
                    output=result.get("output", result),
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    status=status,  # type: ignore[arg-type]
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    cost_usd=result.get("cost_usd", 0.0),
                )
            )

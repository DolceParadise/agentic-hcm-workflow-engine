from __future__ import annotations

from llm import OpenRouterClient
from models import SearchResult
from rag import PolicyIndex
from tracing import TraceCollector


class PolicyAgent:
    def __init__(
        self,
        index: PolicyIndex,
        llm: OpenRouterClient,
        threshold: float,
        top_k: int,
    ) -> None:
        self.index = index
        self.llm = llm
        self.threshold = threshold
        self.top_k = top_k

    def run(self, question: str, trace: TraceCollector) -> tuple[str, list[SearchResult]]:
        with trace.span("policy_retrieval", "retrieval", {"query": question}) as span:
            results = self.index.search(question, self.top_k)
            span["output"] = {
                "matches": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "heading": item.chunk.heading,
                        "score": round(item.score, 4),
                    }
                    for item in results
                ]
            }
        relevant = [result for result in results if result.score >= self.threshold]
        if not relevant:
            return (
                "I couldn't find this in the supplied policy document, so I won't guess. "
                "Please contact HR or the Business Conduct office.",
                [],
            )
        context = "\n\n".join(
            f"[{item.chunk.chunk_id}: {item.chunk.heading}]\n{item.chunk.text}" for item in relevant
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Policy Agent. Answer only from POLICY CONTEXT. "
                    "If the context is insufficient, say so. Keep the answer concise and cite "
                    "claims using [chunk-id]. Never invent benefits, durations, eligibility, "
                    "or law."
                ),
            },
            {"role": "user", "content": f"POLICY CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
        ]
        with trace.span(
            "grounded_policy_answer",
            "llm",
            {"model": self.llm.model, "context_chunks": len(relevant)},
        ) as span:
            response = self.llm.chat(messages, temperature=0.0, max_tokens=350)
            span.update(
                output={"answer": response.content},
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )
        return response.content, relevant

from __future__ import annotations

import numpy as np
import pytest

from hcm_engine.config import Settings
from hcm_engine.llm import LLMResponse
from hcm_engine.rag import PolicyIndex
from hcm_engine.state import SQLiteStateStore


class FakeEmbedder:
    model_name = "Qwen/Qwen3-Embedding-8B"

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        vectors = []
        terms = ("harassment", "confidential", "gift", "outside", "apple")
        for text in texts:
            lowered = text.lower()
            vector = np.array([lowered.count(term) for term in terms], dtype=np.float32)
            if not vector.any():
                vector[-1] = 0.1
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


class FakeLLM:
    model = "openai/gpt-oss-120b:free"

    def chat(self, messages, **kwargs) -> LLMResponse:
        if kwargs.get("json_mode"):
            content = "{}"
        else:
            content = "Report harassment to HR, a manager, or the helpline. [policy-005]"
        return LLMResponse(content=content, input_tokens=100, output_tokens=20, cost_usd=0.0)


@pytest.fixture
def settings(tmp_path, project_root):
    return Settings(
        project_root=project_root,
        openrouter_api_key="test",
        db_path=tmp_path / "hcm.db",
        index_path=tmp_path / "index.npz",
        policy_path=project_root / "data/hr_policy_corpus.txt",
        retrieval_threshold=0.1,
    )


@pytest.fixture
def project_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_index(settings):
    return PolicyIndex(settings.policy_path, settings.index_path, FakeEmbedder())


@pytest.fixture
def state_store(settings):
    return SQLiteStateStore(settings.db_path)


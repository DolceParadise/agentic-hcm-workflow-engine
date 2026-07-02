from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from models import PolicyChunk, SearchResult


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


class QwenEmbedder:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        prompt_name = "query" if query else None
        vectors = self.model.encode(
            texts,
            prompt_name=prompt_name,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def chunk_policy(path: Path, max_words: int = 220, overlap_words: int = 35) -> list[PolicyChunk]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^###\s+", text)
    chunks: list[PolicyChunk] = []
    for section in sections:
        if not section.strip():
            continue
        heading, _, body = section.strip().partition("\n")
        words = body.split()
        start = 0
        part = 1
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                PolicyChunk(
                    chunk_id=f"policy-{len(chunks) + 1:03d}",
                    heading=heading if part == 1 else f"{heading} (continued)",
                    text=chunk_text,
                    source=path.name,
                )
            )
            if end == len(words):
                break
            start = max(end - overlap_words, start + 1)
            part += 1
    return chunks


class PolicyIndex:
    def __init__(self, policy_path: Path, index_path: Path, embedder: Embedder) -> None:
        self.policy_path = policy_path
        self.index_path = index_path
        self.embedder = embedder
        self.chunks: list[PolicyChunk] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)

    def ensure_loaded(self) -> None:
        fingerprint = f"{self.policy_path.stat().st_mtime_ns}:{self.policy_path.stat().st_size}"
        if self.index_path.exists():
            data = np.load(self.index_path, allow_pickle=False)
            metadata = json.loads(str(data["metadata"]))
            is_current = (
                metadata["fingerprint"] == fingerprint
                and metadata["model"] == self.embedder.model_name
            )
            if is_current:
                self.chunks = [PolicyChunk(**item) for item in metadata["chunks"]]
                self.embeddings = data["embeddings"]
                return
        self.chunks = chunk_policy(self.policy_path)
        documents = [f"{chunk.heading}\n{chunk.text}" for chunk in self.chunks]
        self.embeddings = self.embedder.encode(documents)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "fingerprint": fingerprint,
            "model": self.embedder.model_name,
            "chunks": [chunk.__dict__ for chunk in self.chunks],
        }
        np.savez_compressed(
            self.index_path, embeddings=self.embeddings, metadata=json.dumps(metadata)
        )

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        self.ensure_loaded()
        query_vector = self.embedder.encode([query], query=True)[0]
        scores = self.embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(scores[index]))
            for index in indices
        ]

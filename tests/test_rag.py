from rag import chunk_policy


def test_policy_is_chunked_by_heading(settings):
    chunks = chunk_policy(settings.policy_path)
    assert len(chunks) >= 20
    assert chunks[0].heading == "Overview"
    assert all(chunk.source == "hr_policy_corpus.txt" for chunk in chunks)


def test_retrieval_returns_relevant_section(fake_index):
    results = fake_index.search("How do I report harassment?", top_k=3)
    assert results[0].chunk.heading == "Harassment and Discrimination"
    assert results[0].score > 0.5

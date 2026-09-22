"""Unit tests for the JevRelevanceRetriever drop-in filter."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pytest import fixture, mark

from conftest import FakeTypeSafeClassifier
from jev_relevance import (
    JevRelevanceConfig,
    JevRelevanceRetriever,
    build_scoring_strategy,
)
from jev_relevance.constants import (
    CONFIDENCE_METADATA_KEY,
    RELEVANCE_METADATA_KEY,
)


class StubRetriever(BaseRetriever):
    """Returns scripted candidates; used to verify the filter, not retrieval."""

    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        return self.documents


@fixture
def candidates() -> list[Document]:
    return [
        Document(page_content="The Eiffel Tower is in Paris.", metadata={"id": "a"}),
        Document(page_content="Python is a programming language.", metadata={"id": "b"}),
        Document(page_content="The moon is made of cheese.", metadata={"id": "c"}),
    ]


def build_retriever(
    classifier: FakeTypeSafeClassifier,
    candidates: list[Document],
    **config_overrides,
) -> JevRelevanceRetriever:
    config = JevRelevanceConfig(**config_overrides)
    return JevRelevanceRetriever(
        base_retriever=StubRetriever(documents=candidates),
        config=config,
        strategy=build_scoring_strategy(classifier, config),
    )


def test_keeps_only_chunks_above_threshold(candidates) -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9, 0.2, 0.6])
    retriever = build_retriever(classifier, candidates, threshold=0.5)
    results = retriever.invoke("Where is the Eiffel Tower?")
    assert [document.metadata["id"] for document in results] == ["a", "c"]


def test_sorts_by_relevance_descending(candidates) -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.4, 0.9, 0.7])
    retriever = build_retriever(classifier, candidates, threshold=0.5)
    results = retriever.invoke("About what?")
    assert [document.metadata["id"] for document in results] == ["b", "c"]
    assert results[0].metadata[RELEVANCE_METADATA_KEY] == 0.9


def test_attaches_relevance_metadata(candidates) -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9, 0.1, 0.0])
    retriever = build_retriever(classifier, candidates, threshold=0.5)
    results = retriever.invoke("Any query")
    assert results[0].metadata[RELEVANCE_METADATA_KEY] == 0.9


def test_top_k_limits_results(candidates) -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9, 0.8, 0.7])
    retriever = build_retriever(classifier, candidates, threshold=0.5, top_k=2)
    results = retriever.invoke("Any query")
    assert len(results) == 2


def test_empty_candidates_return_empty_list() -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9])
    config = JevRelevanceConfig()
    retriever = JevRelevanceRetriever(
        base_retriever=StubRetriever(documents=[]),
        config=config,
        strategy=build_scoring_strategy(classifier, config),
    )
    assert retriever.invoke("Any query") == []


def test_score_mode_uses_grade_threshold(candidates) -> None:
    classifier = FakeTypeSafeClassifier(score_values=[3.0, 1.0, 2.0])
    retriever = build_retriever(
        classifier, candidates, question_mode="score", score_threshold=2.0
    )
    results = retriever.invoke("Any query")
    assert [document.metadata["id"] for document in results] == ["a", "c"]
    # Score mode also surfaces the confidence metadata.
    assert results[0].metadata[CONFIDENCE_METADATA_KEY] == 0.9


@mark.asyncio
async def test_async_variant_matches_sync(candidates) -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9, 0.2, 0.6])
    retriever = build_retriever(classifier, candidates, threshold=0.5)
    results = await retriever.ainvoke("Where is the Eiffel Tower?")
    assert [document.metadata["id"] for document in results] == ["a", "c"]


def test_fail_open_returns_candidates_on_scoring_error(candidates) -> None:
    class ExplodingClassifier(FakeTypeSafeClassifier):
        def invoke(self, request, config=None, **_: object):
            raise RuntimeError("provider exploded")

    classifier = ExplodingClassifier(noul_values=[0.9])
    retriever = build_retriever(classifier, candidates, fail_open=True)
    results = retriever.invoke("Any query")
    assert len(results) == len(candidates)


def test_raise_on_error_surfaces_scoring_failure(candidates) -> None:
    class ExplodingClassifier(FakeTypeSafeClassifier):
        def invoke(self, request, config=None, **_: object):
            raise RuntimeError("provider exploded")

    classifier = ExplodingClassifier(noul_values=[0.9])
    retriever = build_retriever(classifier, candidates, fail_open=False)
    try:
        retriever.invoke("Any query")
        raise AssertionError("expected RuntimeError to propagate")
    except RuntimeError:
        pass


def test_batched_mode_filters_correctly(candidates) -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9, 0.2, 0.6])
    retriever = build_retriever(
        classifier, candidates, threshold=0.5, scoring_mode="batched"
    )
    results = retriever.invoke("Where is the Eiffel Tower?")
    assert [document.metadata["id"] for document in results] == ["a", "c"]
    assert len(classifier.invoked_requests) == 1


def test_batched_mode_rejects_oversized_candidate_set() -> None:
    many_candidates = [Document(page_content=f"chunk-{index}") for index in range(40)]
    classifier = FakeTypeSafeClassifier(noul_values=[0.9])
    retriever = build_retriever(
        classifier, many_candidates, scoring_mode="batched", fail_open=False
    )
    try:
        retriever.invoke("Any query")
        raise AssertionError("expected ValueError for oversized batch")
    except ValueError:
        pass
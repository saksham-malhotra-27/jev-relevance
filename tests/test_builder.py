"""Unit tests for the fluent builder."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_typesafe import TypeSafeClassifier

from conftest import FakeTypeSafeClassifier
from jev_relevance import JevRelevanceRetrieverBuilder
from jev_relevance.config import JevRelevanceConfig, ProviderConfig
from jev_relevance.strategies import BatchedScorer, PerChunkScorer


class BareRetriever(BaseRetriever):
    """Minimal BaseRetriever stub the builder accepts."""

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return []


def test_builder_requires_base_retriever() -> None:
    with pytest.raises(ValueError):
        JevRelevanceRetrieverBuilder().build()


def test_builder_rejects_non_retriever() -> None:
    with pytest.raises(TypeError):
        JevRelevanceRetrieverBuilder().with_base_retriever("not-a-retriever")  # type: ignore[arg-type]


def test_builder_builds_with_defaults() -> None:
    from langchain_core.retrievers import BaseRetriever

    class StubRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [Document(page_content="doc")]

    retriever = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(StubRetriever())
        .with_provider(ProviderConfig(name="typesafe", api_key="dummy-key-for-build"))
        .build()
    )
    assert retriever.base_retriever is not None


def test_builder_wires_per_chunk_strategy() -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9])
    retriever = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(BareRetriever())  # type: ignore[arg-type]
        .with_classifier(classifier)
        .build()
    )
    assert isinstance(retriever.strategy, PerChunkScorer)


def test_builder_wires_batched_strategy() -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9])
    retriever = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(BareRetriever())  # type: ignore[arg-type]
        .with_classifier(classifier)
        .with_scoring_mode("batched")
        .build()
    )
    assert isinstance(retriever.strategy, BatchedScorer)


def test_builder_validates_classifier_type() -> None:
    with pytest.raises(TypeError):
        JevRelevanceRetrieverBuilder().with_classifier("not-a-classifier")  # type: ignore[arg-type]


def test_builder_accepts_real_classifier_shape() -> None:
    classifier = TypeSafeClassifier(api_key="dummy-key")
    builder = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(BareRetriever())  # type: ignore[arg-type]
        .with_classifier(classifier)
    )
    assert builder._classifier is classifier


def test_builder_config_overrides_apply() -> None:
    classifier = FakeTypeSafeClassifier(noul_values=[0.9])
    builder = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(BareRetriever())  # type: ignore[arg-type]
        .with_classifier(classifier)
        .with_threshold(0.7)
        .with_top_k(3)
    )
    assert builder._config.threshold == 0.7
    assert builder._config.top_k == 3


def test_builder_rejects_invalid_provider_config() -> None:
    with pytest.raises(TypeError):
        JevRelevanceRetrieverBuilder().with_provider("typesafe")  # type: ignore[arg-type]

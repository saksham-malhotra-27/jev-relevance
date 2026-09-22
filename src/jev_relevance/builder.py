"""Fluent builder that assembles a fully-wired :class:`JevRelevanceRetriever`.

The builder owns the composition order:

1. a validated :class:`JevRelevanceConfig` is built (or edited),
2. a classifier is produced by the provider factory (or supplied directly),
3. a :class:`ScoringStrategy` wraps the classifier for the chosen scoring mode,
4. the strategy and config are handed to the retriever, which wraps the caller's
   ``base_retriever``.

Every knob has a named default from :mod:`jev_relevance.constants`, so the
zero-argument path ``JevRelevanceRetrieverBuilder().with_base_retriever(...).build()``
already behaves sensibly.
"""

from __future__ import annotations

from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_typesafe import TypeSafeClassifier

from jev_relevance.config import JevRelevanceConfig, ProviderConfig
from jev_relevance.provider import build_classifier
from jev_relevance.retriever import JevRelevanceRetriever
from jev_relevance.strategies import build_scoring_strategy


class JevRelevanceRetrieverBuilder:
    """Build :class:`JevRelevanceRetriever` instances from validated settings.

    Examples:
        >>> builder = (
        ...     JevRelevanceRetrieverBuilder()
        ...     .with_base_retriever(my_vector_retriever)
        ...     .with_top_k(5)
        ...     .with_threshold(0.6)
        ... )
        >>> retriever = builder.build()

        Use OpenRouter as the Jev provider (instead of TypeSafe default). The
        api_key is a parameter; the library never reads env vars itself:

        >>> from jev_relevance.config import ProviderConfig
        >>> builder.with_provider(
        ...     ProviderConfig(name="openrouter", api_key="sk-...")
        ... )
    """

    def __init__(self) -> None:
        self._config: JevRelevanceConfig = JevRelevanceConfig()
        self._classifier: TypeSafeClassifier | None = None
        self._classifier_builder: Any | None = None
        self._base_retriever: BaseRetriever | None = None

    # ------------------------------------------------------------------
    # Configuration setters (chainable)
    # ------------------------------------------------------------------

    def with_base_retriever(self, retriever: BaseRetriever) -> "JevRelevanceRetrieverBuilder":
        """Set the underlying retrieval engine this filter wraps."""
        if not isinstance(retriever, BaseRetriever):
            raise TypeError(
                f"base_retriever must be a BaseRetriever, got {type(retriever).__name__}."
            )
        self._base_retriever = retriever
        return self

    def with_config(self, config: JevRelevanceConfig) -> "JevRelevanceRetrieverBuilder":
        """Replace the working config with a fully validated one."""
        if not isinstance(config, JevRelevanceConfig):
            raise TypeError(
                f"config must be a JevRelevanceConfig, got {type(config).__name__}."
            )
        self._config = config
        return self

    def with_question_mode(self, question_mode: str) -> "JevRelevanceRetrieverBuilder":
        """Choose ``"noul"`` (yes/no probability) or ``"score"`` (0..3 rubric)."""
        self._config = self._config.model_copy(
            update={"question_mode": question_mode}
        )
        return self

    def with_scoring_mode(self, scoring_mode: str) -> "JevRelevanceRetrieverBuilder":
        """Choose ``"per_chunk"`` (one call per pair) or ``"batched"`` (one call)."""
        self._config = self._config.model_copy(
            update={"scoring_mode": scoring_mode}
        )
        return self

    def with_threshold(self, threshold: float) -> "JevRelevanceRetrieverBuilder":
        """Set the noul-mode relevance probability cutoff (0..1)."""
        self._config = self._config.model_copy(update={"threshold": threshold})
        return self

    def with_score_threshold(self, score_threshold: float) -> "JevRelevanceRetrieverBuilder":
        """Set the score-mode grade cutoff (0..3)."""
        self._config = self._config.model_copy(
            update={"score_threshold": score_threshold}
        )
        return self

    def with_top_k(self, top_k: int | None) -> "JevRelevanceRetrieverBuilder":
        """Cap how many documents are returned after filtering (``None`` = all)."""
        self._config = self._config.model_copy(update={"top_k": top_k})
        return self

    def with_max_concurrency(self, max_concurrency: int) -> "JevRelevanceRetrieverBuilder":
        """Set maximum in-flight Jev calls in per-chunk mode."""
        self._config = self._config.model_copy(
            update={"max_concurrency": max_concurrency}
        )
        return self

    def with_fail_open(self, fail_open: bool) -> "JevRelevanceRetrieverBuilder":
        """Let scoring failures fall back to unfiltered candidates."""
        self._config = self._config.model_copy(update={"fail_open": fail_open})
        return self

    # ------------------------------------------------------------------
    # Classifier wiring (chainable)
    # ------------------------------------------------------------------

    def with_classifier(
        self, classifier: TypeSafeClassifier
    ) -> "JevRelevanceRetrieverBuilder":
        """Inject an already-constructed classifier (any provider/Typesafe client).

        Use this when you have custom clients, proxies, or want to reuse a
        long-lived connection pool.
        """
        if not isinstance(classifier, TypeSafeClassifier):
            raise TypeError(
                f"classifier must be a TypeSafeClassifier, got {type(classifier).__name__}."
            )
        self._classifier = classifier
        return self

    def with_provider(
        self, provider_config: ProviderConfig
    ) -> "JevRelevanceRetrieverBuilder":
        """Build the classifier through the provider factory from ``provider_config``."""
        if not isinstance(provider_config, ProviderConfig):
            raise TypeError(
                f"provider_config must be a ProviderConfig, got "
                f"{type(provider_config).__name__}."
            )
        self._classifier_builder = lambda: build_classifier(provider_config)
        self._classifier = None
        return self

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self) -> JevRelevanceRetriever:
        """Assemble and return the configured retriever.

        Raises:
            ValueError: If no base retriever was set.
            The classifier is constructed lazily so providers without a present
            env-var key only fail at build time, not import time.
        """
        if self._base_retriever is None:
            raise ValueError(
                "Cannot build JevRelevanceRetriever without a base retriever. "
                "Call with_base_retriever(...) first."
            )
        classifier = self._classifier
        if classifier is None:
            if self._classifier_builder is not None:
                classifier = self._classifier_builder()
            else:
                classifier = build_classifier(ProviderConfig.for_provider("typesafe"))
        strategy = build_scoring_strategy(classifier, self._config)
        return JevRelevanceRetriever(
            base_retriever=self._base_retriever,
            config=self._config,
            strategy=strategy,
        )


__all__ = ["JevRelevanceRetrieverBuilder"]
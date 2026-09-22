"""The drop-in relevance-filtered retriever.

:class:`JevRelevanceRetriever` wraps any LangChain ``BaseRetriever`` and turns
it into a relevance-filtered retriever: it forwards the query to the wrapped
engine, scores every candidate with TypeSafe Jev, keeps only chunks that clear
the configured threshold, attaches ``relevance`` (and, in score mode,
``relevance_confidence``) metadata, and sorts by score descending.

Because it subclasses ``BaseRetriever`` it is a 1:1 replacement — every LangChain
RAG chain, ``as_retriever()`` consumer, and agent hook keeps working unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from jev_relevance.config import JevRelevanceConfig
from jev_relevance.strategies import (
    DocumentScore,
    ScoringStrategy,
    build_scoring_strategy,
)

logger = logging.getLogger(__name__)


class JevRelevanceRetriever(BaseRetriever):
    """Filter and re-rank retrieved documents with TypeSafe Jev relevance scores.

    The wrapped retriever produces candidates; Jev decides which of those
    candidates actually answer the query. The filter never invents documents, so
    the output is always a subset of what the base retriever returned.

    Attributes:
        base_retriever: The retrieval engine this class wraps (vector store,
            BM25, hybrid, or anything else exposing ``BaseRetriever``).
        strategy: The scoring strategy used to judge candidates. Defaults to a
            strategy derived from ``config`` if not supplied.
        config: Fully validated settings (question mode, thresholds, ``top_k``,
            concurrency, metadata keys, fail-open).
    """

    base_retriever: BaseRetriever
    config: JevRelevanceConfig = Field(default_factory=JevRelevanceConfig)
    strategy: ScoringStrategy | None = None

    # ------------------------------------------------------------------
    # LangChain callbacks + serialization
    # ------------------------------------------------------------------

    @property
    def _get_strategy(self) -> ScoringStrategy:
        """Return the configured strategy, creating it lazily once.

        The strategy owns the classifier; once built it is cached on the instance.
        """
        if self.strategy is None:
            # The classifier is wired by the builder; see builder module. Default
            # here reads env vars through the standard provider factory.
            from jev_relevance.provider import build_classifier
            from jev_relevance.config import ProviderConfig

            provider_config = ProviderConfig.for_provider("typesafe")
            classifier = build_classifier(provider_config)
            self.strategy = build_scoring_strategy(classifier, self.config)
        return self.strategy

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        candidates = self.base_retriever._get_relevant_documents(
            query, run_manager=run_manager
        )
        if not candidates:
            return []
        try:
            scores = self._get_strategy.score(query, candidates)
        except Exception as exc:
            return self._handle_scoring_failure(exc, candidates)
        return self._retain_relevant(scores)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        candidates = await self.base_retriever._aget_relevant_documents(
            query, run_manager=run_manager
        )
        if not candidates:
            return []
        try:
            scores = await self._get_strategy.ascore(query, candidates)
        except Exception as exc:
            return self._handle_scoring_failure(exc, candidates)
        return self._retain_relevant(scores)

    # ------------------------------------------------------------------
    # Filtering pipeline
    # ------------------------------------------------------------------

    def _handle_scoring_failure(
        self, error: Exception, candidates: list[Document]
    ) -> list[Document]:
        """Decide behavior when Jev scoring fails.

        Returns the raw candidates when ``fail_open`` is on (a slightly worse
        ranking is safer than dropping the user's context); otherwise re-raises.
        """
        if self.config.fail_open:
            logger.warning(
                "Jev relevance scoring failed; returning %d unfiltered candidates. "
                "Reason: %s: %s",
                len(candidates),
                type(error).__name__,
                error,
            )
            return candidates
        raise error

    def _retain_relevant(self, scores: list[DocumentScore]) -> list[Document]:
        """Keep, annotate, and rank documents that clear the relevance threshold.

        Non-destructive: works on copies of each document so caller-held
        documents never pick up Jev metadata unintentionally.
        """
        retained: list[Document] = []
        for score in scores:
            if not self._clears_threshold(score):
                continue
            annotated = Document(
                page_content=score.document.page_content,
                metadata={**score.document.metadata},
            )
            annotated.metadata[self.config.metadata_relevance_key] = score.relevance
            if score.confidence is not None:
                annotated.metadata[
                    self.config.metadata_confidence_key
                ] = score.confidence
            retained.append(annotated)

        retained.sort(
            key=lambda document: float(
                document.metadata.get(self.config.metadata_relevance_key, 0.0)
            ),
            reverse=True,
        )
        return retained[: self.config.top_k] if self.config.top_k is not None else retained

    def _clears_threshold(self, score: DocumentScore) -> bool:
        """Return whether a score passes the configured cutoff."""
        if self.config.question_mode == "noul":
            return score.relevance >= self.config.threshold
        return score.relevance >= self.config.score_threshold


__all__ = ["JevRelevanceRetriever"]
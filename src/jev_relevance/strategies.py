"""Scoring strategies: how relevance questions reach the Jev API.

The retriever depends on the abstract :class:`ScoringStrategy`, never on a
specific transport (Dependency Inversion). Two concrete strategies ship:

* :class:`PerChunkScorer` — one Jev call per query-passage pair, bounded by
  :attr:`~jev_relevance.config.JevRelevanceConfig.max_concurrency`. TyperSafe's
  own rerank guidance: the most accurate design, because each call judges
  exactly one passage against one question.
* :class:`BatchedScorer` — one request that carries every chunk, each embedded
  in its own question's text. Fewest round trips (the "one call, sub-second"
  story) at a small accuracy cost from longer contexts.

The strategy is what produces the raw relevance value + confidence for each
document; filtering, sorting, and metadata are the retriever's job.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_typesafe import ClassifierRequest, ClassifierResponse, TypeSafeClassifier
from pydantic import BaseModel, Field

from jev_relevance.config import JevRelevanceConfig, QuestionMode
from jev_relevance.constants import (
    BATCHED_MAX_QUESTIONS,
    RELEVANCE_QUESTION_KEY,
)
from jev_relevance.scoring import (
    build_batched_state,
    build_noul_question,
    build_noul_question_batched,
    build_score_question,
    build_state,
)

logger = logging.getLogger(__name__)


class DocumentScore(BaseModel):
    """A relevance verdict computed by Jev for a single document.

    Attributes:
        document: The candidate chunk that was judged.
        relevance:
            Noul mode: probability in [0, 1] that the chunk answers the query.
            Score mode: expected value over the 0..3 rubric.
        confidence:
            Score mode only: Jev's confidence in the grade. ``None`` in noul mode
            because a Noul answer carries no separate confidence field.
    """

    document: Document
    relevance: float
    confidence: float | None = Field(default=None)


class ScoringStrategy(ABC):
    """Abstract contract every relevance caller implements.

    Concrete strategies receive the validated :class:`JevRelevanceConfig` and a
    ready-to-call classifier at construction time and expose both sync and async
    scoring across a batch of documents.
    """

    def __init__(
        self, classifier: TypeSafeClassifier, config: JevRelevanceConfig
    ) -> None:
        self.classifier = classifier
        self.config = config

    @abstractmethod
    def score(
        self, query: str, documents: Sequence[Document]
    ) -> list[DocumentScore]:
        """Score every document against ``query`` and return one score each."""

    @abstractmethod
    async def ascore(
        self, query: str, documents: Sequence[Document]
    ) -> list[DocumentScore]:
        """Async variant of :meth:`score`."""

    # ------------------------------------------------------------------
    # Shared answer decoding
    # ------------------------------------------------------------------

    def _read_answer(
        self, response: ClassifierResponse, question_key: str
    ) -> tuple[float, float | None]:
        """Extract (relevance, confidence) from a response for ``question_key``.

        Raises:
            KeyError: When ``question_key`` is absent from the response.
            TypeError: When the answer type does not match the configured mode.
        """
        question_mode: QuestionMode = self.config.question_mode
        answer = response.answers[question_key]
        if question_mode == "noul":
            noul_value = getattr(answer, "noul", None)
            if noul_value is None:
                raise TypeError(
                    f"question_mode='noul' but answer for {question_key!r} is not a Noul."
                )
            return float(noul_value), None
        score_value = getattr(answer, "score", None)
        if score_value is None:
            raise TypeError(
                f"question_mode='score' but answer for {question_key!r} is not a Score."
            )
        confidence = getattr(answer, "confidence", None)
        return float(score_value), float(confidence) if confidence is not None else None


class PerChunkScorer(ScoringStrategy):
    """Score one query-passage pair per Jev request.

    Parallelism is bounded by ``max_concurrency`` (threads sync, semaphore async)
    so a 20-chunk retrieval stays fast without hammering the API or losing
    accuracy to a giant shared context.
    """

    @staticmethod
    def _build_question(config: JevRelevanceConfig):
        if config.question_mode == "noul":
            return build_noul_question(config)
        return build_score_question(config)

    def _score_one(
        self, query: str, document: Document, question_key: str
    ) -> DocumentScore:
        request: ClassifierRequest = {
            "state": build_state(query, document.page_content),
            "questions": {question_key: self._build_question(self.config)},
        }
        response = self.classifier.invoke(request)
        relevance, confidence = self._read_answer(response, question_key)
        return DocumentScore(
            document=document, relevance=relevance, confidence=confidence
        )

    async def _ascore_one(
        self, query: str, document: Document, question_key: str
    ) -> DocumentScore:
        request: ClassifierRequest = {
            "state": build_state(query, document.page_content),
            "questions": {question_key: self._build_question(self.config)},
        }
        response = await self.classifier.ainvoke(request)
        relevance, confidence = self._read_answer(response, question_key)
        return DocumentScore(
            document=document, relevance=relevance, confidence=confidence
        )

    def score(
        self, query: str, documents: Sequence[Document]
    ) -> list[DocumentScore]:
        question_key = RELEVANCE_QUESTION_KEY
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = [
                executor.submit(self._score_one, query, document, question_key)
                for document in documents
            ]
            scores = [future.result() for future in futures]
        return scores

    async def ascore(
        self, query: str, documents: Sequence[Document]
    ) -> list[DocumentScore]:
        question_key = RELEVANCE_QUESTION_KEY
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def score_one(document: Document) -> DocumentScore:
            async with semaphore:
                return await self._ascore_one(query, document, question_key)

        return await asyncio.gather(*[score_one(document) for document in documents])


class BatchedScorer(ScoringStrategy):
    """Score every chunk with a single Jev request.

    The query is the shared state; each chunk is embedded in its own question's
    instructions so the model can tell which passage it is judging. Request
    volume collapses to one round trip, ideal for the small-candidate case that
    motivates the one-call latency story.
    """

    def score(
        self, query: str, documents: Sequence[Document]
    ) -> list[DocumentScore]:
        questions = self._build_questions(query, documents)
        request: ClassifierRequest = {"state": build_batched_state(query), "questions": questions}
        response = self.classifier.invoke(request)
        return self._decode_scores(response, documents)

    async def ascore(
        self, query: str, documents: Sequence[Document]
    ) -> list[DocumentScore]:
        questions = self._build_questions(query, documents)
        request: ClassifierRequest = {"state": build_batched_state(query), "questions": questions}
        response = await self.classifier.ainvoke(request)
        return self._decode_scores(response, documents)

    def _build_questions(
        self, query: str, documents: Sequence[Document]
    ) -> dict[str, object]:
        document_count = len(documents)
        max_questions = self.config.max_batched_questions
        if document_count > max_questions:
            raise ValueError(
                f"BatchedScorer cannot judge {document_count} chunks in one request; "
                f"max_batched_questions={max_questions}. Use scoring_mode='per_chunk' "
                "for larger candidate sets."
            )
        question_mode = self.config.question_mode
        questions: dict[str, object] = {}
        for chunk_index, document in enumerate(documents):
            question_key = f"{RELEVANCE_QUESTION_KEY}_{chunk_index}"
            if question_mode == "noul":
                questions[question_key] = build_noul_question_batched(
                    query, document.page_content
                )
            else:
                questions[question_key] = build_score_question(self.config)
        return questions

    def _decode_scores(
        self, response: ClassifierResponse, documents: Sequence[Document]
    ) -> list[DocumentScore]:
        scores: list[DocumentScore] = []
        for chunk_index, document in enumerate(documents):
            question_key = f"{RELEVANCE_QUESTION_KEY}_{chunk_index}"
            if question_key not in response.answers:
                raise KeyError(
                    f"Jev response missing answer for question {question_key!r}."
                )
            relevance, confidence = self._read_answer(response, question_key)
            scores.append(
                DocumentScore(
                    document=document, relevance=relevance, confidence=confidence
                )
            )
        return scores


def build_scoring_strategy(
    classifier: TypeSafeClassifier, config: JevRelevanceConfig
) -> ScoringStrategy:
    """Create the strategy named by ``config.scoring_mode``.

    Args:
        classifier: Ready-to-call Jev classifier (any provider).
        config: Validated retriever settings.

    Returns:
        A :class:`PerChunkScorer` or :class:`BatchedScorer`.

    Raises:
        ValueError: If ``config.scoring_mode`` names an unknown strategy.
    """
    if config.scoring_mode == "per_chunk":
        return PerChunkScorer(classifier=classifier, config=config)
    if config.scoring_mode == "batched":
        return BatchedScorer(classifier=classifier, config=config)
    raise ValueError(f"Unknown scoring_mode {config.scoring_mode!r}.")


__all__ = [
    "DocumentScore",
    "ScoringStrategy",
    "PerChunkScorer",
    "BatchedScorer",
    "build_scoring_strategy",
]
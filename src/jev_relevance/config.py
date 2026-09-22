"""Fully validated, library-wide configuration for Jev relevance filtering.

Every tunable knob of :class:`~jev_relevance.retriever.JevRelevanceRetriever`
lives here as a typed pydantic field with a named default. Consumers who do not
want to hand-assemble the retriever can use
:class:`~jev_relevance.builder.JevRelevanceRetrieverBuilder`, which wraps this
config and turns it into an instance.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_typesafe import NoulCriteria, Score
from pydantic import BaseModel, Field, field_validator, model_validator

from jev_relevance.constants import (
    BATCHED_MAX_QUESTIONS,
    CONFIDENCE_METADATA_KEY,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_QUESTION_MODE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SCORING_MODE,
    DEFAULT_THRESHOLD,
    DEFAULT_TIMEOUT_S,
    KNOWN_PROVIDERS,
    NAME_MIN_LENGTH,
    NOUL_CRITERIA_FALSE,
    NOUL_CRITERIA_TRUE,
    NOUL_INSTRUCTIONS,
    RELEVANCE_METADATA_KEY,
    SCORE_GRADE_MAX,
    SCORE_GRADE_MIN,
    SCORE_INSTRUCTIONS,
    SCORE_RUBRIC,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    TOP_K_MIN,
)

QuestionMode = Literal["noul", "score"]
ScoringMode = Literal["per_chunk", "batched"]
ProviderName = Literal["typesafe", "openrouter"]

_MAX_CONCURRENCY = Annotated[
    int, Field(gt=0, description="Maximum in-flight Jev scoring calls.")
]
_TOP_K = Annotated[
    int, Field(ge=TOP_K_MIN, description="Maximum documents to return after filtering.")
]
_THRESHOLD = Annotated[
    float,
    Field(
        ge=THRESHOLD_MIN,
        le=THRESHOLD_MAX,
        description="Relevance probability (noul mode) a chunk must clear to be kept.",
    ),
]
_SCORE_THRESHOLD = Annotated[
    float,
    Field(
        ge=SCORE_GRADE_MIN,
        le=SCORE_GRADE_MAX,
        description="Relevance grade (score mode) a chunk must reach to be kept.",
    ),
]

_DEFAULT_NOUL_CRITERIA: NoulCriteria = NoulCriteria(
    true=NOUL_CRITERIA_TRUE, false=NOUL_CRITERIA_FALSE
)
_DEFAULT_SCORE_CRITERIA: list[str] = list(SCORE_RUBRIC)


class JevRelevanceConfig(BaseModel):
    """Validated settings describing *how* relevance filtering should behave.

    Attributes:
        question_mode:
            ``"noul"`` asks a binary yes/no question and keeps the probability in
            ``[0, 1]``. ``"score"`` grades each chunk against the 0..3 rubric and
            keeps the grade plus confidence.
        scoring_mode:
            ``"per_chunk"`` issues one Jev call per query-passage pair (bounded by
            :attr:`max_concurrency`). ``"batched"`` folds every chunk into a single
            request, one question per chunk.
        threshold:
            Noul-mode cutoff; chunks with relevance probability >= threshold survive.
        score_threshold:
            Score-mode cutoff; chunks with grade >= score_threshold survive.
        score_rubric:
            Ordered rubric levels (higher is better) used in score mode.
        max_concurrency:
            Maximum in-flight scoring calls when :attr:`scoring_mode` is ``"per_chunk"``.
        max_batched_questions:
            Safety cap for ``"batched"`` mode; exceeding it triggers a validation
            error because oversized single calls lose accuracy to context rot.
        top_k:
            Maximum documents returned after filtering. ``None`` returns everything
            that cleared the threshold, ranked.
        fail_open:
            If ``True`` and Jev scoring fails, return the base retriever's candidates
            unchanged instead of raising. Keeps the filter from breaking downstream RAG.
        metadata_relevance_key:
            Document metadata key that receives each kept chunk's relevance value.
        metadata_confidence_key:
            Document metadata key (score mode) that receives Jev's confidence value.
    """

    question_mode: QuestionMode = DEFAULT_QUESTION_MODE
    scoring_mode: ScoringMode = DEFAULT_SCORING_MODE
    threshold: _THRESHOLD = DEFAULT_THRESHOLD
    score_threshold: _SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
    score_rubric: tuple[str, ...] = SCORE_RUBRIC
    max_concurrency: _MAX_CONCURRENCY = DEFAULT_MAX_CONCURRENCY
    max_batched_questions: int = BATCHED_MAX_QUESTIONS
    top_k: int | None = None
    fail_open: bool = True
    metadata_relevance_key: str = RELEVANCE_METADATA_KEY
    metadata_confidence_key: str = CONFIDENCE_METADATA_KEY

    # ------------------------------------------------------------------
    # Rubric coercion to Jev's wire types
    # ------------------------------------------------------------------

    def noul_criteria(self) -> NoulCriteria:
        """Return the YES/NO outcome definitions for noul questions."""
        return _DEFAULT_NOUL_CRITERIA

    def score_question(self) -> Score:
        """Return a Jev ``Score`` question built from the configured rubric."""
        return Score(
            instructions=SCORE_INSTRUCTIONS,
            criteria=list(self.score_rubric),
        )


class ProviderConfig(BaseModel):
    """Connection-level settings for the Jev API endpoint.

    Attributes:
        name: Provider identity (``"typesafe"`` or ``"openrouter"``).
        model: Model id used on the provider side (bare or namespaced).
        api_key: Credential, passed explicitly. The library never reads env
            vars; supply the secret through this parameter.
        base_url: Root URL. If omitted, the provider default is used.
        timeout_s: HTTP timeout in seconds.
    """

    name: ProviderName
    model: str = Field(min_length=NAME_MIN_LENGTH, default=DEFAULT_MODEL)
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0.0)

    @field_validator("model")
    @classmethod
    def _strip_and_require_model(cls, model: str) -> str:
        """Reject blank model names (whitespace-only passes min_length)."""
        stripped = model.strip()
        if not stripped:
            raise ValueError("Provider model must not be blank.")
        return stripped

    @model_validator(mode="after")
    def _validate_provider_name(self) -> "ProviderConfig":
        if self.name not in KNOWN_PROVIDERS:
            raise ValueError(
                f"Unknown provider {self.name!r}. Known providers: {KNOWN_PROVIDERS}."
            )
        return self

    @classmethod
    def for_provider(cls, name: ProviderName) -> "ProviderConfig":
        """Create a default config for a known provider name."""
        return cls(name=name)


__all__ = [
    "JevRelevanceConfig",
    "ProviderConfig",
    "QuestionMode",
    "ScoringMode",
    "ProviderName",
]
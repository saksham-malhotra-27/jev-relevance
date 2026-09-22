"""Single source of truth for every default, identifier, and rubric used by
:mod:`jev_relevance`.

No bare numbers or inline strings that carry meaning should live outside this
module. If a value is configurable it must default from here; if it is not
configurable it must still be *named* here so intent is documented.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Provider identities and endpoints
# ---------------------------------------------------------------------------

class Provider:
    """Known Jev providers. New providers register through the factory,
    but the two canonical ones are documented here."""

    TYPESAFE = "typesafe"
    OPENROUTER = "openrouter"

KNOWN_PROVIDERS: tuple[str, ...] = (Provider.TYPESAFE, Provider.OPENROUTER)

#: Default provider used when none is supplied.
DEFAULT_PROVIDER: str = Provider.TYPESAFE

#: Default Jev model id on the TypeSafe side.
DEFAULT_MODEL: str = "jev-latest"

#: OpenRouter namespaces bare TypeSafe ids; ``jev-latest`` -> ``~typesafe/jev-latest``.
OPENROUTER_MODEL_PREFIX: str = "~typesafe/"

#: Root URLs for each provider's System One API. The LangChain classifier
#: appends ``/v1/systemone`` to the configured ``base_url``.
TYPESAFE_DEFAULT_BASE_URL: str = "https://api.typesafe.ai"
OPENROUTER_DEFAULT_BASE_URL: str = "https://openrouter.ai/api"

# ---------------------------------------------------------------------------
# Scoring behavior
# ---------------------------------------------------------------------------

#: Default probability threshold for ``question_mode="noul"`` (0..1). Chunks
#: whose "this chunk answers the query" probability is at least this value are kept.
DEFAULT_THRESHOLD: float = 0.5

#: Default threshold for ``question_mode="score"`` (0..3 rubric). Chunks scoring
#: at least this grade are kept.
DEFAULT_SCORE_THRESHOLD: float = 2.0

#: Default strategy used to dispatch Jev scoring calls to the API.
DEFAULT_SCORING_MODE: str = "per_chunk"

#: Default cardinality question type.
DEFAULT_QUESTION_MODE: str = "noul"

#: Default maximum number of concurrent Jev calls when scoring per chunk.
DEFAULT_MAX_CONCURRENCY: int = 8

#: Default HTTP timeout in seconds for API calls.
DEFAULT_TIMEOUT_S: float = 30.0

#: Default number of candidates to retrieve from the underlying engine.
DEFAULT_CANDIDATE_K: int = 20

# ---------------------------------------------------------------------------
# Question text / rubrics (carefully worded, not tunable)
# ---------------------------------------------------------------------------

#: The question asked about the *query* half of a query/passage state.
NOUL_INSTRUCTIONS: str = "Does this passage answer the query?"

NOUL_CRITERIA_TRUE: str = (
    "The passage contains information that directly answers the query."
)
NOUL_CRITERIA_FALSE: str = (
    "The passage is off-topic, unrelated, or only vaguely about the query."
)

SCORE_INSTRUCTIONS: str = "How well does this passage answer the query?"

#: Ordered 0..3 relevance rubric. Higher is always better.
SCORE_RUBRIC: tuple[str, ...] = (
    "Off-topic; does not relate to the query.",
    "Tangential; mentions the topic but does not answer the query.",
    "Partly answers; relevant but incomplete or missing key details.",
    "Fully answers the query.",
)

#: Identifier for the single relevance question key on every request.
RELEVANCE_QUESTION_KEY: str = "relevance"

# ---------------------------------------------------------------------------
# Batched mode question layout
# ---------------------------------------------------------------------------

#: Batched mode embeds each chunk inside its own question's instructions so
#: Jev never has to guess which question belongs to which passage.
BATCHED_QUESTION_TEMPLATE: str = (
    "Query: {query}\n\nDoes the following chunk answer this query?\n\n"
    "CHUNK: {passage}"
)

#: Maximum number of questions we fold into a single batched request. Beyond
#: this, callers should switch to per_chunk scoring instead of risking
#: context rot from an oversized single call.
BATCHED_MAX_QUESTIONS: int = 32

# ---------------------------------------------------------------------------
# Metadata written onto returned Documents
# ---------------------------------------------------------------------------

#: Metadata key holding the Jev relevance value (noul probability or score grade).
RELEVANCE_METADATA_KEY: str = "relevance"

#: Metadata key holding Jev's confidence (score mode only).
CONFIDENCE_METADATA_KEY: str = "relevance_confidence"

# ---------------------------------------------------------------------------
# Validation bounds
# ---------------------------------------------------------------------------

#: Score rubric is zero-indexed, so valid grades run from 0 to len(rubric)-1.
SCORE_GRADE_MIN: float = 0.0
SCORE_GRADE_MAX: float = float(len(SCORE_RUBRIC) - 1)

#: Lower/upper inclusive bounds enforced by pydantic on every config value.
THRESHOLD_MIN: float = 0.0
THRESHOLD_MAX: float = 1.0
TOP_K_MIN: int = 1
NAME_MIN_LENGTH: int = 1


__all__ = [
    "Provider",
    "KNOWN_PROVIDERS",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "OPENROUTER_MODEL_PREFIX",
    "TYPESAFE_DEFAULT_BASE_URL",
    "OPENROUTER_DEFAULT_BASE_URL",
    "DEFAULT_THRESHOLD",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_SCORING_MODE",
    "DEFAULT_QUESTION_MODE",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_CANDIDATE_K",
    "NOUL_INSTRUCTIONS",
    "NOUL_CRITERIA_TRUE",
    "NOUL_CRITERIA_FALSE",
    "SCORE_INSTRUCTIONS",
    "SCORE_RUBRIC",
    "RELEVANCE_QUESTION_KEY",
    "BATCHED_QUESTION_TEMPLATE",
    "BATCHED_MAX_QUESTIONS",
    "RELEVANCE_METADATA_KEY",
    "CONFIDENCE_METADATA_KEY",
    "SCORE_GRADE_MIN",
    "SCORE_GRADE_MAX",
    "THRESHOLD_MIN",
    "THRESHOLD_MAX",
    "TOP_K_MIN",
    "NAME_MIN_LENGTH",
]
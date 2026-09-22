"""Build the typed questions and states sent to the Jev classifier.

Relevance is judged per query-passage pair. The cleanest Jev shape is
``state = {"query": ..., "passage": ...}`` plus a single ``Noul``/``Score``
question whose ID is :data:`RELEVANCE_QUESTION_KEY` — this module owns both
the single-call and the batched-call phrasing so the retriever never touches
question text, rubric wording, or wire-level schema.
"""

from __future__ import annotations

from typing import Any

from langchain_typesafe import Noul, Score

from jev_relevance.config import JevRelevanceConfig
from jev_relevance.constants import (
    BATCHED_QUESTION_TEMPLATE,
    NOUL_CRITERIA_FALSE,
    NOUL_CRITERIA_TRUE,
    NOUL_INSTRUCTIONS,
    RELEVANCE_QUESTION_KEY,
    SCORE_INSTRUCTIONS,
    SCORE_RUBRIC,
)


def build_noul_question(config: JevRelevanceConfig) -> Noul:
    """Return the binary relevance question used for one query-passage pair.

    Args:
        config: Validated settings; only the criteria wording is used here.

    Returns:
        A ``Noul`` whose YES side means "the passage answers the query".
    """
    return Noul(
        instructions=NOUL_INSTRUCTIONS,
        criteria={
            "true": NOUL_CRITERIA_TRUE,
            "false": NOUL_CRITERIA_FALSE,
        },
    )


def build_noul_question_batched(query: str, passage: str) -> Noul:
    """Return a self-contained Noul that embeds its own passage.

    Batched mode sends many questions in one request against a shared ``state``
    (the query). Because Jev question IDs are invisible to the model, the passage
    must live inside the question text so the model can tell what it is judging.
    """
    return Noul(
        instructions=BATCHED_QUESTION_TEMPLATE.format(
            query=query,
            passage=passage,
        ),
        criteria={
            "true": NOUL_CRITERIA_TRUE,
            "false": NOUL_CRITERIA_FALSE,
        },
    )


def build_score_question(config: JevRelevanceConfig) -> Score:
    """Return the graded (0..3) relevance question from the configured rubric."""
    return Score(instructions=SCORE_INSTRUCTIONS, criteria=list(SCORE_RUBRIC))


def build_state(query: str, passage: str) -> dict[str, str]:
    """Build the minimal Jev state for one query-passage evaluation.

    Args:
        query: The user's retrieval query.
        passage: The candidate chunk text.

    Returns:
        A structured state Jev can judge directly against a single question.
    """
    return {"query": query, "passage": passage}


def build_batched_state(query: str) -> dict[str, str]:
    """Build the shared state used by a batched relevance request.

    The query is the state; every passage lives in its own question's text.
    """
    return {"query": query}


__all__ = [
    "build_noul_question",
    "build_noul_question_batched",
    "build_score_question",
    "build_state",
    "build_batched_state",
    "RELEVANCE_QUESTION_KEY",
]
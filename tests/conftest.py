"""Shared test fixtures: a fake Jev classifier so no API key or network is needed."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_typesafe import (
    ClassifierResponse,
    NoulAnswer,
    ScoreAnswer,
    TypeSafeClassifier,
    Usage,
)
from pydantic import Field


class FakeTypeSafeClassifier(TypeSafeClassifier):
    """TypeSafeClassifier stand-in returning scripted answers per question.

    The real classifier requires a non-empty API key at construction; a dummy
    key satisfies pydantic while `invoke`/`ainvoke` are overridden to return
    the scripted responses instead of hitting the network.

    Configure by setting `noul_values` (noul mode) or `score_values` (score
    mode). The keys in the request are recorded on `invoked_requests`.
    """

    noul_values: list[float] = Field(default_factory=list)
    score_values: list[float] = Field(default_factory=list)
    invoked_requests: list[dict[str, Any]] = Field(default_factory=list)
    _question_count: int = 0

    def __init__(
        self,
        noul_values: list[float] | None = None,
        score_values: list[float] | None = None,
    ) -> None:
        """Build the fake with optional scripted values, then reset request log."""
        super().__init__(api_key="test-key-0000")
        self.noul_values = noul_values or []
        self.score_values = score_values or []
        self.invoked_requests = []
        self._question_count = 0

    def _answers_for(self, questions: dict[str, Any]) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for question_key in questions:
            question = questions[question_key]
            if getattr(question, "type", None) == "score":
                value = (
                    self.score_values[self._question_count % len(self.score_values)]
                    if self.score_values
                    else 0.0
                )
                answers[question_key] = ScoreAnswer(
                    type="score",
                    score=value,
                    confidence=0.9,
                    legend={index: f"level-{index}" for index in range(4)},
                    probabilities={index: 0.25 for index in range(4)},
                )
            else:
                value = (
                    self.noul_values[self._question_count % len(self.noul_values)]
                    if self.noul_values
                    else 0.5
                )
                answers[question_key] = NoulAnswer(type="noul", noul=value)
            self._question_count += 1
        return answers

    def invoke(self, request: Any, config: Any = None, **_: Any) -> ClassifierResponse:
        self.invoked_requests.append(request)
        questions = request["questions"]
        return ClassifierResponse(
            model="jev-latest",
            answers=self._answers_for(questions),
            usage=Usage(),
        )

    async def ainvoke(self, request: Any, config: Any = None, **_: Any) -> ClassifierResponse:
        return self.invoke(request, config)


@pytest.fixture
def fake_classifier() -> FakeTypeSafeClassifier:
    return FakeTypeSafeClassifier()


@pytest.fixture
def fake_classifier_noul() -> FakeTypeSafeClassifier:
    return FakeTypeSafeClassifier(noul_values=[0.9, 0.2, 0.7, 0.3])


@pytest.fixture
def fake_classifier_score() -> FakeTypeSafeClassifier:
    return FakeTypeSafeClassifier(score_values=[3.0, 1.0, 2.0, 0.0])
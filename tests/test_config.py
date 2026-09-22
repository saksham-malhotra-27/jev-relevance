"""Unit tests for configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jev_relevance import JevRelevanceConfig, ProviderConfig
from jev_relevance.constants import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_QUESTION_MODE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SCORING_MODE,
    DEFAULT_THRESHOLD,
    NAME_MIN_LENGTH,
    SCORE_GRADE_MAX,
    SCORE_GRADE_MIN,
)


def test_default_config_uses_named_constants() -> None:
    config = JevRelevanceConfig()
    assert config.question_mode == DEFAULT_QUESTION_MODE
    assert config.scoring_mode == DEFAULT_SCORING_MODE
    assert config.threshold == DEFAULT_THRESHOLD
    assert config.score_threshold == DEFAULT_SCORE_THRESHOLD
    assert config.max_concurrency == DEFAULT_MAX_CONCURRENCY


def test_threshold_is_bounded() -> None:
    with pytest.raises(ValidationError):
        JevRelevanceConfig(threshold=1.01)
    with pytest.raises(ValidationError):
        JevRelevanceConfig(threshold=-0.01)
    # Inclusive bounds are valid.
    assert JevRelevanceConfig(threshold=0.0).threshold == 0.0
    assert JevRelevanceConfig(threshold=1.0).threshold == 1.0


def test_score_threshold_respects_rubric_bounds() -> None:
    assert JevRelevanceConfig(score_threshold=SCORE_GRADE_MIN).score_threshold == 0.0
    assert JevRelevanceConfig(score_threshold=SCORE_GRADE_MAX).score_threshold == 3.0
    with pytest.raises(ValidationError):
        JevRelevanceConfig(score_threshold=SCORE_GRADE_MAX + 0.01)


def test_question_mode_is_literal() -> None:
    with pytest.raises(ValidationError):
        JevRelevanceConfig(question_mode="bogus")  # type: ignore[arg-type]


def test_scoring_mode_is_literal() -> None:
    with pytest.raises(ValidationError):
        JevRelevanceConfig(scoring_mode="everything")  # type: ignore[arg-type]


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        JevRelevanceConfig(max_concurrency=0)


def test_provider_config_defaults() -> None:
    config = ProviderConfig.for_provider("typesafe")
    assert config.model == DEFAULT_MODEL


def test_provider_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(name="bogus")  # type: ignore[typeddict-item]


def test_provider_config_rejects_empty_model() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(name="typesafe", model="   ")


def test_metadata_keys_default_to_named_constants() -> None:
    config = JevRelevanceConfig()
    assert config.metadata_relevance_key == "relevance"
    assert config.metadata_confidence_key == "relevance_confidence"
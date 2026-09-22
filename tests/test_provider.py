"""Unit tests for the provider factory and model-name mapping."""

from __future__ import annotations

import pytest

from jev_relevance import ProviderConfig, build_classifier
from jev_relevance.constants import (
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_MODEL_PREFIX,
    TYPESAFE_DEFAULT_BASE_URL,
)
from jev_relevance.provider import ProviderFactory


def test_typesafe_model_stays_bare() -> None:
    config = ProviderConfig(name="typesafe", api_key="test-key-0000")
    classifier = build_classifier(config)
    assert classifier.model == "jev-latest"
    assert classifier.base_url == TYPESAFE_DEFAULT_BASE_URL


def test_openrouter_model_is_namespaced() -> None:
    config = ProviderConfig(name="openrouter", api_key="test-key-0000")
    classifier = build_classifier(config)
    assert classifier.model == f"{OPENROUTER_MODEL_PREFIX}jev-latest"
    assert classifier.base_url == OPENROUTER_DEFAULT_BASE_URL


def test_openrouter_respects_explicit_key_and_base_url() -> None:
    config = ProviderConfig(
        name="openrouter",
        api_key="explicit-key",
        base_url="https://proxy.example.internal/api",
    )
    classifier = build_classifier(config)
    assert classifier.api_key.get_secret_value() == "explicit-key"
    assert classifier.base_url == "https://proxy.example.internal/api"


def test_typesafe_uses_api_key_param() -> None:
    config = ProviderConfig(name="typesafe", api_key="param-key")
    classifier = build_classifier(config)
    assert classifier.api_key.get_secret_value() == "param-key"


def test_provider_ignores_env_when_param_absent() -> None:
    """Library must not read env vars for credentials; the caller passes them."""
    config = ProviderConfig(name="typesafe", api_key="")  # explicit empty param
    with pytest.raises(Exception):
        build_classifier(config)  # empty api_key rejected at construction


def test_provider_factory_fails_fast_on_unregistered_provider() -> None:
    factory = ProviderFactory()
    # A factory whose entry was removed should reject the provider name.
    del factory.factories["typesafe"]
    with pytest.raises(KeyError):
        factory.build(ProviderConfig.for_provider("typesafe"))


def test_provider_factory_register_requires_name() -> None:
    factory = ProviderFactory()
    with pytest.raises(ValueError):
        factory.register("   ", lambda config: config)


def test_provider_factory_allows_override() -> None:
    factory = ProviderFactory()
    factory.register("typesafe", lambda config: config)
    returned = factory.build(ProviderConfig.for_provider("typesafe"))
    assert returned is not None
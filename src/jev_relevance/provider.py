"""Provider abstraction: build a configured :class:`TypeSafeClassifier`.

A single factory turns a :class:`ProviderConfig` into a ready-to-call
``TypeSafeClassifier``. Both TypeSafe and OpenRouter speak the same System One
request/response shape; the only differences are the model id mapping and the
default base URL. Third parties can register additional providers so the rest of
the library never needs to know about endpoints or key locations.

The library is deliberately passive about credentials: every connection detail
(api_key, base_url, model, timeout) arrives as a parameter on
:class:`ProviderConfig`. Secret management — env variables, .env files, vaults —
is the caller's job, kept out of the library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_typesafe import TypeSafeClassifier

from jev_relevance.config import ProviderConfig
from jev_relevance.constants import (
    DEFAULT_MODEL,
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_MODEL_PREFIX,
    TYPESAFE_DEFAULT_BASE_URL,
)

#: Signature of a provider factory: it receives a ProviderConfig and returns a
#: classifier-shaped object. Defaults to a TypeSafeClassifier, but any object
#: exposing ``invoke``/``ainvoke`` (LangChain Runnable) is accepted.
ClassifierFactory = Callable[[ProviderConfig], TypeSafeClassifier]
ProviderCanvas = dict[str, ClassifierFactory]


def _typesafe_factory(config: ProviderConfig) -> TypeSafeClassifier:
    """Build a TypeSafeClassifier pointed at the TypeSafe API."""
    return TypeSafeClassifier(
        model=config.model,
        api_key=config.api_key or "",
        base_url=config.base_url or TYPESAFE_DEFAULT_BASE_URL,
        timeout=config.timeout_s,
    )


def _openrouter_factory(config: ProviderConfig) -> TypeSafeClassifier:
    """Build a TypeSafeClassifier pointed at OpenRouter's System One API.

    Bare TypeSafe model ids (``jev-latest``) are namespaced to OpenRouter's
    ``~typesafe/...`` alias. The key must be supplied on ``config.api_key``.
    """
    model = config.model
    if model == DEFAULT_MODEL and "/" not in model and not model.startswith("~"):
        model = f"{OPENROUTER_MODEL_PREFIX}{model}"
    return TypeSafeClassifier(
        model=model,
        api_key=config.api_key or "",
        base_url=config.base_url or OPENROUTER_DEFAULT_BASE_URL,
        timeout=config.timeout_s,
    )


@dataclass
class ProviderFactory:
    """Registry of provider name -> classifier factory.

    Registering a provider is how the library stays open for extension without
    changing the retriever core (Open/Closed). The built-in providers are
    already registered; call :meth:`register` to add your own.

    Attributes:
        factories: Mapping of provider name to its classifier factory.
    """

    factories: ProviderCanvas = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.register("typesafe", _typesafe_factory)
        self.register("openrouter", _openrouter_factory)

    def register(
        self, name: str, factory: ClassifierFactory
    ) -> "ProviderFactory":
        """Register (or override) a provider factory under ``name``."""
        if not name or not name.strip():
            raise ValueError("Provider name must be a non-empty string.")
        self.factories[name] = factory
        return self

    def build(self, config: ProviderConfig) -> TypeSafeClassifier:
        """Build a classifier from a validated provider config.

        Raises:
            KeyError: If `config.name` has not been registered.
        """
        factory = self.factories.get(config.name)
        if factory is None:
            known = ", ".join(sorted(self.factories))
            raise KeyError(
                f"No factory registered for provider {config.name!r}. "
                f"Known providers: {known}."
            )
        return factory(config)


#: Module-level convenience instance; most callers use :func:`build_classifier`.
_default_factory = ProviderFactory()


def build_classifier(config: ProviderConfig) -> TypeSafeClassifier:
    """Build a classifier for a :class:`ProviderConfig`.

    Convenience wrapper around the shared :data:`_default_factory`.

    Args:
        config: Validated provider settings (name, model, key, base url, timeout).

    Returns:
        A configured ``TypeSafeClassifier`` ready for ``invoke``/``ainvoke``.
    """
    return _default_factory.build(config)


__all__ = [
    "ClassifierFactory",
    "ProviderFactory",
    "build_classifier",
]
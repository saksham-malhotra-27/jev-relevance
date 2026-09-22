"""jev-relevance: a drop-in relevance filter for LangChain RAG retrievers.

Wrap any ``BaseRetriever``; Jev scores every candidate in one bounded pass and
the wrapper returns only the chunks that answer the query, ranked by relevance.

Basic usage (TypeSafe Provider):

    from jev_relevance import JevRelevanceRetrieverBuilder

    retriever = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(my_vector_retriever)
        .with_top_k(5)
        .build()
    )

OpenRouter Provider (api_key passed as a param):

    from jev_relevance import JevRelevanceRetrieverBuilder
    from jev_relevance.config import ProviderConfig

    retriever = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(my_vector_retriever)
        .with_provider(ProviderConfig(name="openrouter", api_key="sk-..."))
        .build()
    )
"""

from jev_relevance.builder import JevRelevanceRetrieverBuilder
from jev_relevance.config import (
    JevRelevanceConfig,
    ProviderConfig,
    ProviderName,
    QuestionMode,
    ScoringMode,
)
from jev_relevance.constants import Provider
from jev_relevance.provider import ProviderFactory, build_classifier
from jev_relevance.retriever import JevRelevanceRetriever
from jev_relevance.scoring import (
    build_noul_question,
    build_state,
)
from jev_relevance.strategies import (
    BatchedScorer,
    DocumentScore,
    PerChunkScorer,
    ScoringStrategy,
    build_scoring_strategy,
)

__version__ = "0.1.0"

__all__ = [
    "JevRelevanceRetriever",
    "JevRelevanceRetrieverBuilder",
    "JevRelevanceConfig",
    "ProviderConfig",
    "ProviderFactory",
    "build_classifier",
    "build_scoring_strategy",
    "build_state",
    "build_noul_question",
    "DocumentScore",
    "ScoringStrategy",
    "PerChunkScorer",
    "BatchedScorer",
    "Provider",
    "ProviderName",
    "QuestionMode",
    "ScoringMode",
    "__version__",
]
"""Drop-in example: swap any LangChain retriever for the Jev-filtered one.

Run with a real key (either provider) to see live filtering:

    # TypeSafe provider (default)
    set TYPESAFE_API_KEY=your-key
    python examples/drop_in.py --provider typesafe

    # OpenRouter provider
    set OPENROUTER_API_KEY=your-key
    python examples/drop_in.py --provider openrouter
"""

from __future__ import annotations

import argparse
import os

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from jev_relevance import ProviderConfig, JevRelevanceRetrieverBuilder

CORPUS_TEXT = [
    "Chromosomes carry DNA; humans have 23 pairs.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Louvre museum is located in Paris, France.",
    "Neural networks are trained with backpropagation.",
    "The Sahara is the largest hot desert on Earth.",
]


class MemoryRetriever(BaseRetriever):
    """Stands in for a real vector store."""

    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        # Naive overlap: anything sharing a token comes back as a candidate.
        query_tokens = set(query.lower().split())
        return [
            document
            for document in self.documents
            if query_tokens & set(document.page_content.lower().split())
        ]


def build_retriever(provider: str) -> JevRelevanceRetrieverBuilder:
    documents = [
        Document(page_content=text, metadata={"source": f"doc-{index}"})
        for index, text in enumerate(CORPUS_TEXT)
    ]

    builder = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(MemoryRetriever(documents=documents))
        .with_threshold(0.5)
        .with_top_k(5)
    )

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY is not set. "
                "Set it or re-run with --provider typesafe."
            )
        return builder.with_provider(
            ProviderConfig(name="openrouter", api_key=api_key)
        )
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit(
            "TYPESAFE_API_KEY is not set. "
            "Set it or re-run with --provider openrouter."
        )
    return builder.with_provider(ProviderConfig(name="typesafe", api_key=api_key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["typesafe", "openrouter"],
        default="typesafe",
        help="Which Jev provider to use (default: typesafe).",
    )
    parser.add_argument(
        "--query",
        default="Where are the chromosomes and DNA found in cells?",
        help="Sample query to filter candidates against.",
    )
    args = parser.parse_args()

    retriever = build_retriever(args.provider).build()
    results = retriever.invoke(args.query)

    print(f"Query: {args.query}\n")
    print(f"Retained {len(results)} / {len(CORPUS_TEXT)} candidate chunks:\n")
    for index, document in enumerate(results):
        relevance = document.metadata.get("relevance")
        print(f"  [{index}] relevance={relevance:.2f}")
        print(f"       {document.page_content[:80]}")


if __name__ == "__main__":
    main()
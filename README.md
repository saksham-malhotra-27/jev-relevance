# jev-relevance

A drop-in relevance filter for LangChain RAG retrievers, powered by
[TypeSafe Jev](https://typesafe.ai). Wrap any `BaseRetriever`, and Jev decides
which retrieved chunks actually answer the query. No chain plumbing, no prompt
engineering, one (or a few) API calls.

```python
from jev_relevance import JevRelevanceRetrieverBuilder

retriever = (
    JevRelevanceRetrieverBuilder()
    .with_base_retriever(my_vector_retriever)  # any LangChain BaseRetriever
    .with_top_k(5)
    .build()
)

docs = retriever.invoke("What is the capital of France?")
```

Because `JevRelevanceRetriever` **is a** `BaseRetriever`, you can swap it into an
existing chain without touching anything else. It returns only the chunks that
clear the relevance threshold, sorted by Jev score, with the score attached in
`Document.metadata["relevance"]`.

## Features

- **Drop-in**: subclass of `BaseRetriever`; `astream`, `with_retriever`, agent
  and chain wiring all keep working.
- **Two scoring modes**:
  - `per_chunk` (default): one Jev call per query-passage pair, bounded by
    `max_concurrency`. Most accurate.
  - `batched`: every chunk judged in a single request for the "sub-second"
    latency case (max 32 chunks).
- **Two providers**: TypeSafe (`typesafe`) and OpenRouter (`openrouter`), plus a
  factory that lets you register more without touching the core.
- **fail-open**: on scoring errors you get the unfiltered candidates instead of
  empty context (configurable).
- **Zero magic numbers**: every default lives in `constants.py` and every
  behavior is builder-parameterized.

## Installation

```bash
pip install jev-relevance
# or, from this repo:
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.10+. The library itself only needs `langchain-core` and
`langchain-typesafe`.

## Providers

| Provider | Default model | Base URL |
| --- | --- | --- |
| `typesafe` | `jev-latest` | `https://api.typesafe.ai` |
| `openrouter` | `~typesafe/jev-latest` | `https://openrouter.ai/api` |

Credentials are **always passed as parameters** — the library never reads env
vars or `.env` files itself; that's the caller's job. Supply the key (and any
custom base URL) directly on `ProviderConfig`:

```python
from jev_relevance import JevRelevanceRetrieverBuilder, ProviderConfig

retriever = (
    JevRelevanceRetrieverBuilder()
    .with_base_retriever(my_vector_retriever)
    .with_provider(
        ProviderConfig(name="openrouter", api_key=os.environ["OPENROUTER_API_KEY"])
    )
    .build()
)
```

## Configuration

Everything is available through the builder chain:

| Builder method | Default | Meaning |
| --- | --- | --- |
| `.with_question_mode("noul" \| "score")` | `noul` | Noul = binary "answers or not"; score = 0..3 rubric |
| `.with_threshold(0.5)` | `0.5` | Noul keep-if `score >= threshold` |
| `.with_score_threshold(2.0)` | `2.0` | Score-mode keep-if `grade >= threshold` |
| `.with_scoring_mode("per_chunk" \| "batched")` | `per_chunk` | One call per chunk, or one call for all |
| `.with_top_k(20)` | `20` | Max results returned after filtering |
| `.with_max_concurrency(8)` | `8` | Concurrent Jev calls in per-chunk mode |
| `.with_fail_open(True)` | `True` | Return unfiltered candidates on scoring error |
| `.with_classifier(classifier)` | auto | Inject a ready `TypeSafeClassifier` |
| `.with_provider(config)` | `typesafe` | Which provider config to build from |

The same values can be passed directly to `JevRelevanceConfig`, which validates
every field (pydantic) so invalid values fail fast at construction time.

Metadata attached to returned documents:

- `relevance`: the Jev value (noul probability 0..1, or score grade 0..3)
- `relevance_confidence`: Jev's confidence, score mode only

## Question modes

- **noul** — "Does this passage answer the query?" Binary yes/no probability.
  The default; best for a strict relevance gate.
- **score** — "How well does this passage answer the query?" Rubric of 0..3
  (`Off-topic`, `Tangential`, `Partly answers`, `Fully answers`), plus Jev's
  confidence.

## Examples

See [`examples/drop_in.py`](examples/drop_in.py) for a runnable swap-in with a
naive keyword retriever:

```bash
set TYPESAFE_API_KEY=your-key
python examples/drop_in.py --provider typesafe

set OPENROUTER_API_KEY=your-key
python examples/drop_in.py --provider openrouter
```

## Benchmarks & dashboard

The [jev-relevance-evals](https://github.com/saksham-malhotra-27/jev-relevance-evals)
companion repo holds the BEIR/NFCorpus benchmark harness and a Streamlit
dashboard. It scores BM25 candidates three ways on the same frozen pool — plain
keyword retrieval, the Jev relevance filter (drop-in retriever), and a cheap
OpenRouter LLM judge — and writes full IR metrics plus per-call cost to JSON.
On nfcorpus at the default 0.5 threshold, Jev roughly doubles precision at the
cost of some recall; see the companion repo for the full results tables and
interactive per-query drill-downs.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

Tests cover builder validation, provider model-mapping, per-chunk/batched
scoring, fail-open behavior, and async parity using a mocked classifier — no
API keys required.
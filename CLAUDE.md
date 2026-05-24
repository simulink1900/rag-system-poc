# Disneyland Reviews RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions about Disneyland visitor experiences across different parks, seasons, and locations. Retrieves relevant reviews and uses an LLM to synthesize context-aware answers.

## Quick Start

```bash
# Install dependencies (uv handles venv automatically)
uv sync

# Run the interactive chat
uv run python chat.py

# Or activate venv manually
source .venv/bin/activate
python chat.py
```

## Project Structure

```
rag-system-poc/
├── pyproject.toml             # Project config + dependencies
├── uv.lock                    # Locked dependency versions
├── README.md                  # Quick reference
├── CLAUDE.md                  # This file (project guide)
├── .env                       # Configuration (LLM settings)
├── chat.py                    # Interactive CLI for asking questions
├── tests/
│   ├── test_filters.py        # Tests for filter parsing logic
│   └── test_evaluation.py     # Tests for LLM-as-judge evaluation
├── data/
│   └── DisneylandReviews.csv  # Visitor review dataset
├── chroma_db/                 # Persistent vector store (auto-generated)
├── notebooks/
│   └── dashboard_analytics.ipynb  # Analytics & visualizations
├── src/rag/
│   ├── __init__.py
│   ├── config.py              # Centralized configuration (LLM, paths, filters)
│   ├── ingest.py              # CSV loading and document creation
│   ├── embeddings.py          # Sentence-transformers embedding setup
│   ├── vectorstore.py         # ChromaDB collection management
│   ├── filter_parser.py       # Extract filters from natural language queries
│   ├── retriever.py           # Retrieve reviews with optional filtering
│   └── chain.py               # RAG chain: prompt + LLM + formatting
└── .venv/                     # Virtual environment (auto-generated, not tracked)
```

## Technology Stack

### Core LLM & Orchestration
- **langchain** (>=1.3.0) - LLM application framework
- **litellm** (>=1.0.0) - LLM proxy/abstraction layer (OpenRouter, Anthropic, OpenAI, etc.)
- **langchain-litellm** (>=0.1.0) - LangChain integration for LiteLLM

### Vector Database & Embeddings
- **chromadb** (>=1.5.0) - Vector database with persistence
- **sentence-transformers** (>=5.5.0) - HuggingFace embedding models (`all-MiniLM-L6-v2` by default)

### Data & Processing
- **pandas** (>=2.0.0) - CSV loading and data manipulation
- **pypdf** (>=6.11.0) - PDF document processing (future use)
- **torch** (>=2.12.0) - Deep learning framework (embedding inference)
- **tqdm** (>=4.66.0) - Progress bars for long-running operations

### Visualization & Analysis
- **matplotlib** (>=3.8.0) - Static plotting
- **seaborn** (>=0.13.0) - Statistical visualization

### Utilities
- **python-dotenv** (>=1.2.0) - Environment variable management
- **requests** (>=2.34.0) - HTTP client (LiteLLM calls)

## How It Works

### Data Ingestion (`src/rag/ingest.py`)
- Loads Disneyland visitor reviews from CSV (`DisneylandReviews.csv`)
- Extracts metadata: branch, reviewer location, season, rating, year/month
- Creates LangChain Document objects with text content and metadata

### Embeddings (`src/rag/embeddings.py`)
- Uses `sentence-transformers` model (`all-MiniLM-L6-v2`) to convert review text to vectors
- Batch processing with configurable batch size (512 by default)
- One embedding per review document

### Vector Storage (`src/rag/vectorstore.py`)
- ChromaDB collection (`disneyland_reviews`) persisted to disk
- Stores embeddings + metadata for fast retrieval
- Lazy initialization: rebuilds if collection missing

### Filter Extraction (`src/rag/filter_parser.py`)
- LLM analyzes user query to extract structured filters:
  - **Branch**: California, Hong Kong, Paris (normalized from free-text)
  - **Season**: Winter, Spring, Summer, Autumn
  - **Rating**: Minimum star rating (1-5)
  - **Reviewer Location**: Geographic location
  - **Date**: Year, month, or specific YYYY-M format
  - **Recency Bias**: Prefer recent reviews (configurable default: True)
- Returns `ReviewFilters` TypedDict with None for unmentioned filters

### Retrieval (`src/rag/retriever.py`)
- Vector similarity search (top-K, configurable: 30 by default)
- Applies ChromaDB where-clauses for branch/location/season/rating filters
- Post-filtering for partial date matches (month-only, year-only)
- Recency scoring: boosts recent reviews if requested
- Returns formatted review context with location, branch, date, rating

### RAG Chain (`src/rag/chain.py`)
- System prompt: instructs LLM to answer based on reviews only
- Retrieval: fetch reviews for user question
- Context formatting: each review shows metadata header + snippet (300 chars)
- LLM response: ChatLiteLLM generates answer using context
- Configurable: temperature, max_tokens, timeout

### Evaluation (`src/rag/evaluation.py`)
- LLM-as-judge evaluates answer quality on 4 metrics:
  - **Relevance** (0-1): How well does the answer address the question?
  - **Conciseness** (0-1): Is it appropriately brief?
  - **Helpfulness** (0-1): How useful to someone seeking info?
  - **Hallucination** (0-1): How much false/unsupported info? (0=none, 1=full hallucination)
- Scores on discrete scale: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Uses dedicated judge LLM (default: `litellm_proxy/openrouter/google/gemma-4-26b-a4b-it`, configurable via `LLMASAJUDGE_MODEL_NAME`)
- Enabled via config or `evaluation=True` parameter in `ask()`

### Interactive Chat (`chat.py`)
- CLI loop: user enters question → extract filters → retrieve → generate answer
- Shows number of matching reviews and query parameters
- Continues until user exits

## Development

### Package Management
Uses `uv` for fast, reliable dependency management:
- `uv sync` - Install/sync all dependencies
- `uv pip list` - List installed packages
- `uv run python <script>` - Run Python in the project environment

### Code Quality (Optional)
Dev dependencies (not installed by default):
- pytest - Testing framework
- pytest-cov - Coverage reporting
- black - Code formatter
- ruff - Fast linter
- mypy - Type checking

Install dev dependencies:
```bash
uv sync --extra dev
```

Run tests:
```bash
pytest test_filters.py -v
```

## Configuration

All settings are centralized in `src/rag/config.py`:

```python
# Data paths
DATA_PATH = "data/DisneylandReviews.csv"
CHROMA_PERSIST_DIR = "chroma_db"

# Model selection
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # HuggingFace model ID
LLM_MODEL_NAME = "litellm_proxy/openrouter/openai/gpt-4-mini"  # Configurable

# LLM parameters (read from .env or defaults)
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 120

# Retrieval settings
TOP_K_RETRIEVAL = 30           # Number of reviews to fetch
PREFER_RECENT_BY_DEFAULT = True # Boost recent reviews

# Evaluation
EVALUATION_ENABLED = False     # Enable LLM-as-judge scoring
LLMASAJUDGE_MODEL_NAME = "litellm_proxy/openrouter/google/gemma-4-26b-a4b-it"  # Model for evaluation

# Filter defaults
EMBED_BATCH_SIZE = 512
CSV_ENCODING = "latin-1"
COLLECTION_NAME = "disneyland_reviews"

# Valid filter values
VALID_BRANCHES = {"Disneyland_California", "Disneyland_HongKong", "Disneyland_Paris"}
SEASON_MAP = {12: "winter", 1: "winter", ..., 11: "autumn"}
```

Override via environment variables:
```bash
export LLM_MODEL_NAME="litellm_proxy/openrouter/anthropic/claude-3-sonnet"
export LLM_TEMPERATURE="0.5"
export LLM_MAX_TOKENS="2048"
export LLMASAJUDGE_MODEL_NAME="litellm_proxy/openrouter/google/gemma-4-26b-a4b-it"
```

### Coding Guidelines

All code follows the **karpathy-guidelines** skill—behavioral guidelines to reduce common LLM coding mistakes.

#### 1. Think Before Coding
State assumptions, surface tradeoffs, don't hide confusion.

#### 2. Simplicity First
Minimum code that solves the problem. No speculative features or abstractions.

#### 3. Surgical Changes
Touch only what you must. Match existing style. Every line traces to the user's request.

#### 4. Goal-Driven Execution
Define success criteria. Loop until verified. Strong success criteria prevent constant clarification.

## Claude Code Skills

Use these skills with `/skill-name` for development:

| Skill | Purpose |
|-------|---------|
| `/karpathy-guidelines` | Coding best practices to reduce mistakes |
| `/run` | Launch chat.py and test changes interactively |
| `/verify` | Verify code changes work as expected |
| `/code-review` | Review code for quality issues |

## Environment Variables

Create a `.env` file for local configuration:
```bash
# Main LLM (for question answering)
LLM_MODEL_NAME=litellm_proxy/openrouter/anthropic/claude-3-sonnet
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=180

# Evaluation
EVALUATION_ENABLED=true  # Enable LLM-as-judge scoring (default: false)
LLMASAJUDGE_MODEL_NAME=litellm_proxy/openrouter/google/gemma-4-26b-a4b-it  # Model for evaluation

# LiteLLM proxy (if using local proxy)
LITELLM_PROXY_URL=http://localhost:8000
```

**Default models:**
- Main LLM: `litellm_proxy/openrouter/openai/gpt-4-mini`
- Judge LLM: `litellm_proxy/openrouter/google/gemma-4-26b-a4b-it`

### Evaluation Output Format

When `EVALUATION_ENABLED=true`, the chat shows evaluation scores for each answer:
```
✨ Answer:
[Answer text here]

📊 Evaluation Scores:
   Relevance:    0.8
   Conciseness:  0.6
   Helpfulness:  0.8
   Hallucination (↓ = better): 0.2
```

All scores are on [0, 0.2, 0.4, 0.6, 0.8, 1.0] scale. Higher is better for all metrics except hallucination (lower is better).

## Data

**DisneylandReviews.csv** contains visitor reviews with:
- Text content: visitor feedback and experience descriptions
- Metadata: branch (California/HongKong/Paris), reviewer_location, season, year_month, rating (1-5)

Example query:
```
"What do people say about visiting California in summer with high ratings?"
```
System extracts: `branch=California, season=summer, min_rating=4`

## Completed Features

- [x] Data ingestion from CSV
- [x] Embedding model setup (sentence-transformers)
- [x] Vector storage (ChromaDB with persistence)
- [x] Filter extraction from natural language (LLM-based)
- [x] Retrieval with filtering, ranking, and recency bias
- [x] RAG chain (retrieval + LLM synthesis)
- [x] Interactive CLI chat
- [x] Dashboard analytics (notebook with visualizations)
- [x] LLM-as-judge evaluation (relevance, conciseness, helpfulness, hallucination)
- [x] Comprehensive test suite (filter parsing, evaluation)
- [x] Type hints and configuration management

## Future Enhancements

- [ ] Chunking strategy for very long reviews
- [ ] Web UI or REST API interface
- [ ] Batch evaluation mode for benchmarking
- [ ] Query result caching layer
- [ ] Advanced filtering (date ranges, multi-location OR queries)
- [ ] Semantic similarity clustering of reviews

## Python Version

- **Required:** >= 3.11
- **Tested with:** 3.13.9
- **CUDA support:** 13.0 (optional, for GPU-accelerated embeddings)

## Notes

- All dependencies are pinned in `uv.lock` for reproducibility
- The `.venv/` directory is auto-generated and not tracked in git
- Vector embeddings are cached in `chroma_db/` (not tracked in git)
- CSV file must be in `data/DisneylandReviews.csv`

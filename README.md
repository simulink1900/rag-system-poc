# Disneyland Reviews RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions about Disneyland visitor experiences across different parks, seasons, and locations. Uses advanced retrieval with smart filtering, LLM-based question answering, and LLM-as-judge evaluation.

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Set environment variables (optional)
export LITELLM_MASTER_KEY=your_key
export LITELLM_PROXY_URL=your_url
```

### Run Interactive Chat

```bash
uv run python chat.py
```

Example questions:
- "What do visitors from Australia say about Disneyland in HongKong?"
- "Is spring a good time to visit Disneyland?"
- "Is Disneyland California usually crowded in June?"
- "Is the staff in Paris friendly?"

### Run Jupyter Notebooks

```bash
# EDA & analytics dashboard
jupyter notebook notebooks/dashboard_analytics.ipynb

# RAG system demo
jupyter notebook notebooks/rag_demo.ipynb
```

## Project Overview

This system combines three core capabilities:

1. **Intelligent Retrieval** - Semantic search with structured filtering
2. **Context-Aware Answers** - LLM synthesizes answers from retrieved reviews
3. **Quality Evaluation** - Separate LLM judges answer quality on 4 metrics

## Data

### DisneylandReviews.csv
- **Size**: 42,636 visitor reviews
- **Source**: Disneyland parks across 3 locations
- **Time span**: Multiple years of visitor feedback
- **Metadata fields**:
  - `branch`: Disneyland_California, Disneyland_HongKong, Disneyland_Paris
  - `reviewer_location`: Geographic origin of reviewer
  - `season`: Winter, Spring, Summer, Autumn
  - `year_month`: Date of visit (YYYY-M format)
  - `rating`: Star rating (1-5)
  - `text`: Visitor review text

### Data Pipeline

```
CSV Load → Document Creation → Embedding Generation → Vector Storage
   ↓            ↓                    ↓                      ↓
raw reviews  LangChain Docs   sentence-transformers   ChromaDB
```

## Architecture & Pipelines

### 1. Data Ingestion Pipeline (`src/rag/ingest.py`)
- Loads CSV with proper encoding (latin-1)
- Creates LangChain Document objects
- Extracts and validates metadata
- Outputs: Document list with metadata

### 2. Embedding Pipeline (`src/rag/embeddings.py`)
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Batch processing (512 docs/batch)
- GPU acceleration available (CUDA 13.0)
- Output: Vector embeddings (384-dim)

### 3. Vector Storage (`src/rag/vectorstore.py`)
- Backend: ChromaDB with persistent disk storage
- Collection: `disneyland_reviews` (42,636 documents)
- Lazy initialization: Rebuilds on first run, loads from cache afterwards
- Location: `./chroma_db/`

### 4. Filter Extraction Pipeline (`src/rag/filter_parser.py`)
- LLM analyzes user query
- Extracts structured filters:
  - Branch (California/HongKong/Paris)
  - Season (Winter/Spring/Summer/Autumn)
  - Rating (1-5 minimum)
  - Location (reviewer's origin)
  - Date (YYYY, YYYY-M, M formats)
  - Recency bias (prefer recent reviews)
- Output: ReviewFilters TypedDict

### 5. Retrieval Pipeline (`src/rag/retriever.py`)
- Vector similarity search: top-30 reviews (configurable)
- ChromaDB where-clauses for exact filters
- Post-filtering for partial date matches
- Recency scoring: boosts recent reviews
- Output: Ranked list of Document objects

### 6. RAG Chain (`src/rag/chain.py`)
- **Input**: User question
- **Filter extraction**: LLM extracts structured filters
- **Retrieval**: Fetch top-K reviews matching filters
- **Context formatting**: Reviews → formatted text with metadata
- **LLM generation**: Answer based on context
- **Output**: Synthesized answer string

### 7. Evaluation Pipeline (`src/rag/evaluation.py`)
- **Separate Judge LLM**: `litellm_proxy/openrouter/google/gemma-4-26b-a4b-it`
- **4 Metrics** (0-1 scale, 0.2 increments):
  - Relevance: How well does answer address question?
  - Conciseness: Appropriately brief?
  - Helpfulness: Useful to information seekers?
  - Hallucination: False/unsupported info? (0=none, 1=full)
- **Output**: EvaluationScores dict

### 8. Interactive Chat Loop (`chat.py`)
```
User Question
    ↓
Filter Extraction (LLM)
    ↓
Display Extracted Filters
    ↓
Retrieval (Vector + Filters)
    ↓
RAG Chain (LLM Answer)
    ↓
[Optional] Evaluation (Judge LLM)
    ↓
Display Answer + Scores
```

## Features

### Core RAG
- ✅ Semantic vector search with ChromaDB
- ✅ Smart filter extraction from natural language
- ✅ Metadata-aware retrieval (branch, season, location, rating, date)
- ✅ LLM-powered question answering

### Evaluation
- ✅ LLM-as-judge scoring (4 metrics)
- ✅ Separate judge LLM (cost-effective model)
- ✅ Discrete scoring scale [0, 0.2, 0.4, 0.6, 0.8, 1.0]
- ✅ Optional per-query evaluation

### Analytics
- ✅ Dashboard notebook (matplotlib/seaborn)
- ✅ Review distribution by park/season/rating
- ✅ Filter effectiveness analysis
- ✅ EDA and data quality insights

### Development
- ✅ Comprehensive test suite
- ✅ Modular architecture (easy to extend)
- ✅ Configuration management (environment variables)
- ✅ Type hints throughout (TypedDict for data structures)

## Configuration

All settings in `src/rag/config.py`. Override via environment variables:

```bash
# Main LLM (question answering)
export LLM_MODEL_NAME="litellm_proxy/openrouter/anthropic/claude-3-sonnet"
export LLM_TEMPERATURE="0.5"
export LLM_MAX_TOKENS="2048"
export LLM_TIMEOUT="180"

# Evaluation
export EVALUATION_ENABLED="true"
export LLMASAJUDGE_MODEL_NAME="litellm_proxy/openrouter/google/gemma-4-26b-a4b-it"

# Retrieval
export TOP_K_RETRIEVAL="50"
```

See `CLAUDE.md` for full configuration details.

## Development

### Run Tests

```bash
# Test filter parsing
pytest tests/test_filters.py -v

# Test evaluation
pytest tests/test_evaluation.py -v
```

### Project Structure

```
.
├── chat.py                         # Interactive CLI
├── src/rag/
│   ├── ingest.py                  # CSV loading
│   ├── embeddings.py              # Embedding setup
│   ├── vectorstore.py             # ChromaDB management
│   ├── filter_parser.py           # Filter extraction
│   ├── retriever.py               # Retrieval logic
│   ├── chain.py                   # RAG chain
│   ├── evaluation.py              # LLM-as-judge
│   └── config.py                  # Centralized config
├── notebooks/
│   ├── dashboard_analytics.ipynb  # Analytics & visualizations
│   ├── rag_demo.ipynb             # System demonstration
│   └── lightllm_test.ipynb        # LiteLLM testing
├── tests/
│   ├── test_filters.py            # Filter extraction tests
│   └── test_evaluation.py         # Evaluation tests
├── data/
│   └── DisneylandReviews.csv      # Review dataset
└── chroma_db/                     # Vector store (auto-generated)
```

## Technology Stack

- **LLM Framework**: LangChain 1.3+
- **LLM Proxy**: LiteLLM (OpenRouter, Anthropic, OpenAI)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB 1.5+
- **Data**: pandas, PyPDF
- **Visualization**: matplotlib, seaborn
- **ML**: torch 2.12+ (with CUDA 13.0 support)

## Performance

- **Latency**: ~2-5s per query (retrieval + generation + optional evaluation)
- **Vector Search**: <100ms for 42K documents
- **Embedding Generation**: ~3-5 minutes for full dataset (cached on disk)
- **Evaluation**: +1-3s per answer (separate judge LLM)

## Python Requirements

- **Required**: Python >= 3.11
- **Tested with**: Python 3.13.9
- **Optional**: CUDA 13.0 for GPU-accelerated embeddings

## Next Steps

- [ ] Chunking strategy for longer reviews
- [ ] GitHub repository setup
- [ ] Multi-document answer synthesis
- [ ] Caching layer for frequent queries

## License

MIT

## Contact

Questions? See `CLAUDE.md` for detailed implementation guide.

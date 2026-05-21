# RAG System POC

Proof of concept for a Retrieval-Augmented Generation (RAG) system. Combines document retrieval with LLM capabilities for context-aware question answering.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Or use uv to run directly
uv run python your_script.py

# Sync dependencies if needed
uv sync
```

## Project Structure

```
rag-system-poc/
├── pyproject.toml          # Project config + dependencies
├── uv.lock                 # Locked dependency versions
├── .venv/                  # Virtual environment (auto-generated, not tracked)
├── .claude/
│   └── skills/
│       └── karpathy-guidelines/  # Coding guidelines skill
└── CLAUDE.md               # This file
```

## Technology Stack

### Core LLM & Orchestration
- **langchain** (1.3.1) - LLM application framework
- **openai** (2.37.0) - OpenAI API client
- **langgraph** (1.2.0) - Stateful graph-based workflows

### Vector Database & Embeddings
- **chromadb** (1.5.9) - In-memory vector database
- **sentence-transformers** (5.5.1) - Embedding models (HuggingFace)
- **transformers** (5.9.0) - NLP models

### ML/Compute
- **torch** (2.12.0) - Deep learning framework with CUDA 13.0 support
- **scikit-learn** (1.8.0) - ML utilities
- **onnxruntime** (1.26.0) - Model inference optimization

### Utilities
- **pypdf** (6.11.0) - PDF processing for document ingestion
- **python-dotenv** (1.2.2) - Environment variable management
- **requests** (2.34.2) - HTTP client

## Development

### Package Management
Uses `uv` for fast, reliable dependency management:
- `uv sync` - Install/sync all dependencies
- `uv pip list` - List installed packages
- `uv run` - Run Python in the project environment

### Code Quality (Optional)
Dev dependencies available (not installed by default):
- pytest - Testing framework
- black - Code formatter
- ruff - Fast linter
- mypy - Type checking

Install dev dependencies:
```bash
uv sync --extra dev
```

### Coding Guidelines
Follow the **karpathy-guidelines** skill:
1. **Think before coding** - Surface assumptions, don't hide confusion
2. **Simplicity first** - Minimum code, no speculative features
3. **Surgical changes** - Touch only what you must
4. **Goal-driven** - Define verifiable success criteria

## Next Steps

- [ ] Set up data ingestion pipeline (PDF/text files)
- [ ] Implement document chunking strategy
- [ ] Configure embedding model selection
- [ ] Build retrieval + LLM chain
- [ ] Add example queries/tests
- [ ] Performance optimization & evaluation

## Python Version

- **Required:** >= 3.11
- **Tested with:** 3.13.9
- **CUDA support:** 13.0 (for GPU acceleration with torch)

## Environment Variables

Create a `.env` file for local configuration:
```
OPENAI_API_KEY=your_key_here
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Notes

- All dependencies are pinned in `uv.lock` for reproducibility
- The `.venv/` directory is auto-generated and not tracked in git
- GPU support is pre-configured (CUDA 13.0) but CPU-only is also supported

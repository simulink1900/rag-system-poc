#!/usr/bin/env python3
"""Debug script to test filter extraction."""

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_litellm import ChatLiteLLM
from rag.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
from rag.filter_parser import extract_filters

# Initialize LLM
proxy_url = os.getenv("LITELLM_PROXY_URL")
api_key = os.getenv("LITELLM_MASTER_KEY")

llm = ChatLiteLLM(
    model=LLM_MODEL_NAME,
    api_base=proxy_url,
    api_key=api_key,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    timeout=LLM_TIMEOUT,
)

# Test questions
test_questions = [
    "What do visitors from Australia say about Disneyland in HongKong?",
    "Is spring a good time to visit Disneyland?",
    "Is Disneyland California usually crowded in June?",
    "Is the staff in Paris friendly?",
    "What are recent positive reviews from 2023?",
]

print("Testing filter extraction...\n")

for question in test_questions:
    print(f"Question: {question}")
    print("-" * 70)

    filters = extract_filters(question, llm)

    print(f"Extracted filters:")
    for key, value in filters.items():
        print(f"  {key}: {value}")

    print("=" * 70 + "\n")

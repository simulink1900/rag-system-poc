#!/usr/bin/env python3
"""Debug script to test filter extraction."""

import sys
from pathlib import Path
import os
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_litellm import ChatLiteLLM
from rag.config import LLM_MODEL_NAME

# Initialize LLM
proxy_url = os.getenv("LITELLM_PROXY_URL", "https://litellm.gke-prod.linnovate.net")
api_key = os.getenv("LITELLM_MASTER_KEY")

llm = ChatLiteLLM(
    model=LLM_MODEL_NAME,
    api_base=proxy_url,
    api_key=api_key,
    timeout=120,
)

# Test questions
test_questions = [
    "What do visitors from Australia say about Disneyland in HongKong?",
    "Is spring a good time to visit Disneyland?",
    "Is Disneyland California usually crowded in June?",
    "Is the staff in Paris friendly?",
]

print("Testing filter extraction...\n")

for question in test_questions:
    print(f"Question: {question}")
    print("-" * 70)

    prompt = f"""Extract metadata filters from this Disneyland question.
Return ONLY valid JSON with keys: branch, reviewer_location, season. No other text.
- branch: one of "Disneyland_California", "Disneyland_HongKong", "Disneyland_Paris", or null
- reviewer_location: country/region string or null
- season: "spring", "summer", "autumn", "winter", or null

Question: {question}

JSON:"""

    response = llm.invoke(prompt)
    response_text = response.content.strip()

    print(f"Raw LLM Response:\n{response_text}\n")

    # Try to parse JSON
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        print(f"Extracted JSON:\n{json_str}\n")
        try:
            parsed = json.loads(json_str)
            print(f"Parsed: {parsed}\n")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}\n")
    else:
        print("No JSON found in response\n")

    print("=" * 70 + "\n")

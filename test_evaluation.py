#!/usr/bin/env python3
"""
Test the LLM-as-judge evaluation module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
from rag.evaluation import evaluate_answer, EvaluationScores
from langchain_litellm import ChatLiteLLM
import os


def test_evaluation():
    """Test evaluation with mock data."""
    # Setup LLM
    proxy_url = os.getenv("LITELLM_PROXY_URL", "https://litellm.gke-prod.linnovate.net")
    api_key = os.getenv("LITELLM_MASTER_KEY")
    llm = ChatLiteLLM(
        model=LLM_MODEL_NAME,
        api_base=proxy_url,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
    )

    # Test data
    question = "What do visitors say about Disneyland California in summer?"
    answer = "Visitors generally report that summer is a popular time at Disneyland California, with mixed reviews about crowds and heat."
    context = """[Review from USA, Disneyland_California, 2024-6, Rating: 4/5]
Summer visit was hot but fun. Staff was helpful. Would come back.

[Review from Canada, Disneyland_California, 2024-7, Rating: 3/5]
Too crowded in July. Long wait times. Park was still enjoyable."""

    print("Testing evaluation...")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()

    try:
        scores = evaluate_answer(question, answer, context, llm)
        print("✅ Evaluation successful!")
        print(f"\nEvaluation Scores:")
        print(f"  Relevance:     {scores['relevance']:.1f}")
        print(f"  Conciseness:   {scores['conciseness']:.1f}")
        print(f"  Helpfulness:   {scores['helpfulness']:.1f}")
        print(f"  Hallucination: {scores['hallucination']:.1f}")

        # Validate scores are in the correct range
        valid_scores = {0, 0.2, 0.4, 0.6, 0.8, 1.0}
        for metric, value in scores.items():
            assert value in valid_scores, f"{metric} score {value} not in valid range"

        print("\n✅ All scores in valid range [0, 0.2, 0.4, 0.6, 0.8, 1.0]")
        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


if __name__ == "__main__":
    success = test_evaluation()
    sys.exit(0 if success else 1)

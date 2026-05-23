#!/usr/bin/env python3
"""
Interactive CLI chat for Disneyland RAG system.
Ask questions about visitor reviews across different parks and seasons.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.config import DATA_PATH, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT, EVALUATION_ENABLED
from rag.ingest import load_reviews
from rag.embeddings import SentenceTransformerEmbeddings
from rag.vectorstore import get_or_build_collection
from rag.chain import ask, AnswerWithEvaluation
from rag.filter_parser import extract_filters
from langchain_litellm import ChatLiteLLM
import os


def main():
    print("=" * 70)
    print("DISNEYLAND RAG CHAT SYSTEM")
    print("=" * 70)
    print("\nInitializing system...")

    # Initialize components
    print("  • Loading reviews from CSV...")
    documents = load_reviews(DATA_PATH)
    print(f"    Loaded {len(documents)} reviews")

    print("  • Initializing embeddings...")
    embeddings = SentenceTransformerEmbeddings()
    print(f"    Using model: all-MiniLM-L6-v2")

    print("  • Building/loading ChromaDB collection...")
    collection = get_or_build_collection(documents, embeddings)
    print(f"    Collection ready: {collection.count()} documents")

    print("  • Initializing LLM...")
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
    print(f"    Connected to {proxy_url}")
    print(f"    Using model: {LLM_MODEL_NAME}")
    print(f"    Temperature: {LLM_TEMPERATURE}, Max tokens: {LLM_MAX_TOKENS}")

    print("\n" + "=" * 70)
    print("Ready! Ask questions about Disneyland visitor experiences.")
    print(f"Evaluation: {'✅ Enabled' if EVALUATION_ENABLED else '❌ Disabled'}")
    print("Examples:")
    print("  • What do visitors from Australia say about Disneyland in HongKong?")
    print("  • Is spring a good time to visit Disneyland?")
    print("  • Is Disneyland California usually crowded in June?")
    print("  • Is the staff in Paris friendly?")
    print("\nType 'exit' or 'quit' to exit.")
    print("=" * 70)

    # Chat loop
    while True:
        try:
            question = input("\n📝 Question: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                print("\nGoodbye! 👋")
                break

            print("\n⏳ Processing...")

            # Extract filters
            try:
                filters = extract_filters(question, llm)
            except Exception as e:
                print(f"⚠️  Filter extraction error: {e}")
                filters = {}

            print(f"\n🔍 Extracted filters:")
            print(f"   Branch: {filters.get('branch') or 'any'}")
            print(f"   Location: {filters.get('reviewer_location') or 'any'}")
            print(f"   Season: {filters.get('season') or 'any'}")
            print(f"   Rating: {filters.get('min_rating') or 'any'}")
            print(f"   Year/Month: {filters.get('year_month') or 'any'}")
            print(f"   Prefer Recent: {filters.get('prefer_recent', False)}")

            # Get answer (with optional evaluation)
            result = ask(question, collection, embeddings, llm, auto_extract_filters=False, filters=filters, evaluation=EVALUATION_ENABLED)

            print(f"\n✨ Answer:")
            if isinstance(result, dict) and "answer" in result:
                answer_text = result["answer"]
                print(f"{answer_text}")

                # Display evaluation scores
                if "evaluation" in result:
                    scores = result["evaluation"]
                    print(f"\n📊 Evaluation Scores:")
                    print(f"   Relevance:    {scores['relevance']:.1f}")
                    print(f"   Conciseness:  {scores['conciseness']:.1f}")
                    print(f"   Helpfulness:  {scores['helpfulness']:.1f}")
                    print(f"   Hallucination (↓ = better): {scores['hallucination']:.1f}")
            else:
                print(f"{result}")
            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()

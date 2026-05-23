import chromadb
from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap, Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_litellm import ChatLiteLLM
from .embeddings import SentenceTransformerEmbeddings
from .retriever import retrieve
from .filter_parser import extract_filters, ReviewFilters
from .evaluation import evaluate_answer, EvaluationScores
from .config import TOP_K_RETRIEVAL, LLMASAJUDGE_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
import os


RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions about Disneyland parks
based only on the visitor reviews provided below. If the reviews do not contain enough
information to answer the question, say so honestly. Base your answer on the reviews provided.

Visitor Reviews:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


class AnswerWithEvaluation(TypedDict):
    answer: str
    evaluation: EvaluationScores


def format_docs(documents: list[Document]) -> str:
    """Format retrieved documents into context string."""
    if not documents:
        return "No reviews found."

    formatted = []
    for doc in documents:
        meta = doc.metadata
        location = meta.get("reviewer_location", "Unknown")
        branch = meta.get("branch", "Unknown")
        year_month = meta.get("year_month", "Unknown")
        rating = meta.get("rating", "Unknown")

        header = f"[Review from {location}, {branch}, {year_month}, Rating: {rating}/5]"
        text = doc.page_content[:300]
        formatted.append(f"{header}\n{text}")

    return "\n\n".join(formatted)


def build_rag_chain(
    collection: chromadb.Collection,
    embeddings: SentenceTransformerEmbeddings,
    llm: ChatLiteLLM,
    filters: ReviewFilters | None = None,
    n_results: int = TOP_K_RETRIEVAL,
) -> Runnable:
    """
    Build LCEL RAG chain with optional pre-specified filters.
    Chain: question → retrieve+format → prompt → LLM → answer
    """
    if filters is None:
        filters = {}

    def retrieve_with_filters(question: str) -> list[Document]:
        return retrieve(
            question,
            collection,
            embeddings,
            n_results=n_results,
            branch=filters.get("branch"),
            reviewer_location=filters.get("reviewer_location"),
            season=filters.get("season"),
            min_rating=filters.get("min_rating"),
            year_month=filters.get("year_month"),
            prefer_recent=filters.get("prefer_recent"),
        )

    chain = (
        RunnableMap(
            {
                "context": RunnableLambda(retrieve_with_filters) | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return chain


def ask(
    question: str,
    collection: chromadb.Collection,
    embeddings: SentenceTransformerEmbeddings,
    llm: ChatLiteLLM,
    auto_extract_filters: bool = True,
    filters: ReviewFilters | None = None,
    n_results: int = TOP_K_RETRIEVAL,
    evaluation: bool = False,
) -> str | AnswerWithEvaluation:
    """
    Ask a question and get an answer.

    If auto_extract_filters=True, uses LLM to extract filters from the question.
    Otherwise uses the provided filters dict (or empty if None).

    If evaluation=True, returns AnswerWithEvaluation dict with answer + evaluation scores.
    Otherwise returns answer string.
    """
    if auto_extract_filters:
        filters = extract_filters(question, llm)
    elif filters is None:
        filters = {}

    chain = build_rag_chain(collection, embeddings, llm, filters, n_results)
    answer = chain.invoke(question)

    if not evaluation:
        return answer

    from .retriever import retrieve
    docs = retrieve(
        question,
        collection,
        embeddings,
        n_results=n_results,
        branch=filters.get("branch"),
        reviewer_location=filters.get("reviewer_location"),
        season=filters.get("season"),
        min_rating=filters.get("min_rating"),
        year_month=filters.get("year_month"),
        prefer_recent=filters.get("prefer_recent"),
    )
    context = format_docs(docs)

    # Create judge LLM for evaluation
    proxy_url = os.getenv("LITELLM_PROXY_URL")
    api_key = os.getenv("LITELLM_MASTER_KEY")
    judge_llm = ChatLiteLLM(
        model=LLMASAJUDGE_MODEL_NAME,
        api_base=proxy_url,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
    )

    scores = evaluate_answer(question, answer, context, judge_llm)
    return AnswerWithEvaluation(answer=answer, evaluation=scores)

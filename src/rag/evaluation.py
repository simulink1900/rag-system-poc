import json
import os
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_litellm import ChatLiteLLM
from .config import LLMASAJUDGE_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT


class EvaluationScores(TypedDict):
    relevance: float
    conciseness: float
    helpfulness: float
    hallucination: float


EVALUATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert evaluator of AI-generated answers about Disneyland visitor reviews.
Evaluate the provided answer on the following metrics using a 0-1 scale with 0.2 increments (0, 0.2, 0.4, 0.6, 0.8, 1.0):

1. **Relevance** (0-1): How well does the answer address the user's question?
2. **Conciseness** (0-1): Is the answer appropriately brief without losing important information?
3. **Helpfulness** (0-1): How useful is this answer to someone seeking information about Disneyland?
4. **Hallucination** (0-1): How much false or unsupported information does the answer contain? (0=no hallucination, 1=full hallucination)

Respond ONLY with a valid JSON object in this exact format (no additional text):
{{"relevance": X, "conciseness": Y, "helpfulness": Z, "hallucination": W}}

Where X, Y, Z, W are numbers from {{0, 0.2, 0.4, 0.6, 0.8, 1.0}}.""",
        ),
        (
            "human",
            """Question: {question}

Answer to evaluate:
{answer}

Visitor Reviews (context used for answer):
{context}""",
        ),
    ]
)


def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    judge_llm: ChatLiteLLM | None = None,
) -> EvaluationScores:
    """
    Evaluate an answer using LLM-as-judge on 4 metrics.
    Returns scores on [0, 0.2, 0.4, 0.6, 0.8, 1.0] scale.

    If judge_llm is not provided, creates one using LLMASAJUDGE_MODEL_NAME.
    """
    if judge_llm is None:
        proxy_url = os.getenv("LITELLM_PROXY_URL", "https://litellm.gke-prod.linnovate.net")
        api_key = os.getenv("LITELLM_MASTER_KEY")
        judge_llm = ChatLiteLLM(
            model=LLMASAJUDGE_MODEL_NAME,
            api_base=proxy_url,
            api_key=api_key,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT,
        )

    chain = EVALUATION_PROMPT_TEMPLATE | judge_llm | StrOutputParser()

    response = chain.invoke(
        {
            "question": question,
            "answer": answer,
            "context": context,
        }
    )

    try:
        scores_dict = json.loads(response)
        return EvaluationScores(
            relevance=float(scores_dict["relevance"]),
            conciseness=float(scores_dict["conciseness"]),
            helpfulness=float(scores_dict["helpfulness"]),
            hallucination=float(scores_dict["hallucination"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Failed to parse evaluation scores: {response}") from e

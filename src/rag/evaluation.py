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
Evaluate the provided answer on the following metrics using a 0-1 continuous scale with 0.2 increments (0, 0.2, 0.4, 0.6, 0.8, 1.0):

1. **Relevance** (0-1): Evaluate the relevance of the generation on a continuous scale from 0 to 1. A generation can be considered relevant (Score: 1) if it enhances or clarifies the response, adding value to the user's comprehension of the topic in question. Relevance is determined by the extent to which the provided information addresses the specific question asked, staying focused on the subject without straying into unrelated areas or providing extraneous details.

2. **Conciseness** (0-1): Evaluate the conciseness of the generation on a continuous scale from 0 to 1. A generation can be considered concise (Score: 1) if it directly and succinctly answers the question posed, focusing specifically on the information requested without including unnecessary, irrelevant, or excessive details.

3. **Helpfulness** (0-1): Evaluate the helpfulness of the generation on a continuous scale from 0 to 1. A generation can be considered helpful (Score: 1) if it not only effectively addresses the user's query by providing accurate and relevant information, but also does so in a friendly and engaging manner. The content should be clear and assist in understanding or resolving the query.

4. **Hallucination** (0-1): Evaluate the degree of hallucination in the generation on a continuous scale from 0 to 1. A generation can be considered to hallucinate (Score: 1) if it does not align with established knowledge, verifiable data, or logical inference, and often includes elements that are implausible, misleading, or entirely fictional. (0=no hallucination, 1=full hallucination)

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

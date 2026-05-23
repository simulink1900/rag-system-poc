import json
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_litellm import ChatLiteLLM


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
    llm: ChatLiteLLM,
) -> EvaluationScores:
    """
    Evaluate an answer using LLM-as-judge on 4 metrics.
    Returns scores on [0, 0.2, 0.4, 0.6, 0.8, 1.0] scale.
    """
    chain = EVALUATION_PROMPT_TEMPLATE | llm | StrOutputParser()

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

import json
from typing import TypedDict
from langchain_litellm import ChatLiteLLM
from .config import VALID_BRANCHES


class ReviewFilters(TypedDict, total=False):
    branch: str | None
    reviewer_location: str | None
    season: str | None


BRANCH_NAME_MAPPING = {
    "california": "Disneyland_California",
    "hong kong": "Disneyland_HongKong",
    "paris": "Disneyland_Paris",
}

SEASON_VARIATIONS = {
    "spring": "spring",
    "summer": "summer",
    "autumn": "autumn",
    "fall": "autumn",
    "winter": "winter",
}


def _normalize_branch(value: str | None) -> str | None:
    """Map free-text branch names to canonical values."""
    if not value:
        return None
    value_str = str(value).strip()
    # If already canonical, return as-is
    if value_str in VALID_BRANCHES:
        return value_str
    # Otherwise try to map it
    value_lower = value_str.lower()
    return BRANCH_NAME_MAPPING.get(value_lower)


def _normalize_season(value: str | None) -> str | None:
    """Map season variations to canonical values."""
    if not value or value is None:
        return None
    value_str = str(value).strip()
    if not value_str or value_str.lower() == "none" or value_str.lower() == "null":
        return None
    value_lower = value_str.lower()
    return SEASON_VARIATIONS.get(value_lower)


def extract_filters(question: str, llm: ChatLiteLLM) -> ReviewFilters:
    """
    Use LLM to extract metadata filters from a free-text question.
    Returns a ReviewFilters dict with normalized values.
    """
    prompt = f"""Extract metadata filters from this Disneyland question.
Return ONLY valid JSON with keys: branch, reviewer_location, season. No other text.
- branch: one of "Disneyland_California", "Disneyland_HongKong", "Disneyland_Paris", or null
- reviewer_location: country/region string or null
- season: "spring", "summer", "autumn", "winter", or null

Question: {question}

JSON:"""

    response = llm.invoke(prompt)
    response_text = response.content.strip()

    # Try to extract JSON from response (in case LLM adds extra text)
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        response_text = response_text[json_start:json_end]

    try:
        extracted = json.loads(response_text)
    except json.JSONDecodeError:
        # If JSON parsing fails, return empty filters
        return {"branch": None, "reviewer_location": None, "season": None}

    branch = extracted.get("branch")
    branch = _normalize_branch(branch)
    if branch and branch not in VALID_BRANCHES:
        branch = None

    reviewer_location = extracted.get("reviewer_location")
    if reviewer_location:
        reviewer_location = str(reviewer_location).strip()

    season = extracted.get("season")
    season = _normalize_season(season)

    return {
        "branch": branch,
        "reviewer_location": reviewer_location,
        "season": season,
    }

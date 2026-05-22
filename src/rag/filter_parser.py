import json
from typing import TypedDict
from langchain_litellm import ChatLiteLLM
from .config import VALID_BRANCHES


class ReviewFilters(TypedDict, total=False):
    branch: str | None
    reviewer_location: str | None
    season: str | None
    min_rating: int | None  # Filter for reviews >= this rating (1-5)
    year_month: str | None  # Filter for specific month (YYYY-M format)


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
Return ONLY valid JSON with keys: branch, reviewer_location, season, sentiment, year_month. No other text.
- branch: one of "Disneyland_California", "Disneyland_HongKong", "Disneyland_Paris", or null
- reviewer_location: country/region string or null
- season: "spring", "summer", "autumn", "winter", or null
- sentiment: "positive" (5-4 stars), "neutral" (3 stars), "negative" (2-1 stars), or null
- year_month: specific month in "YYYY-M" format (e.g. "2023-6" for June 2023), or null for any time

Use sentiment to filter:
- "good/great/love/excellent/best" → positive
- "bad/terrible/hate/worst/complaint" → negative
- "meh/ok/average" → neutral

Use year_month for temporal queries:
- "2023-6" for June 2023
- "2024-8" for August 2024
- null if no specific time mentioned

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
        return {
            "branch": None,
            "reviewer_location": None,
            "season": None,
            "min_rating": None,
            "year_month": None,
        }

    branch = extracted.get("branch")
    branch = _normalize_branch(branch)
    if branch and branch not in VALID_BRANCHES:
        branch = None

    reviewer_location = extracted.get("reviewer_location")
    if reviewer_location:
        reviewer_location = str(reviewer_location).strip()

    season = extracted.get("season")
    season = _normalize_season(season)

    # Convert sentiment to min_rating
    sentiment = extracted.get("sentiment")
    min_rating = None
    if sentiment:
        sentiment_lower = str(sentiment).lower().strip()
        if sentiment_lower == "positive":
            min_rating = 4  # Show 4-5 star reviews
        elif sentiment_lower == "negative":
            min_rating = 1  # Show 1-2 star reviews
        elif sentiment_lower == "neutral":
            min_rating = 3  # Show 3 star reviews

    # Validate year_month format (YYYY-M)
    year_month = extracted.get("year_month")
    if year_month:
        year_month = str(year_month).strip()
        # Basic validation: should be YYYY-M or YYYY-MM format
        if not (len(year_month) >= 5 and "-" in year_month):
            year_month = None

    return {
        "branch": branch,
        "reviewer_location": reviewer_location,
        "season": season,
        "min_rating": min_rating,
        "year_month": year_month,
    }

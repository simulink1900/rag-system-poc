import json
from typing import TypedDict
from langchain_litellm import ChatLiteLLM
from .config import VALID_BRANCHES, PREFER_RECENT_BY_DEFAULT


class ReviewFilters(TypedDict, total=False):
    branch: str | None
    reviewer_location: str | None
    season: str | None
    min_rating: int | None  # Filter for reviews >= this rating (1-5)
    year_month: str | None  # Filter for specific month/year (YYYY-M, YYYY, M, or null)
    prefer_recent: bool  # Prioritize recent reviews in ranking


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


def _normalize_year_month(value: str | None) -> str | None:
    """Normalize temporal filter supporting flexible formats.

    Accepts:
    - YYYY-M or YYYY-MM: Full date (e.g., "2023-6", "2023-06")
    - YYYY: Year only (e.g., "2023")
    - M or MM: Month only (e.g., "6", "06")
    Returns normalized value or None if invalid.
    """
    if not value:
        return None
    value_str = str(value).strip()
    if not value_str or value_str.lower() in ("none", "null"):
        return None

    # Check if it contains a hyphen (YYYY-M format)
    if "-" in value_str:
        parts = value_str.split("-")
        if len(parts) == 2:
            try:
                year = int(parts[0])
                month = int(parts[1])
                # Validate year and month ranges
                if 1900 <= year <= 2100 and 1 <= month <= 12:
                    return f"{year}-{month}"  # Normalize to YYYY-M
            except ValueError:
                return None
    else:
        # No hyphen: could be year-only or month-only
        try:
            value_int = int(value_str)
            # If 4 digits, treat as year; if 1-2 digits, treat as month
            if value_int > 100:  # Likely a year
                if 1900 <= value_int <= 2100:
                    return str(value_int)
            elif 1 <= value_int <= 12:  # Valid month
                return str(value_int)
        except ValueError:
            return None

    return None


def extract_filters(question: str, llm: ChatLiteLLM) -> ReviewFilters:
    """
    Use LLM to extract metadata filters from a free-text question.
    Returns a ReviewFilters dict with normalized values.
    """
    prompt = f"""Extract metadata filters from this Disneyland question.
Return ONLY valid JSON with keys: branch, reviewer_location, season, sentiment, year_month, prefer_recent. No other text.
- branch: one of "Disneyland_California", "Disneyland_HongKong", "Disneyland_Paris", or null
- reviewer_location: country/region string or null
- season: "spring", "summer", "autumn", "winter", or null
- sentiment: "positive" (5-4 stars), "neutral" (3 stars), "negative" (2-1 stars), or null
- year_month: temporal info - can be "YYYY-M" (e.g. "2023-6"), "YYYY" (e.g. "2023"), "M" (e.g. "6" for June), or null
- prefer_recent: true if query asks for recent/latest/newest reviews, false otherwise

Use sentiment to filter:
- "good/great/love/excellent/best" → positive
- "bad/terrible/hate/worst/complaint" → negative
- "meh/ok/average" → neutral

Use year_month for temporal queries:
- "2023-6" for June 2023
- "2024" for any month in 2024
- "6" for June of any year
- null if no specific time mentioned

Set prefer_recent to true for queries mentioning: recent, latest, newest, recent months, recent years, current, etc.

If only month is mentioned, prefer season (e.g., "6" → "summer" for June).

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
            "prefer_recent": PREFER_RECENT_BY_DEFAULT,
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

    # Normalize year_month (supports flexible formats: YYYY-M, YYYY, M)
    year_month = extracted.get("year_month")
    year_month = _normalize_year_month(year_month)

    # Fallback: if we have month-only (1-12) and no season yet, infer season from month
    if year_month and season is None and "-" not in year_month:
        try:
            month_int = int(year_month)
            if 1 <= month_int <= 12:
                # Map month to season
                if month_int in (12, 1, 2):
                    season = "winter"
                elif month_int in (3, 4, 5):
                    season = "spring"
                elif month_int in (6, 7, 8):
                    season = "summer"
                elif month_int in (9, 10, 11):
                    season = "autumn"
        except (ValueError, TypeError):
            pass

    # Extract prefer_recent flag (default from config)
    prefer_recent = extracted.get("prefer_recent", PREFER_RECENT_BY_DEFAULT)
    if prefer_recent is not None:
        prefer_recent = bool(prefer_recent)
    else:
        prefer_recent = PREFER_RECENT_BY_DEFAULT

    return {
        "branch": branch,
        "reviewer_location": reviewer_location,
        "season": season,
        "min_rating": min_rating,
        "year_month": year_month,
        "prefer_recent": prefer_recent,
    }

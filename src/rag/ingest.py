import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from .config import SEASON_MAP, CSV_ENCODING


def _derive_season(year_month: str) -> str | None:
    """Parse 'YYYY-M' format → season string. Returns None if parsing fails."""
    if not year_month or year_month.strip() == "" or year_month == "missing":
        return None
    try:
        parts = year_month.split("-")
        month = int(parts[1])
        return SEASON_MAP.get(month)
    except (ValueError, IndexError):
        return None


def load_reviews(csv_path: Path | str, encoding: str = CSV_ENCODING) -> list[Document]:
    """
    Load DisneylandReviews.csv and return one Document per row.

    page_content = Review_Text
    metadata keys: review_id, rating (int), year_month (str),
                   reviewer_location, branch, season (str)
    """
    df = pd.read_csv(csv_path, encoding=encoding)

    documents = []
    for _, row in df.iterrows():
        review_text = str(row["Review_Text"]).strip()
        if not review_text or review_text == "nan":
            continue

        year_month = str(row["Year_Month"]).strip()
        if year_month == "missing":
            year_month = ""
        season = _derive_season(year_month) or ""

        doc = Document(
            page_content=review_text,
            metadata={
                "review_id": str(row["Review_ID"]),
                "rating": int(row["Rating"]),
                "year_month": year_month,
                "reviewer_location": str(row["Reviewer_Location"]),
                "branch": str(row["Branch"]),
                "season": season,
            },
        )
        documents.append(doc)

    return documents

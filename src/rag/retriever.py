from langchain_core.documents import Document
import chromadb
from .embeddings import SentenceTransformerEmbeddings
from .config import TOP_K_RETRIEVAL, PREFER_RECENT_BY_DEFAULT


def build_where_clause(
    branch: str | None = None,
    reviewer_location: str | None = None,
    season: str | None = None,
    min_rating: int | None = None,
    year_month: str | None = None,
) -> dict | None:
    """
    Construct a ChromaDB where-clause dict from optional filter parameters.

    Only applies full year_month format (YYYY-M) to ChromaDB.
    Partial formats (year-only, month-only) are handled via post-filtering.

    Single filter:    {"field": {"$eq": value}}
    Multiple filters: {"$and": [...]}
    No filters:       None
    """
    conditions = []

    if branch and branch.strip():
        conditions.append({"branch": {"$eq": branch}})

    if reviewer_location and reviewer_location.strip():
        conditions.append({"reviewer_location": {"$eq": reviewer_location}})

    if season and season.strip():
        conditions.append({"season": {"$eq": season}})

    if min_rating is not None:
        conditions.append({"rating": {"$gte": min_rating}})

    if year_month and year_month.strip():
        year_month_clean = year_month.strip()
        # Only apply to ChromaDB if full format (contains hyphen)
        # Partial formats are handled in retrieve() via post-filtering
        if "-" in year_month_clean:
            conditions.append({"year_month": {"$eq": year_month_clean}})

    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embeddings: SentenceTransformerEmbeddings,
    n_results: int = TOP_K_RETRIEVAL,
    branch: str | None = None,
    reviewer_location: str | None = None,
    season: str | None = None,
    min_rating: int | None = None,
    year_month: str | None = None,
    prefer_recent: bool | None = None,
) -> list[Document]:
    """
    Embed query, call collection.query() with optional where clause,
    return list of Document objects with metadata restored.

    If prefer_recent=True, results are sorted by year_month (newest first).
    """
    if prefer_recent is None:
        prefer_recent = PREFER_RECENT_BY_DEFAULT

    query_embedding = embeddings.embed_query(query)
    where_clause = build_where_clause(branch, reviewer_location, season, min_rating, year_month)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_clause,
    )

    documents = []
    if results["documents"] and results["documents"][0]:
        for doc_text, metadata in zip(
            results["documents"][0], results["metadatas"][0]
        ):
            doc = Document(page_content=doc_text, metadata=metadata)
            documents.append(doc)

    # Post-filter by partial year_month if needed (couldn't use regex in ChromaDB)
    if year_month:
        year_month_clean = year_month.strip()
        if "-" not in year_month_clean:
            # Partial format: filter documents in Python
            documents = _filter_by_partial_year_month(documents, year_month_clean)

    # Sort by recency if requested (newest first)
    if prefer_recent:
        documents.sort(
            key=lambda doc: _parse_year_month_for_sort(doc.metadata.get("year_month")),
            reverse=True,
        )

    return documents


def _filter_by_partial_year_month(documents: list, year_month_partial: str) -> list:
    """Filter documents by partial year_month (year-only or month-only).

    Examples:
    - "2023": keeps docs with year_month="2023-1", "2023-12", etc.
    - "6": keeps docs with year_month="2020-6", "2023-6", etc.
    """
    try:
        value_int = int(year_month_partial)
        if value_int > 100:
            # Year-only: filter by year
            return [
                doc for doc in documents
                if doc.metadata.get("year_month", "").startswith(f"{value_int}-")
            ]
        elif 1 <= value_int <= 12:
            # Month-only: filter by month
            return [
                doc for doc in documents
                if doc.metadata.get("year_month", "").endswith(f"-{value_int}")
            ]
    except ValueError:
        pass
    return documents


def _parse_year_month_for_sort(year_month_str: str | None) -> tuple[int, int]:
    """Parse year_month string for sorting. Returns (year, month) tuple.

    Examples:
    - "2023-6" → (2023, 6)
    - "2023" → (2023, 12)  # Treat year-only as end of year
    - "6" → (2000, 6)      # Treat month-only as year 2000
    - None → (0, 0)        # Unknown dates sort first
    """
    if not year_month_str:
        return (0, 0)

    year_month_str = str(year_month_str).strip()

    if "-" in year_month_str:
        try:
            parts = year_month_str.split("-")
            year = int(parts[0])
            month = int(parts[1])
            return (year, month)
        except (ValueError, IndexError):
            return (0, 0)
    else:
        try:
            value = int(year_month_str)
            if value > 100:  # Likely a year
                return (value, 12)  # End of year
            elif 1 <= value <= 12:  # Likely a month
                return (2000, value)  # Default year for month-only
        except ValueError:
            pass

    return (0, 0)

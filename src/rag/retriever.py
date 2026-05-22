from langchain_core.documents import Document
import chromadb
from .embeddings import SentenceTransformerEmbeddings
from .config import TOP_K_RETRIEVAL


def build_where_clause(
    branch: str | None = None,
    reviewer_location: str | None = None,
    season: str | None = None,
    min_rating: int | None = None,
    year_month: str | None = None,
) -> dict | None:
    """
    Construct a ChromaDB where-clause dict from optional filter parameters.

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
        conditions.append({"year_month": {"$eq": year_month}})

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
) -> list[Document]:
    """
    Embed query, call collection.query() with optional where clause,
    return list of Document objects with metadata restored.
    """
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

    return documents

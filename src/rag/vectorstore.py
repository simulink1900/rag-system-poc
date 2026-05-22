import chromadb
from pathlib import Path
from langchain_core.documents import Document
from tqdm import tqdm
from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from .embeddings import SentenceTransformerEmbeddings


def get_chroma_client(
    persist_dir: Path | str = CHROMA_PERSIST_DIR,
) -> chromadb.PersistentClient:
    """Return (or create) a persistent ChromaDB client."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def collection_exists(
    client: chromadb.PersistentClient, collection_name: str = COLLECTION_NAME
) -> bool:
    """Check whether the named collection already has documents."""
    try:
        collection = client.get_collection(name=collection_name)
        return collection.count() > 0
    except Exception:
        return False


def load_collection(
    client: chromadb.PersistentClient, collection_name: str = COLLECTION_NAME
) -> chromadb.Collection:
    """Load an existing collection without re-embedding."""
    return client.get_collection(name=collection_name)


def build_collection(
    documents: list[Document],
    embeddings: SentenceTransformerEmbeddings,
    client: chromadb.PersistentClient,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 512,
) -> chromadb.Collection:
    """
    Embed all documents and upsert into ChromaDB in batches.
    Uses upsert so re-running is idempotent.
    """
    print(f"🔨 Creating embeddings (this takes 3-5 minutes)...")
    collection = client.get_or_create_collection(name=collection_name)

    num_batches = (len(documents) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="📦 Embedding and storing"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(documents))
        batch_docs = documents[start_idx:end_idx]

        batch_texts = [doc.page_content for doc in batch_docs]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        batch_ids = [doc.metadata["review_id"] for doc in batch_docs]
        batch_metadatas = [doc.metadata for doc in batch_docs]

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas,
        )

    print(f"✅ Embeddings created and saved to {CHROMA_PERSIST_DIR}")
    return collection


def get_or_build_collection(
    documents: list[Document],
    embeddings: SentenceTransformerEmbeddings,
    persist_dir: Path | str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Load existing collection if it exists and is complete,
    otherwise build it from scratch. Detects incomplete collections.
    """
    client = get_chroma_client(persist_dir)
    if collection_exists(client, collection_name):
        collection = load_collection(client, collection_name)
        stored_count = collection.count()
        input_count = len(documents)

        if stored_count == input_count:
            print(f"\n📚 EMBEDDINGS LOADED FROM CACHE (instant)")
            print(f"   Location: {persist_dir}")
            print(f"   Documents: {stored_count:,}")
            print(f"   Status: Ready to use\n")
            return collection
        else:
            print(f"\n⚠️  EMBEDDINGS INCOMPLETE - REBUILDING...")
            print(f"   Expected: {input_count:,} documents")
            print(f"   Found: {stored_count:,} documents")
            print(f"   Restarting build...\n")
            return build_collection(documents, embeddings, client, collection_name)
    else:
        print(f"\n📝 EMBEDDINGS NOT FOUND - CREATING NEW...")
        print(f"   Input: {len(documents):,} documents")
        print(f"   Location: {persist_dir}\n")
        return build_collection(documents, embeddings, client, collection_name)

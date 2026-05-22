from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME, EMBED_BATCH_SIZE


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode documents in batches, return list of float lists."""
        embeddings = self.model.encode(
            texts, batch_size=self.batch_size, convert_to_tensor=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode single query, return float list."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

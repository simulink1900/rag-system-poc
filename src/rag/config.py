from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "DisneylandReviews.csv"
CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME: str = "disneyland_reviews"

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "litellm_proxy/openrouter/openai/gpt-4.1-mini")
EMBED_BATCH_SIZE: int = 512
TOP_K_RETRIEVAL: int = 30
CSV_ENCODING: str = "latin-1"

SEASON_MAP: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

VALID_BRANCHES: set[str] = {
    "Disneyland_California",
    "Disneyland_HongKong",
    "Disneyland_Paris",
}

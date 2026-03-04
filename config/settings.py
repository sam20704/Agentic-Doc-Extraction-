"""
Configuration and environment variable management.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ─── Azure Document Intelligence ───
AZURE_DOC_KEY = os.getenv("AZURE_DOC_KEY")
AZURE_DOC_ENDPOINT = os.getenv("AZURE_DOC_ENDPOINT")

# ─── Azure OpenAI (Chat / Vision) ───
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# ─── Azure OpenAI (Embeddings) ───
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)

# ─── Retrieval Settings ───
USE_RETRIEVAL = os.getenv("USE_RETRIEVAL", "true").lower() == "true"
RETRIEVAL_CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "6"))
RETRIEVAL_CHUNK_OVERLAP = int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "2"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "15"))

# ─── Poppler (Windows pdf2image) ───
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

# ─── Application Settings ───
BATCH_SIZE = 2
MAX_TEXT_DISPLAY = 150      # Non-retrieval fallback limit
TEMP_UPLOAD_DIR = "data/uploads"


def validate_config():
    """Validate all required environment variables."""
    required = {
        "AZURE_DOC_KEY": AZURE_DOC_KEY,
        "AZURE_DOC_ENDPOINT": AZURE_DOC_ENDPOINT,
        "AZURE_OPENAI_KEY": AZURE_OPENAI_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please check your .env file."
        )

    if USE_RETRIEVAL and EMBEDDING_PROVIDER == "azure":
        if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
            raise ValueError(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT required when "
                "USE_RETRIEVAL=true and EMBEDDING_PROVIDER=azure."
            )

    return True


os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

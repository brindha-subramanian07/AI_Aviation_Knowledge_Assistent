"""
Configuration Module for AI-Powered Aviation Knowledge Assistant
Centralizes all application settings, directory paths, and model constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: root of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from the project root
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Directory Paths
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"
DOCUMENTS_METADATA_FILE = DATA_DIR / "documents_registry.json"

# Ensure runtime directories exist
for directory in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

# Supported Gemini Models for user selection / fallback
SUPPORTED_GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

# Ensure model from .env is always included in choices
if DEFAULT_GEMINI_MODEL and DEFAULT_GEMINI_MODEL not in SUPPORTED_GEMINI_MODELS:
    SUPPORTED_GEMINI_MODELS.insert(0, DEFAULT_GEMINI_MODEL)

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "aviation_knowledge").strip()

# Chunking Parameters
# Target: 500-800 words/tokens, overlap 50-100
DEFAULT_CHUNK_SIZE_WORDS = 600
DEFAULT_CHUNK_OVERLAP_WORDS = 80

# Retrieval Parameters
DEFAULT_TOP_K = 6
MIN_TOP_K = 2
MAX_TOP_K = 10
# Relevance distance threshold: Chroma uses L2/cosine distance; lower distance = higher similarity
# Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
DEFAULT_RELEVANCE_DISTANCE_THRESHOLD = 0.85

# Supported Document Formats
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# Safety Disclaimer
AVIATION_DISCLAIMER = (
    "This assistant provides information based on uploaded institutional documents. "
    "Always verify safety-critical aviation information against official and current regulations, "
    "manuals, and authorized sources."
)

"""
Helper utilities for text cleaning, file hashing, formatting, and UI presentation.
"""

import hashlib
import re
from datetime import datetime
from typing import Optional


def compute_file_hash(file_bytes: bytes) -> str:
    """Generate a stable SHA-256 hex digest for document deduplication."""
    return hashlib.sha256(file_bytes).hexdigest()


def format_file_size(size_in_bytes: int) -> str:
    """Return a human-readable file size string."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"


def get_current_timestamp() -> str:
    """Return current timestamp in ISO-like readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(text: str) -> str:
    """
    Cleans and normalizes extracted document text:
    - Normalizes strange unicode whitespace and quotes
    - Strips null bytes and unprintable characters
    - Fixes hyphenated line breaks (e.g., 'mainte-\\nnance' -> 'maintenance')
    - Collapses multiple consecutive spaces and empty lines
    - Preserves logical paragraph flow
    """
    if not text:
        return ""

    # Replace null bytes
    text = text.replace("\x00", "")

    # Normalize various Unicode spaces to standard space
    text = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", text)

    # Normalize quotation marks and dashes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)

    # Fix words split by hyphen at end of line (common in PDFs)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace three or more newlines with two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces/tabs within a line with a single space
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    text = "\n".join(lines)

    # Trim leading and trailing whitespace
    return text.strip()


def truncate_preview(text: str, max_chars: int = 250) -> str:
    """Truncate text cleanly at word boundary for UI preview."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."

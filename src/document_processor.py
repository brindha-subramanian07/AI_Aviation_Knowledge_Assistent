"""
Document Processing Module for AI-Powered Aviation Knowledge Assistant
Handles extraction of raw text and page-level metadata from PDF, DOCX, and TXT files.
Includes OCR-detection checks, robust encoding fallbacks, and text sanitization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import io

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

from src.config import SUPPORTED_EXTENSIONS
from utils.helpers import clean_text, compute_file_hash


@dataclass
class ExtractedPage:
    """Represents text extracted from a single document page or section."""
    page_number: int
    text: str
    char_count: int
    word_count: int


@dataclass
class ProcessedDocument:
    """Represents a fully extracted and validated document."""
    document_id: str
    filename: str
    document_type: str
    total_pages: int
    total_characters: int
    total_words: int
    pages: List[ExtractedPage]
    raw_hash: str
    ocr_warning: bool = False
    warning_message: Optional[str] = None


class DocumentProcessingError(Exception):
    """Custom exception raised when document extraction fails."""
    pass


class DocumentProcessor:
    """Extracts, cleans, and structures text from PDF, DOCX, and TXT files."""

    def __init__(self):
        self.supported_extensions = SUPPORTED_EXTENSIONS

    def process_file(
        self,
        file_bytes: bytes,
        filename: str
    ) -> ProcessedDocument:
        """
        Main entry point for extracting text from any supported file.
        Computes SHA-256 hash, extracts page-by-page text, cleans content,
        and flags empty/scanned PDFs requiring OCR.
        """
        if not file_bytes:
            raise DocumentProcessingError(f"File '{filename}' is empty (0 bytes).")

        ext = Path(filename).suffix.lower()
        if ext not in self.supported_extensions:
            raise DocumentProcessingError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(self.supported_extensions)}"
            )

        doc_hash = compute_file_hash(file_bytes)
        # Use first 16 characters of SHA-256 for human-friendly yet collision-resistant ID
        document_id = f"doc_{doc_hash[:16]}"

        if ext == ".pdf":
            pages, ocr_warning, warning_msg = self._extract_pdf(file_bytes, filename)
            doc_type = "PDF"
        elif ext == ".docx":
            pages, ocr_warning, warning_msg = self._extract_docx(file_bytes, filename)
            doc_type = "DOCX"
        elif ext == ".txt":
            pages, ocr_warning, warning_msg = self._extract_txt(file_bytes, filename)
            doc_type = "TXT"
        else:
            raise DocumentProcessingError(f"Unsupported document extension: {ext}")

        total_chars = sum(p.char_count for p in pages)
        total_words = sum(p.word_count for p in pages)

        # If after extraction there's literally no text extracted at all
        if total_words == 0 and not ocr_warning:
            raise DocumentProcessingError(
                f"No readable text could be extracted from '{filename}'. The file may be blank."
            )

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            document_type=doc_type,
            total_pages=len(pages),
            total_characters=total_chars,
            total_words=total_words,
            pages=pages,
            raw_hash=doc_hash,
            ocr_warning=ocr_warning,
            warning_message=warning_msg
        )

    def _extract_pdf(
        self,
        file_bytes: bytes,
        filename: str
    ) -> Tuple[List[ExtractedPage], bool, Optional[str]]:
        """
        Extracts text page-by-page from a PDF using PyMuPDF (fitz).
        Detects scanned/image-only PDFs where extractable text is zero.
        """
        if fitz is None:
            raise DocumentProcessingError("PyMuPDF (fitz) is not installed. Run `pip install pymupdf`.")

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise DocumentProcessingError(f"Could not open PDF file '{filename}'. It may be corrupted or encrypted: {str(e)}")

        if doc.page_count == 0:
            raise DocumentProcessingError(f"PDF '{filename}' contains 0 pages.")

        pages: List[ExtractedPage] = []
        total_extracted_chars = 0

        for page_index in range(doc.page_count):
            page_num = page_index + 1
            try:
                page = doc.load_page(page_index)
                raw_text = page.get_text("text") or ""
                cleaned = clean_text(raw_text)
                words = len(cleaned.split()) if cleaned else 0
                chars = len(cleaned)

                pages.append(ExtractedPage(
                    page_number=page_num,
                    text=cleaned,
                    char_count=chars,
                    word_count=words
                ))
                total_extracted_chars += chars
            except Exception as e:
                # Keep page entry to maintain page numbering continuity
                pages.append(ExtractedPage(
                    page_number=page_num,
                    text="",
                    char_count=0,
                    word_count=0
                ))

        doc.close()

        ocr_warning = False
        warning_msg = None

        # Check if the PDF has virtually no extractable text across all pages
        if total_extracted_chars < 50:
            ocr_warning = True
            warning_msg = (
                f"Notice: PDF '{filename}' appears to be a scanned document or image-only PDF "
                f"({total_extracted_chars} characters extracted). Optical Character Recognition (OCR) "
                "is required to extract text from scanned images. Text search will be limited."
            )

        return pages, ocr_warning, warning_msg

    def _extract_docx(
        self,
        file_bytes: bytes,
        filename: str
    ) -> Tuple[List[ExtractedPage], bool, Optional[str]]:
        """
        Extracts text from a DOCX document using python-docx.
        Extracts both paragraphs and table cells, maintaining logical flow.
        """
        if docx is None:
            raise DocumentProcessingError("python-docx is not installed. Run `pip install python-docx`.")

        try:
            doc_file = io.BytesIO(file_bytes)
            doc = docx.Document(doc_file)
        except Exception as e:
            raise DocumentProcessingError(f"Could not open DOCX file '{filename}'. It may be corrupted: {str(e)}")

        extracted_sections: List[str] = []

        # Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                extracted_sections.append(text)

        # Extract table data
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    table_rows.append(" | ".join(row_cells))
            if table_rows:
                extracted_sections.append("\n".join(table_rows))

        full_text = "\n\n".join(extracted_sections)
        cleaned = clean_text(full_text)

        # Approximate logical pages (e.g. ~400 words per page) for DOCX
        words = cleaned.split()
        if not words:
            return [ExtractedPage(page_number=1, text="", char_count=0, word_count=0)], False, None

        words_per_page = 400
        pages: List[ExtractedPage] = []

        if len(words) <= words_per_page:
            pages.append(ExtractedPage(
                page_number=1,
                text=cleaned,
                char_count=len(cleaned),
                word_count=len(words)
            ))
        else:
            current_page = 1
            for i in range(0, len(words), words_per_page):
                page_words = words[i : i + words_per_page]
                page_text = " ".join(page_words)
                pages.append(ExtractedPage(
                    page_number=current_page,
                    text=page_text,
                    char_count=len(page_text),
                    word_count=len(page_words)
                ))
                current_page += 1

        return pages, False, None

    def _extract_txt(
        self,
        file_bytes: bytes,
        filename: str
    ) -> Tuple[List[ExtractedPage], bool, Optional[str]]:
        """
        Extracts text from a plain TXT file with multiple encoding fallbacks
        (UTF-8, UTF-8-sig, Latin-1, CP1252).
        """
        decoded_text = ""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for enc in encodings:
            try:
                decoded_text = file_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if not decoded_text and file_bytes:
            # Fallback with replacement
            decoded_text = file_bytes.decode("utf-8", errors="replace")

        cleaned = clean_text(decoded_text)
        words = cleaned.split()

        # Approximate page count for TXT (~400 words per page)
        words_per_page = 400
        pages: List[ExtractedPage] = []

        if len(words) <= words_per_page:
            pages.append(ExtractedPage(
                page_number=1,
                text=cleaned,
                char_count=len(cleaned),
                word_count=len(words)
            ))
        else:
            current_page = 1
            for i in range(0, len(words), words_per_page):
                page_words = words[i : i + words_per_page]
                page_text = " ".join(page_words)
                pages.append(ExtractedPage(
                    page_number=current_page,
                    text=page_text,
                    char_count=len(page_text),
                    word_count=len(page_words)
                ))
                current_page += 1

        return pages, False, None

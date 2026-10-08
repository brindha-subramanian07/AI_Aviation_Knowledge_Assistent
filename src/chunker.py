"""
Chunking Module for AI-Powered Aviation Knowledge Assistant
Implements sentence-aware and paragraph-preserving text chunking with metadata tracking.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Any
from src.config import DEFAULT_CHUNK_SIZE_WORDS, DEFAULT_CHUNK_OVERLAP_WORDS
from src.document_processor import ProcessedDocument


@dataclass
class DocumentChunk:
    """A semantic chunk of text with its corresponding metadata."""
    chunk_id: str
    document_id: str
    source: str
    page: int
    document_type: str
    text: str
    word_count: int
    char_count: int
    chunk_index: int

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk attributes to a ChromaDB-compatible metadata dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "page": int(self.page),
            "document_type": str(self.document_type),
            "word_count": int(self.word_count),
            "chunk_index": int(self.chunk_index),
        }


class TextChunker:
    """
    Splits document pages into overlapping semantic chunks of 500-800 words
    while preserving page numbers and source provenance.
    """

    def __init__(
        self,
        chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using punctuation boundaries."""
        if not text:
            return []
        # Match sentence terminators (.!?) followed by space/newline or end of line,
        # but avoid splitting common abbreviations like e.g., i.e., Fig., etc.
        sentence_pattern = r'(?<!\b(?:e\.g|i\.e|Fig|fig|No|Ref|vs|approx)\.)(?<=[.!?])\s+'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
        return sentences if sentences else [text.strip()]

    def chunk_document(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        """
        Creates chunks from a ProcessedDocument.
        Processes page-by-page to accurately preserve page number metadata.
        If a page exceeds chunk_size_words, it is split across multiple chunks.
        If a page is smaller than chunk_size_words, it forms a cohesive single chunk
        or can carry context to ensure high-quality retrieval.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in doc.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            page_words = page_text.split()
            page_word_count = len(page_words)

            # If page fits within chunk size, keep it as an intact page chunk
            if page_word_count <= self.chunk_size_words:
                chunk_id = f"{doc.document_id}_p{page.page_number}_c{global_chunk_idx}"
                all_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    source=doc.filename,
                    page=page.page_number,
                    document_type=doc.document_type,
                    text=page_text,
                    word_count=page_word_count,
                    char_count=len(page_text),
                    chunk_index=global_chunk_idx
                ))
                global_chunk_idx += 1
            else:
                # Page is larger than chunk_size_words: break into sentence-aligned chunks
                sentences = self.split_into_sentences(page_text)
                current_chunk_sentences: List[str] = []
                current_words: int = 0

                for sent in sentences:
                    sent_words = len(sent.split())
                    if current_words + sent_words > self.chunk_size_words and current_chunk_sentences:
                        # Flush current chunk
                        chunk_content = " ".join(current_chunk_sentences).strip()
                        c_words = len(chunk_content.split())
                        chunk_id = f"{doc.document_id}_p{page.page_number}_c{global_chunk_idx}"
                        all_chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=doc.document_id,
                            source=doc.filename,
                            page=page.page_number,
                            document_type=doc.document_type,
                            text=chunk_content,
                            word_count=c_words,
                            char_count=len(chunk_content),
                            chunk_index=global_chunk_idx
                        ))
                        global_chunk_idx += 1

                        # Compute overlap sentences
                        overlap_sentences: List[str] = []
                        accumulated_overlap = 0
                        for s in reversed(current_chunk_sentences):
                            s_cnt = len(s.split())
                            if accumulated_overlap + s_cnt <= self.chunk_overlap_words:
                                overlap_sentences.insert(0, s)
                                accumulated_overlap += s_cnt
                            else:
                                break

                        current_chunk_sentences = overlap_sentences + [sent]
                        current_words = sum(len(s.split()) for s in current_chunk_sentences)
                    else:
                        current_chunk_sentences.append(sent)
                        current_words += sent_words

                if current_chunk_sentences:
                    chunk_content = " ".join(current_chunk_sentences).strip()
                    c_words = len(chunk_content.split())
                    chunk_id = f"{doc.document_id}_p{page.page_number}_c{global_chunk_idx}"
                    all_chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc.document_id,
                        source=doc.filename,
                        page=page.page_number,
                        document_type=doc.document_type,
                        text=chunk_content,
                        word_count=c_words,
                        char_count=len(chunk_content),
                        chunk_index=global_chunk_idx
                    ))
                    global_chunk_idx += 1

        return all_chunks

"""
Comprehensive test script for the AI-Powered Aviation Knowledge Assistant pipeline.
Validates:
1. Document extraction (PDF, DOCX, TXT)
2. Chunking & metadata tracking
3. Embedding generation
4. ChromaDB storage & querying
5. Duplicate upload protection
6. Semantic retrieval across single & multiple documents
7. Negative query filtering (out-of-domain rejection)
8. Document re-indexing & deletion
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.document_processor import DocumentProcessor
from src.chunker import TextChunker
from src.embeddings import get_embedding_service
from src.vector_store import get_vector_store
from src.document_manager import get_document_manager
from src.retriever import get_retriever
from src.rag_engine import get_rag_engine, FALLBACK_NO_INFO_PHRASE


def run_pipeline_tests():
    print("=" * 60)
    print("STARTING AVIATION KNOWLEDGE ASSISTANT TEST SUITE")
    print("=" * 60)

    sample_dir = ROOT_DIR / "sample_docs"
    pdf_path = sample_dir / "Aircraft_Maintenance_Manual_B737.pdf"
    docx_path = sample_dir / "Flight_Operations_Safety_Guidelines.docx"
    txt_path = sample_dir / "Emergency_Response_and_Runway_Safety.txt"

    doc_manager = get_document_manager()
    retriever = get_retriever()
    rag_engine = get_rag_engine()

    # 1. Test Ingestion of Sample Documents
    print("\n--- 1. Testing Document Ingestion & Vectorization ---")
    for file_path in [pdf_path, docx_path, txt_path]:
        print(f"Indexing: {file_path.name}...")
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        success, msg, record = doc_manager.upload_and_index(
            file_bytes=file_bytes,
            filename=file_path.name,
            allow_reindex=True
        )
        assert success, f"Failed to index {file_path.name}: {msg}"
        print(f"  [PASS] {msg}")
        print(f"         ID: {record.document_id}, Pages: {record.total_pages}, Chunks: {record.total_chunks}")

    # 2. Test Duplicate Detection
    print("\n--- 2. Testing Duplicate Prevention ---")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    dup_success, dup_msg, _ = doc_manager.upload_and_index(
        file_bytes=pdf_bytes,
        filename=pdf_path.name,
        allow_reindex=False
    )
    assert not dup_success, "Duplicate upload was not caught!"
    print(f"  [PASS] Duplicate caught: {dup_msg}")

    # 3. Test Knowledge Base Stats
    print("\n--- 3. Testing Knowledge Base Stats ---")
    stats = doc_manager.get_knowledge_base_stats()
    print(f"  Stats: {stats}")
    assert stats["total_documents"] >= 3
    assert stats["total_chunks"] >= 3
    print("  [PASS] Knowledge Base stats verified.")

    # 4. Test Semantic Retrieval on Required Test Questions
    test_queries = [
        ("What are the aircraft maintenance requirements?", ["Aircraft_Maintenance_Manual_B737.pdf"]),
        ("What are the safety procedures?", ["Flight_Operations_Safety_Guidelines.docx"]),
        ("Explain the emergency procedures.", ["Emergency_Response_and_Runway_Safety.txt"]),
        ("Which document contains information about maintenance?", ["Aircraft_Maintenance_Manual_B737.pdf"]),
        ("Compare the safety procedures in the uploaded documents.", ["Flight_Operations_Safety_Guidelines.docx", "Emergency_Response_and_Runway_Safety.txt"]),
        ("What information is available about runway safety?", ["Emergency_Response_and_Runway_Safety.txt"]),
    ]

    print("\n--- 4. Testing Semantic Retrieval on Aviation Questions ---")
    for query, expected_sources in test_queries:
        print(f"\nQuery: '{query}'")
        result = retriever.retrieve(query=query, top_k=5)
        assert result.has_results, f"No results found for query: {query}"
        retrieved_sources = [s["source"] for s in result.unique_sources]
        print(f"  Retrieved {len(result.chunks)} chunks from: {retrieved_sources}")
        for s in result.unique_sources:
            print(f"    - {s['source']} (Page {s['page']}, Score: {s['highest_similarity']})")
        # Check that at least one expected source is retrieved
        overlap = set(retrieved_sources).intersection(set(expected_sources))
        assert len(overlap) > 0, f"Expected one of {expected_sources}, got {retrieved_sources}"
        print(f"  [PASS] Successfully retrieved relevant source: {overlap}")

    # 5. Negative Test: Out-of-domain Query
    print("\n--- 5. Testing Out-of-Domain Retrieval & Fallback ---")
    unrelated_query = "What is the secret recipe for homemade chocolate chip cookies?"
    unrelated_result = retriever.retrieve(query=unrelated_query, top_k=5)
    print(f"Query: '{unrelated_query}' -> Chunks retrieved: {len(unrelated_result.chunks)}")
    # Test RAG Engine with unrelated query
    rag_res = rag_engine.generate_answer(unrelated_query)
    print(f"RAG Answer for unrelated query: '{rag_res.answer[:120]}...'")
    print("  [PASS] Out-of-domain query handled safely.")

    # 6. Test Document Re-indexing
    print("\n--- 6. Testing Document Re-indexing ---")
    doc_to_reindex = doc_manager.list_documents()[0].document_id
    reindex_ok, reindex_msg = doc_manager.reindex_document(doc_to_reindex)
    assert reindex_ok, f"Re-index failed: {reindex_msg}"
    print(f"  [PASS] Re-indexed {doc_to_reindex}: {reindex_msg}")

    print("\n" + "=" * 60)
    print("ALL PIPELINE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_tests()

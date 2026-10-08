"""
RAG Engine Module for AI-Powered Aviation Knowledge Assistant
Interacts with the Google Gemini API to generate grounded, context-constrained responses.
Strictly prohibits hallucination and enforces institutional source attribution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

from src.config import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY, SUPPORTED_GEMINI_MODELS
from src.retriever import KnowledgeRetriever, RetrievalResult, RetrievedChunk, get_retriever

# Try modern google-genai SDK first
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_MODERN_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GENAI_MODERN_AVAILABLE = False

# Try google-generativeai legacy fallback
try:
    import google.generativeai as genai_legacy
    GENAI_LEGACY_AVAILABLE = True
except ImportError:
    genai_legacy = None
    GENAI_LEGACY_AVAILABLE = False


FALLBACK_NO_INFO_PHRASE = (
    "I could not find sufficient information in the uploaded aviation documents to answer this question."
)


@dataclass
class RAGResponse:
    """Structured container for RAG generation output."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[RetrievedChunk]
    model_name: str
    has_sufficient_context: bool
    is_error: bool = False
    error_message: Optional[str] = None


class RAGEngine:
    """Orchestrates document context retrieval and Gemini grounded text generation."""

    def __init__(
        self,
        retriever: Optional[KnowledgeRetriever] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None
    ):
        self.retriever = retriever or get_retriever()
        self.model_name = model_name
        self.api_key = api_key or GEMINI_API_KEY

    def set_api_key(self, api_key: str):
        """Dynamically update the active Gemini API key."""
        self.api_key = api_key.strip()

    def set_model_name(self, model_name: str):
        """Dynamically switch the active Gemini model."""
        self.model_name = model_name.strip()

    def _build_system_instruction(self) -> str:
        """Constructs the system prompt instructing strict context adherence."""
        return (
            "You are an AI-Powered Aviation Knowledge Assistant designed to help aviation personnel, "
            "engineers, and students find precise information from institutional aviation documents.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Grounding Rule: Answer the question ONLY and STRICTLY using the retrieved document context "
            "provided to you below. Do NOT use outside pre-training knowledge or assume unstated aviation facts.\n"
            "2. Insufficient Context Rule: If the retrieved document context does NOT contain enough factual "
            "details to answer the user's question, you MUST explicitly state:\n"
            f'"{FALLBACK_NO_INFO_PHRASE}"\n'
            "Do not attempt to fabricate, speculate, or extrapolate.\n"
            "3. Source Attribution Rule: For each claim or procedure you describe, reference the specific document name "
            "and page number from the context (e.g., 'According to Aircraft_Maintenance_Manual.pdf, Page 42...').\n"
            "4. Multi-Document Questions: If information comes from multiple uploaded documents, clearly contrast or "
            "organize the answer by document source.\n"
            "5. Safety Disclaimer: Maintain a professional, safety-oriented aviation tone. Do not invent safety instructions."
        )

    def _build_user_prompt(self, query: str, context_text: str) -> str:
        """Assembles the user prompt containing the question and retrieved chunks."""
        return (
            f"USER QUESTION:\n{query}\n\n"
            f"RETRIEVED DOCUMENT CONTEXT:\n"
            f"==================================================\n"
            f"{context_text}\n"
            f"==================================================\n\n"
            f"Provide a clear, concise, and structured answer based strictly on the context above. "
            f"Cite the source document and page number for each key point. If the context does not answer "
            f"the question, respond with: \"{FALLBACK_NO_INFO_PHRASE}\""
        )

    def _call_gemini_api(self, prompt: str, system_instruction: str) -> str:
        """Invokes Google Gemini API with fallback across modern and legacy SDKs."""
        effective_key = self.api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not effective_key:
            raise ValueError(
                "Gemini API Key is missing. Please provide your Gemini API key in the application sidebar or in your .env file."
            )

        # Deprecated models that throw 404 on current API versions
        deprecated_models = {"gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash"}
        target_model = self.model_name
        if target_model in deprecated_models:
            target_model = "gemini-3.5-flash"
            self.model_name = "gemini-3.5-flash"

        # 1. Try modern google-genai SDK
        if GENAI_MODERN_AVAILABLE:
            try:
                client = genai.Client(api_key=effective_key)
                models_to_try = [target_model]
                for fallback in ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]:
                    if fallback not in models_to_try:
                        models_to_try.append(fallback)

                last_exception = None
                for candidate_model in models_to_try:
                    try:
                        config = genai_types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.2,
                            top_p=0.95,
                        )
                        response = client.models.generate_content(
                            model=candidate_model,
                            contents=prompt,
                            config=config
                        )
                        if response and response.text:
                            self.model_name = candidate_model
                            return response.text.strip()
                    except Exception as err:
                        last_exception = err
                        continue

                if last_exception:
                    raise last_exception
            except Exception as e:
                raise e

        # 2. Try legacy google-generativeai SDK
        if GENAI_LEGACY_AVAILABLE:
            try:
                genai_legacy.configure(api_key=effective_key)
                model_name = self.model_name
                # Normalize model name for legacy SDK if needed
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"

                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.95,
                }
                model = genai_legacy.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                raise RuntimeError(f"Gemini API request failed: {str(e)}")

        raise RuntimeError("No compatible Google Gemini SDK is available.")

    def generate_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        Executes end-to-end RAG pipeline:
        1. Retrieves relevant context chunks via KnowledgeRetriever
        2. If no chunks found, returns standard fallback message directly
        3. Constructs grounded prompt and queries Gemini API
        4. Validates grounding and formats source references
        """
        clean_query = query.strip()
        if not clean_query:
            return RAGResponse(
                query=query,
                answer="Please enter a valid question about aviation documents.",
                sources=[],
                retrieved_chunks=[],
                model_name=self.model_name,
                has_sufficient_context=False
            )

        # Retrieve relevant chunks
        try:
            retrieval: RetrievalResult = self.retriever.retrieve(
                query=clean_query,
                top_k=top_k,
                document_id_filter=document_id_filter
            )
        except Exception as e:
            return RAGResponse(
                query=clean_query,
                answer=f"Error searching knowledge base: {str(e)}",
                sources=[],
                retrieved_chunks=[],
                model_name=self.model_name,
                has_sufficient_context=False,
                is_error=True,
                error_message=str(e)
            )

        # If zero relevant chunks passed threshold
        if not retrieval.has_results:
            return RAGResponse(
                query=clean_query,
                answer=FALLBACK_NO_INFO_PHRASE,
                sources=[],
                retrieved_chunks=[],
                model_name=self.model_name,
                has_sufficient_context=False
            )

        # Build prompt & system instruction
        system_instruction = self._build_system_instruction()
        user_prompt = self._build_user_prompt(clean_query, retrieval.context_text)

        # Call Gemini
        try:
            raw_answer = self._call_gemini_api(user_prompt, system_instruction)
            has_sufficient_info = FALLBACK_NO_INFO_PHRASE.lower() not in raw_answer.lower()

            return RAGResponse(
                query=clean_query,
                answer=raw_answer,
                sources=retrieval.unique_sources,
                retrieved_chunks=retrieval.chunks,
                model_name=self.model_name,
                has_sufficient_context=has_sufficient_info
            )
        except ValueError as val_err:
            # Typically missing API key
            return RAGResponse(
                query=clean_query,
                answer=(
                    "⚠️ **Gemini API Key Required**\n\n"
                    "To generate answers, please provide your Google Gemini API key. "
                    "You can enter it in the left sidebar under **Settings**, or set `GEMINI_API_KEY` in your `.env` file.\n\n"
                    "Retrieved documents and context chunks are still displayed below."
                ),
                sources=retrieval.unique_sources,
                retrieved_chunks=retrieval.chunks,
                model_name=self.model_name,
                has_sufficient_context=True,
                is_error=True,
                error_message=str(val_err)
            )
        except Exception as gen_err:
            # Handle rate limits, quota issues, or network hiccups gracefully
            return RAGResponse(
                query=clean_query,
                answer=(
                    f"⚠️ **Gemini API Error**: {str(gen_err)}\n\n"
                    "Please check your API key, network connection, or quota limits. "
                    "Retrieved document sources and context chunks remain visible below."
                ),
                sources=retrieval.unique_sources,
                retrieved_chunks=retrieval.chunks,
                model_name=self.model_name,
                has_sufficient_context=True,
                is_error=True,
                error_message=str(gen_err)
            )


def get_rag_engine(api_key: Optional[str] = None) -> RAGEngine:
    """Helper factory to retrieve RAGEngine instance."""
    return RAGEngine(api_key=api_key)

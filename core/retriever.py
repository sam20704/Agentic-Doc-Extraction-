"""
Semantic retrieval over document content using ChromaDB.

Indexes OCR lines, table content, and figure captions as
overlapping chunks with metadata, so the agent receives only
the most relevant context per query.
"""
from typing import List, Dict, Optional
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    EMBEDDING_PROVIDER,
    RETRIEVAL_CHUNK_SIZE,
    RETRIEVAL_CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
)


class DocumentRetriever:
    """
    Builds a Chroma vector store from document content and
    returns the top-k most relevant chunks per user query.
    """

    def __init__(
        self,
        ordered_text: List[dict],
        layout_regions: List,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.ordered_text = ordered_text
        self.layout_regions = layout_regions
        self.chunk_size = chunk_size or RETRIEVAL_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or RETRIEVAL_CHUNK_OVERLAP
        self.chunks: List[dict] = []

        self._init_embeddings()
        self._build_chunks()
        self._index_chunks()

    # ────────────────────────────────────
    # Embedding initialisation
    # ────────────────────────────────────

    def _init_embeddings(self):
        """Create the embedding function."""
        if EMBEDDING_PROVIDER == "azure":
            from langchain_openai import AzureOpenAIEmbeddings

            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            )
            print(f"✓ Embeddings: Azure OpenAI ({AZURE_OPENAI_EMBEDDING_DEPLOYMENT})")

        elif EMBEDDING_PROVIDER == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✓ Embeddings: HuggingFace (all-MiniLM-L6-v2)")
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

    # ────────────────────────────────────
    # Chunk creation
    # ────────────────────────────────────

    def _build_chunks(self):
        """Create overlapping text chunks + region chunks."""
        self.chunks = []

        # ── 1. OCR text chunks (sliding window) ──
        step = max(1, self.chunk_size - self.chunk_overlap)

        for start in range(0, len(self.ordered_text), step):
            window = self.ordered_text[start : start + self.chunk_size]
            if not window:
                continue

            lines = []
            pages = set()
            positions = []

            for item in window:
                page = item.get("page", "?")
                pos = item.get("position", "?")
                text = item.get("text", "")
                lines.append(f"[P{page}:L{pos}] {text}")
                pages.add(str(page))
                positions.append(pos)

            self.chunks.append(
                {
                    "text": "\n".join(lines),
                    "type": "ocr_chunk",
                    "pages": ",".join(sorted(pages)),
                    "start_pos": str(positions[0]),
                    "end_pos": str(positions[-1]),
                }
            )

        # ── 2. Table content chunks ──
        for region in self.layout_regions:
            if region.region_type != "table":
                continue

            table_text = self._table_to_text(region)
            if table_text:
                self.chunks.append(
                    {
                        "text": table_text,
                        "type": "table",
                        "region_id": str(region.region_id),
                        "pages": str(region.page_number),
                    }
                )

        # ── 3. Figure caption chunks ──
        for region in self.layout_regions:
            if region.region_type != "figure":
                continue

            caption = ""
            if region.content and isinstance(region.content, dict):
                caption = region.content.get("caption", "") or ""

            fig_text = (
                f"[Figure Region {region.region_id}, Page {region.page_number}] "
                f"{caption}"
            ).strip()

            if fig_text:
                self.chunks.append(
                    {
                        "text": fig_text,
                        "type": "figure",
                        "region_id": str(region.region_id),
                        "pages": str(region.page_number),
                    }
                )

        # ── 4. Text block chunks ──
        for region in self.layout_regions:
            if region.region_type != "text":
                continue

            content_text = ""
            if region.content and isinstance(region.content, dict):
                content_text = region.content.get("text", "")

            if content_text and len(content_text) > 30:
                self.chunks.append(
                    {
                        "text": (
                            f"[Text Block Region {region.region_id}, "
                            f"Page {region.page_number}] {content_text}"
                        ),
                        "type": "text_block",
                        "region_id": str(region.region_id),
                        "pages": str(region.page_number),
                    }
                )

        print(f"✓ Created {len(self.chunks)} retrieval chunks")

    @staticmethod
    def _table_to_text(region) -> str:
        """Convert table region to searchable text."""
        content = region.content
        if not content or "cells" not in content:
            return ""

        row_count = content.get("row_count", 0)
        col_count = content.get("column_count", 0)
        if row_count == 0 or col_count == 0:
            return ""

        grid = [["" for _ in range(col_count)] for _ in range(row_count)]
        for cell in content["cells"]:
            r = cell.get("row_index", 0)
            c = cell.get("column_index", 0)
            if r < row_count and c < col_count:
                grid[r][c] = cell.get("content", "").strip()

        lines = [
            f"[Table Region {region.region_id}, Page {region.page_number}]"
        ]
        for row in grid:
            lines.append(" | ".join(row))

        return "\n".join(lines)

    # ────────────────────────────────────
    # Chroma indexing
    # ────────────────────────────────────

    def _index_chunks(self):
        """Embed all chunks and store in Chroma."""
        from langchain_community.vectorstores import Chroma

        texts = [c["text"] for c in self.chunks]
        metadatas = [
            {k: v for k, v in c.items() if k != "text"} for c in self.chunks
        ]

        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name="document_chunks",
        )
        print(f"✓ Indexed {len(texts)} chunks into Chroma")

    # ────────────────────────────────────
    # Search
    # ────────────────────────────────────

    def search(self, query: str, k: int = None) -> List[dict]:
        """
        Retrieve the top-k most relevant chunks.

        Args:
            query: User question
            k: Number of results (default from config)

        Returns:
            List of dicts with 'text', 'type', 'score', and metadata
        """
        k = k or RETRIEVAL_TOP_K

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )

        retrieved = []
        for doc, score in results:
            entry = {
                "text": doc.page_content,
                "score": round(score, 4),
                **doc.metadata,
            }
            retrieved.append(entry)

        return retrieved

    def format_retrieved_context(self, query: str, k: int = None) -> str:
        """
        Retrieve and format context as a string ready
        for injection into the user message.

        Args:
            query: User question
            k: Number of results

        Returns:
            Formatted context string with source annotations
        """
        results = self.search(query, k)

        if not results:
            return "[No relevant document excerpts found]"

        lines = []
        for i, r in enumerate(results, 1):
            source_type = r.get("type", "unknown")
            pages = r.get("pages", "?")
            score = r.get("score", 0)
            region_id = r.get("region_id", "")

            header = f"── Excerpt {i} [{source_type}]"
            if region_id:
                header += f" Region {region_id}"
            header += f" (Page {pages}, relevance: {score:.2f}) ──"

            lines.append(header)
            lines.append(r["text"])
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return retriever statistics."""
        from collections import Counter

        type_counts = Counter(c.get("type", "unknown") for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks_by_type": dict(type_counts),
        }

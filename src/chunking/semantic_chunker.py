import json
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import numpy as np
from tqdm import tqdm

from .token_utils import estimate_token_count
from .buffer_merger import merge_with_buffer


class SemanticChunker:
    """
    Implements a SemRAG-style semantic chunker:
    - sentence-level embeddings
    - cosine-based grouping with buffer
    - token limit + overlapping sub-chunks
    """

    def __init__(
        self,
        model_name: str,
        buffer_size: int,
        cosine_threshold: float,
        max_tokens_chunk: int,
        subchunk_tokens: int,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.buffer_size = buffer_size
        self.cosine_threshold = cosine_threshold
        self.max_tokens_chunk = max_tokens_chunk
        self.subchunk_tokens = subchunk_tokens

    @staticmethod
    def load_pdf(path: str) -> str:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        # Simple sentence split; you can switch to nltk or spaCy if you prefer
        import re

        candidates = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in candidates if s.strip()]
        return sentences

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences, show_progress_bar=False)

    def _group_semantic_chunks(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[List[str]]:
        """
        Groups sentences into semantically coherent chunks using cosine similarity
        between adjacent sentences.
        """
        if not sentences:
            return []

        chunks: List[List[str]] = []
        current_chunk: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            prev_emb = embeddings[i - 1].reshape(1, -1)
            curr_emb = embeddings[i].reshape(1, -1)
            sim = cosine_similarity(prev_emb, curr_emb)[0][0]
            dist = 1.0 - sim

            if dist <= self.cosine_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(current_chunk)
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _apply_token_limits(
        self, chunks: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Enforces a maximum token limit per chunk and creates overlapping
        sub-chunks of ~subchunk_tokens when necessary.
        """
        final_chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        for chunk in chunks:
            text = " ".join(chunk)
            tokens = estimate_token_count(text)

            if tokens <= self.max_tokens_chunk:
                final_chunks.append(
                    {
                        "id": f"chunk_{chunk_id}",
                        "text": text,
                        "tokens": tokens,
                    }
                )
                chunk_id += 1
            else:
                words = text.split()
                step = self.subchunk_tokens // 2
                start = 0
                while start < len(words):
                    sub_words = words[start : start + self.subchunk_tokens]
                    sub_text = " ".join(sub_words)
                    sub_tokens = estimate_token_count(sub_text)
                    final_chunks.append(
                        {
                            "id": f"chunk_{chunk_id}",
                            "text": sub_text,
                            "tokens": sub_tokens,
                        }
                    )
                    chunk_id += 1
                    start += step

        return final_chunks

    def chunk_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        raw_text = self.load_pdf(pdf_path)
        sentences = self.split_into_sentences(raw_text)
        sentences = merge_with_buffer(sentences, self.buffer_size)
        embeddings = self.embed_sentences(sentences)
        semantic_groups = self._group_semantic_chunks(sentences, embeddings)
        final_chunks = self._apply_token_limits(semantic_groups)
        return final_chunks

    @staticmethod
    def save_chunks(chunks: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

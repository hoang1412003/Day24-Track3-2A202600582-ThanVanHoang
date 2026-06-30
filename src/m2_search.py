from __future__ import annotations

"""
Module 2: Hybrid Search — BM25 Vietnamese + Dense + RRF
=======================================================

Implement:
1. Vietnamese segmentation
2. BM25 sparse search
3. Dense vector search with Qdrant
4. Reciprocal Rank Fusion (RRF)
5. HybridSearch = BM25 + Dense + RRF

Test:
    pytest tests/test_m2.py
"""

import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BM25_TOP_K,
    DENSE_TOP_K,
    HYBRID_TOP_K,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


# =========================================================
# Vietnamese segmentation
# =========================================================

def segment_vietnamese(text: str) -> str:
    """
    Segment Vietnamese text into words.

    Ưu tiên dùng underthesea.
    Nếu underthesea chưa cài, fallback sang regex tokenizer đơn giản.

    Lưu ý:
    underthesea có thể trả về token kiểu "nghỉ_phép".
    Với BM25, ta replace "_" thành " " để query "nghỉ phép" vẫn match.
    """
    text = text or ""

    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")

    except Exception:
        # Fallback nhẹ để test không crash nếu thiếu underthesea.
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return " ".join(tokens)


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    """
    segmented = segment_vietnamese(text)
    return segmented.lower().split()


def _get_chunk_text(chunk: dict) -> str:
    """
    Lấy text từ chunk dạng dict.
    """
    return str(chunk.get("text", "") or "")


def _get_chunk_metadata(chunk: dict) -> dict:
    """
    Lấy metadata từ chunk dạng dict.
    """
    metadata = chunk.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


# =========================================================
# BM25 Search
# =========================================================

class BM25Search:
    """
    Sparse search using BM25.

    Phù hợp với:
    - keyword exact match
    - tên riêng
    - số điều luật
    - thuật ngữ xuất hiện trực tiếp trong tài liệu
    """

    def __init__(self) -> None:
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from chunks.
        """
        self.documents = chunks or []
        self.corpus_tokens = [
            _tokenize_for_bm25(_get_chunk_text(chunk))
            for chunk in self.documents
        ]

        if not self.corpus_tokens:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)

        except Exception as exc:
            raise RuntimeError(
                "Không thể khởi tạo BM25. Hãy cài rank_bm25 bằng: "
                "pip install rank-bm25"
            ) from exc

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """
        Search using BM25.
        """
        if self.bm25 is None or not self.documents:
            return []

        query_tokens = _tokenize_for_bm25(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []

        for i in top_indices:
            score = float(scores[i])

            # Bỏ tài liệu không liên quan.
            if score <= 0:
                continue

            chunk = self.documents[i]
            results.append(
                SearchResult(
                    text=_get_chunk_text(chunk),
                    score=score,
                    metadata=_get_chunk_metadata(chunk),
                    method="bm25",
                )
            )

        return results


# =========================================================
# Dense Search
# =========================================================

class DenseSearch:
    """
    Dense vector search.

    Bình thường:
    - Encode chunk bằng SentenceTransformer.
    - Lưu vector vào Qdrant.
    - Query bằng Qdrant.

    Fallback:
    - Nếu Qdrant chưa chạy, dùng in-memory vectors.
    - Điều này giúp test/dev chạy được mà không cần bật Docker Qdrant.
    """

    def __init__(self) -> None:
        self._encoder = None
        self.client = None
        self.documents: list[dict] = []
        self._memory_vectors: list[list[float]] = []
        self._use_memory = False

    def _get_encoder(self):
        """
        Lazy-load embedding model.

        Model chỉ được tải khi thật sự cần dense index/search.
        """
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)

        return self._encoder

    def _get_client(self):
        """
        Lazy-connect Qdrant.

        Nếu Qdrant không chạy thì trả None để dùng in-memory fallback.
        """
        if self.client is not None:
            return self.client

        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            return self.client

        except Exception:
            self.client = None
            return None

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Encode list of texts into vectors.
        """
        if not texts:
            return []

        vectors = self._get_encoder().encode(texts, show_progress_bar=True)

        encoded: list[list[float]] = []
        for vector in vectors:
            if hasattr(vector, "tolist"):
                encoded.append(vector.tolist())
            else:
                encoded.append(list(vector))

        return encoded

    def _encode_query(self, query: str) -> list[float]:
        """
        Encode query into vector.
        """
        vector = self._get_encoder().encode(query)

        if hasattr(vector, "tolist"):
            return vector.tolist()

        return list(vector)

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Cosine similarity không cần numpy.
        """
        if not vec_a or not vec_b:
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def index(
        self,
        chunks: list[dict],
        collection: str = COLLECTION_NAME,
    ) -> None:
        """
        Build dense vector index.

        Nếu Qdrant chạy được:
            lưu vectors vào Qdrant.

        Nếu Qdrant không chạy:
            lưu vectors trong RAM.
        """
        self.documents = chunks or []
        texts = [_get_chunk_text(chunk) for chunk in self.documents]

        if not texts:
            self._memory_vectors = []
            return

        vectors = self._encode_texts(texts)

        client = self._get_client()

        if client is None:
            self._use_memory = True
            self._memory_vectors = vectors
            print("  ⚠️  Qdrant chưa chạy, DenseSearch dùng in-memory fallback.")
            return

        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            # recreate_collection phù hợp lab/dev.
            # Production thật thì nên dùng create_collection nếu chưa tồn tại.
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )

            points = []

            for i, vector in enumerate(vectors):
                chunk = self.documents[i]
                text = texts[i]
                metadata = _get_chunk_metadata(chunk)

                payload = {
                    **metadata,
                    "text": text,
                    "doc_id": metadata.get("doc_id", i),
                    "chunk_index": metadata.get("chunk_index", i),
                }

                points.append(
                    PointStruct(
                        id=i,
                        vector=vector,
                        payload=payload,
                    )
                )

            if points:
                client.upsert(
                    collection_name=collection,
                    points=points,
                )

            self._use_memory = False
            self._memory_vectors = vectors

        except Exception as exc:
            # Nếu Qdrant lỗi giữa chừng, vẫn fallback RAM để không chết test.
            self._use_memory = True
            self._memory_vectors = vectors
            print(f"  ⚠️  Qdrant lỗi, DenseSearch dùng in-memory fallback: {exc}")

    def _search_memory(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Dense search bằng in-memory cosine similarity.
        """
        if not self.documents or not self._memory_vectors:
            return []

        query_vector = self._encode_query(query)

        scored: list[tuple[int, float]] = []

        for i, vector in enumerate(self._memory_vectors):
            score = self._cosine_similarity(query_vector, vector)
            scored.append((i, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        results: list[SearchResult] = []

        for i, score in scored[:top_k]:
            chunk = self.documents[i]

            results.append(
                SearchResult(
                    text=_get_chunk_text(chunk),
                    score=float(score),
                    metadata=_get_chunk_metadata(chunk),
                    method="dense",
                )
            )

        return results

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """
        Search using dense vectors.
        """
        if not query or not query.strip():
            return []

        if self._use_memory:
            return self._search_memory(query, top_k)

        client = self._get_client()

        if client is None:
            return self._search_memory(query, top_k)

        query_vector = self._encode_query(query)

        try:
            response = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
            )

            results: list[SearchResult] = []

            for pt in response.points:
                payload: dict[str, Any] = pt.payload or {}

                metadata = dict(payload)
                text = str(metadata.pop("text", "") or "")

                results.append(
                    SearchResult(
                        text=text,
                        score=float(pt.score),
                        metadata=metadata,
                        method="dense",
                    )
                )

            return results

        except Exception:
            return self._search_memory(query, top_k)


# =========================================================
# Reciprocal Rank Fusion
# =========================================================

def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """
    Merge ranked lists using Reciprocal Rank Fusion.

    Formula:
        score(d) = Σ 1 / (k + rank)

    Trong code:
        rank bắt đầu từ 1, nên dùng rank_index + 1.
    """
    fused: dict[str, dict] = {}

    for results in results_list:
        for rank_index, result in enumerate(results):
            # Dùng text làm key đơn giản.
            # Nếu production, nên dùng doc_id + chunk_index ổn định hơn.
            key = result.text

            if key not in fused:
                fused[key] = {
                    "score": 0.0,
                    "result": result,
                    "methods": set(),
                }

            fused[key]["score"] += 1.0 / (k + rank_index + 1)
            fused[key]["methods"].add(result.method)

    ranked_items = sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]

    hybrid_results: list[SearchResult] = []

    for item in ranked_items:
        base_result: SearchResult = item["result"]
        methods = sorted(item["methods"])

        metadata = {
            **base_result.metadata,
            "source_methods": methods,
        }

        hybrid_results.append(
            SearchResult(
                text=base_result.text,
                score=float(item["score"]),
                metadata=metadata,
                method="hybrid",
            )
        )

    return hybrid_results


# =========================================================
# Hybrid Search
# =========================================================

class HybridSearch:
    """
    Combines BM25 + Dense + RRF.

    Luồng:
        index(chunks)
            ├── BM25 index
            └── Dense index

        search(query)
            ├── bm25.search(query)
            ├── dense.search(query)
            └── reciprocal_rank_fusion(...)
    """

    def __init__(self) -> None:
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(
        self,
        query: str,
        top_k: int = HYBRID_TOP_K,
    ) -> list[SearchResult]:
        bm25_results = self.bm25.search(
            query,
            top_k=BM25_TOP_K,
        )

        dense_results = self.dense.search(
            query,
            top_k=DENSE_TOP_K,
        )

        return reciprocal_rank_fusion(
            [bm25_results, dense_results],
            top_k=top_k,
        )


# =========================================================
# Smoke test
# =========================================================

if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm"

    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")

    chunks = [
        {
            "text": "Nhân viên được nghỉ phép năm theo quy định của công ty.",
            "metadata": {"source": "policy.md", "chunk_index": 0},
        },
        {
            "text": "Dữ liệu cá nhân cần được bảo vệ theo quy định pháp luật.",
            "metadata": {"source": "privacy.md", "chunk_index": 1},
        },
    ]

    bm25 = BM25Search()
    bm25.index(chunks)

    print("\nBM25 results:")
    for result in bm25.search("nghỉ phép năm"):
        print(result)
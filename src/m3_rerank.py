from __future__ import annotations

"""
Module 3: Reranking
===================

Reranking = lấy top-k documents từ retriever rồi sắp xếp lại bằng model mạnh hơn.

Implement:
1. CrossEncoderReranker
2. FlashrankReranker
3. Fallback lexical reranker để test/dev không bị treo vì tải model
4. Latency benchmark

Test:
    pytest tests/test_m3.py
"""

import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


# =========================================================
# Helper functions
# =========================================================

def _get_doc_text(doc: dict) -> str:
    """
    Lấy text từ document dict.
    """
    return str(doc.get("text", "") or "")


def _get_doc_score(doc: dict) -> float:
    """
    Lấy score gốc từ retriever.
    """
    try:
        return float(doc.get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _get_doc_metadata(doc: dict) -> dict:
    """
    Lấy metadata từ document.
    """
    metadata = doc.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _tokenize(text: str) -> list[str]:
    """
    Tokenizer đơn giản cho fallback reranker.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _lexical_rerank_score(query: str, document_text: str) -> float:
    """
    Fallback rerank score không cần model.

    Ý tưởng:
    - Tính overlap giữa query và document.
    - Có normalize để document dài không tự động thắng.
    - Đủ nhẹ để chạy unit test nhanh.
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document_text)

    if not query_tokens or not doc_tokens:
        return 0.0

    query_set = set(query_tokens)
    doc_set = set(doc_tokens)

    overlap = query_set & doc_set

    # Precision-like + recall-like score.
    recall = len(overlap) / len(query_set)
    precision = len(overlap) / len(doc_set)

    if recall + precision == 0:
        return 0.0

    f1 = 2 * recall * precision / (recall + precision)

    # Bonus nhẹ nếu nguyên query xuất hiện trong doc.
    phrase_bonus = 0.2 if query.lower() in document_text.lower() else 0.0

    return float(f1 + phrase_bonus)


def _to_rerank_results(
    scored_documents: list[tuple[float, dict]],
    top_k: int,
) -> list[RerankResult]:
    """
    Convert scored documents sang list[RerankResult].
    Rank bắt đầu từ 1.
    """
    ranked = sorted(
        scored_documents,
        key=lambda item: item[0],
        reverse=True,
    )[:top_k]

    results: list[RerankResult] = []

    for rank_index, (rerank_score, doc) in enumerate(ranked, start=1):
        results.append(
            RerankResult(
                text=_get_doc_text(doc),
                original_score=_get_doc_score(doc),
                rerank_score=float(rerank_score),
                metadata=_get_doc_metadata(doc),
                rank=rank_index,
            )
        )

    return results


# =========================================================
# Cross Encoder Reranker
# =========================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranker.

    Cross-encoder nhận trực tiếp pair:
        (query, document)

    rồi trả về relevance score.

    Mặc định KHÔNG tự load model để unit test không bị đứng/tải model nặng.
    Muốn dùng model thật, set:

        PowerShell:
            $env:LOAD_RERANKER_MODEL="1"

        Bash:
            export LOAD_RERANKER_MODEL=1
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """
        Lazy-load cross encoder model.

        Chỉ load nếu LOAD_RERANKER_MODEL=1.
        """
        should_load = os.getenv("LOAD_RERANKER_MODEL", "0") == "1"

        if not should_load:
            raise RuntimeError(
                "Reranker model is disabled. "
                "Set LOAD_RERANKER_MODEL=1 to enable it."
            )

        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

        return self._model

    def _model_scores(
        self,
        query: str,
        documents: list[dict],
    ) -> list[float]:
        """
        Score documents bằng cross encoder model thật.
        """
        model = self._load_model()

        pairs = [
            (query, _get_doc_text(doc))
            for doc in documents
        ]

        scores = model.predict(pairs)

        if isinstance(scores, (int, float)):
            return [float(scores)]

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        return [float(score) for score in scores]

    def _fallback_scores(
        self,
        query: str,
        documents: list[dict],
    ) -> list[float]:
        """
        Score documents bằng lexical fallback.
        """
        return [
            _lexical_rerank_score(query, _get_doc_text(doc))
            for doc in documents
        ]

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """
        Rerank documents.

        Input documents thường có dạng:
            {
                "text": "...",
                "score": 0.82,
                "metadata": {...}
            }
        """
        if not query or not query.strip():
            return []

        if not documents:
            return []

        try:
            scores = self._model_scores(query, documents)
        except Exception:
            scores = self._fallback_scores(query, documents)

        scored_documents = list(zip(scores, documents))

        return _to_rerank_results(
            scored_documents=scored_documents,
            top_k=top_k,
        )


# =========================================================
# Flashrank Reranker
# =========================================================

class FlashrankReranker:
    """
    Flashrank reranker.

    Ưu tiên dùng thư viện flashrank nếu có.
    Nếu chưa cài hoặc lỗi, fallback sang lexical reranker.

    Cài nếu cần:
        pip install flashrank
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        self.model_name = model_name
        self._ranker = None

    def _load_model(self):
        """
        Lazy-load Flashrank model.

        Chỉ load nếu LOAD_FLASHRANK_MODEL=1 để tránh unit test tải model.
        """
        should_load = os.getenv("LOAD_FLASHRANK_MODEL", "0") == "1"

        if not should_load:
            raise RuntimeError(
                "Flashrank model is disabled. "
                "Set LOAD_FLASHRANK_MODEL=1 to enable it."
            )

        if self._ranker is None:
            from flashrank import Ranker

            self._ranker = Ranker(model_name=self.model_name)

        return self._ranker

    def _model_scores(
        self,
        query: str,
        documents: list[dict],
    ) -> list[float]:
        """
        Score bằng Flashrank thật.
        """
        from flashrank import RerankRequest

        ranker = self._load_model()

        passages = [
            {
                "id": str(i),
                "text": _get_doc_text(doc),
                "metadata": _get_doc_metadata(doc),
            }
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(
            query=query,
            passages=passages,
        )

        ranked_passages = ranker.rerank(request)

        # Flashrank trả về list đã sort.
        # Để thống nhất với _to_rerank_results, ta map score về đúng doc index.
        scores = [0.0 for _ in documents]

        for passage in ranked_passages:
            idx = int(passage["id"])
            scores[idx] = float(passage.get("score", 0.0))

        return scores

    def _fallback_scores(
        self,
        query: str,
        documents: list[dict],
    ) -> list[float]:
        """
        Fallback lexical scoring.
        """
        return [
            _lexical_rerank_score(query, _get_doc_text(doc))
            for doc in documents
        ]

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """
        Rerank documents bằng Flashrank hoặc fallback.
        """
        if not query or not query.strip():
            return []

        if not documents:
            return []

        try:
            scores = self._model_scores(query, documents)
        except Exception:
            scores = self._fallback_scores(query, documents)

        scored_documents = list(zip(scores, documents))

        return _to_rerank_results(
            scored_documents=scored_documents,
            top_k=top_k,
        )


# =========================================================
# Benchmark
# =========================================================

def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """
    Benchmark latency over n_runs.

    Return:
        {
            "avg_ms": ...,
            "min_ms": ...,
            "max_ms": ...,
            "n_runs": ...
        }
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be greater than 0")

    times: list[float] = []

    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "n_runs": n_runs,
    }


# =========================================================
# Smoke test
# =========================================================

if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"

    docs = [
        {
            "text": "Nhân viên được nghỉ 12 ngày/năm.",
            "score": 0.8,
            "metadata": {"source": "policy.md"},
        },
        {
            "text": "Mật khẩu thay đổi mỗi 90 ngày.",
            "score": 0.7,
            "metadata": {"source": "security.md"},
        },
        {
            "text": "Thời gian thử việc là 60 ngày.",
            "score": 0.75,
            "metadata": {"source": "hr.md"},
        },
    ]

    reranker = CrossEncoderReranker()

    print("CrossEncoderReranker results:")
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

    print("\nBenchmark:")
    print(benchmark_reranker(reranker, query, docs, n_runs=3))
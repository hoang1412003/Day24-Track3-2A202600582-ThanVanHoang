from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================

Implement:
1. Basic chunking - baseline
2. Semantic chunking - cắt theo độ tương đồng ngữ nghĩa giữa các câu
3. Hierarchical chunking - parent chunks + child chunks
4. Structure-aware chunking - cắt theo header markdown / cấu trúc văn bản pháp luật

Test:
    pytest tests/test_m1.py
"""

import glob
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR,
    HIERARCHICAL_PARENT_SIZE,
    HIERARCHICAL_CHILD_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


# =========================================================
# Document Loading
# =========================================================

def _extract_pdf_text(path: str) -> str:
    """
    Extract text layer từ PDF.
    Trả về "" nếu PDF là scan ảnh, không có text layer.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Load tất cả markdown và PDF có text layer từ data/.

    - .md: đọc trực tiếp.
    - .pdf: trích text bằng pypdf.
    - PDF scan ảnh không có text layer sẽ bị bỏ qua và in cảnh báo.
    """
    docs: list[dict] = []

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append(
                {
                    "text": f.read(),
                    "metadata": {"source": os.path.basename(fp)},
                }
            )

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append(
                {
                    "text": text,
                    "metadata": {"source": os.path.basename(fp)},
                }
            )
        else:
            print(
                f"  ⚠️  Bỏ qua {os.path.basename(fp)}: "
                f"PDF scan ảnh, không có text layer (cần OCR)."
            )

    return docs


# =========================================================
# Helper functions
# =========================================================

def _normalize_whitespace(text: str) -> str:
    """
    Làm sạch khoảng trắng nhưng không phá cấu trúc đoạn.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    """
    Tách paragraph theo dòng trống.
    """
    text = _normalize_whitespace(text)
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """
    Tách câu tương đối an toàn cho tiếng Việt.

    Giữ lại dấu câu cuối câu.
    Không hoàn hảo 100%, nhưng tốt hơn split('.') thô.
    """
    text = _normalize_whitespace(text)
    if not text:
        return []

    # Tách sau . ! ? ; : hoặc xuống dòng, nếu sau đó là khoảng trắng.
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    sentences = [p.strip() for p in parts if p.strip()]

    # Nếu văn bản không có dấu câu, fallback theo paragraph.
    if len(sentences) <= 1:
        sentences = _split_paragraphs(text)

    return sentences


def _chunks_from_units(
    units: Iterable[str],
    max_size: int,
    metadata: dict,
    strategy: str,
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    """
    Gom các đơn vị nhỏ như sentence/paragraph thành chunk <= max_size.
    """
    extra_metadata = extra_metadata or {}

    chunks: list[Chunk] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return

        chunk_text = "\n\n".join(current).strip()
        if not chunk_text:
            return

        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    **extra_metadata,
                    "strategy": strategy,
                    "chunk_index": len(chunks),
                },
            )
        )

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        current_text = "\n\n".join(current)
        would_exceed = current and len(current_text) + len(unit) + 2 > max_size

        if would_exceed:
            flush()
            current = [unit]
        else:
            current.append(unit)

    flush()
    return chunks


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


def _lexical_vector(sentence: str) -> dict[str, float]:
    """
    Fallback vector đơn giản nếu không dùng được sentence-transformers.
    Dùng bag-of-words để vẫn có semantic-ish chunking mà không crash.
    """
    words = re.findall(r"\w+", sentence.lower(), flags=re.UNICODE)
    vector: dict[str, float] = {}
    for w in words:
        vector[w] = vector.get(w, 0.0) + 1.0
    return vector


def _dict_cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Cosine similarity cho dict vector.
    """
    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# =========================================================
# Baseline: Basic Chunking
# =========================================================

def chunk_basic(
    text: str,
    chunk_size: int = 500,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph.

    Đây là baseline để so sánh với các strategy nâng cao.
    """
    metadata = metadata or {}
    paragraphs = _split_paragraphs(text)

    chunks: list[Chunk] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(
                Chunk(
                    text=current.strip(),
                    metadata={
                        **metadata,
                        "strategy": "basic",
                        "chunk_index": len(chunks),
                    },
                )
            )
            current = ""

        current += para + "\n\n"

    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(),
                metadata={
                    **metadata,
                    "strategy": "basic",
                    "chunk_index": len(chunks),
                },
            )
        )

    return chunks


# =========================================================
# Strategy 1: Semantic Chunking
# =========================================================

def _get_sentence_embeddings(sentences: list[str]) -> list[list[float]] | None:
    """
    Tạo embedding bằng sentence-transformers nếu có thể.

    Nếu không import/load được model thì trả về None để fallback lexical.
    Model mặc định có thể đổi bằng biến môi trường:
        HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    """
    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        model = SentenceTransformer(model_name)
        embeddings = model.encode(sentences, convert_to_numpy=False)

        return [list(map(float, emb)) for emb in embeddings]

    except Exception:
        return None


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Semantic chunking.

    Ý tưởng:
    - Tách văn bản thành câu.
    - Tính similarity giữa câu hiện tại và câu trước.
    - Nếu similarity thấp hơn threshold thì mở chunk mới.
    - Nếu chunk quá dài thì cũng cắt để tránh chunk khổng lồ.

    Nếu sentence-transformers không khả dụng, fallback sang lexical similarity
    để file vẫn chạy được trong môi trường test nhẹ.
    """
    metadata = metadata or {}
    text = _normalize_whitespace(text)

    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    max_chunk_size = int(os.getenv("SEMANTIC_MAX_CHUNK_SIZE", "900"))

    embeddings = _get_sentence_embeddings(sentences)

    chunks: list[Chunk] = []
    current: list[str] = [sentences[0]]

    if embeddings is not None:
        for i in range(1, len(sentences)):
            prev_vec = embeddings[i - 1]
            curr_vec = embeddings[i]
            similarity = _cosine_similarity(prev_vec, curr_vec)

            current_len = len(" ".join(current))
            should_split = similarity < threshold
            too_long = current_len + len(sentences[i]) > max_chunk_size

            if current and (should_split or too_long):
                chunks.append(
                    Chunk(
                        text=" ".join(current).strip(),
                        metadata={
                            **metadata,
                            "strategy": "semantic",
                            "chunk_index": len(chunks),
                            "split_reason": "similarity" if should_split else "max_size",
                        },
                    )
                )
                current = [sentences[i]]
            else:
                current.append(sentences[i])

    else:
        lexical_vectors = [_lexical_vector(s) for s in sentences]

        for i in range(1, len(sentences)):
            similarity = _dict_cosine_similarity(
                lexical_vectors[i - 1],
                lexical_vectors[i],
            )

            current_len = len(" ".join(current))
            should_split = similarity < threshold
            too_long = current_len + len(sentences[i]) > max_chunk_size

            if current and (should_split or too_long):
                chunks.append(
                    Chunk(
                        text=" ".join(current).strip(),
                        metadata={
                            **metadata,
                            "strategy": "semantic",
                            "chunk_index": len(chunks),
                            "split_reason": "similarity" if should_split else "max_size",
                            "embedding_backend": "lexical_fallback",
                        },
                    )
                )
                current = [sentences[i]]
            else:
                current.append(sentences[i])

    if current:
        chunks.append(
            Chunk(
                text=" ".join(current).strip(),
                metadata={
                    **metadata,
                    "strategy": "semantic",
                    "chunk_index": len(chunks),
                },
            )
        )

    return chunks


# =========================================================
# Strategy 2: Hierarchical Chunking
# =========================================================

def _split_text_into_sized_chunks(
    text: str,
    max_size: int,
) -> list[str]:
    """
    Cắt text thành các chunk <= max_size, ưu tiên không cắt giữa câu.

    Thứ tự ưu tiên:
    1. Paragraph
    2. Sentence
    3. Hard cut nếu một sentence quá dài
    """
    paragraphs = _split_paragraphs(text)

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            chunk_text = "\n\n".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current = []

    for para in paragraphs:
        current_text = "\n\n".join(current)

        if current and len(current_text) + len(para) + 2 > max_size:
            flush()

        if len(para) <= max_size:
            current.append(para)
            continue

        # Paragraph quá dài thì tách tiếp theo sentence.
        sentences = _split_sentences(para)
        sent_buffer: list[str] = []

        for sent in sentences:
            sent_current = " ".join(sent_buffer)

            if sent_buffer and len(sent_current) + len(sent) + 1 > max_size:
                if current:
                    flush()
                chunks.append(" ".join(sent_buffer).strip())
                sent_buffer = []

            if len(sent) <= max_size:
                sent_buffer.append(sent)
            else:
                # Sentence quá dài thì hard cut.
                if sent_buffer:
                    if current:
                        flush()
                    chunks.append(" ".join(sent_buffer).strip())
                    sent_buffer = []

                for start in range(0, len(sent), max_size):
                    hard_piece = sent[start:start + max_size].strip()
                    if hard_piece:
                        if current:
                            flush()
                        chunks.append(hard_piece)

        if sent_buffer:
            if current:
                flush()
            chunks.append(" ".join(sent_buffer).strip())

    flush()
    return chunks


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Hierarchical chunking.

    Tạo:
    - parent chunks: chunk lớn, giữ nhiều context.
    - child chunks: chunk nhỏ hơn để retrieve chính xác hơn.

    Trong RAG:
    - Search bằng child chunks.
    - Khi tìm được child tốt, có thể trả về parent tương ứng để có context rộng.
    """
    metadata = metadata or {}
    text = _normalize_whitespace(text)

    if not text:
        return [], []

    parent_texts = _split_text_into_sized_chunks(text, parent_size)

    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"

        parent_chunk = Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "strategy": "hierarchical",
                "chunk_type": "parent",
                "parent_id": parent_id,
                "parent_index": parent_index,
                "chunk_index": parent_index,
            },
            parent_id=None,
        )
        parents.append(parent_chunk)

        child_texts = _split_text_into_sized_chunks(parent_text, child_size)

        for child_index, child_text in enumerate(child_texts):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "strategy": "hierarchical",
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "chunk_index": len(children),
                    },
                    parent_id=parent_id,
                )
            )

    return parents, children


# =========================================================
# Strategy 3: Structure-aware Chunking
# =========================================================

def _detect_structure_headers(text: str) -> list[re.Match]:
    """
    Detect header trong markdown và văn bản pháp luật.

    Hỗ trợ:
    - Markdown: #, ##, ###
    - Legal docs:
        Chương I ...
        Mục 1 ...
        Điều 1. ...
        Article 1 ...
        Section 1 ...
    """
    header_pattern = re.compile(
        r"""
        (?m)^(
            \#{1,6}\s+.+ |
            Chương\s+[IVXLC\d]+\.?.* |
            Mục\s+\d+\.?.* |
            Điều\s+\d+[a-zA-Z]?\s*\.?.* |
            Article\s+\d+[a-zA-Z]?\s*\.?.* |
            Section\s+\d+[a-zA-Z]?\s*\.?.*
        )$
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    return list(header_pattern.finditer(text))


def _clean_section_name(header: str) -> str:
    """
    Làm sạch tên section.
    """
    header = header.strip()
    header = re.sub(r"^#{1,6}\s+", "", header)
    return header.strip()


def chunk_structure_aware(
    text: str,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Structure-aware chunking.

    Mục tiêu:
    - Markdown thì chunk theo header.
    - Văn bản pháp luật thì chunk theo Chương / Mục / Điều.
    - Không cắt ngang bảng, list, code block nếu chúng nằm trong cùng section.
    """
    metadata = metadata or {}
    text = _normalize_whitespace(text)

    if not text:
        return []

    matches = _detect_structure_headers(text)

    if not matches:
        return [
            Chunk(
                text=text,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": "document",
                    "chunk_index": 0,
                },
            )
        ]

    chunks: list[Chunk] = []

    # Nội dung trước header đầu tiên.
    intro = text[: matches[0].start()].strip()
    if intro:
        chunks.append(
            Chunk(
                text=intro,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": "intro",
                    "chunk_index": len(chunks),
                },
            )
        )

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        header = match.group(0).strip()
        section_name = _clean_section_name(header)

        if not section_text:
            continue

        chunks.append(
            Chunk(
                text=section_text,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": section_name,
                    "raw_header": header,
                    "chunk_index": len(chunks),
                },
            )
        )

    return chunks


# =========================================================
# A/B Test: Compare All Strategies
# =========================================================

def compare_strategies(documents: list[dict]) -> dict:
    """
    Run tất cả chunking strategies trên documents và so sánh thống kê cơ bản.
    """
    def _stats(chunk_list: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunk_list]

        if not lengths:
            return {
                "count": 0,
                "avg_len": 0,
                "min_len": 0,
                "max_len": 0,
            }

        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {
            **_stats(children),
            "parents": len(parents),
        },
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(
            f"{name:<15} "
            f"{s['count']:>7} "
            f"{s['avg_len']:>5} "
            f"{s['min_len']:>5} "
            f"{s['max_len']:>5}"
        )

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    results = compare_strategies(docs)

    for name, stats in results.items():
        print(f"  {name}: {stats}")
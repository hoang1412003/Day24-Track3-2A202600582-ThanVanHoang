from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================

Làm giàu chunks TRƯỚC khi embed:
1. Chunk Summarization
2. HyQA - Hypothesis Questions
3. Contextual Prepend
4. Auto Metadata Extraction
5. Combined Single-Call Enrichment

Test:
    pytest tests/test_m5.py
"""

import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full", "combined"


# =========================================================
# Helper functions
# =========================================================

def _has_openai_key() -> bool:
    """
    Check whether OpenAI key is available.

    Ưu tiên:
    - OPENAI_API_KEY trong config
    - OPENAI_API_KEY trong environment
    """
    return bool(OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))


def _get_openai_client():
    """
    Lazy-create OpenAI client.
    """
    from openai import OpenAI

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

    if api_key:
        return OpenAI(api_key=api_key)

    return OpenAI()




def _default_cache_path() -> str:
    """Path lưu cache enrichment để có thể resume/reuse giữa các lần chạy."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.getenv("M5_ENRICH_CACHE", os.path.join(root, "cache", "m5_enrichment_cache.json"))


def _chunk_cache_key(text: str, source: str, methods: list[str]) -> str:
    """Stable hash cho nội dung chunk + source + methods."""
    payload = json.dumps(
        {
            "text": text or "",
            "source": source or "",
            "methods": sorted(methods or []),
            "schema": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_enrichment_cache(cache_path: str) -> dict:
    if os.getenv("M5_DISABLE_CACHE", "0") == "1":
        return {}
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"  ⚠️  Could not load M5 cache: {exc}")
        return {}


def _save_enrichment_cache(cache: dict, cache_path: str) -> None:
    if os.getenv("M5_DISABLE_CACHE", "0") == "1" or not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        print(f"  ⚠️  Could not save M5 cache: {exc}")


def _enriched_from_cache(data: dict) -> EnrichedChunk | None:
    try:
        return EnrichedChunk(
            original_text=str(data.get("original_text", "")),
            enriched_text=str(data.get("enriched_text", "")),
            summary=str(data.get("summary", "")),
            hypothesis_questions=list(data.get("hypothesis_questions", [])),
            auto_metadata=dict(data.get("auto_metadata", {})),
            method=str(data.get("method", "combined")),
        )
    except Exception:
        return None

def _clean_json_text(text: str) -> str:
    """
    Clean JSON returned by LLM.

    LLM đôi khi trả:
        ```json
        {...}
        ```

    Hàm này bỏ markdown fence.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())

    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """
    Tách câu đơn giản cho tiếng Việt/English.
    """
    text = (text or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    return sentences


def _detect_language(text: str) -> str:
    """
    Detect language đơn giản.
    """
    vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"

    lower = (text or "").lower()

    if any(ch in lower for ch in vietnamese_chars):
        return "vi"

    return "en"


def _guess_category(text: str) -> str:
    """
    Guess category bằng keyword heuristic.
    """
    lower = (text or "").lower()

    hr_keywords = [
        "nhân viên",
        "nghỉ phép",
        "lương",
        "thưởng",
        "thử việc",
        "hợp đồng lao động",
        "employee",
        "leave",
        "salary",
    ]

    it_keywords = [
        "mật khẩu",
        "tài khoản",
        "bảo mật",
        "hệ thống",
        "phần mềm",
        "password",
        "account",
        "security",
    ]

    finance_keywords = [
        "chi phí",
        "hóa đơn",
        "thanh toán",
        "ngân sách",
        "tài chính",
        "invoice",
        "payment",
        "budget",
    ]

    legal_keywords = [
        "điều",
        "nghị định",
        "luật",
        "quy định",
        "pháp luật",
        "article",
        "law",
        "regulation",
    ]

    if any(k in lower for k in hr_keywords):
        return "hr"

    if any(k in lower for k in it_keywords):
        return "it"

    if any(k in lower for k in finance_keywords):
        return "finance"

    if any(k in lower for k in legal_keywords):
        return "legal"

    return "policy"


def _extract_entities_simple(text: str, max_entities: int = 8) -> list[str]:
    """
    Extract entities đơn giản bằng regex.

    Bắt:
    - Cụm viết hoa
    - Số + đơn vị
    - Tên văn bản như Nghị định, Điều, Chương
    """
    text = text or ""

    entities: list[str] = []

    patterns = [
        r"\b(?:Nghị định|Luật|Thông tư|Quyết định)\s+số\s+[\w\-\/\.]+",
        r"\b(?:Điều|Chương|Mục)\s+\d+[A-Za-z]?",
        r"\b\d+\s*(?:ngày|năm|tháng|giờ|%|VNĐ|đồng|USD)\b",
        r"\b[A-ZĐ][A-Za-zÀ-ỹ]*(?:\s+[A-ZĐ][A-Za-zÀ-ỹ]*){1,5}",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text):
            entity = match.strip()
            if entity and entity not in entities:
                entities.append(entity)

            if len(entities) >= max_entities:
                return entities

    return entities


def _fallback_summary(text: str) -> str:
    """
    Extractive fallback summary.
    """
    sentences = _split_sentences(text)

    if not sentences:
        return ""

    if len(sentences) == 1:
        return sentences[0]

    return " ".join(sentences[:2])


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate simple hypothesis questions without LLM.
    """
    sentences = _split_sentences(text)
    questions: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip(" .!?;:")

        if len(sentence) < 10:
            continue

        questions.append(f"Đoạn này nói gì về {sentence[:60].lower()}?")

        if len(questions) >= n_questions:
            break

    if not questions:
        questions = ["Đoạn văn này cung cấp thông tin gì?"]

    return questions[:n_questions]


def _fallback_metadata(text: str) -> dict:
    """
    Extract metadata without LLM.
    """
    summary = _fallback_summary(text)

    return {
        "topic": summary[:100] if summary else "general",
        "entities": _extract_entities_simple(text),
        "category": _guess_category(text),
        "language": _detect_language(text),
    }


# =========================================================
# Technique 1: Chunk Summarization
# =========================================================

def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.

    Embed summary thay vì hoặc cùng với raw chunk để giảm noise.
    """
    text = text or ""

    if not text.strip():
        return ""

    if _has_openai_key():
        try:
            client = _get_openai_client()

            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_ENRICH_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
                            "Chỉ giữ thông tin quan trọng, không bịa thêm."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                max_tokens=150,
                temperature=0,
            )

            content = resp.choices[0].message.content or ""
            return content.strip()

        except Exception as exc:
            print(f"  ⚠️  OpenAI summarize failed: {exc}")

    return _fallback_summary(text)


# =========================================================
# Technique 2: Hypothesis Question-Answer / HyQA
# =========================================================

def generate_hypothesis_questions(
    text: str,
    n_questions: int = 3,
) -> list[str]:
    """
    Generate các câu hỏi mà chunk có thể trả lời.

    Mục đích:
    - Index thêm câu hỏi giả định.
    - Giúp query match tốt hơn nếu query dùng từ khác với chunk gốc.
    """
    text = text or ""

    if not text.strip():
        return []

    if n_questions <= 0:
        return []

    if _has_openai_key():
        try:
            client = _get_openai_client()

            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_ENRICH_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên một dòng. "
                            "Không trả lời câu hỏi, chỉ viết câu hỏi."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                max_tokens=220,
                temperature=0,
            )

            content = resp.choices[0].message.content or ""
            raw_questions = content.strip().splitlines()

            questions = []
            for q in raw_questions:
                q = q.strip()
                q = re.sub(r"^\s*[\-\*\d]+[\.\)]\s*", "", q)
                q = q.strip()

                if q and q not in questions:
                    questions.append(q)

                if len(questions) >= n_questions:
                    break

            if questions:
                return questions[:n_questions]

        except Exception as exc:
            print(f"  ⚠️  OpenAI HyQA failed: {exc}")

    return _fallback_questions(text, n_questions=n_questions)


# =========================================================
# Technique 3: Contextual Prepend
# =========================================================

def contextual_prepend(
    text: str,
    document_title: str = "",
) -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.

    Contextual retrieval style:
        "Trích từ tài liệu X, phần này nói về Y."

    Sau đó mới đến raw chunk.
    """
    text = text or ""
    document_title = document_title or ""

    if not text.strip():
        return ""

    if _has_openai_key():
        try:
            client = _get_openai_client()

            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_ENRICH_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết đúng 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này "
                            "nằm trong tài liệu nào và nói về chủ đề gì. "
                            "Không tóm tắt dài, không bịa thông tin."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
                temperature=0,
            )

            context = (resp.choices[0].message.content or "").strip()

            if context:
                return f"{context}\n\n{text}"

        except Exception as exc:
            print(f"  ⚠️  OpenAI contextual failed: {exc}")

    if document_title:
        return f"Trích từ tài liệu {document_title}. {text}"

    return text


# =========================================================
# Technique 4: Auto Metadata Extraction
# =========================================================

def extract_metadata(text: str) -> dict:
    """
    Extract metadata tự động:
    - topic
    - entities
    - category
    - language
    """
    text = text or ""

    if not text.strip():
        return {
            "topic": "general",
            "entities": [],
            "category": "policy",
            "language": "vi",
        }

    if _has_openai_key():
        try:
            client = _get_openai_client()

            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_ENRICH_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ, không markdown. '
                            'Schema: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance|legal|general", "language": "vi|en"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                max_tokens=180,
                temperature=0,
            )

            content = resp.choices[0].message.content or ""
            data = json.loads(_clean_json_text(content))

            return {
                "topic": str(data.get("topic", "general")),
                "entities": data.get("entities", []) if isinstance(data.get("entities", []), list) else [],
                "category": str(data.get("category", "policy")),
                "language": str(data.get("language", _detect_language(text))),
            }

        except Exception as exc:
            print(f"  ⚠️  OpenAI metadata failed: {exc}")

    return _fallback_metadata(text)


# =========================================================
# Combined Single-Call Mode
# =========================================================

def _enrich_single_call(
    text: str,
    source: str,
) -> dict:
    """
    Single LLM call để lấy:
    - summary
    - questions
    - context
    - metadata

    Tối ưu chi phí: 1 API call/chunk thay vì 4 calls riêng.
    """
    text = text or ""
    source = source or ""

    fallback = {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ tài liệu {source}." if source else "",
        "metadata": extract_metadata(text),
    }

    if not text.strip():
        return fallback

    if _has_openai_key():
        try:
            client = _get_openai_client()

            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_ENRICH_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": """
Phân tích đoạn văn và trả về JSON hợp lệ, không markdown, đúng schema:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {
    "topic": "...",
    "entities": ["..."],
    "category": "policy|hr|it|finance|legal|general",
    "language": "vi|en"
  }
}
Không bịa thông tin ngoài đoạn văn.
""".strip(),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=450,
                temperature=0,
            )

            content = resp.choices[0].message.content or ""
            data = json.loads(_clean_json_text(content))

            summary = str(data.get("summary", fallback["summary"]))
            questions = data.get("questions", fallback["questions"])
            context = str(data.get("context", fallback["context"]))
            metadata = data.get("metadata", fallback["metadata"])

            if not isinstance(questions, list):
                questions = fallback["questions"]

            if not isinstance(metadata, dict):
                metadata = fallback["metadata"]

            return {
                "summary": summary,
                "questions": questions[:3],
                "context": context,
                "metadata": {
                    "topic": str(metadata.get("topic", fallback["metadata"].get("topic", "general"))),
                    "entities": metadata.get("entities", []),
                    "category": str(metadata.get("category", fallback["metadata"].get("category", "policy"))),
                    "language": str(metadata.get("language", fallback["metadata"].get("language", "vi"))),
                },
            }

        except Exception as exc:
            print(f"  ⚠️  Enrichment API failed: {exc}")

    return fallback


# =========================================================
# Full Enrichment Pipeline
# =========================================================

def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    cache_path: str | None = None,
    use_cache: bool = True,
    save_every: int = 5,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Có 2 chế độ:

    1. Combined mode:
        methods=None hoặc methods=["combined"]

        Gọi _enrich_single_call() để lấy summary/questions/context/metadata.

    2. Separate mode:
        methods=["summary", "hyqa", "contextual", "metadata"]

        Gọi từng function riêng để dễ học/debug.

    Args:
        chunks:
            List of {"text": str, "metadata": dict}

        methods:
            Options:
            - "summary"
            - "hyqa"
            - "contextual"
            - "metadata"
            - "combined"

        cache_path:
            File JSON lưu cache enrichment. Mặc định: cache/m5_enrichment_cache.json.

        use_cache:
            Nếu True, reuse chunk đã enrich và chỉ gọi API cho cache miss.

        save_every:
            Checkpoint cache sau mỗi N cache miss để chạy lại không mất tiến độ.
    """
    if methods is None:
        methods = ["combined"]

    if not chunks:
        return []

    allowed_methods = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown = set(methods) - allowed_methods

    if unknown:
        raise ValueError(f"Unknown enrichment methods: {sorted(unknown)}")

    use_combined = "combined" in methods
    cache_path = cache_path or _default_cache_path()
    cache = _load_enrichment_cache(cache_path) if use_cache else {}

    enriched: list[EnrichedChunk] = []
    cache_hits = 0
    cache_misses = 0

    for i, chunk in enumerate(chunks):
        text = str(chunk.get("text", "") or "")
        chunk_metadata = chunk.get("metadata", {})
        chunk_metadata = chunk_metadata if isinstance(chunk_metadata, dict) else {}

        source = str(chunk_metadata.get("source", "") or "")
        cache_key = _chunk_cache_key(text, source, methods)

        if use_cache and cache_key in cache:
            cached = _enriched_from_cache(cache[cache_key])
            if cached is not None:
                enriched.append(cached)
                cache_hits += 1
                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    print(
                        f"  Enriched {i + 1}/{len(chunks)} chunks "
                        f"({cache_hits} cache hits, {cache_misses} new)...",
                        flush=True,
                    )
                continue

        cache_misses += 1

        if use_combined:
            result = _enrich_single_call(text, source)

            summary = str(result.get("summary", ""))
            questions = result.get("questions", [])
            context_line = str(result.get("context", "") or "")
            auto_meta = result.get("metadata", {})

            if not isinstance(questions, list):
                questions = []

            if not isinstance(auto_meta, dict):
                auto_meta = {}

            enriched_text = f"{context_line}\n\n{text}" if context_line else text

        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        item = EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={
                **chunk_metadata,
                **auto_meta,
            },
            method="+".join(methods),
        )
        enriched.append(item)

        if use_cache:
            cache[cache_key] = asdict(item)
            if cache_misses % max(save_every, 1) == 0:
                _save_enrichment_cache(cache, cache_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(
                f"  Enriched {i + 1}/{len(chunks)} chunks "
                f"({cache_hits} cache hits, {cache_misses} new)...",
                flush=True,
            )

    if use_cache:
        _save_enrichment_cache(cache, cache_path)
        print(f"  M5 cache saved: {cache_path} ({len(cache)} entries)", flush=True)

    return enriched


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}\n")

    chunks = [
        {
            "text": sample,
            "metadata": {
                "source": "Sổ tay nhân viên VinUni 2024",
                "chunk_index": 0,
            },
        }
    ]

    enriched = enrich_chunks(chunks)
    print(f"Enriched: {enriched[0]}")
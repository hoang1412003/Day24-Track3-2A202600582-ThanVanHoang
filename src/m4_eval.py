from __future__ import annotations

"""
Module 4: RAGAS Evaluation — 4 metrics + failure analysis
=========================================================

Implement:
1. Load test set from JSON
2. Evaluate RAG outputs with RAGAS
3. Fallback heuristic evaluation if RAGAS/API key is unavailable
4. Failure analysis with diagnostic tree
5. Save JSON report

Test:
    pytest tests/test_m4.py
"""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


# =========================================================
# Load test set
# =========================================================

def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """
    Load test set from JSON.

    Expected format:
        [
            {
                "question": "...",
                "ground_truth": "...",
                ...
            }
        ]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Test set must be a list of dictionaries.")

    return data


# =========================================================
# Helper functions
# =========================================================

def _safe_float(value, default: float = 0.0) -> float:
    """
    Convert value to float safely.
    """
    try:
        if value is None:
            return default

        value = float(value)

        if value < 0:
            return 0.0
        if value > 1:
            return 1.0

        return value

    except Exception:
        return default


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for Vietnamese/English fallback metrics.
    """
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _f1_overlap(a: str, b: str) -> float:
    """
    Token overlap F1 between two strings.
    Used as lightweight fallback when RAGAS cannot run.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    overlap = set_a & set_b

    if not overlap:
        return 0.0

    precision = len(overlap) / len(set_a)
    recall = len(overlap) / len(set_b)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _context_text(contexts: list[str]) -> str:
    """
    Join retrieved contexts into one text.
    """
    return "\n\n".join(c for c in contexts if c)


def _fallback_eval_one(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> EvalResult:
    """
    Heuristic fallback evaluation.

    Không thay thế RAGAS thật, nhưng giúp:
    - unit test chạy được
    - không cần OPENAI_API_KEY
    - vẫn có signal để failure_analysis hoạt động
    """
    joined_context = _context_text(contexts)

    # Answer relevancy: answer có trả lời đúng ý question không.
    answer_relevancy = _f1_overlap(question, answer)

    # Faithfulness: answer có được support bởi contexts không.
    faithfulness = _f1_overlap(answer, joined_context)

    # Context precision: contexts có liên quan tới question không.
    context_precision = _f1_overlap(question, joined_context)

    # Context recall: contexts có chứa thông tin trong ground truth không.
    context_recall = _f1_overlap(ground_truth, joined_context)

    # Nếu answer gần ground truth, cộng nhẹ cho relevancy.
    answer_gt_overlap = _f1_overlap(answer, ground_truth)
    answer_relevancy = max(answer_relevancy, answer_gt_overlap)

    return EvalResult(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        faithfulness=_safe_float(faithfulness),
        answer_relevancy=_safe_float(answer_relevancy),
        context_precision=_safe_float(context_precision),
        context_recall=_safe_float(context_recall),
    )


def _aggregate(per_question: list[EvalResult]) -> dict:
    """
    Aggregate per-question metrics.
    """
    if not per_question:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    n = len(per_question)

    return {
        "faithfulness": sum(r.faithfulness for r in per_question) / n,
        "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
        "context_precision": sum(r.context_precision for r in per_question) / n,
        "context_recall": sum(r.context_recall for r in per_question) / n,
        "per_question": per_question,
    }


def _validate_eval_inputs(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> None:
    """
    Ensure all input lists have the same length.
    """
    lengths = {
        "questions": len(questions),
        "answers": len(answers),
        "contexts": len(contexts),
        "ground_truths": len(ground_truths),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input lengths must match, got: {lengths}")


# =========================================================
# RAGAS Evaluation
# =========================================================

def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    Run RAGAS evaluation.

    Trả về:
        {
            "faithfulness": float,
            "answer_relevancy": float,
            "context_precision": float,
            "context_recall": float,
            "per_question": list[EvalResult]
        }

    Nếu RAGAS không chạy được, fallback sang heuristic evaluation.
    """
    _validate_eval_inputs(questions, answers, contexts, ground_truths)

    if not questions:
        return _aggregate([])

    use_ragas = os.getenv("USE_RAGAS", "1") == "1"

    if not use_ragas:
        per_question = [
            _fallback_eval_one(q, a, c, gt)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return _aggregate(per_question)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        df = result.to_pandas()

        per_question: list[EvalResult] = []

        for _, row in df.iterrows():
            per_question.append(
                EvalResult(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    contexts=list(row.get("contexts", [])),
                    ground_truth=str(row.get("ground_truth", "")),
                    faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                    answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                    context_precision=_safe_float(row.get("context_precision", 0.0)),
                    context_recall=_safe_float(row.get("context_recall", 0.0)),
                )
            )

        return _aggregate(per_question)

    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed, using fallback metrics: {exc}")

        per_question = [
            _fallback_eval_one(q, a, c, gt)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]

        return _aggregate(per_question)


# =========================================================
# Failure Analysis
# =========================================================

def _diagnose_metric(metric_name: str) -> tuple[str, str]:
    """
    Map worst metric to likely root cause and suggested fix.
    """
    diagnostic_tree = {
        "faithfulness": (
            "LLM có thể đang hallucinate hoặc trả lời vượt quá context.",
            "Siết prompt: bắt buộc chỉ trả lời từ context, thêm citation, giảm temperature.",
        ),
        "answer_relevancy": (
            "Câu trả lời chưa bám sát câu hỏi.",
            "Cải thiện prompt template, rewrite query, hoặc thêm bước intent detection.",
        ),
        "context_precision": (
            "Retriever lấy quá nhiều chunk không liên quan.",
            "Thêm reranking, metadata filter, hoặc giảm top_k retrieval.",
        ),
        "context_recall": (
            "Retriever bỏ sót chunk quan trọng.",
            "Cải thiện chunking, thêm BM25/hybrid search, tăng top_k trước rerank.",
        ),
    }

    return diagnostic_tree.get(
        metric_name,
        (
            "Không xác định rõ nguyên nhân.",
            "Kiểm tra lại retrieval, prompt và dữ liệu test.",
        ),
    )


def _metric_dict(result: EvalResult) -> dict[str, float]:
    """
    Convert EvalResult metrics to dict.
    """
    return {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
    }


def failure_analysis(
    eval_results: list[EvalResult],
    bottom_n: int = 10,
) -> list[dict]:
    """
    Analyze bottom-N worst questions using diagnostic tree.

    Output:
        [
            {
                "question": "...",
                "avg_score": 0.42,
                "worst_metric": "context_recall",
                "worst_metric_score": 0.1,
                "diagnosis": "...",
                "suggested_fix": "..."
            }
        ]
    """
    if not eval_results:
        return []

    rows: list[dict] = []

    for result in eval_results:
        metrics = _metric_dict(result)
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        worst_metric_score = metrics[worst_metric]

        diagnosis, suggested_fix = _diagnose_metric(worst_metric)

        rows.append(
            {
                "question": result.question,
                "answer": result.answer,
                "ground_truth": result.ground_truth,
                "avg_score": float(avg_score),
                "worst_metric": worst_metric,
                "worst_metric_score": float(worst_metric_score),
                "metrics": metrics,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    rows.sort(key=lambda row: row["avg_score"])

    return rows[:bottom_n]


# =========================================================
# Save report
# =========================================================

def _serialize_eval_result(result: EvalResult) -> dict:
    """
    Convert EvalResult dataclass to JSON-serializable dict.
    """
    return asdict(result)


def save_report(
    results: dict,
    failures: list[dict],
    path: str = "ragas_report.json",
) -> None:
    """
    Save evaluation report to JSON.
    """
    per_question = results.get("per_question", [])

    report = {
        "aggregate": {
            k: v
            for k, v in results.items()
            if k != "per_question"
        },
        "num_questions": len(per_question),
        "per_question": [
            _serialize_eval_result(r) if isinstance(r, EvalResult) else r
            for r in per_question
        ],
        "failures": failures,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to {path}")


# =========================================================
# Smoke test
# =========================================================

if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

    sample_questions = ["Nhân viên được nghỉ phép bao nhiêu ngày?"]
    sample_answers = ["Nhân viên được nghỉ 12 ngày mỗi năm."]
    sample_contexts = [["Nhân viên được nghỉ phép năm 12 ngày/năm theo chính sách công ty."]]
    sample_ground_truths = ["Nhân viên được nghỉ 12 ngày phép năm."]

    results = evaluate_ragas(
        questions=sample_questions,
        answers=sample_answers,
        contexts=sample_contexts,
        ground_truths=sample_ground_truths,
    )

    failures = failure_analysis(results["per_question"], bottom_n=3)
    save_report(results, failures, path="ragas_report.json")

    print(results)
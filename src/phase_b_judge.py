from __future__ import annotations

"""Phase B: LLM-as-Judge - pairwise, swap-and-average, Cohen kappa, bias analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE))


def _overlap_score(answer: str, reference: str) -> float:
    answer_tokens = _tokens(answer)
    ref_tokens = _tokens(reference)
    if not answer_tokens or not ref_tokens:
        return 0.0
    return len(answer_tokens & ref_tokens) / len(ref_tokens)


def _heuristic_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = 0.45 * _overlap_score(answer_a, question) + 0.35 * min(len(answer_a) / 220, 1.0)
    score_b = 0.45 * _overlap_score(answer_b, question) + 0.35 * min(len(answer_b) / 220, 1.0)

    policy_numbers = ["15", "12", "55", "50", "25", "8", "80", "3", "5", "120", "90"]
    for n in policy_numbers:
        if n in answer_a and n in question:
            score_a += 0.08
        if n in answer_b and n in question:
            score_b += 0.08

    score_a = max(0.0, min(score_a, 1.0))
    score_b = max(0.0, min(score_b, 1.0))
    if abs(score_a - score_b) < 0.05:
        winner = "tie"
    else:
        winner = "A" if score_a > score_b else "B"
    return {
        "winner": winner,
        "reasoning": "Heuristic fallback based on overlap, completeness proxy, and policy-number cues.",
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    prompt = f"""Ban la expert danh gia chat luong cau tra loi RAG.

Cau hoi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Danh gia dua tren do chinh xac, day du va suc tich. Tra loi JSON duy nhat:
{{"winner":"A|B|tie","reasoning":"giai thich ngan","scores":{{"A":0.0,"B":0.0}}}}
"""
    use_llm = bool(OPENAI_API_KEY) and os.getenv("USE_LLM_JUDGE", "1") == "1" and "PYTEST_CURRENT_TEST" not in os.environ
    if use_llm:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Ban la expert danh gia RAG. Chi tra loi JSON hop le."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                timeout=30,
            )
            data = json.loads(resp.choices[0].message.content)
            winner = data.get("winner", "tie")
            if winner not in {"A", "B", "tie"}:
                winner = "tie"
            scores = data.get("scores", {}) or {}
            return {
                "winner": winner,
                "reasoning": str(data.get("reasoning", "LLM judge completed.")),
                "scores": {
                    "A": max(0.0, min(float(scores.get("A", 0.0)), 1.0)),
                    "B": max(0.0, min(float(scores.get("B", 0.0)), 1.0)),
                },
            }
        except Exception as exc:
            print(f"LLM judge failed, using heuristic fallback: {exc}")
    return _heuristic_pairwise(question, answer_a, answer_b)


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")
    winner_pass1 = pass1.get("winner", "tie")
    final = winner_pass1 if winner_pass1 == winner_pass2 else "tie"
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=winner_pass1,
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=(winner_pass1 == winner_pass2),
        scores_pass1=pass1.get("scores", {"A": 0.0, "B": 0.0}),
        scores_pass2={"A": pass2_raw.get("scores", {}).get("B", 0.0), "B": pass2_raw.get("scores", {}).get("A", 0.0)},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    labels = sorted(set(judge_labels) | set(human_labels))
    p_e = sum((judge_labels.count(label) / n) * (human_labels.count(label) / n) for label in labels)
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    return max(-1.0, min((p_o - p_e) / (1 - p_e), 1.0))


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "No judge results available.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    decisive = [r for r in judge_results if r.final_winner != "tie"]
    a_wins_a_longer = sum(1 for r in decisive if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b))
    b_wins_b_longer = sum(1 for r in decisive if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a))
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    position_bias_rate = position_bias_count / total
    interpretation = "Position bias cao - nen tiep tuc dung swap-and-average." if position_bias_rate > 0.3 else "Position bias thap - judge kha on dinh."
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


def _main() -> None:
    os.makedirs("reports", exist_ok=True)
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)

    judge_results: list[JudgeResult] = []
    judge_labels: list[int] = []
    human_labels = [int(item["human_label"]) for item in human_data]
    for item in human_data:
        model_answer = item["model_answer"]
        reference = item["human_note"] + "\n" + item["question"]
        result = swap_and_average(item["question"], model_answer, reference)
        judge_results.append(result)
        judge_labels.append(1 if result.final_winner in {"A", "tie"} else 0)

    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(judge_results)
    report = {
        "total_questions": len(human_data),
        "cohen_kappa": round(kappa, 4),
        "judge_labels": judge_labels,
        "human_labels": human_labels,
        "bias_report": bias,
        "judge_results": [asdict(r) for r in judge_results],
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("analysis/bias_report.md", "w", encoding="utf-8") as f:
        f.write("# Bias Report\n\n")
        f.write(f"- Cohen kappa: {kappa:.3f}\n")
        f.write(f"- Position bias rate: {bias['position_bias_rate']}\n")
        f.write(f"- Verbosity bias: {bias['verbosity_bias']}\n")
        f.write(f"- Interpretation: {bias['interpretation']}\n")
    print("Phase B report saved -> reports/judge_results.json")
    print(f"Cohen kappa: {kappa:.3f}")
    print(f"Bias: {bias}")


if __name__ == "__main__":
    _main()

# Lab 24 Report - Production Eval + Guardrail Stack

## Executive Summary

Lab 24 xây dựng lớp đánh giá và bảo vệ cho RAG pipeline từ Day 18. Pipeline đã generate đủ `answers_50q.json` từ Qdrant, chạy đủ 3 phase và pass toàn bộ validation.

| Hạng mục | Kết quả |
|---|---:|
| Test suite | 40/40 passed |
| `check_lab.py` | 22/22 checks passed |
| Answers generated | 50/50 |
| Guard adversarial pass rate | 20/20 (100.0%) |
| Guard P95 latency | 0.05ms |

## Phase A - RAGAS Production Eval

### Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.3303 | 0.2854 | 0.1702 |
| answer_relevancy | 0.7249 | 0.4080 | 0.2781 |
| context_precision | 0.2436 | 0.3017 | 0.2810 |
| context_recall | 0.4403 | 0.3197 | 0.2674 |
| **avg_score** | 0.4348 | 0.3287 | 0.2492 |

Kết quả cho thấy factual là nhóm tốt nhất, multi-hop thấp hơn do cần kết hợp nhiều chính sách, và adversarial thấp nhất vì có bẫy version conflict, phủ định và policy contradiction. Đây là xu hướng hợp lý cho một bộ stress-test production.

### Failure Clusters

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 3 | 7 | 4 | 14 |
| answer_relevancy | 0 | 4 | 3 | 7 |
| context_precision | 17 | 8 | 1 | 26 |
| context_recall | 0 | 1 | 2 | 3 |

Dominant failure metric là `context_precision`. Nguyên nhân chính là retrieval lấy nhiều context nhiễu, đặc biệt khi corpus có nhiều phiên bản policy hoặc nhiều tài liệu cùng nhắc tới một chủ đề. Suggested fix ưu tiên: metadata filter theo version/status, cải thiện reranking, và prompt yêu cầu ưu tiên chính sách hiện hành.

### Bottom 5 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | factual | Cơ cấu điểm đánh giá hiệu suất gồm những thành phần nào và tỷ lệ ra sao? | 0.0681 | faithfulness |
| 2 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.0708 | answer_relevancy |
| 3 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.0810 | answer_relevancy |
| 4 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.1050 | answer_relevancy |
| 5 | adversarial | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.1175 | answer_relevancy |

## Phase B - LLM-as-Judge

| Metric | Value |
|---|---:|
| Total judged | 10 |
| Cohen kappa | 0.0000 |
| Position bias rate | 0.000 |
| Position bias count | 0 |
| Verbosity bias | 1.000 |

Position bias rate bằng 0.0, nghĩa là kết quả judge nhất quán giữa pass gốc và pass swap trong bộ hiện tại. Verbosity bias bằng 1.0, cho thấy judge có xu hướng chọn câu trả lời dài hơn hoặc chứa nhiều thông tin hơn. Cohen kappa bằng 0.0 vì nhãn judge fallback chưa đồng thuận tốt với human labels.

Ghi chú: Phase B hiện chạy bằng heuristic fallback để tránh phụ thuộc API khi test cục bộ; nếu bật LLM judge thật, cần chạy lại với USE_LLM_JUDGE=1.

## Phase C - Guardrails

| Metric | Value |
|---|---:|
| Adversarial passed | 20/20 |
| Pass rate | 100.0% |
| Presidio P95 | 0.02ms |
| Input rail P95 | 0.03ms |
| Total P95 | 0.05ms |
| Latency budget | 500ms |
| Budget OK | True |

Guard stack đạt 20/20 adversarial inputs, vượt yêu cầu tối thiểu 15/20 và đạt ngưỡng bonus 18/20. Các lớp chặn gồm PII detection cho CCCD, CMND, số điện thoại, email; input rail cho jailbreak, prompt injection, off-topic và yêu cầu truy xuất thông tin cá nhân.

## Production Recommendations

1. Thêm metadata filter cho `version`, `effective_date`, `policy_status` để ưu tiên chính sách hiện hành.
2. Giảm nhiễu retrieval bằng reranking chặt hơn và giới hạn context sau rerank.
3. Với câu multi-hop, thêm bước query decomposition để truy xuất từng phần chính sách rồi tổng hợp.
4. Với adversarial, thêm rule ưu tiên bản mới nhất và cảnh báo khi context chứa policy đã hết hiệu lực.
5. Chạy lại Phase B bằng LLM judge thật để có Cohen kappa phản ánh đúng chất lượng judge.

## Conclusion

Hệ thống đã hoàn thành đủ eval stack và guardrail stack cho Day24. Kết quả chính: RAG pipeline còn yếu ở `context_precision`, nhưng guardrail hoạt động tốt với pass rate 100% và latency thấp. Trọng tâm cải thiện tiếp theo nên nằm ở retrieval precision, version-aware filtering và prompt trả lời bám sát câu hỏi.

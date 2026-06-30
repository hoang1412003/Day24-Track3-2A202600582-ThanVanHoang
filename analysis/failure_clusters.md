# Failure Cluster Analysis - Phase A

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.3303 | 0.2854 | 0.1702 |
| answer_relevancy | 0.7249 | 0.4080 | 0.2781 |
| context_precision | 0.2436 | 0.3017 | 0.2810 |
| context_recall | 0.4403 | 0.3197 | 0.2674 |
| **avg_score** | 0.4348 | 0.3287 | 0.2492 |

Nhìn theo `avg_score`, nhóm factual đạt cao nhất (0.4348), tiếp theo là multi_hop (0.3287), và thấp nhất là adversarial (0.2492). Điều này hợp lý vì adversarial chứa version conflict, phủ định, và policy contradiction.

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric | Diagnosis |
|---:|---|---|---:|---|---|
| 1 | factual | Cơ cấu điểm đánh giá hiệu suất gồm những thành phần nào và tỷ lệ ra sao? | 0.0681 | faithfulness | LLM hallucinating |
| 2 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.0708 | answer_relevancy | Answer does not match question |
| 3 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.0810 | answer_relevancy | Answer does not match question |
| 4 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.1050 | answer_relevancy | Answer does not match question |
| 5 | adversarial | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.1175 | answer_relevancy | Answer does not match question |
| 6 | multi_hop | Nhân viên thử việc tháng thứ 3 phát hiện vi phạm bảo mật. Họ nên và không nên làm gì theo chính sách? | 0.1217 | faithfulness | LLM hallucinating |
| 7 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.1223 | answer_relevancy | Answer does not match question |
| 8 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.1375 | answer_relevancy | Answer does not match question |
| 9 | multi_hop | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | 0.1506 | answer_relevancy | Answer does not match question |
| 10 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.1554 | faithfulness | LLM hallucinating |

Bottom 10 có 1 factual, 6 multi_hop và 3 adversarial. Các câu yếu nhất tập trung vào password policy, phép năm v2024, lương thử việc, mua sắm thiết bị và câu hỏi kết hợp nhiều chính sách.

## 3. Failure Cluster Matrix

Mỗi ô là số câu có `worst_metric` bằng metric ở hàng và thuộc distribution ở cột.

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 3 | 7 | 4 | 14 |
| answer_relevancy | 0 | 4 | 3 | 7 |
| context_precision | 17 | 8 | 1 | 26 |
| context_recall | 0 | 1 | 2 | 3 |

## 4. Dominant Failure Analysis

**Dominant distribution:** `factual`  
**Dominant metric:** `context_precision`

Metric yếu nhất là `context_precision`, xuất hiện nhiều nhất trong failure matrix. Điều này cho thấy retriever lấy được context nhưng còn lẫn nhiều chunk chưa liên quan trực tiếp, làm giảm chất lượng bằng chứng cho câu trả lời. Với factual, số lượng câu bị `context_precision` thấp cao nhất vì câu hỏi đơn giản cần đúng một chính sách, nhưng retrieval có thể kéo thêm chunk nhiễu từ policy khác hoặc version khác. Với multi_hop và adversarial, lỗi chuyển dần sang `answer_relevancy` và `faithfulness`, do câu hỏi cần tính toán hoặc ưu tiên phiên bản chính sách hiện hành.

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Answer không được support đủ bởi context hoặc LLM suy diễn thêm | Siết system prompt, yêu cầu chỉ trả lời từ context, thêm citation và giảm temperature |
| context_recall | Retriever bỏ sót chunk quan trọng | Tăng top_k trước rerank, cải thiện chunking, thêm query expansion cho câu multi-hop |
| context_precision | Retriever lấy nhiều chunk nhiễu | Thêm metadata filter theo policy/version, rerank mạnh hơn, loại policy hết hiệu lực khi câu hỏi hỏi bản hiện hành |
| answer_relevancy | Answer chưa bám sát câu hỏi | Cải thiện prompt template, tách intent, yêu cầu trả lời trực tiếp trước rồi mới giải thích |

## 6. Nhận xét về Adversarial Distribution

Adversarial có `avg_score` thấp nhất (0.2492), thấp hơn factual (0.4348) và multi_hop (0.3287). Đây là tín hiệu tốt cho bộ eval vì nhóm adversarial thực sự khó hơn. Các câu adversarial trong bottom 10 gồm câu đổi mật khẩu, độ dài mật khẩu và phép năm hiện hành. Những câu này dễ làm pipeline nhầm giữa policy cũ và policy mới, hoặc trả lời thiếu trọng tâm khi context chứa nhiều phiên bản.

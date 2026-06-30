## CI/CD Blueprint: RAG Eval + Guardrail Stack
**Sinh viên thực hiện:** Thân Văn Hoàng
### Guard Stack Pipeline
| Layer           | Tool          | Latency P95 | Failure Action |
|-----------------|---------------|-------------|----------------|
| PII Detection   | Presidio/Regex | 0.03ms | Reject + log |
| Topic/Jailbreak | NeMo/Input Rule | 0.02ms | 503 + reason |
| RAG Pipeline    | Day 18        | <2000ms     | Fallback |
| Output Check    | NeMo/Output Rule | <300ms | Block + log |

### CI Gates (phai pass truoc khi merge to main)
- [ ] RAGAS faithfulness >= 0.75 (measured on 50q test set)
- [x] Adversarial suite pass rate >= 75% (20/20)
- [x] P95 total guard latency < 500ms

### Monitoring
- P95 latency thuc te: 0.05ms
- Adversarial pass rate: 20/20
- Worst RAGAS metric: xem reports/ragas_50q.json -> failure_clusters.dominant_failure_metric
- Dominant failure distribution: xem reports/ragas_50q.json -> failure_clusters.dominant_failure_distribution

### Operating Notes
Guardrail stack blocks PII first, then blocks jailbreak/off-topic/prompt-injection input before RAG. Output rail rejects sensitive or PII-bearing answers before returning to user.

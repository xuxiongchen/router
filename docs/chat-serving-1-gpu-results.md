# Chat Serving 1 — finite GPU results, 2026-09-27

These two-Worker results are preserved. The separately authorized
[three-Worker follow-up](chat-serving-1-three-workers.md) uses the same measured
trace and native, with its own evidence, source SHA and memory configuration.

## Exact scope and artifacts

Fresh user-authorized window ended no later than 2026-09-28 01:00 Asia/Shanghai.
Work completed and both owned Workers stopped at approximately 23:12 on Sep 27;
post-stop GPU memory was 0 MiB. No instance deletion, unrelated process operation,
model/cache deletion, package installation, Worker patch installation, push or PR.

- Semantic harness: `a64c7d111eb8a80c6bd1e9dfe0078026dbe0bf79`.
- Focused cancellation fix and performance: `1763ecf0192b4228b5599a8e11a67d2348e3653a`.
- Actual production native source: `2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110`.
- Native SHA-256: `f17ccc17cc343b5cea68145a3e73d6442cbd38e2413f6f9a75606a23928e683b`.
- Production wheel SHA-256: `4c7bf72c70c1311d9fe9883bfa1767434a86cf1b95b806f49447779695b38d27`.
- Downloaded evidence archive SHA-256:
  `7778a690acbf17fa065f2a99ad5df1ebfed818531862351ed4ec9d4313583f3f`.

Same feature-off production artifact throughout; no experimental ablation build.
Ancestry and a fail-closed non-product diff allowlist establish artifact reuse.
The compiled native is **not** relabeled with either harness SHA. No Rust or
production Python module changed. Existing capability Worker proposal is an
explicit dependency, reused read-only and hash-verified, not a stock HTTP API.

One RTX 4080 SUPER, 32760 MiB, UUID
`GPU-3a378f4a-55d3-93b2-c0dc-c0239eda93bb`; two independent DP/TP/PP=1 Workers
share that GPU. vLLM 0.29.0, pinned `98dff2a81d747d1dba01a47f939f48c3526d4206`
plus recorded capability proposal; official Hermes tool and Qwen3 reasoning
parsers. Actual Chat serving file SHA-256 matches the pinned source:
`ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92`.
Qwen3-0.6B ModelScope revision `09b42cad3d112e832108974449ccb5e8e0f5b5d1`;
weight SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`
rechecked against the preserved fixed-revision listing after measurement.
This is on-disk provenance, not GPU-memory attestation or HF revision equivalence.

## Semantic acceptance and the retained failed attempt

`chat-agent-first/summary.json` remains **FAIL**: 62 cases passed, then the Chat
cancellation log check failed. The Worker really aborted and both loads reached
zero; the reused Completion matcher incorrectly required `response.id + '-0'`.
Pinned single-input Chat uses `response.id` before the engine's optional eight-hex
suffix. The fix adds an explicit Chat mode, exact full matching, wrong-ID/PID/suffix
negative tests, and `--chat-cancel-only`. No production cancellation change.

`chat-cancel-fixed/summary.json` is **PASS**, two focused cases: actual Chat abort
and a subsequent successful request. Request-ID/PID-correlated Worker abort log,
active-before-close, no natural completion, and zero Worker/Router load verified.
No model semantic case was retried to obtain a success. These are complementary
records, **not** one rewritten all-green full-matrix report.

The 62 passing initial cases comprise 35 existing functional cases and 27 new
Chat cases: three request-error comparisons plus 12 semantic shapes in JSON/SSE.
These include strings, text-parts/Unicode, history, thinking and suppression,
actual JSON object/schema/choice outputs, auto/required/named tools and tool-none.
Full actual Worker input IDs agree with the official renderer/Router observations.

Three tool-choice forms × JSON/SSE × direct/Router produced **12 real tool loops**.
Each uses the model-generated unique ID and validated `synthetic_add` arguments,
executes bounded local `17 + 25`, inserts its actual `{"result":42}`, then obtains
the follow-up answer. No arbitrary tool execution, canned replacement call,
model retry, model switch or download. Multiple calls are CPU-replay tested but
not a real-model claim from this GPU slice.

Chat remains raw forwarding. Per-case actual Completion optimization counters
remain unchanged on Chat; there is no invented Chat prepared/fallback metric.
CPU native/SDK wire replay separately verifies exact raw bodies and ordinary SDK
headers. Prepared Chat is not implemented. JSON/SSE usage/finish semantics are
checked, while request IDs and model text need not match byte-for-byte.

Meaningful SSE timing is recorded separately for content/reasoning/tool events,
excluding empty role frames. Example single samples, not benchmark quantiles:
Router thinking case first reasoning 65.9 ms versus first content 1507.6 ms;
auto-tool case first tool delta 233.0 ms. Counting only content would misdescribe
the former's first useful output, and tools may have no content event.

## Requested 128 / 32 / 4 comparison

One round, 128 requests per arm, 32 prefixes each four times, concurrency 4.
Fixed interleaved group order; first occurrences cold, **no owner warmup or burn-in**.
Fresh Worker processes/cache epochs before every arm. Within an arm the Workers
persist, so the next three occurrences can reuse cache. APC enabled in all arms.
Actual prompt lengths 1057–1069 tokens, target shared prefix 768 tokens, exactly
32 output tokens, text Chat with thinking disabled. Full input oracle is outside
timing. Ingress bytes/order and complete prepared-token hashes match across arms.

| Production policy | Successful | TTFT p50 / p95 ms | E2E p50 / p95 ms | Output tokens/s | Token cache hit | Worker requests |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RR | 128/128 | 70.71 / 89.65 | 606.12 / 729.53 | 202.99 | 55.21% | 64 / 64 |
| cache_aware | 128/128 | 72.66 / 97.26 | 595.95 / 724.89 | 209.86 | 36.80% | 62 / 66 |
| KV-aware + load guard (CL) | 128/128 | 83.63 / 118.76 | 613.76 / 788.60 | 200.95 | 54.65% | 63 / 65 |

Compared with RR, cache_aware throughput is +3.39% with TTFT p50 +2.76%; CL
throughput is -1.01% with TTFT p50 +18.26%. One short synthetic round establishes
neither significance nor a general speedup/regression rate. All 384 measured
requests pass; none is removed. A performance harness PASS means its correctness
and comparison contract passed, **not** that KV-aware outperformed RR.

Interpretation grounded in this run:

- Repeating after 32 submissions with two round-robin destinations can preserve
  RR affinity. Its measured 55.21% token hit rate already matches CL's 54.65%:
  this trace gives CL no incremental cache savings. Prefix frequency alone does
  not imply an advantage over RR; this ordering was fixed before results.
- Worker queue means are 0.0098 / 0.0188 / 0.0186 ms for RR/cache_aware/CL;
  sampled maximum waiting is zero on both Workers. No material Worker queue
  backlog was observed. RR Router load is **UNKNOWN_UNMAINTAINED**, not zero.
- Worker mean prefill is 35.87 / 36.56 / 37.20 ms; mean decode is
  549.54 / 530.37 / 549.02 ms. Decode dominates this small-model shared-GPU
  workload; higher token cache hit did not produce faster prefill here.
- Router measured-window CPU is 0.60 / 0.50 / 2.05 seconds. Peak process RSS is
  45,816 / 50,668 / 1,412,836 KiB; CL includes its long-lived Python/vLLM runtime.
  This supports extra Router work, not an isolated per-request render duration
  or a memory-leak claim. No stage instrumentation or A/B render isolation was
  enabled in this comparison; **do not label the 12.9 ms TTFT difference as
  pure render cost**. Different placement/batching and one-round noise remain.

These are cumulative-window Worker histogram means, not differences of p95s or
request-correlated stage sums. One shared GPU is not multi-GPU capacity evidence.

## Public reproduction and remaining gates

Use the existing finite GPU runner with newly authorized own Workers and the
PID/source/native/config/log arguments from `kv-perf-2-gpu-runbook.md`. Add:

```bash
--native-source-candidate 2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110 \
--production-validation --arms product_rr product_cache_aware CL \
--requests 128 --groups 32 --scenarios repeat --trace-order interleaved \
--concurrencies 4 --rounds 1 --request-kind chat --prompt-format text \
--input-tokens 1024 --prefix-tokens 768 --output-tokens 32 \
--seed cmb-chat-serving-50577-20260927 --max-seconds 1800 \
--cache-state fresh-cohort --cohort-preparation-hook /absolute/approved-hook \
--allow-cohort-preparation --cohort-timeout 300
```

Full Chat: `render_bridge_gpu_validate.py --chat-agent` with automatic capabilities,
production validation and load guard. Focused recovery additionally sets
`--chat-cancel-only`; that slice must never be reported as the full semantic matrix.

New CPU follow-up: 32 performance harness tests, GPU/CUDA helper self-checks,
14 actual-vLLM/native Chat replay methods, changed-file Ruff, and selected-file
Black all pass. A wrong local CPU asset path attempt is retained separately;
the corrected run passes. No new Rust build is claimed for test/doc-only changes.
Inherited full-repository formatting/CI gates are not silently cleared.

Remaining: applicable full CI, human semantic/authorship/license review,
publication approval, broader model/parser/Agent coverage, and a separately
coordinated receiving-Worker admission contract for prepared Chat. Existing CT
stays default-off with immutable-cohort limitations. No prepared Chat speedup claim.

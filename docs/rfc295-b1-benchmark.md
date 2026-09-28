# RFC295-B1: official vLLM four-policy comparison

## Scope and identity

2026-09-28, one NVIDIA RTX 4080 SUPER (32,760 MiB), two independently
addressable DP/TP/PP=1 Workers, Qwen3-0.6B. This is one serial trial per policy,
not a saturation, multi-GPU or statistically significant speedup claim.
No Router/Worker production code was changed for the comparison.

Production native source: `3238b56d1bb265b8ea0d1f381e3f3f20ffd46099`.
Harness source: `807f1bfd13934a0c3ea8623dd2dedaf03605c005`.
The installed optimized production extension, with experimental performance
features disabled, is the same CPU-verified artifact in all four processes:

- Native SHA256: `f2613d588ffc8a551c7b7166c6dd900837306767643adc6e2a366e42fa13f898`.
- Wheel SHA256: `e59cc9e4870cd0f6a14902c02d315900092a21fce110125baee6040c2b50d553`.

vLLM 0.29.0 at `98dff2a81d747d1dba01a47f939f48c3526d4206` and the existing
separately proposed capability-export snapshot are reused read-only. This is
not stock zero-engine-change automatic discovery; no new patch was installed.
See [runtime dependencies and GPU identity](rfc295-b1-gpu-results.md).

## Method

Unmodified official `vllm bench serve`, OpenAI Chat, 150 requests, 50 distinct
prefixes, three consecutive occurrences each, seed 1234 and shuffle disabled.
Nominal input is 896 shared + 128 unique tokens; output is 128 tokens.
Concurrency 1, offered rate infinity, no generation readiness check or warmups,
ignore EOS, temperature 0.01, thinking disabled consistently for this workload.
The prefix-count argument is **50**, not the repetition count 3.

Each arm starts fresh Workers sequentially and a fresh Router. No physical
cache or Router history is carried between arms; disk compilation caches are
retained. Before benchmarking, Worker completion counters must both be zero.
All arms use the same logging, queue/rate settings and Worker sampler. Router
load counters are marked unknown; Worker metrics are the comparison source.

RR and standalone cache_aware are their actual product paths. Both KV arms use
the same-process vLLM input backend, raw Chat forwarding and accepted CL load
guard. Only the fourth enables bounded exact-history fallback (TTL 300 s).
That fallback is not the standalone string-prefix cache_aware policy and does
not create physical ownership or increase reusable-token estimates.

```sh
vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:19000 \
  --endpoint /v1/chat/completions --model Qwen/Qwen3-0.6B \
  --tokenizer "$MODEL_PATH" --dataset-name prefix_repetition \
  --num-prompts 150 --prefix-repetition-num-prefixes 50 \
  --prefix-repetition-prefix-len 896 --prefix-repetition-suffix-len 128 \
  --prefix-repetition-output-len 128 --disable-shuffle \
  --request-rate inf --max-concurrency 1 --ready-check-timeout-sec 0 \
  --num-warmups 0 --ignore-eos --temperature 0.01 \
  --percentile-metrics ttft,tpot,itl,e2el --seed 1234 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --save-result --save-detailed --result-dir "$RESULT_DIR" \
  --result-filename official.json
```

`MODEL_PATH` is the existing fixed ModelScope `09b42cad` snapshot, not a
download instruction. Actual commands and complete launch arguments are
retained in each arm's `bench-command.json` and `router-command.json`.
Rerunning requires fresh hardware authorization.

## Measurement definitions

TTFT/E2E/output throughput retain the official client's definitions. Official
Chat TTFT is the first SSE choice, not a custom semantic-content-token metric.
Output throughput is total output tokens divided by benchmark wall duration.
The official displayed peak-concurrency count may be 2 despite the explicit
client semaphore of 1: its inclusive one-second time buckets can contain two
adjacent serial requests. Worker running/waiting samples are retained separately.

Prefix hit rate is each Worker's delta of `vllm:prefix_cache_hits_total` divided
by its `vllm:prefix_cache_queries_total`: **token** hit rate, not the percentage
of requests with any hit. A Worker with zero query tokens has N/A hit rate.
Request share is that Worker's completion-count delta divided by all completed
requests; it is not CPU/GPU utilization or instantaneous in-flight load.

## Results

All four arms completed 150/150 requests with zero failures, exactly 19,200
output tokens per arm and 128 output tokens per request. Actual input was
1036–1037 tokens (mean 1036.02) in every arm: nominal length plus mean 12.02
tokens of template/text-token round-trip overhead, not exactly 1024 on the wire.

| Policy | Mean TTFT ms | Median / P99 TTFT ms | Output tok/s | Mean / P99 E2E ms | Mean TPOT ms |
|---|---:|---:|---:|---:|---:|
| RR | 85.97 | 82.71 / 113.08 | 47.40 | 2699.89 / 2836.39 | 20.582 |
| cache_aware | 74.81 | 73.92 / 96.67 | 48.27 | 2651.00 / 2756.26 | 20.285 |
| kv_aware + CL | 94.29 | 92.05 / 153.19 | 47.17 | 2712.64 / 2862.21 | 20.617 |
| kv_aware + CL + history fallback | 92.29 | 90.50 / 134.78 | 48.97 | 2613.41 / 2695.20 | 19.851 |

| Policy | W0 hit/query tokens (rate) | W1 hit/query tokens (rate) | W0:W1 requests | Request share W0:W1 |
|---|---|---|---|---|
| RR | 22,400 / 77,701 (28.83%) | 22,400 / 77,702 (28.83%) | 75:75 | 50%:50% |
| cache_aware | 89,600 / 155,403 (57.66%) | 0 / 0 (N/A) | 150:0 | 100%:0% |
| kv_aware + CL | 44,800 / 77,702 (57.66%) | 44,800 / 77,701 (57.66%) | 75:75 | 50%:50% |
| kv_aware + CL + history fallback | 44,800 / 77,702 (57.66%) | 44,800 / 77,701 (57.66%) | 75:75 | 50%:50% |

Worker sampling had no errors. Mean sampled running-request counts W0/W1
were respectively 0.479/0.471, 0.933/0, 0.465/0.459 and 0.466/0.476; each
Worker's maximum was 1 except the idle cache_aware Worker (0). All sampled
waiting counts were zero. These are sampled observations, not proof that no
short scheduling wait ever occurred. Router load counters remain unknown.

## Interpretation and limits

- Relative to RR, standalone cache_aware had 12.98% lower mean TTFT, 1.84%
  higher output throughput and 1.81% lower mean E2E, but sent all requests to
  Worker 0. Existing low-match `min_by_key(worker.load())` selection prefers
  the first eligible Worker when serial requests leave both idle; affinity
  retains repeated prefixes there. No balancing modification was introduced.
- KV routing doubled physical token-hit rate from 28.83% to 57.66% while
  distributing 75 requests to each Worker. Nonetheless its mean TTFT was
  9.68% higher than RR, throughput 0.47% lower and E2E 0.47% higher. Better
  locality alone did not produce a net speedup in this small-model/C=1 run.
- With history enabled, TTFT remained 7.35% higher than RR; throughput was
  3.31% higher and E2E 3.20% lower. Both KV arms have identical physical-hit
  totals and request allocation. This run has no per-attempt history-stage
  instrumentation and does **not** establish that history caused the gain.
  Physical ownership is primary; enabling fallback does not prove it was used.
- The E2E difference between the two KV arms is about 99.23 ms, whereas mean
  TTFT differs by only 2.00 ms. Most of that difference is after the first
  chunk: mean TPOT changed by about 0.766 ms over 127 intervals. It must not
  be presented as a measured renderer or history lookup optimization.
- Decode accounts for roughly 2.5–2.6 seconds of the 2.6–2.7 second E2E. The
  workload's 128 output tokens limit how much prefill savings change total
  throughput. RR/cache_aware also omit Router vLLM rendering, unlike the KV
  paths; this comparison cannot isolate rendering cost. No new stage trace
  was enabled, and no precise extra-cost attribution is claimed here.
- Fixed grouped ordering, one trial, one shared GPU, small model and C=1
  cannot establish saturated two-node capacity, multi-card scaling, CL under
  overload, robustness to shuffled traffic, or statistical significance.
  Trial order and runtime variability remain confounders.

Mean TTFT for the first/second/third occurrence of each prefix was
93.10/84.69/80.13 ms (RR), 77.36/74.74/72.33 (cache_aware),
103.11/90.06/89.70 (KV), and 99.68/89.84/87.35 (KV+history).
These are aggregates over the saved official samples, not a second benchmark.

## Evidence and closeout

`cmb-rfc295-b1-review/bench-150-50/remote-final/bench-150-50/` retains official
detailed results, unmodified benchmark-source hashes, exact launch commands,
before/after raw metrics, per-Worker descriptors/epochs, mapped native identity,
load samples, logs, analysis and closeout. Each arm's saved-result hash is
verified by the analysis. Archive SHA256:
`097d25c446623f3f7a6036cca8dfced4e51269bc440cd6c68d20c365e300d6f1`.

All task-owned Worker/Router/client processes exited at approximately 22:18
Asia/Shanghai, before both the original and extended 23:10 deadline. Follow-up
checks found no tracked process, no GPU compute process, all six task ports
closed and GPU memory at 1 MiB. Remote source remained clean. The minimal
image lacks `ss`; socket probes and process/GPU checks were used instead.
No models/caches were deleted, no new code or patch deployed, and no push/PR
performed. Existing [CI and human release gates](rfc295-b1-cpu-results.md)
remain; this performance comparison does not close them.

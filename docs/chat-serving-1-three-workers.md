# Three-Worker finite policy comparison

Follow-up to `chat-serving-1-gpu-results.md`, under a fresh explicit user
confirmation for three owned Workers and isolated KV event port 5559.
The existing two-Worker results/evidence are preserved, not overwritten.

## Results

All **384/384 measured requests succeeded**. Post-window full token-ID probes
also passed for all three direct Workers and the CL Router. Offline recomputation
verified request identities, exact four occurrences per prefix, response usage,
percentiles and all per-Worker counters. Complete ingress and prepared-token
hashes match across all three policies **and the previous two-Worker run**.

| Three-Worker policy | TTFT p50 / p95 ms | E2E p50 / p95 ms | Output tokens/s | Token cache hit | Worker completed requests |
| --- | ---: | ---: | ---: | ---: | --- |
| RR | 66.58 / 86.58 | 577.28 / 704.87 | 210.32 | 18.40% | 42 / 43 / 43 |
| cache_aware | 62.72 / 78.99 | 553.32 / 690.82 | 221.15 | 28.19% | 33 / 61 / 34 |
| KV-aware + load guard (CL) | 77.30 / 105.63 | 599.24 / 718.09 | 206.21 | 53.49% | 40 / 42 / 46 |

Compared with the same-run RR, cache_aware output throughput is **+5.15%** and
median TTFT **-5.80%**. CL throughput is **-1.95%**, median TTFT **+16.10%**.
These are one-round observations, not statistically established differences.
The policy comparison passes its correctness contract, not a required speedup.

The repeated-prefix count remains four, but 32 submissions no longer align
with a three-destination RR cycle. RR's token hit ratio drops from 55.21% in
the previous two-Worker trace to 18.40%; CL retains 53.49%. This shows a real
cache-locality difference, **but higher cache hit still does not guarantee lower
latency or greater throughput in this workload**.

Measured diagnostics, policy order RR / cache_aware / CL:

- Router CPU in the measured window: **0.55 / 0.53 / 2.06 seconds**.
- Peak Router process RSS: **45,732 / 47,108 / 1,417,120 KiB**, including CL's
  long-lived Python/vLLM runtime; not per-request memory or a leak diagnosis.
- Worker mean queue: **0.0132 / 0.0176 / 0.0239 ms**. Sampled maximum waiting
  is zero on every Worker. RR Router load remains UNKNOWN_UNMAINTAINED.
- Worker mean prefill: **36.50 / 34.67 / 37.03 ms**; mean decode:
  **535.84 / 511.56 / 538.30 ms**. These are before/after cumulative histogram
  windows across all three Workers, not quantile subtraction or additive stages.

CL's extra observed Router CPU and the decode-dominated small-model/shared-card
execution are consistent with its lack of end-to-end gain. This is not an
isolated render-cost experiment: do not label the 10.72 ms median TTFT difference
as pure render cost. Placement, batching and single-round variation remain.
cache_aware's request distribution is less even yet it is fastest here; request
count balance alone is not a performance verdict.

For context only, previous two-Worker output rates were RR 202.99, cache_aware
209.86, CL 200.95 tokens/s. All three are higher in this run, but per-Worker
memory allocation, batching and sampling differ, and each comparison has only
one round. No causal scaling or multi-GPU benefit is established.

All four owned three-Worker cohorts reached EXITED without cleanup errors, and
all three Router children exited successfully. GPU returned to 0 MiB by about
00:02 on 2026-09-28; task HTTP/metrics/event ports were independently confirmed
free at 00:03:29, well before the 01:00 cutoff. No model/cache deletion, package
or Worker patch installation, unrelated process operation, push or PR.
Preserved evidence archive SHA-256:
`bc00a624428a9e91b809c30bb56a7af445f49c724a1237ecc26f5232eebf6557`.
Its remote/local hashes match. The original zero-phase preflight FAIL is retained.

## Scope and reproducibility

This is a test-runner extension, not a Router policy or Worker protocol change.
Public `scripts/kv_capabilities_performance.py` retains two Workers by default
and optionally accepts `--worker-count 3` with all third-Worker identity fields.
Full Chat semantic/cancellation harness still has its original two-Worker CLI;
no three-Worker Agent-matrix claim is made here.

Final executable test candidate: `0daedb010faedfe95bd3bc2232508e2978b72c6d`.
Actual feature-off production native source remains
`2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110`, SHA-256
`f17ccc17cc343b5cea68145a3e73d6442cbd38e2413f6f9a75606a23928e683b`.
The source/native non-product diff proof is retained; no recompilation claim.
Commit `c7bde73` adds third-Worker validation; `0daedb0` removes the inherited
two-only runtime-version gate while retaining exact version checks for every
Worker. Its initial preflight failure is retained and contains **zero measured
phases**. No measured performance result was retried or selected for being faster.

Same Qwen3-0.6B assets, pinned vLLM 0.29/capability proposal, official parsers,
same one RTX 4080 SUPER and native as the two-Worker run. Three independent
DP/TP/PP=1 Workers share that single GPU. Per-Worker memory utilization is 0.27
(aggregate 0.81), versus 0.40 each (aggregate 0.80) in the previous two-Worker run.
This is not a three-GPU test or a controlled pure Worker-count scaling claim.

Exactly one round per policy: product RR, product cache_aware, production
KV-aware + load guard (CL). Each measured arm has 128 raw Chat requests,
32 prefixes each four times, fixed interleaved order, concurrency 4, thinking
disabled and 32 output tokens. Seed `cmb-chat-serving-50577-20260927` and all
request settings unchanged. Each arm starts fresh Worker processes and cache
epochs; no owner-prefix warmup or Router burn-in. Full token-ID probes run only
outside timing; post-window probes cannot warm the next freshly restarted arm.

Third-Worker checks include actual runtime version, complete tokens, compatible
capability contract, distinct/new event epoch, HTTP/EngineCore lineage, log file
identity and endpoint binding. CPU tests cover wrong third-Worker tokens,
contract, epoch, duplicate PID and wrong runtime version. All 36 performance
contract tests and both helper self-checks pass; changed-file Ruff and selected
test-file Black pass (inherited Python-version warning retained). No Rust or
production Python edits; no full CI or new native build claim.

To reproduce, retain the two-Worker command's exact source/native/config/log
arguments, update render configuration to all three Worker URLs and add:

```bash
--worker-count 3 --worker2 http://127.0.0.1:8102 \
--worker2-pid "$WORKER2_PID" --engine2-pid "$ENGINE2_PID" \
--worker2-log /absolute/task/worker2.log \
--event2 tcp://127.0.0.1:5559 --publisher2 'tcp://*:5559'
```

The separately authorized fresh-cohort hook must also return `worker2_pid` and
`engine2_pid`. This does not authorize deployment, patch installation or port
exposure. New SSH/GPU authority is required for any subsequent window.

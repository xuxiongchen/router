# Official long-input comparison: concurrency 1 and 2

## Outcome and frozen identity

All eight arms passed on 2026-09-28: each completed 150/150 Chat requests,
zero failures, exactly 19,200 output tokens, 128 per response. Every actual
input was 10,252 tokens (nominal 10,240 plus 12 tokens of template/round-trip
overhead). No truncation is inferred from a nominal argument: actual usage
length arrays were checked across all arms. No production changes or rebuild.

Production source candidate: `3238b56d1bb265b8ea0d1f381e3f3f20ffd46099`.
Remote clean harness source: `807f1bfd13934a0c3ea8623dd2dedaf03605c005`.
Installed production native SHA256:
`f2613d588ffc8a551c7b7166c6dd900837306767643adc6e2a366e42fa13f898`.
Wheel SHA256:
`e59cc9e4870cd0f6a14902c02d315900092a21fce110125baee6040c2b50d553`.
Actual mapped native identity was checked for every Router process; final
on-disk native remained identical. Experimental performance features and
per-attempt stage tracing were disabled.

One 4080 SUPER, 32,760 MiB, driver 595.71.05, GPU UUID
`GPU-48d73674-8d81-b607-e377-cdb6add02b5a`; two independent DP/TP/PP=1 Workers
on that card. This UUID differs from the earlier 1024-input comparison despite
the same reported GPU model: do not attribute cross-run differences solely to
input length. vLLM 0.29.0, torch 2.13.0, transformers 5.17.0, tokenizers 0.23.2,
pydantic 2.13.5, pyzmq 27.2.0. Same existing capability-export snapshot, reused
read-only; no Worker patch installed and no stock zero-engine-change claim.

## Method

Unmodified official `vllm bench serve`, 150 requests, 50 prefixes, each repeated
three times consecutively. Nominal input **10112 shared + 128 unique tokens**,
output 128; seed 1234, shuffle disabled, thinking disabled, temperature 0.01,
ignore EOS, infinite offered rate, zero warmups and no generation readiness
probe. Four policies at concurrency 1, then the same four at concurrency 2.
Each arm gets fresh Workers (started sequentially) and a fresh Router; no
physical cache or history survives across arms. Disk compilation caches remain.

Both Worker and same-process vLLM input-backend max-model-len changed from
4096 to **12288** to fit the workload. Other Worker settings stayed unchanged:
eager execution, memory utilization 0.40 each, block size 16, sha256_cbor,
fixed Qwen3-0.6B ModelScope snapshot 09b42cad. All 16 Worker initializations
reported exactly **103,648 KV tokens** of capacity each.

RR/cache_aware use real product paths without Router vLLM rendering. Both KV
arms use the accepted input backend, raw Chat forwarding and CL load guard;
only the fourth enables bounded exact-history fallback (TTL 300 s). This is
not prepared Chat, an isolated-render microbenchmark or a load-guard ablation.

The [earlier benchmark command](rfc295-b1-benchmark.md#method) is unchanged
except `--prefix-repetition-prefix-len 10112` and `--max-concurrency 1` or `2`.
The saved `bench-command.json` / `router-command.json` per arm, task-local
`start_worker.sh`, `render-config.json` and `run_bench.py` specify exact launches.
The same existing sampler collects Worker metrics every 0.5 seconds; Router
load counters are explicitly unknown. No forced balancing, tuning or repeats.
Future hardware execution requires fresh authorization.

## Client performance

Official mean TTFT, output throughput and E2E; TTFT is first SSE choice, not
a custom semantic-content-token timestamp. Throughput is output tokens divided
by whole benchmark duration. Official detailed samples and tails are retained.

| Concurrency | Policy | Mean TTFT ms | P99 TTFT ms | Output tok/s | Mean E2E s | P99 E2E s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | RR | 349.49 | 482.80 | 42.50 | 3.011 | 3.279 |
| 1 | cache_aware | 252.49 | 458.14 | 43.95 | 2.911 | 3.180 |
| 1 | kv_aware + CL | 325.43 | 546.75 | 42.55 | 3.008 | 3.278 |
| 1 | KV + CL + history | 341.43 | 587.17 | 42.40 | 3.018 | 3.282 |
| 2 | RR | 398.01 | 685.29 | 82.68 | 3.081 | 3.378 |
| 2 | cache_aware | 405.27 | 734.75 | 81.93 | 3.122 | 3.420 |
| 2 | kv_aware + CL | 377.27 | 803.54 | 80.65 | 3.172 | 3.592 |
| 2 | KV + CL + history | 384.93 | 724.03 | 80.23 | 3.178 | 3.565 |

Relative to the RR baseline at the **same concurrency**:

| Concurrency | Policy | Mean TTFT change | Output throughput change | Mean E2E change |
|---:|---|---:|---:|---:|
| 1 | cache_aware | -27.75% | +3.43% | -3.31% |
| 1 | kv_aware + CL | -6.89% | +0.13% | -0.12% |
| 1 | KV + CL + history | -2.31% | -0.21% | +0.22% |
| 2 | cache_aware | +1.82% | -0.91% | +1.35% |
| 2 | kv_aware + CL | -5.21% | -2.46% | +2.95% |
| 2 | KV + CL + history | -3.29% | -2.96% | +3.17% |

## Actual cache hits and request distribution

Hit rate is each Worker's hit-token/query-token counter delta, not the fraction
of requests with any hit. Allocation uses actual Worker completion deltas,
not Router load counters. Worker 0/1 mean ports 8100/8101, irrespective of
asynchronous Router registration order. N/A denotes no query tokens.

| C | Policy | W0 hit/query tokens (rate) | W1 hit/query tokens (rate) | W0:W1 requests | W0:W1 share |
|---:|---|---|---|---|---|
| 1 | RR | 252800/768900 (32.88%) | 252800/768900 (32.88%) | 75:75 | 50%:50% |
| 1 | cache_aware | 0/0 (N/A) | 1011200/1537800 (65.76%) | 0:150 | 0%:100% |
| 1 | KV | 505600/768900 (65.76%) | 505600/768900 (65.76%) | 75:75 | 50%:50% |
| 1 | KV + history | 505600/768900 (65.76%) | 505600/768900 (65.76%) | 75:75 | 50%:50% |
| 2 | RR | 252800/768900 (32.88%) | 252800/768900 (32.88%) | 75:75 | 50%:50% |
| 2 | cache_aware | 242688/758648 (31.99%) | 262912/779152 (33.74%) | 74:76 | 49.33%:50.67% |
| 2 | KV | 485376/748396 (64.86%) | 505600/789404 (64.05%) | 73:77 | 48.67%:51.33% |
| 2 | KV + history | 505600/768900 (65.76%) | 505600/768900 (65.76%) | 75:75 | 50%:50% |

### Important standalone cache_aware clarification

In this candidate, `ChatCompletionRequest::extract_text_for_routing()` in
`src/protocols/spec.rs` returns session_id or empty text, **not message text**.
The HTTP Router passes that to standalone cache_aware. Official benchmark
requests have no session_id; the policy's empty-input match score is zero,
so its existing minimum-in-flight-load selection applies. At C=1, repeated
idle ties concentrate traffic on one eligible node, incidentally preserving
locality. At C=2, it distributes almost evenly and its overall token-hit rate
equals RR's 32.88%. This is an actual product comparison, not proof of
standalone Chat message-prefix matching. The previous report's affinity
explanation was corrected; no routing code was changed for favorable results.
KV selection instead consumes the existing exact tokens and physical events.

## Worker timing and interpretation

Below are Worker-reported histogram sum deltas divided by 150 completions,
aggregated across both nodes. They are not a disjoint client-side latency
decomposition or an isolated CPU/GPU/render cost measurement.

| C | Policy | Worker mean prefill ms | Worker mean decode ms |
|---:|---|---:|---:|
| 1 | RR | 219.37 | 2683.40 |
| 1 | cache_aware | 131.54 | 2678.64 |
| 1 | KV | 132.57 | 2700.97 |
| 1 | KV + history | 132.09 | 2694.83 |
| 2 | RR | 284.91 | 2701.22 |
| 2 | cache_aware | 288.38 | 2735.53 |
| 2 | KV | 171.26 | 2813.07 |
| 2 | KV + history | 167.30 | 2811.42 |

- Physical reuse and reduced prefill are observed, not merely advisory Router
  scores. Nevertheless, C=1 KV output throughput is effectively unchanged;
  C=2 KV output throughput is lower and its P99 TTFT is worse than RR.
- At C=2, KV saves about 114 ms of Worker-reported mean prefill versus RR but
  adds about 112 ms of reported mean decode. KV+history shows a similar
  tradeoff. These observations help explain why prefill savings alone do not
  guarantee total-throughput gains; they do not prove why decode slowed.
- Both KV arms have sampled per-Worker running-request maxima of 2 at C=2;
  RR/cache_aware maxima were 1 each. Affinity can co-locate concurrent requests
  within CL's accepted slack. Shared-GPU contention, batching and timing are
  possible factors, not isolated causal findings from this experiment.
- All eight arms have zero Worker preemption deltas and no sampler errors.
  All sampled waiting gauges are zero, but continuous queue-time histograms
  reveal short waits: C=2 history W1 totals 0.173 seconds over its 75 requests.
  Do not claim absence of all queuing based on half-second samples.
- History adds no demonstrated throughput advantage here. This trace has
  physical hits, unlike the prior zero-block history correctness fixture.
  No per-attempt history-use trace was enabled; enabling the setting is not
  proof that fallback caused any observed allocation/timing difference.
- C=2 increases total throughput for every strategy, but this remains a
  small-model, one-card/two-Worker, grouped-prefix experiment. One trial per
  arm cannot establish statistical significance, peak capacity, multi-card
  scaling or performance on shuffled/many-client workloads.

## Evidence and closeout

Local evidence: `cmb-rfc295-b1-review/bench-10240-c1-c2/remote-final/` includes
official detailed JSON, client/Router/Worker logs, raw before/after metrics,
per-Worker capabilities/epochs, launch commands, native identity, load samples,
runtime identity, analysis and closeout. Benchmark source-file hashes match
the earlier unmodified official vLLM implementation. Archive SHA256:
`1400a5ec735e71b89f51d1767d95a533251fe6fcb534c557d869dcca91b32c10`.

The controller and all task-owned processes exited about 23:40 Asia/Shanghai,
before the user-authorized 2026-09-29 00:00 cutoff. Closeout confirmed all 24
tracked Worker/Router process IDs absent, no GPU compute processes, task ports
8100/8101/5557/5558/19000/19001 closed and GPU memory 1 MiB. No cache/model
deletion, new package installation, cloud patch, native rebuild, push or PR.
Source clean on the GPU host; only local documentation changes. Existing
human review, maintainer-interface agreement and CI release gates remain.

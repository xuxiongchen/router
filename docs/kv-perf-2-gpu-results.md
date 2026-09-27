# Perf-2 finite GPU results — 2026-09-27

This report separates functional correctness, measured performance and release
readiness. Both new options remain **off by default**. This is one shared GPU
with two DP=1 Workers, not two independent GPU devices.

Final finite matrix: **13 invocations, 56 phases, 3808/3808 measured requests,
zero measured errors**, with every paired cell independently checked from saved
literal ingress, full token arrays, raw timings and native identity. This is a
functional/evidence PASS, **not** a universal performance or SLO PASS. Hotspot
CLT improves C0 in all three production rounds; true product RR remains faster
in TTFT there, and throughput superiority to RR is not consistent across workloads.

## Artifact and environment

- Exact compiled, deployed and measured candidate:
  `2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110`.
- Production wheel SHA256:
  `4c7bf72c70c1311d9fe9883bfa1767434a86cf1b95b806f49447779695b38d27`.
- Actual production native SHA256:
  `f17ccc17cc343b5cea68145a3e73d6442cbd38e2413f6f9a75606a23928e683b`.
- Ablation wheel SHA256:
  `e84d026ed0e4fa06b9fa2b880a0bd80cf00e358b165250d1f7cd403c1cea4f8c`.
- Actual ablation native SHA256:
  `eec963959d61272400a89d81c7630fcfb45c9fae51afdac7c5852c0795b325c4`.
- Cargo.lock SHA256:
  `000665f280cab3ba36fd6392afd5e9c39f72fcb10d35521e8161ded0d1b4d614`.
- Original optimized CPU-build wheels were transferred and actually installed
  into separate new GPU-host venvs, not recompiled or relabeled. Clean remote
  checkout, build manifests, installed Python files, live native mappings and
  feature handshakes were checked. Later documentation commits are not compiled
  into these artifacts. Production has the experimental feature disabled.
- RTX 4080 SUPER, 32760 MiB, UUID
  `GPU-26a10dcd-fe18-dd86-1f13-a7015a4334f7`, driver 580.76.05.
  Fresh preflight identifies this actual device; no older device receipt reused.
- Linux x86_64, Python 3.12.3, vLLM 0.29.0, torch 2.13.0+cu130,
  transformers 5.17.0, tokenizers 0.23.2, pydantic 2.13.5, orjson 3.12.0.
  128 visible CPU threads, cgroup quota 16 CPUs; no affinity or render-pool change.
- Process-local system libstdc++ preload for Router/harness only; no system
  library replaced. Existing capability proposal runtime/model assets reused
  read-only, with pinned proposal file hashes checked. No new Worker patch.
- Qwen3-0.6B from fixed ModelScope commit
  `09b42cad3d112e832108974449ccb5e8e0f5b5d1`; SmolLM2-135M-Instruct for correctness.
  Actual config/tokenizer receipts retained. This is not a model download or
  a claim to have retested other model sizes.

The local evidence root is `cmb-kv-perf-2-review/gpu-50577/`. `PLAN.md` was
frozen before testing; `PRODUCTION-DECISION.md` records selection and remaining
samples before production performance. Raw per-request timings, literal bounded
ingress, full Worker prompt IDs, descriptors/epochs, process start identities,
metric snapshots, Router exits and invocation records are retained.

## Functional gates

The actual production artifact with **both** options enabled passed Qwen 35/35
and Smol 35/35 GPU cases. These include prepared Completion JSON/SSE, ordinary
generation and stop termination, exact full Worker IDs and deterministic output/
finish/core usage versus the text path, raw echo/logprobs fallback, invalid
requests with no dispatch, and active prepared-stream cancellation with lease
release. They also retain generic Chat and true warmed-owner routing tests.

The N=464 boundary has 29 stored matched blocks but only 448 reusable tokens;
both models' actual warm Worker counter is 448 hit / 464 query tokens. Matched
blocks, input IDs and reusable tokens are distinct and not a complete cache key.

Output equivalence excludes request IDs/timestamps, SSE chunk boundaries,
cache-usage details and floating-point logprobs. Full raw responses remain.
These finite deterministic cases do not prove all API/sampling/model behavior.

The supplemental Qwen production invocation passed **47/47**, including the
same 35-case baseline plus 12 added existing public Chat fixtures: thinking
on/off, reasoning-effort none, dropping historical thinking, and both tool
schema/history key orders, each JSON and SSE. Saved actual generation prompt
IDs equal the real in-process vLLM result. All 12 have Chat qualification false
for Completion token input and preserve the original forwarding path. The
fixture-only adapter's bytes/hash are retained; Router/Worker code is unchanged.
This proves these inputs and transports, not complete generated tool arguments,
all reasoning-parser behavior, or Chat tokens-in/out. Repeated baseline cases
are not counted as new unique coverage.

## Exploration: one ablation artifact, all five arms

C0: render+KV, original forwarding. CL: load guard only. CT: eligible Completion
prepared IDs only. CLT: both. RR is the true product round-robin path, not a
different synthetic shared path used to estimate pure render cost.

Each row below is TTFT p50 in ms / completed requests per second. One round,
32 measured requests/arm, output 32 tokens, approximately 1K input and 768
shared prefix where applicable. Actual first locality/C4 input is 1048–1052
tokens. All four cells completed: 20 phases, 640/640 requests, no errors.

| Workload | Product RR | C0 | CL | CT | CLT |
| --- | --- | --- | --- | --- | --- |
| Locality C4 | 59.23 / 6.344 | 82.54 / 5.752 | 77.40 / 5.856 | 87.01 / 5.820 | 67.21 / 6.178 |
| Cold C4 | 59.01 / 6.328 | 76.80 / 6.020 | 73.24 / 5.741 | 69.93 / 6.172 | 75.97 / 5.953 |
| Locality C1 | 57.34 / 1.710 | 66.15 / 1.818 | 60.90 / 1.817 | 58.10 / 1.800 | 58.93 / 1.749 |
| Cold C1 | 56.41 / 1.796 | 69.20 / 1.600 | 67.39 / 1.643 | 60.90 / 1.576 | 59.64 / 1.700 |

Locality/C4 CLT versus C0: TTFT p50 -18.58%, RPS +7.40%; versus RR:
TTFT p50 +13.47%, RPS -2.61%. Cold/C4 CLT versus C0 has slightly worse RPS;
CT alone is better there. Locality/C1 CLT also loses RPS versus C0. These are
exploration outcomes, not proof of universal benefit or stable p95/p99/SLO.

Locality/C4 real prefix hit ratios: RR/CL/CLT 64.99%, C0/CT 74.28%. Sampled
Worker running peaks fall from 4/4 (C0) to 3/3 (CLT), while Worker waiting
samples stay zero. C0 completions are 16/16; CLT 17/15. Aggregate request-count
fairness is not instantaneous balance. The guard sees Router leases, not GPU
capacity. RR's unmaintained Router load is **unknown**, never substituted with 0.

Prepared counters prove CT/CLT took the new path: 32 prepared, zero raw per
Completion phase; C0/CL have 32 raw. In locality/C4 the total backend body
shrinks from 208920 to 178670 bytes (~14.5%); that is workload-specific, not an
assumption that token arrays are always smaller. No request-level remote render
or metadata call was added.

## Actual production wheel: three-round locality confirmation

Nine phases, 128 requests/arm/round, 1152/1152 successful. Order was
RR/C0/CLT, RR/CLT/C0, CLT/RR/C0. All three cells pass saved-literal/full-token/
metric offline consistency checks. Timing/trace disabled, experimental feature
disabled in the actual loaded native. TTFT and E2E columns are milliseconds.

| Round | Arm | TTFT p50 | TTFT p95 | E2E p50 | RPS | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | RR | 57.91 | 74.47 | 575.14 | 6.825 | 218.41 |
| 1 | C0 | 72.16 | 126.26 | 578.43 | 6.447 | 206.31 |
| 1 | CLT | 63.07 | 72.52 | 547.24 | 7.086 | 226.74 |
| 2 | RR | 63.82 | 77.30 | 556.14 | 6.947 | 222.29 |
| 2 | C0 | 70.23 | 120.26 | 570.40 | 6.541 | 209.32 |
| 2 | CLT | 65.00 | 88.31 | 560.33 | 6.839 | 218.86 |
| 3 | RR | 62.38 | 69.85 | 551.51 | 6.917 | 221.34 |
| 3 | C0 | 75.58 | 128.93 | 591.52 | 6.406 | 205.00 |
| 3 | CLT | 65.72 | 97.95 | 570.03 | 6.717 | 214.96 |

CLT versus C0, rounds 1/2/3: TTFT p50 **-12.60/-7.45/-13.04%**, RPS
**+9.90/+4.56/+4.86%**. This finite hotspot workload demonstrates a repeatable
improvement over this candidate's original KV path, not over every alternative.
Versus real RR: TTFT p50 **+8.91/+1.84/+5.36%**, RPS **+3.81/-1.54/-2.88%**.
It does **not** demonstrate a consistent win over RR or a production SLO.
Do not combine these phase p95s into a fictitious pooled percentile.

C0 hit ratios by round are 74.30/74.23/74.60%; CLT 71.98/71.92/72.27%;
RR 71.97/71.89/72.27%. RR learns the shared prefixes during the longer window,
so the ~9.3 percentage-point cache advantage in 32-request exploration shrinks
to ~2.3 points here. Higher hit ratio alone again does not predict better latency.
All CLT phases have 128 prepared / zero raw attempts; C0 has the inverse.

### Production cold regression check

One predeclared round, 128 requests/arm, 384/384 successful, actual prefix hit
ratio **0** in every arm. Same production artifact and independently restarted
cohorts; strict offline comparison passes. This is not three-round cold evidence.

| Arm | TTFT p50 | TTFT p95 | E2E p50 | RPS | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| RR | 63.20 | 89.74 | 552.35 | 6.939 | 222.04 |
| C0 | 73.37 | 98.11 | 558.17 | 6.748 | 215.92 |
| CLT | 66.08 | 90.28 | 554.94 | 6.851 | 219.23 |

CLT still loses to RR: TTFT p50 +4.55%, RPS -1.27%. It improves C0 here, but
the earlier cold 32-request exploration had an RPS regression versus C0; both
results remain. No universal cold-traffic improvement or non-regression promise.

### Original supplemental two-arm windows (retained)

These finished before using the extended 14:30 authorization. Each has one
round of 32 measured requests/arm; natural also has 32 routed burn-in requests
per arm, on the same measured cohort after burn-in. All six phases/192 requests
passed strict offline checks. C0 was absent under the original time budget.
The later complete three-arm runs are separate evidence, not spliced controls.

| Window | Arm | Actual input tokens | TTFT p50 | TTFT p95 | E2E p50 | RPS | Hit % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Natural | RR | 1052–1057 | 62.74 | 77.10 | 544.98 | 7.275 | 74.44 |
| Natural | CLT | 1052–1057 | 60.07 | 78.11 | 548.62 | 6.979 | 74.44 |
| Long | RR | 3096–3100 | 93.09 | 185.42 | 697.51 | 5.602 | 58.18 |
| Long | CLT | 3096–3100 | 91.07 | 206.68 | 698.14 | 5.596 | 58.18 |
| Text Chat | RR | 1060–1064 | 66.81 | 98.80 | 660.86 | 6.006 | 64.59 |
| Text Chat | CL | 1060–1064 | 80.89 | 171.14 | 666.37 | 5.978 | 64.59 |

Natural CLT lowers TTFT but loses ~4.07% RPS versus RR; long-input throughput
is essentially unchanged, not a demonstrated gain; Chat remains materially
slower in TTFT versus RR. These short windows are not stability or SLO evidence.

### Extended authorization: three-round production cold confirmation

After the user extended the deadline to 14:30 and prioritized complete controls,
`EXTENDED-PLAN.md` froze additional runs before execution. This new cold run has
128 requests/arm/round, order RR/C0/CLT, RR/CLT/C0, CLT/RR/C0; 1152/1152 pass.
All three cells pass independent saved-literal/raw-timing/native consistency
checks and all actual prefix hit ratios are zero. The earlier runs above remain.

| Round | Arm | TTFT p50 | TTFT p95 | E2E p50 | RPS | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | RR | 62.17 | 73.58 | 558.23 | 6.814 | 218.05 |
| 1 | C0 | 76.08 | 97.24 | 562.17 | 6.784 | 217.07 |
| 1 | CLT | 68.45 | 83.11 | 549.08 | 6.961 | 222.74 |
| 2 | RR | 63.71 | 89.05 | 555.46 | 6.771 | 216.66 |
| 2 | C0 | 72.66 | 88.94 | 559.57 | 6.905 | 220.96 |
| 2 | CLT | 65.61 | 79.76 | 547.77 | 6.959 | 222.70 |
| 3 | RR | 54.30 | 68.19 | 547.81 | 7.098 | 227.13 |
| 3 | C0 | 69.19 | 97.09 | 560.69 | 6.678 | 213.71 |
| 3 | CLT | 69.49 | 107.98 | 553.08 | 6.860 | 219.51 |

CLT versus C0, rounds 1/2/3: TTFT p50 -10.03/-9.70/**+0.44%**, RPS
+2.61/+0.79/+2.71%. Versus RR: TTFT p50 **+10.11/+2.98/+27.97%**, RPS
+2.15/+2.79/**-3.35%**. The third round removes even the C0 TTFT advantage;
neither a cold latency guarantee nor consistent superiority to RR is supported.
Worker/cache/artifact prerequisites still pass: performance regressions are
valid measurements, not functional failures to rerun until they disappear.

### Extended complete supplemental controls

Each new invocation has all three arms, a fresh cohort per arm, one round of
32 measured requests/arm, and all checks enabled. Natural retains 32 routed
burn-in requests per arm on the subsequently measured cohort. All 9 phases /
288 measured requests pass the independent offline checks. These are finite
supplemental comparisons, not repeated tail-latency or steady-state guarantees.

| Window | Arm | TTFT p50 | TTFT p95 | E2E p50 | RPS | Output tokens/s | Hit % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural | RR | 60.09 | 76.18 | 548.67 | 7.261 | 232.35 | 74.44 |
| Natural | C0 | 75.39 | 99.20 | 550.78 | 7.204 | 230.52 | 74.44 |
| Natural | CLT | 68.19 | 87.16 | 552.31 | 7.144 | 228.61 | 74.44 |
| Long | RR | 89.53 | 185.52 | 678.69 | 5.928 | 189.70 | 58.18 |
| Long | C0 | 132.92 | 255.01 | 749.04 | 5.212 | 166.77 | 66.49 |
| Long | CLT | 89.99 | 232.09 | 682.44 | 5.667 | 181.34 | 58.18 |
| Text Chat | RR | 72.09 | 128.84 | 659.05 | 6.153 | 196.90 | 64.59 |
| Text Chat | C0 | 101.96 | 172.87 | 707.46 | 5.643 | 180.59 | 73.81 |
| Text Chat | CL | 82.42 | 164.84 | 669.25 | 5.909 | 189.08 | 64.59 |

- Natural CLT versus C0: TTFT p50 -9.56%, RPS **-0.83%**; versus RR:
  TTFT +13.48%, RPS -1.61%. Even equal 74.44% token hit ratios do not imply
  equal latency or throughput.
- Long CLT versus C0: TTFT -32.29%, RPS +8.74%; versus RR:
  TTFT +0.52%, RPS **-4.40%**. Actual input is 3096–3100 tokens, unchanged
  4096 context limit; this does not establish a long-input throughput win over RR.
- Chat CL versus C0: TTFT -19.16%, RPS +4.70%; versus RR:
  TTFT **+14.33%**, RPS **-3.97%**. This exercises the shared load guard only;
  it is not Chat prepared-token forwarding or a Completion result extrapolation.

## Closeout and evidence receipt

The user's deadline progressed from 14:00 to 14:30 and then 15:00. The fixed
matrix completed without needing to extend the internal 14:20 measurement /
14:25 supervisor safety limits again. At **14:14:56 Asia/Shanghai**, read-only
closeout verified all 59 recorded Worker cohorts and 59 Routers exited, no task
ports listening, no GPU compute processes and GPU memory 0 MiB. No model/cache
deletion was necessary. Instance billing/shutdown remains the owner's action.

The candidate checkout remained clean at 2a0d179. Both wheel/native/Python
installations were reverified after testing; existing Worker proposal source
hashes remained unchanged. Post-run actual weight files matched the retained
fixed ModelScope API listings (disk verification, not GPU-memory attestation):

- Qwen3-0.6B: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- SmolLM2-135M-Instruct: `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`,
  listing revision `c134cb42e51e0d1f29041149173377c623c99b25`.

`gpu-evidence-final.tar.gz` was retrieved with matching remote/local SHA256:
`07ac1acd2cc682a1baf772570f1a5a0be96de138b63546c6af6bcb22555803fb`.
It contains all original evidence, plans, helper versions, manifests and
closeout receipts, not model weights or duplicate wheel binaries. Canonical
extraction: `gpu-50577/remote-final/`; full independent phase analysis:
`gpu-50577/final-performance-analysis.json`; aggregate assertions:
`gpu-50577/FINAL-EVIDENCE-VERIFICATION.json` (PASS). An earlier incomplete-copy
analysis is retained under an explicitly named file; after transfer completed,
all source data passed. No GPU run was repeated to conceal that transfer state.

## Measurement discipline and interpretation

Every arm receives a freshly started, verified owned Worker cohort. The prior
Router exits and Workers drain before replacement, preserving the required
immutable CT cohort boundary. Startup/oracles/warmups are outside the timed
window. Equal persistent clients, sampling, connection/resource budgets,
ordered ingress and complete prepared IDs are checked within each paired cell.
The saved-literal offline analyzer passes all four exploratory cells with no
missing/inconsistent arms. Instrumentation is off in headline measurements.

Removing duplicate Worker text tokenization does not remove Router render,
derived-body transport/validation, GPU prefill or decode. In actual vLLM 0.29,
the token-array path bypasses text encoding and, for this non-echo subset,
does not request input detokenization. Do not explain a regression by inventing
an input decode/encode pass. Source-path verification is not a CPU timing result.

Worker queue/prefill/decode cumulative sum/count windows and per-request times
are saved. Nested stage means are not additive; no subtraction of unrelated
p95 values or attribution of unmeasured costs to GIL/GPU kernels. Same-GPU
batching/contention and run variation can affect decode; neither is isolated
by these observations. Closed-loop RPS is not open-loop production capacity.

## Release boundaries

Chat continues original text forwarding with existing tools/reasoning. Full
Chat tokens-in/out is not implemented and Completion gains do not apply to it.
CT requires an immutable Worker endpoint/model/tokenizer/config cohort; stop
ingress, drain/cancel, stop Router, replace Worker, then restart and reconform.
Polling/current local generation is not atomic validation on the remote POST;
live/rolling same-URL replacement remains unsupported.

No pool, default affinity, tokenizer cache, offload restructuring, new layout,
model-specific branch or new Worker patch is part of this increment. Full
applicable CI/format ownership, ABI/install matrix, human correctness and
author/license review, capability-interface/base coordination and explicit
publication approval remain gates. No push or PR is authorized by this report.

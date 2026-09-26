# Finite capability validation: exact evidence boundary

Final measured code candidate: `60ca5ec61fa473141aa09d8f639198e6b8bda554`.
Tree: `2fa78343b8db35c6912d74658742500bf274d9f5`.
Base: `f0f02adb64a26d819b0e6e9e501a37b8a9d71f09`.
Actual built and mapped Router debug extension SHA-256:
`3b3cd861c54e343a3031773da5edb1ac96a865ec9f2441dd4d9b46dbd933bc1f`.

This report may be added in a later documentation-only commit. It does not
relabel that later commit as the executed candidate, or reuse the old Render
Bridge artifact. The measured source was clean before/after each finite run.
Earlier runs at `50d72d93500f6e2c43a87072a64d54e6020db70a` are retained
under their original identity. The only change to the final code candidate
allows legal empty decoded text in untimed one-token performance warmup;
timed requests still require nonempty generated text. Production code is
unchanged, and the final incremental build independently verifies the same
native artifact. Neither an earlier result nor an old build is relabeled.
The first Smol performance attempt stopped during untimed warmup on a legal
empty-decoded one-token response, before any measured phase. Its failure is
preserved separately; the final complete run is the result after the public
test-runner correction and its two added regression checks.

## Runtime and deployment dependency

2026-09-26 authorized isolated run: one NVIDIA GeForce RTX 4080 SUPER,
32760 MiB, driver 595.71.05; Python 3.12.3; vLLM 0.29.0 at
`98dff2a81d747d1dba01a47f939f48c3526d4206`; Torch 2.13.0+cu130;
Transformers 5.17.0; tokenizers 0.23.2; PyO3 0.26.0; Rust 1.95.0.
Cargo.lock SHA-256:
`000665f280cab3ba36fd6392afd5e9c39f72fcb10d35521e8161ded0d1b4d614`.

The separately reviewed Worker patch is required, SHA-256
`c3898a75488608b8286f8707a8b8f92211fb3da273c81b5bf024f0d1f5367b08`.
All 16 base/audit files matched the pinned installed source. The authorized
finite deployment copied the complete package into its own directory, applied
the nine-file Python proposal there, preserved all 18 native artifacts, and
verified scoped imports. The original installed package was not patched. This
source snapshot is not a production wheel/editable-install or ABI validation.
Only owned Workers were launched, with loopback HTTP and user-isolated event
ports. No DEV_MODE, arbitrary RPC or Router-triggered patch installation.

The debug Router probe used a process-scoped system libstdc++ preload because
the inherited Miniconda Python resolves an older library. It was not installed
globally, used to alter Worker libraries, or offered as a production recipe.

## Correctness results

| Model | Actual initialized group | Finite matrix | Artifact |
|---|---|---|---|
| Qwen3-0.6B | 28 layers, one Full Attention group, 16-token units | 27/27 PASS | Same native SHA above |
| SmolLM2-135M-Instruct (Llama architecture) | 30 layers, one Full Attention group, 16-token units | 27/27 PASS | Same native SHA above |

Both independently addressed Worker copies report Normal execution,
DP/TP/PP/DCP/PCP=1, full-byte 32-byte `sha256_cbor`, numeric seed 0,
terminal recomputation of one token and separate boot-unique event topics.
No model-specific Router code or manual Profile was added for SmolLM2.

Each matrix covers 10 supported input shapes, eight first-route warmed-owner
cases across Completion/Chat and JSON/SSE, seven cold/warm boundary groups,
four salted fallbacks and one canceled stream. The 27 top-level cells are not
27 requests: each run has 32 complete exact requests, four salted fallback
requests and an additional canceled stream. All 36 saved complete generation
responses are compared with both Worker render token arrays; the 32 eligible
exact requests are additionally compared with the local facade and Rust digest.
Raw input forwarding remains unchanged; tokens are not claimed to be the whole
cache key. Salt remains ineligible for affinity.

| Prepared N | Warm-owner raw matches | Predicted reusable tokens | Actual hit tokens |
|---:|---:|---:|---:|
| 15 | 0 | 0 | 0 |
| 16 | 1 | 0 | 0 |
| 17 | 1 | 16 | 16 |
| 31 | 1 | 16 | 16 |
| 32 | 2 | 16 | 16 |
| 33 | 2 | 32 | 32 |
| 464 | 29 | 448 | 448 |

The same boundaries are checked on both models. At N=16 the stored match has
zero reuse, so selecting the warmed owner is not required. Separate fresh
prefixes isolate cold and direct-only-warm cases. Metadata reads are bounded
control-plane startup/revalidation/background work, not one query per request;
generation does not issue a request-level remote `/render` call.

Test assets are pinned for reproduction, not production admission:

- Qwen3-0.6B, ModelScope commit `09b42cad3d112e832108974449ccb5e8e0f5b5d1`;
  weight SHA `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- SmolLM2-135M-Instruct, ModelScope commit `c134cb42e51e0d1f29041149173377c623c99b25`;
  weight SHA `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.

## Real owned-Worker restart

The final `60ca5ec` SmolLM2 probe passed with the same native artifact. It first
established positive ownership, stopped only the recorded Worker0, and started
a replacement from the same verified deployment. Worker0's publisher epoch
changed; surviving Worker1's epoch remained unchanged. After the replacement
became ready, old Router contract rejection was observed within 3.912 seconds
with a two-second test health interval. Before rejection, the replaced Worker's
old ownership did not influence selection; the surviving Worker may learn
fresh cache entries from successful intervening requests.

Restarting the Router repeated automatic token conformance. A prefix warmed
before the new subscription had zero observed ownership despite successful
metadata discovery; a subsequent new event yielded a positive owner. This
checks an empty observed subset, not replay or complete-cache recovery. Both
probe Routers and the replacement Worker were stopped. Real cache clear was
**not run**: no supported non-DEV clear API was available, and development mode
was not enabled. Gap/clear/delayed-reply cases remain source/Rust test evidence.

The preserved first Qwen restart probe at `50d72d9` failed an overly strong
test assertion: it demanded zero cache score even for surviving Worker1 after
that Worker had served a fresh request. The retained decisions showed no
positive old Worker0 ownership. The evidence-only probe was corrected locally
to distinguish stale replacement ownership from valid surviving-Worker events,
self-tested, transferred and rerun; Router production code was not changed.
That first failure remains a failure in the evidence, not a retroactive PASS.

## CPU and native gates

Same measured candidate: focused Rust KV tests 55 PASS and bridge tests 13
PASS; the real loopback HTTP/ZMQ lifecycle case also passed independently.
fmt, locked check, Clippy and debug native build passed. This is not the entire
upstream test suite or a release build. Python render tests 23, capability
entries 53 and entrypoint tests 13 passed on the GPU host in this final build.

Independent Linux x86 CPU revalidation at `50d72d9` with Python 3.11.2/vLLM 0.29.0+cpu
also passed: Dense oracle 6 methods/18 explicit fixtures, capability entries
53, entrypoint 13, Worker exporter 39, actual SmolLM CPU rendering of 10 shapes
and four exclusions, GPU-runner self-check 21 and performance self-check 16.
That CPU-only run has no new native artifact and is not relabeled CUDA evidence.
The final `60ca5ec` GPU-host build reran the focused Rust/Python gates above;
the performance runner's CPU self-check now has 18 tests, including empty-text
warmup acceptance and strict timed-text rejection.

## Finite performance: no net speedup established

One round, two Workers on one GPU, debug Router, actual text Completion,
32 requests per phase, 32 generated tokens, four prewarmed prefix groups,
concurrency 1/4. RR omits Router rendering while KV includes it. Startup,
direct warmup and setup-only Worker render oracles are excluded from timing.
Fresh first-block namespaces isolate phases without cache resets. Logical
order and target lengths match, not bytes or exact prepared token lengths
within each pair. This is a finite end-to-end comparison, not a component
profile or production benchmark. Each model has eight phases/256 requests.

### Qwen3-0.6B

All eight phases/256 timed requests at `60ca5ec` passed; actual prompt lengths
were 1044–1056 tokens. The earlier `50d72d9` comparison is retained separately
and is not pooled with this final run or used to select a best result.

| Scenario / concurrency | Prefix token hit ratio RR → KV | TTFT p50 ms RR → KV | TTFT p95 ms RR → KV | Throughput requests/s RR → KV |
|---|---:|---:|---:|---:|
| locality / 1 | 64.97% → 74.47% | 61.87 → 74.40 | 89.04 → 101.25 | 1.490 → 1.470 |
| locality / 4 | 64.96% → 74.23% | 71.20 → 108.36 | 108.02 → 157.80 | 5.453 → 5.157 |
| cold / 1 | 0% → 0% | 65.59 → 77.25 | 70.68 → 98.74 | 1.474 → 1.462 |
| cold / 4 | 0% → 0% | 66.92 → 93.71 | 98.09 → 133.97 | 5.523 → 5.422 |

Every phase's actual Worker completed-request counts were 16/16 (Jain=1).
This is finite request-distribution evidence, not universal load balance.
RR's ordinary path does not maintain the same `/workers.load` counter as KV's
owned lease; its zero samples cannot be interpreted as an idle GPU or compared
numerically with KV. The only lower latency percentile was locality/C1
end-to-end p95 (712.63 → 693.38 ms); all TTFT percentiles increased and
throughput decreased in all four comparisons.
No causal attribution to a particular renderer/GIL/hash component is proven.

### SmolLM2-135M-Instruct

All eight phases/256 timed requests at `60ca5ec` passed. The 16 untimed one-token
warmups completed with exact input IDs and usage but no decoded text; they
correctly have no TTFT measurement. Timed requests all produced nonempty text.
Actual prompt lengths were 1045–1057 tokens. The same restrictions above apply.

| Scenario / concurrency | Prefix token hit ratio RR → KV | TTFT p50 ms RR → KV | TTFT p95 ms RR → KV | Throughput requests/s RR → KV |
|---|---:|---:|---:|---:|
| locality / 1 | 64.93% → 74.38% | 76.72 → 86.37 | 111.18 → 112.61 | 1.568 → 1.567 |
| locality / 4 | 65.07% → 74.16% | 85.11 → 113.84 | 122.23 → 167.80 | 5.886 → 5.616 |
| cold / 1 | 0% → 0% | 79.76 → 89.95 | 85.29 → 120.09 | 1.569 → 1.538 |
| cold / 4 | 0% → 0% | 84.33 → 103.13 | 105.05 → 145.09 | 5.859 → 5.793 |

All per-Worker completed-request counts were 16/16, with Jain=1. Locality hit
ratios increased by 9.44 and 9.09 percentage points, but TTFT p50 increased
12.57% and 33.75%; throughput changed -0.11% and -4.59%. All four end-to-end
p50/p95 comparisons also increased. These are observations from one finite run,
not statistically established effects or proof of which component caused them.

## Evidence retention and closeout

Final GPU build manifest SHA-256:
`19328b9e8e38ec8bdcb83c9eb084f27d4a687b1d14cd289958d3b1ee34255641`.
The complete remote evidence/configuration/supervision archive is retained
locally as `evidence-final.tgz`, SHA-256:
`0554201777c1f03ac19e144ecc393fa49fa91706f0804693eb4e259fdd698dfa`.
The actual 199,484,176-byte native artifact is retained separately, with the
SHA at the top of this report. The archive contains raw responses, descriptors,
process and build records, counter/trace data, current probe scripts, and failed
attempts; it does not contain model weights or the native artifact itself.

Closeout passed at **2026-09-26 09:25:22 Asia/Shanghai**, before the 11:00 limit:
all 27 owned supervisor records had exited, none had a matching live PID, all
task ports were closed, and GPU compute-process output was empty (1 MiB reported
device memory). Original installed package and isolated patched snapshot source/
native inventories remained unchanged; Python bytecode caches are excluded
from those inventory hashes. Remote source remained clean at `60ca5ec`, with
the same native hash. No model files were deleted. Only task-owned processes
were stopped; the cloud instance itself was not shut down.

## Remaining gates

Human correctness, authorship/license and upstream endpoint/base review remain
required. Production installation/wheel ABI, release performance, multiple-GPU
load, longer traces and repeated statistics are unvalidated. Do not advertise
universal model support or a speedup, add excluded cache mechanisms, or publish
without approval. Another hardware run needs fresh authority.

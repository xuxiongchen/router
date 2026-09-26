# KV performance investigation (bounded, opt-in)

This follow-up does not change the supported models/layouts, Worker proposal,
cache-key eligibility, capability generations, or the production KV selector.
The base is `346c3c3d35f63979d26b8d2706778283b3a35479`; its diff from the measured
`60ca5ec61fa473141aa09d8f639198e6b8bda554` contains only five documentation files.
PR1 and the accepted Render Bridge/capabilities branches are unchanged.

The earlier debug results remain valid **negative performance evidence**:
more cache hits did not demonstrate end-to-end acceleration. They used different
forwarding/load paths, different phase namespaces and one round of 32 requests.
They cannot identify a pure render/GIL/scorer cost. Single-GPU, two-process
results also cannot demonstrate balancing across independent devices.

## Build the extension actually loaded by Python

Use an isolated Linux environment and the existing `setup.py`/PEP 517 build,
not just the Rust executable. For the measured environment, build dependencies
are setuptools 80.9.0, setuptools-rust 1.13.0 and wheel 0.48.0. An older
setuptools 66.1.1 rejects this repository's SPDX license metadata before build.
Keep build tooling separate from the runtime: the actual vLLM 0.29.0+cpu wheel
pins runtime setuptools 77.0.3. The baseline and candidate runtime environments
use that pin; retain both build and runtime dependency records rather than
silently ignoring the dependency conflict from using 80.9.0 at runtime.
The optional renderer remains vLLM 0.29.0; no new Worker patch is needed beyond
the already documented capabilities dependency.

```sh
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
export SETUPTOOLS_RUST_CARGO_PROFILE=release
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_OPT_LEVEL=3
export CARGO_PROFILE_RELEASE_LTO=thin
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
export CARGO_PROFILE_RELEASE_DEBUG=0
python -m pip wheel --no-deps --no-build-isolation --verbose . -w artifacts/production
```

Record `rustc -Vv`, Cargo config, all `CARGO_*`/`RUST*`/`PYO3_*` build overrides,
Cargo.lock hash, source SHA/dirty diff, wheel hash, wheel tag and actual native
hash. Verify the main crate's effective rustc flags in the verbose log; a
`release/` directory alone proves nothing. Current packaging requests
`pyo3/abi3-py38`; this differs from a plain CPython-specific `cargo build --lib`.
The optional vLLM dependency's supported Python version is a separate constraint.

The original `346c3c3` fails this actual abi3 wheel build in three existing
`extract::<&str>()` calls (PyO3 0.26.0 does not expose that extraction below the
limited-API 3.10 floor). The isolated ABI-only baseline is
`c888e8d3b987a3b22a487973d57600d80b88b510`, containing only the strict `Cow<str>`
extraction fix. The performance candidate carries the same fix. This still
rejects non-strings and invalid UTF-8/surrogates, and owns the final contract ID;
the abi3-py38 fallback copies UTF-8 rather than borrowing it. Do not label the
unmodified baseline as a successful wheel build or count ABI compatibility as a
performance gain. Compare the two optimized artifacts at the same ABI/profile.

Install the wheel without dependency replacement in a separate venv. Check both
`vllm_router.__file__` and `vllm_router_rs.__file__`, the native hash, and the live
Linux `/proc/self/maps` entry. Do not let a checkout/PYTHONPATH silently replace
the installed Python package. Profiler artifacts, if any, need separate hashes.

## Diagnostic durations

`VLLM_ROUTER_KV_STAGE_TIMING=1` enables bounded-cardinality metrics. Default is off.
`VLLM_ROUTER_KV_STAGE_TRACE=1` additionally permits at most 65,536 stage log events
per process, within the ordinary logging filter. Use a separate short diagnostic
window; request IDs are carried in spans, never Prometheus labels. No prompts,
token arrays, exception payloads or model paths are emitted by this observer.

- `vllm_router_kv_stage_duration_seconds{stage}`: queue wait, executor occupied
  wall time, Python attach/entry, call, result/token conversion, asset checks,
  schema/raw JSON, eligibility, serving/renderer/tokenizer/template combinations,
  block hashing, candidate/index work, policy and Router selection, dispatch to
  upstream headers (including failures/cancellation).
- `vllm_router_kv_stage_operations_total{operation}`: facade asset scan, explicit
  stat/is_file/exists, content read/hash calls and bytes. These are facade
  operations, **not all OS syscalls or internal Transformers file operations**.
- `vllm_router_kv_bridge_usage{resource}`: admitted jobs, input bytes, reserved
  tokens, active callbacks and pending jobs. Pending includes admitted work just
  before enqueue. Active/reservations survive caller timeout until real return.

All cross-language measurements are durations. No Rust/Python clock origins
are subtracted. Attach/entry is not pure GIL contention; tokenizer/template
async intervals include their existing offload queue. Nested intervals must not
be added, and stage p95 values do not sum to request p95. Missing stages are
unavailable, not zero. Gauges are snapshots, not exact per-request timelines.

The client separately measures response headers, first complete SSE event,
first nonempty reasoning/content and response completion. TTFT means first
nonempty generated text, not headers or an empty SSE event. Worker queue/prefill
histograms remain aggregate window metrics, not request-correlated timings.

## Controlled ablation artifact

Build a **separate, non-publishing** wheel with
`VLLM_ROUTER_BUILD_KV_PERF=1` and the same optimization/ABI settings. The default
production build rejects an accidental `VLLM_ROUTER_KV_PERF_MODE` setting.
The native `kv_perf_capabilities()` handshake identifies the feature and mode.
The diagnostic build still defaults to ordinary production behavior.

| Arm | Local preparation | Selection | Forward/load lifetime |
| --- | --- | --- | --- |
| product_rr | ordinary product path | product RR | ordinary product path |
| A (`shared_rr`) | no render | RR | shared raw KV ingress/stream/lease |
| B (`render_rr`) | real render once | RR, ignores tokens for selection | same as A/C |
| C (`render_kv`) | real render once | unchanged KV policy | same as A/B |

The feature requires literal loopback HTTP Workers and Router bind, static
Regular DP=1 and an active automatically verified capability bridge. No PD,
discovery or program scheduling. A intentionally skips per-request facade
asset/schema checks and relies on the original request's Worker validation;
it also skips bridge admission, copies, queueing and its provider timeout.
It is not a production policy. All three A/B/C arms retain active deployment/epoch fencing,
subscriptions, connection pools and retry/stream lifetime. B/C do not fabricate
tokens or scores. B-A measures the changed preprocessing path under controlled
conditions, not isolated renderer CPU; C-B additionally includes Worker
placement, not just scorer CPU.

## Minimal asset metadata optimization

The signature now reads one stat result per entry and uses both its size and
mtime. Each request still enumerates assets and checks the same signature
fields; order, duplicate entries, oversize checks and invalidation fences remain.
No immutable asset cache, background detection window or token cache is added.
The operation saving depends on the actual asset count. A concurrent mutation
can be observed differently from two separate stat calls; neither version
proves content integrity or eliminates TOCTOU. Performance is measured
separately from this reduction in redundant operations.

## Measurement discipline and release gates

Reuse `py_test/benchmark_render_bridge_paths.py` for CPU/native checks and
`scripts/kv_capabilities_performance.py` for the finite GPU comparison. A real
facade-only 1/2/4 pre-screen is **not** a production pool, Rust bridge or GPU
benchmark. It cannot independently establish a Rust executor queue bottleneck.

Headline windows use the same keep-alive clients and diagnostic settings,
without prompt-ID echo. Full actual-Worker token oracles run outside timing.
The product RR Router load counter is **unknown**, not zero: compare actual
Worker running/waiting/phase metrics. Preserve every error, round and ordering;
32-request exploratory runs do not support p99 or production capacity claims.

Byte-identical paired requests also require controlled initial caches, via an
explicitly authorized fresh cohort or an actually supported reset. A restarted
Worker requires a newly verified Router contract. Namespaced phases are useful
exploration but not strict pairing. No DEV_MODE, unsupported reset API or new
Worker patch is enabled by these tools. GPU/SSH operations need fresh authority.

Before release: bind final results to candidate/wheel/native, run affected
correctness/lifecycle checks on that extension, measure instrumentation on/off,
then perform the applicable CI and bounded GPU subset. Longer stability, real
cache clear, representative device topology, human correctness/license/base
review and publication approval remain separate gates. No performance SLO or
non-regression budget is invented in the absence of an agreed one.

## Bounded CPU results (2026-09-26)

The final packaged runtime source is
`9a9d968952be05e061fa9b6805fd4b04a15b664b`. Actual installed artifacts:

```text
production native SHA256: 32e9de024c52c1fc5d5e665c5f7577ef9c1adcf783796cb00b96ea4ab68c0c28
production wheel SHA256:  d815586d130d22796c306135295a37052e249ab0caa8b45444d13898747c57ee
ablation native SHA256:   a252b763b9aec5626b7ed6fcc19303f15e2f3f46741cba115c107591085583f8
ablation wheel SHA256:    ab5f761be940a93a2c0496b734603a27876e51490e0a8b0b3010abf9145a5781
```

Both are `cp38-abi3-linux_x86_64` wheels, actually loaded in Python 3.11.2 with
vLLM 0.29.0+cpu, Torch 2.13.0+cpu, Transformers 5.17.0 and tokenizers 0.23.2.
The host was x86 with a four-CPU quota and about 8 GiB visible memory. These
are not Python 3.8 execution tests, portable manylinux certification or GPU
loadability proof. The production handshake has the experiment feature absent;
the separate ablation artifact enables it but defaults to no experimental mode.

The pre-stat candidate was `17a158fc321d159a6857a76f68c6881008f4ed06` with the
same production native hash above. Its ABI-only baseline was `c888e8d3` (full
source identity above), native
`cc1eac37c6ba9879ba3c1adcf773488e2d50a61cfa1d2d0730d49f9a11834327`.
Two rounds completed 540 cells and 21,600 timed requests, covering native,
in-process vLLM, official HTTP and facade-only N=1/2/4 paths. These are not the
GPU shared-forwarding A/B/C contrasts. Complete facade token arrays and epochs
were checked, including a shared 29-shape mixed-input oracle.

Selected same-window means in the pre-stat CPU fixture:

| Interval | Range (ms) |
| --- | ---: |
| C4 Rust queue, 1k/4k Completion and Chat | 4.815–22.419 |
| Corresponding executor occupied wall time | 3.946–11.113 |
| Asset check across timing-on vLLM cells | 0.4263–0.7399 |
| Serving | 1.2211–9.2924 |
| Async tokenization, including offload wait | 0.6159–6.8414 |
| Python attach/entry, not pure GIL | 0.0143–0.0382 |
| Router selection in this CPU fixture | 0.0775–0.2857 |

These nested intervals are not additive and do not explain all of the old
debug GPU regression. Median per-cell RPS changes for candidate-off versus
ABI-baseline-off were +0.04%/+0.96% by round; timing-on versus candidate-off
were −4.16%/−2.37%. These are unweighted medians of 18 cell ratios, not a
workload-wide speedup. Individual regressions and sign reversals remain.

No production render pool is included. For example, N=2/C4 improved Chat4k
RPS by +73.2%/+38.6%, but short Completion regressed −32.8%/−28.2%, and
Completion4k flipped +79.9% to −37.4%. N=4 was also mixed. Retained state/RSS
and ordering limit the screening; snapshots are not per-renderer memory costs.
Actual dependency inspection also found shared assistant-mask template state.
Independent facade instances alone do not prove arbitrary concurrency safe.

### Stat-only comparison

Four fresh processes in pre/post/post/pre order completed 144 cells and 11,520
timed requests: N=1, C=1/2/4, six input shapes, observation off/on, 80 samples
per cell. The harness source was `2e090a8ff855fed74041b7ff28c7d8505c996d47`;
its only changes after the packaged runtime were AST-preserving formatting of
three test modules. Both actually installed runtimes and source stability were
verified. The native was recompiled at `9a9d968` and remained byte-identical.

In the unchanged eight-file fixture, explicit facade stat calls fell from
26 to 17; scans and zero content-read/hash counts were unchanged. This directory
includes three preserved oracle JSON files: the count is fixture-specific,
not every model's count or total OS syscalls. Full tokens/epochs remained equal.
With observation enabled, asset-check mean across equal-sized cells changed
0.833→0.784 ms and 0.925→0.786 ms by round. This is a limited local reduction
signal, not an end-to-end gain. Four of 18 first-round asset means increased.
The on-state saving includes nine fewer observer counter updates, so it is not
pure OS stat cost and does not directly quantify the disabled observer's stage.

Overall facade RPS cell-ratio medians were +0.48%/+8.70% with observation off,
but −0.67%/−3.86% with it on; eight of 18 off cells changed direction across
rounds. Keep the small redundant-operation removal, not a universal speedup
claim. No Rust bridge timed path or GPU inference ran in this stat comparison.

### Correctness and remaining gates

At `9a9d968`, ablation KV subsets passed 60 tests per timing state, prompt
subsets 21 per state, and two local token goldens explicitly passed. Both
production/ablation release checks and all-targets/all-features Clippy passed
(no diagnostics; no `-D warnings`). Production prompt/KV subsets had already
passed at `17a158f`; the later Rust change only fixed a test negative control
that had incorrectly kept its RR override while expecting KV behavior.
The original 59/60 failure is retained in the local evidence, not relabeled.

The 38 Python boundary tests and real vLLM CPU oracle passed (27 named cases,
seven startup checks, five unsupported and two invalid cases). Each final
installed extension passed three native lifecycle probes with timing off and
three with it on. The three formatted test modules passed 20+11+4 tests with
their documented unittest-discovery entry points; their ASTs were unchanged.
Changed Python files pass Ruff 0.16.0, and the three new modules pass Black.
Repository-wide Black/Ruff gates are not green: the base already has eleven
formatting failures and an unused import in `test_kv_dense_source_oracle.py`.
Do not treat focused checks as the unfiltered repository suite or hosted CI.

GPU product-RR/A/B/C comparison, actual GPU-environment artifact loading,
affected two-model checks, longer stability and publication approval remain
outstanding. No new SSH/GPU authority was used. The existing Worker capability
dependency is unchanged; no additional Worker patch is required by this diff.

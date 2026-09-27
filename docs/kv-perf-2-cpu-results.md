# Perf-2 CPU acceptance: scope and exact artifacts

This is CPU functionality/build evidence, **not a GPU or performance result**.
Both production increments default off. No push or PR publication was performed.

## Source and build identity

- Base: `6b17f5a54f4713bdf75474d2b40131fafd8fd499` (its runtime is unchanged
  from the previously measured `9a9d968952be05e061fa9b6805fd4b04a15b664b`).
- Load guard feature: `91339da`; local pre-dispatch lifecycle correction: `c467d39`.
- Completion prepared-input feature: `b446f5d`.
- Harness/functional gates: `bf52ba3` and `6b81e6f`.
- **Compiled and tested candidate:** `2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110`.
  This includes the equivalent Boolean simplification required by strict Clippy.
- Later documentation-only commits do not change or relabel that build identity.
  New GPU execution must build and verify the exact selected checkout SHA.

Both wheels were actually installed and imported in separate Linux x86_64
environments. Build manifests retain loaded native mappings, the effective
Rust compilation command and source/package hashes. Rust 1.95.0, release opt-level
3, thin LTO, one codegen unit, debug 0, incremental off; Python 3.11.2; ABI3 wheel
floor CPython 3.8. This does not assert an exercised multi-Python/OS ABI matrix.

| Artifact | Production (no `kv-perf`) | Ablation (`kv-perf`) |
| --- | --- | --- |
| Wheel SHA-256 | `4c7bf72c70c1311d9fe9883bfa1767434a86cf1b95b806f49447779695b38d27` | `e84d026ed0e4fa06b9fa2b880a0bd80cf00e358b165250d1f7cd403c1cea4f8c` |
| Native SHA-256 | `f17ccc17cc343b5cea68145a3e73d6442cbd38e2413f6f9a75606a23928e683b` | `eec963959d61272400a89d81c7630fcfb45c9fae51afdac7c5852c0795b325c4` |
| Artifact handshake | enabled=false, modes=[], selected=null | enabled=true, selected=null during CPU probes |

`Cargo.lock` is unchanged, SHA-256
`000665f280cab3ba36fd6392afd5e9c39f72fcb10d35521e8161ded0d1b4d614`.
PyO3 0.26.0 and serde_json 1.0.151 remain locked; only serde_json's existing
`raw_value` feature is enabled. No new Rust crate or Worker patch was added.
The existing capability Worker proposal is still a dependency, not a stock API.

Actual CPU preprocessing used vLLM `0.29.0+cpu` (fixed source
`98dff2a81d747d1dba01a47f939f48c3526d4206`), torch `2.13.0+cpu`, transformers
`5.17.0` and Python tokenizers `0.23.2`. Full runtime freezes and native manifests
are retained in the task's review evidence. Both final dependency checks pass.
The new environments initially lacked declared dependency orjson; only these new
environments received `orjson==3.12.0`. The initial failed check remains retained.

## Measured CPU gates

| Gate | Production | Ablation |
| --- | --- | --- |
| Focused Rust `kv_` tests | 71 PASS | 75 PASS |
| Rust `prompt_tokens::`, including local-asset cases | 27 PASS | 27 PASS |
| `cargo check --lib --tests` | PASS | PASS |
| Installed wheel argument/configuration/startup suites | 105 PASS, 4 existing SKIP | 105 PASS, 4 existing SKIP |
| Actual extension, C0/CL/CT/CLT flag combinations | All four PASS | All four PASS |

`cargo fmt` and all-targets/all-features strict Clippy (`-D warnings`) pass at the
compiled candidate. The first strict-Clippy failure was fixed and retained;
warnings were not suppressed to obtain a pass. Focused filters are not a unique
test total or an unfiltered full-suite result. The four Python skips are explicit
unchanged legacy tokenizer-configuration tests, not new exclusions.

The public `py_test/integration/render_bridge_native_probe.py` runs three scenarios
per invocation: JSON/SSE/derived and raw forwarding, execution deadline, and client
disconnect/shutdown. Both binaries passed all four flag combinations, plus one
separate production CLT run with diagnostic timing enabled (nine invocations,
27 scenarios). It checks actual extension hashes, Python progress/lifetime,
bounded admission, late synchronous completion, teardown, ingress/backend bytes,
safe-header behavior and forwarding/payload counters. The facade and capability
descriptor are explicitly synthetic. This is **not** actual vLLM rendering,
KV-event/cache-hit or GPU/TTFT evidence. CPU probes use ordinary dispatch in both
binaries; the experimental performance-mode arms still require GPU execution.

Separate actual-vLLM CPU tests supply the input-equivalence evidence:

- Qwen3-0.6B and SmolLM2-135M-Instruct: each 10 positive fixtures and five negative
  fixtures pass. Full final IDs and every effective sampling parameter match the
  original text path. The token-array path makes zero text-tokenizer calls.
  Invalid-then-valid behavior remains correct. No inference engine is initialized.
- Existing Qwen renderer regression: all 27 corpus cases pass, including supported
  tools/reasoning, public goldens and startup mismatch rejection; two invalid and
  five unsupported request cases remain checked.
- These two gates were repeated after production wheel installation. Tests load
  the source module explicitly; its bytes were first checked equal to the installed
  wheel's render module. The installed native was tested separately as above.
- Additional public boundary/entrypoint/build-contract and harness self-checks
  passed; these include inherited/overlapping fixtures and are not additive counts.

No GPU JSON/SSE generation, actual Worker prepared-token identity, production
throughput, TTFT benefit, stable percentile/SLO or cache hit improvement is claimed.
The runnable four-arm and production gates are in [the GPU runbook](kv-perf-2-gpu-runbook.md).

## Limitations and release gates

- Prepared Completion input requires the [immutable Worker cohort deployment
  contract](kv-perf-2-completion-input.md#required-immutable-worker-cohort).
  Observed local generation checks do not atomically bind a POST to a remote
  tokenizer/process; live same-URL replacement is unsupported.
- Chat remains original forwarding with its existing tools/reasoning path.
  Full Chat tokens-in/out is not implemented, and no feature was removed to
  claim completion. GPU generation/performance regressions remain required.
- The load guard sees this Router's in-flight leases, not global GPU capacity.
  Its fixed slack is experimental, not a universally optimal threshold.
- Full Python CI is not green: base and candidate Black checks report the same
  11 files, seven edited by this increment and four untouched. Full Ruff retains
  an unused `SimpleNamespace` import in untouched
  `py_test/test_kv_dense_source_oracle.py`; changed-file Ruff passes. The audit used
  Python 3.11.2/Black 26.5.1/Ruff 0.16.0, not the exact public Python 3.12 environment.
  Formatting ownership must be resolved before claiming green full CI.
- Full unfiltered Rust, Python integration, standalone CLI/WASM and supported
  install/ABI lanes have not all run. No internal `codex_verify.sh` was executed.
- New GPU authorization, finite Qwen four-arm/real-RR measurements, same-native
  Smol correctness and actual production-wheel confirmation are pending. Keep
  cold regressions, errors and larger derived payloads visible.
- Human correctness/author/license review, capability-interface and dependency
  coordination, and explicit publication approval remain required. No community
  division of work or other contributor's PR is assumed approved.

CPU functional acceptance, performance benefit and publication readiness are
separate states. The latter two remain pending. No render pool, affinity default,
new tokenizer cache or offload restructuring is included.

# Chat Serving 1: raw text Agent semantic acceptance

## Scope and status

CPU acceptance and the newly authorized finite real-model GPU acceptance are
executed; see [GPU results and retained failure](chat-serving-1-gpu-results.md). Chat was
already supported; this change does not implement or advertise Chat tokens-in/out.

Dependency base: `98c024e81b67a0eba85e7312cc07ad0c6b248566` (Perf-2 documentation
HEAD). New branch: `codex/cmb-chat-serving-1`. Frozen branches remain unchanged.
Only public acceptance helpers/tests and documentation change. No production
renderer, Rust parser, Worker protocol, model whitelist or scheduling change.
CPU semantic replay was rerun at `a64c7d111eb8a80c6bd1e9dfe0078026dbe0bf79`.
GPU semantic matrix uses that SHA; focused abort/recovery and performance use
`1763ecf0192b4228b5599a8e11a67d2348e3653a`. The latter only fixes the test
log matcher and adds a focused test slice; no production code changed.

The separately attached Release-Closeout task is not executed implicitly. Its
remaining full-CI/publication gates still apply. A fresh GPU window followed the
CPU handoff. No Worker patch installation, push or PR was performed.

## Evidence layers and supported contract

Selected runtime: vLLM `0.29.0+cpu`, recorded source
`98dff2a81d747d1dba01a47f939f48c3526d4206`; Python 3.11.2, torch
2.13.0+cpu, transformers 5.17.0, tokenizers 0.23.2, OpenAI Python SDK 3.19.2.
Qwen3-0.6B public config/tokenizer assets only; no CPU model generation/weights.
Official parser configuration: `hermes` tools, `qwen3` reasoning, automatic
tool choice enabled, generation config `vllm`. This is not a family-wide claim.

`py_test/test_chat_serving.py` uses the real OnlineRenderer, official incremental
detokenizer, and OpenAIServingChat full/stream output generators. Only engine
outputs are fixed synthetic fixtures. The parser implementation is not copied.
Its actual source-file hashes and bounded CPU measurements are retained in
`replay-25089b9.json` in the task evidence directory. Earlier failed harness
development runs are retained; the wrong generic required/named output fixture
was corrected to the pinned official Hermes structural-tag grammar, not by
altering the production parser.

| Text case | Actual CPU input/render | Official JSON/SSE replay | Router transport | Real model |
| --- | --- | --- | --- | --- |
| String, text-parts, Unicode, history | PASS, inherited actual renderer corpus rerun | PASS Unicode at steps 1/3/11 | Existing raw path; SDK wire replay PASS | PASS selected JSON/SSE fixtures |
| Auto, required, named tools | PASS selected configuration | PASS, official structural-tag semantics | SDK auto-tool JSON/SSE PASS | PASS; 12 direct/Router real tool loops |
| Multiple tool calls and actual tool results in history | PASS replay follow-up inputs | PASS; unique IDs, validated arguments, real local addition | Single-call loop JSON/SSE PASS | Single-call loop PASS; multiple calls NOT RUN on model |
| Thinking/reasoning and suppression | PASS selected flags/history | PASS, no reasoning leakage into content | Unchanged raw forwarding | PASS selected flags |
| JSON object/schema and structured choice | PASS official preprocessing | PASS actual fixture output validation | Unchanged raw forwarding | PASS selected actual outputs, not arbitrary schemas |
| Engine errors / cancelled generator / next request | Not model generation | PASS official output behavior | Existing native lifecycle probes PASS | Request errors and actual Chat abort/recovery PASS; retained initial matcher failure |
| Ordinary SDK headers | Not a tokenizer feature | SDK parses official outputs | PASS auth, user-agent, stainless headers and raw body | NOT RUN |
| n>1, logprobs, other parser/model configurations | Outside this new acceptance slice | NOT RUN | No new API prohibition introduced | NOT RUN |
| Prepared Chat | Not implemented | Not applicable | Always original body | Not implemented |

Do not collapse these columns into a single “Chat supported” checkbox. Cache-key
eligibility is additionally governed by the existing actual-render and Worker
capability gates. Salt/adapters/embeddings/multimodal input do not gain eligibility.

The sole test tool `synthetic_add` validates the complete call set before doing
bounded integer addition. It rejects unknown names, duplicate IDs/argument keys,
booleans, extra keys, oversized or malformed arguments. It never executes model
code, shell commands, network calls or external tools. Follow-up history uses
the returned IDs and the **actual computed results**, not a canned tool result.
Fixed replay proves transport/parser semantics, not model tool-use competence.

The native SDK probe uses the actual production extension with a clearly
synthetic render facade and event-free mock Workers. Four observed backend Chat
dispatches (two two-turn loops), all raw; zero prepared Chat dispatches. Actual
Completion counters remain unchanged. There is no dedicated production Chat
prepared/fallback counter today: the four raw observations are test wire evidence,
not a fabricated Prometheus Chat metric or a cache-hit count.

## CPU execution and provenance

Executed in Linux x86_64 container `cmb-chat-serving-1-dev-x86`, cwd
`/chat-serving`, never the original `/workspace`. Reused the existing CPU image;
no macOS package installation. Prior model assets and production artifacts are
read-only mounts.

- New Chat suite: **14 test methods PASS**, with nested fixtures (not 14 unique
  model requests). Includes actual native/SDK replay, cancellation and dependency
  rejection. Intentional engine-error fixtures emit expected error logs.
- Existing actual renderer regression: **2 test methods PASS**, covering its
  existing 27 named cases, 2 invalid and 5 unsupported cases; overlapping tests,
  not additional unique model-quality coverage.
- Existing performance harness plus the 128/32/four-repeat fixture: **32 tests PASS**.
- Existing render GPU runner and CUDA helper CPU self-checks: PASS. An initial
  invocation used `--self-check` for the older helper; its documented positional
  `self-check` invocation then passed. Preserve the initial usage error.
- Existing actual production native lifecycle probe with CL/CT enabled:
  **3 scenarios PASS** (wire JSON/SSE, timeout, disconnect/shutdown). Synthetic
  facade evidence, not Chat GPU abort evidence.
- No Rust/product source or Cargo.lock change, so no native recompilation or new
  Rust build claim. Full CI is not claimed green; inherited Python formatting
  failures remain a separately tracked release gate.

Actual native is still built from
`2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110`, SHA-256
`f17ccc17cc343b5cea68145a3e73d6442cbd38e2413f6f9a75606a23928e683b`.
Original production wheel SHA-256:
`4c7bf72c70c1311d9fe9883bfa1767434a86cf1b95b806f49447779695b38d27`.
The new tests do not relabel that artifact with their own commit SHA.
Cargo.lock SHA-256 is unchanged:
`000665f280cab3ba36fd6392afd5e9c39f72fcb10d35521e8161ded0d1b4d614`;
PyO3 0.26.0 / Tokio 1.53.1. No new runtime dependency.

Bounded official serving parser CPU observations (one sample per cell, x86
Docker/emulation environment, step=8, not GPU TTFT or an asymptotic proof):

| Output tokens | Plain output CPU ms | Reasoning output tokens | Reasoning CPU ms |
| --- | ---: | --- | ---: |
| 64 | 4.74 | 67 | 1.91 |
| 256 | 10.77 | 259 | 3.43 |
| 1024 | 63.01 | 1027 | 13.21 |

These measure the pinned **ordinary serving** path, including detokenization,
parsing and serialization, excluding prompt rendering. They do not measure the
newer streaming derender replay implementation. No speedup claim follows.
The entire final CPU suite process took 18.27 seconds wall / 14.63 seconds user
CPU / 2.40 seconds system CPU; peak test-process RSS was 1,087,124 KiB. This
includes vLLM imports/initialization and is not per-request or whole-container
memory. A missing `/usr/bin/time` attempt did not execute tests; the retained
final run uses Python's standard `resource` measurement without installing tools.

## Reproduce CPU tests

Use a dedicated Linux environment with the pinned optional vLLM runtime and
public model config/tokenizer assets (no weights needed). Explicit opt-in avoids
silently installing vLLM or downloading assets in ordinary CI.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS=''
export VLLM_CPU_KVCACHE_SPACE=0 PYTHONDONTWRITEBYTECODE=1
export CMB_CHAT_MODEL_DIR=/absolute/public/model-assets
export CMB_CHAT_NATIVE=/absolute/production/vllm_router_rs.abi3.so
export CMB_CHAT_REPLAY_REPORT=/absolute/new-evidence/replay.json
python -m unittest discover -s py_test -p test_chat_serving.py -v
python -m unittest discover -s py_test -p test_kv_performance_harness.py
python scripts/render_bridge_gpu_validate.py --self-check
python scripts/kv_aware_cuda_validate.py self-check
```

Without the opt-in variables, dependency-free helper tests run and optional
tests explicitly SKIP. Do not report these skips as real-vLLM acceptance.

## Reproduce GPU acceptance (new permission required for another window)

**One NVIDIA GPU with 24–32 GiB is sufficient for the proposed correctness
window**, with two exclusive independent DP=1/TP=1 Workers sharing that card.
This is not multi-GPU performance evidence. Proposed initial budget: two hours.
Begin with Qwen3-0.6B; if real tool-use fails, retain that failure. A separately
approved Qwen3-1.7B may be used for tool competence, not repeated attempts until
the smaller model happens to pass. No automatic model download or patch install.

Reuse the existing GPU runbook and capability dependency. Worker arguments must
match Router rendering, including `--enable-auto-tool-choice --tool-call-parser
hermes --reasoning-parser qwen3 --generation-config vllm`. HTTP must be loopback;
two KV event ports must be isolated. Task directory, model downloads, Worker
lifecycle, ports and deadline require explicit authorization.

Add these switches to the existing `render_bridge_gpu_validate.py` command from
`kv-perf-2-gpu-runbook.md`, retaining its PID/log/native/Worker-source arguments:

```bash
--candidate "$(git rev-parse HEAD)" \
--native-source-candidate 2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110 \
--automatic-capabilities --production-validation --kv-load-guard --chat-agent
```

`--native-source-candidate` checks ancestry and permits only docs/Python tests/four reviewed harness files
differences; any product/build-input change requires a new build. The build
manifest still names the actual old native source. The report records both
harness and native identities. No old manifest is rewritten.

The finite extension adds baseline-error cases, 12 text/Agent shapes in JSON
and SSE, actual tool-result follow-ups, Chat cancellation with request-correlated
Worker abort evidence, zero leases and a recovery request. No retry-to-PASS.
It keeps full Worker input-ID equality checks, facade observations, raw responses,
actual Completion metric snapshots and first content/reasoning/tool delta times.
Empty role frames are not useful TTFT. Unexpected schema/tool/model behavior is
retained as failure or explicitly classified unsupported, never a performance PASS.

Before interpreting performance, use existing true product RR and CL raw Chat
on the same artifact/config/trace. Target is correct text-Agent serving, not an
acceleration claim; there is no proposed prepared arm in this increment. Any
timed comparison should use at most three independent rounds and report all
rounds, failures, CPU/RSS, useful TTFT and E2E. No acceptable-overhead threshold
is asserted retroactively from successful samples.

## Remaining gates

Complete applicable CI; broader models/parser combinations and real multiple-call
coverage; human authorship/license and semantic review; upstream interface
coordination; publication approval. The finite Qwen3-0.6B GPU slice is now
recorded separately, not generalized to all Agent behavior.
See `chat-serving-1-reuse-route.md` for why prepared Chat is not enabled.

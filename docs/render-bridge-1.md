# CMB Render Bridge 1: optional in-process preprocessing

## Scope and status

This is a separate increment based on PR1 candidate
`624d8408b2610046db40e48f6d512b3f0741fc02`, on branch
`codex/cmb-render-bridge-1`. PR1's historical CPU/CUDA results do not validate
these changes. No public push, PR, maintainer-approved API, or new GPU result is
claimed here. Publication and any new GPU operation require separate approval.

The optional Python-launched Router reuses installed vLLM text preprocessing
in-process. The native Rust input path remains the default. The cache matcher
remains PR1's static Regular HTTP, independently addressable DP=1, Normal Dense
Qwen3 scope: one full-attention group, unchanged event wire/hash/index/scoring.
Rendering more request shapes does not add another model's cache-layout support.
There is no PD, Hybrid/MTP, dynamic discovery, exact-history, Agent affinity,
tokens-in/out, or inference-engine integration in this increment.

## Production path and source map

The inspected implementation family is vLLM `0.29.0`, tag commit
`98dff2a81d747d1dba01a47f939f48c3526d4206` (also recorded in `Cargo.lock`).
The Python adapter checks the installed vLLM version; this is not a cryptographic
attestation of its Python sources. Record the actual installed wheel/source
hashes with each result, including CPU-build suffixes and local patches.

| Responsibility | Source reused or changed |
| --- | --- |
| Python launch/configuration | `py_src/vllm_router/{launch_router,router_args,router}.py`, `src/lib.rs` |
| Original HTTP bytes | `src/server.rs` → `src/routers/http/router.rs::route_kv_bytes` |
| Admission and dedicated executor | `src/prompt_tokens/bridge.rs::{prepare_bytes,run_executor}` |
| Optional persistent facade | `py_src/vllm_router/render_bridge.py::RenderFacade` |
| Official CLI defaults and model configuration | vLLM `entrypoints/launchers/cli_args.py`, `AsyncEngineArgs.create_model_config` |
| CPU render initialization being followed | vLLM `entrypoints/launchers/render/{entry,app_state}.py` |
| Serving/schema/model/sampling validation | vLLM `entrypoints/scale_out/render/serving.py::ServingRender`, OpenAI Chat/Completion request schemas and `OpenAIModelRegistry` |
| Actual prompt preprocessing | vLLM `renderers/online_renderer.py::OnlineRenderer`, `renderer_from_config` and its installed renderer/tokenizer |
| Independent public HTTP oracle | vLLM `entrypoints/scale_out/render/api_router.py` and official `init_render_app_state` |

The facade creates one persistent event loop and CPU renderer in `startup()` on
the dedicated execution thread. It does not create EngineCore, load model
weights, or allocate GPU KV cache. Like the official CPU render launcher, it
clears model quantization for render-only configuration. Platform/package
coexistence still needs validation in the actual environment; clearing that
field is not a general compatibility guarantee for every model or plugin.

For each request, the bridge owns one admitted raw-byte copy. Python validates
those original bytes through the vLLM schema, then calls the serving facade,
not a copied template engine. An observing wrapper captures the final
`EngineInput` without replacing rendering. Only one supported text token input
with matching public-render output and no unsupported cache identity can
produce owned exact token IDs. Rust stores those IDs plus the contract/epoch
once in the request context and reuses them across compatible retries.

Generation still receives the original bytes at the original Chat or Completion
endpoint; it is not rewritten to a tokens endpoint. The Worker therefore still
preprocesses the request again. This work removes a per-request remote render
call, not repeated Worker preprocessing. It is not tokens-in/out or zero-copy.

## Configuration and package boundaries

Native-only installations need no vLLM Python dependency. The `render` optional
extra declares `vllm==0.29.0`; the present adapter admits the implemented
`0.29.0` version, including a build suffix. Install only a reviewed candidate
artifact and compatible optional dependencies in an isolated environment. No
installation, upgrade, model download, remote-code trust, plugin loading, or
development-mode switch happens automatically at Router startup.

There is one user-facing selection: `--kv-input-backend native|vllm`. For `vllm`,
`--kv-render-config` is the reviewed shared deployment configuration. Its local
model/tokenizer assets, serving arguments, worker URLs, and cache settings must
describe the actual workers. The Python wrapper rejects conflicting Router
overrides. There is no user-managed per-model golden/profile directory.

Save this example as `/absolute/path/to/render-deployment.json`, replacing the
model directory with an existing verified public snapshot. No weights are read
by the facade, although the generation workers need them. A ModelScope snapshot
is acceptable when its actual immutable revision and file hashes are recorded;
do not assert an unproved equivalence to a different Hugging Face revision.

```json
{
  "serving_args": [
    "--model", "/absolute/path/to/Qwen3-0.6B",
    "--tokenizer", "/absolute/path/to/Qwen3-0.6B",
    "--served-model-name", "Qwen/Qwen3-0.6B",
    "--max-model-len", "4096",
    "--generation-config", "vllm",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    "--reasoning-parser", "qwen3"
  ],
  "worker_urls": ["http://127.0.0.1:8000", "http://127.0.0.1:8001"],
  "cache_layout": {
    "kind": "qwen3_dense_full_attention",
    "block_size": 16,
    "hash_algorithm": "sha256_cbor",
    "hash_seed": 0
  },
  "bridge_limits": {
    "max_pending_jobs": 32,
    "max_input_bytes": 1048576,
    "max_tokens_per_request": 65536,
    "max_reserved_tokens": 262144,
    "queue_timeout_ms": 1000,
    "execution_timeout_ms": 10000
  },
  "conformance_timeout_seconds": 10
}
```

If authentication is needed, add `"worker_api_key_env": "CMB_WORKER_API_KEY"`
and supply the secret through that environment variable, not this JSON or logs.
The bridge refuses startup HTTP redirects. No authentication secret enters the
render-contract identity.

After both matching workers are ready, launch the candidate's Python entrypoint
from the isolated environment containing its matching native extension:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
python -m vllm_router.launch_router \
  --host 127.0.0.1 --port 3001 \
  --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --policy kv_aware --kv-input-backend vllm \
  --kv-render-config /absolute/path/to/render-deployment.json \
  --kv-hash-algo sha256_cbor --kv-hash-seed 0 \
  --kv-events-topic-filter kv \
  --kv-events-endpoint http://127.0.0.1:8000=tcp://127.0.0.1:5557 \
  --kv-events-endpoint http://127.0.0.1:8001=tcp://127.0.0.1:5558 \
  --log-level debug
```

Startup rendering and actual-worker public-render conformance must finish before
the Router listens. `worker_startup_timeout_secs` supplies the overall startup
wait limit; the JSON timeout bounds each conformance HTTP call. Requested
configuration, resolved `effective_config`, asset hashes and conformance token
digests are distinct evidence. Finite probes plus a reviewed shared deployment
source are not complete remote attestation: the Worker does not export its full
effective preprocessing identity through this adapter. Operators must not change
worker preprocessing in place while retaining the same declared cohort.

## Safety and unsupported boundaries

- A single OS thread runs Python startup, rendering and close; `PyO3 0.26`
  `Python::detach` releases the GIL around the long-running Rust server. No
  policy/index lock spans Python calls or awaits.
- Admission precedes the bridge's byte copy. The limits bound total admitted
  jobs (including active), retained request bytes and reserved maximum output
  tokens. Only one call executes at a time. With the example limits, token
  reservations permit at most four jobs despite the higher job-count ceiling.
  These are not hard limits on tokenizer intermediates, arbitrary native
  allocations, original HTTP bodies, or total process RSS.
- Dropping a request cancels queued work. Started synchronous Python work may
  outlive the caller; it retains its reservation until actual completion.
  Deadline/cancelled/late results cannot become an exact dispatch. Busy or
  unsupported input can use PR1's existing fair non-affinity fallback only while
  the verified deployment contract remains current.
- Explicit invalidation or a reported contract/epoch mismatch permanently fences
  the provider. Both first dispatch and retries check the saved contract before
  reserving Worker load. Startup identity failure is not a fallback condition.
- Shutdown stops admission and discards queued/late work. The binding waits at
  most 30 seconds for disposal, then reports still-active execution. One Python
  non-daemon `Event.wait` thread prevents normal interpreter finalization while
  the Rust callback remains active; it is released after facade cleanup. It
  performs no rendering and owns no event loop. Bounded `start()`/shutdown return
  does **not** promise bounded process exit. A stuck GIL/native extension needs
  external process management; forceful process exit/crashes are not recoverable
  in-process.
- Python exception messages/tracebacks are not forwarded as request errors or
  logs. Invalid input, unsupported/cache-ineligible input, provider unavailability,
  overload, deadline and cancellation remain distinct outcomes.
  The optional facade temporarily attaches content-free filters to exactly
  `vllm.renderers.hf`, `vllm.entrypoints.chat_utils`, and
  `vllm.entrypoints.scale_out.render.serving`, because v0.29 can log template
  exceptions/tool-history identifiers internally, including from renderer pool
  threads. Logger name and severity remain; message arguments, exception and
  stack text are removed. These three loggers are process-shared, so other
  consumers of them also lose diagnostic detail while any facade is active.
  Each facade removes only its own filters on close; native-only operation and
  unrelated loggers are unchanged. This is a scoped adapter measure, not a
  guarantee about arbitrary third-party libraries or every vLLM module.
- Ordered tool arguments/schema, null/absent fields, supported reasoning fields,
  thinking flags and output constraints go to the actual vLLM schema/renderer.
  No claim is made that these fields are prompt-neutral. JSON duplicate handling
  follows the worker ingress's last-key-wins behavior, not a sorted Rust request
  reconstruction. The Router still forwards the original bytes.
- Batches/multiple EngineInputs, multimodal/embedding inputs, non-null cache salt,
  LoRA/transfer/session cache extras, unknown cache-identity fields and untrusted
  request template overrides never enter the exact KV path. A rendering-capable
  request can still be cache-ineligible. Determinism screening is conservative,
  not a sandbox or proof that arbitrary custom templates are safe.

The first implementation is restricted to local Qwen3 Dense assets. The public
0.6B fixtures remain an independent oracle; a second Qwen3 Dense tokenizer/model
variant exercises the same facade without another Rust renderer. This does not
claim generic support for arbitrary template families, DeepSeek, Hybrid or MTP.

## Validation and measurements

Current-branch results and artifact hashes belong in the separate, candidate-SHA-
bound validation evidence. Commands below are reproduction instructions, not
PASS claims; attach that evidence when publishing this proposal.
Never reuse PR1's native SHA/CUDA report as evidence for a modified extension.
For every result record candidate source identity (and dirty diff when applicable),
actual extension/native hash, Python/vLLM/toolchain versions, asset/config hashes,
cwd, exact command and complete bounded output.

The focused suites are `prompt_tokens::bridge::tests`, Router raw-byte/retry
tests, `py_test/test_render_bridge.py`, and
`py_test/unit/test_kv_render_entrypoint.py`. The opt-in actual-extension probe
exercises GIL progress, dedicated-thread execution, timeout/overload, cancellation
and lifecycle cleanup with a deliberately synthetic facade:

```sh
python py_test/integration/render_bridge_native_probe.py \
  --extension /absolute/path/to/candidate/vllm_router_rs.so \
  --output /absolute/path/to/new-probe-evidence
```

Real CPU render/public-HTTP equivalence uses existing public local assets and
the actual official HTTP render initializer, without an inference engine:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
CMB_RENDER_TEST_MODEL_DIRS=/absolute/assets/Qwen3-0.6B:/absolute/assets/Qwen3-4B-Instruct-2507 \
python -m unittest discover -s py_test -p 'test_render_bridge*.py' -v
```

`CMB_RENDER_BENCHMARK=1` additionally enables the bounded **serial Python-facade
versus persistent official loopback HTTP** measurement in that suite. Warmup is
outside its timed windows; it reports p50/p95/p99, throughput, CPU, RSS and actual
token counts. Model-length-invalid cases are NOT_RUN. This is not the full
native Rust → in-process Rust/Python → HTTP three-way benchmark, does not measure
Rust queue/GIL/conversion intervals, and is not TTFT.
Safe conversion still copies bytes into Python and token IDs back into Rust;
no zero-copy or negligible-conversion-overhead claim is made.

The separate finite three-path harness compares the compiled native Router,
the compiled Router with the actual in-process facade, and persistent official
HTTP render at concurrency 1/2/4. It starts only its own loopback CPU services;
its generation endpoint is explicitly a mock, with no inference or KV-hit claim.
Router timings include both client and mock-worker HTTP; they are not the same
envelope as standalone render HTTP and must not be subtracted to invent a
conversion cost. Test-only observation requires every timed facade call to be
exact, preventing fair fallback from being mistaken for a fast render.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
python py_test/benchmark_render_bridge_paths.py \
  --native-library /absolute/path/to/candidate/vllm_router_rs.so \
  --model-directory /absolute/assets/Qwen3-0.6B \
  --output /absolute/path/to/new-cpu-evidence
```

This records the immutable native artifact, source manifest, warmup-separated
latencies, throughput, CPU/RSS and cancellation/timeout counts. Source files and
HEAD must not change during the run. Tail percentiles from the deliberately
small samples have low statistical confidence. Unsupported native request shapes
and model-length-invalid cases are explicitly NOT_RUN, not measured fallbacks.

## Optional finite GPU extension (not authorized or executed here)

Obtain fresh GPU authorization and a user-owned isolated directory first. Use
two independently addressable vLLM 0.29 workers, not aliases for one process.
Record candidate/extension hashes and the real Worker HTTP/EngineCore PIDs before
running. Start workers sequentially so memory profiling does not overlap; size
memory limits for the selected device. These commands are examples to execute
only after that approval, with the same local assets and preprocessing settings
as the JSON above:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 \
VLLM_USE_RUST_FRONTEND=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
vllm serve /absolute/path/to/Qwen3-0.6B \
  --tokenizer /absolute/path/to/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 \
  --data-parallel-size 1 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --enforce-eager \
  --generation-config vllm --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --enable-log-requests \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv"}'
```

Use a second terminal/process with HTTP port `8001` and KV port `5558`. In this
vLLM publisher, a fixed `tcp://127.0.0.1:PORT` endpoint means **connect**, not bind;
two connecting sockets do not establish event delivery. The wildcard publisher
example is permitted only on an explicitly authorized host whose firewall or
network namespace blocks public access to KV ports. Confirm that protection
before launch. HTTP stays loopback-only. Do not enable broad development mode or
clear live caches as part of these steps.

Then start the Python Router above. This bounded sequence uses one public prompt
long enough for several 16-token blocks:

```sh
CMB_GPU_CHAT='{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Public render bridge validation prefix. Explain how rain forms, how clouds move, and why rivers flow downhill. Use ordinary words and compare evaporation, condensation, precipitation, and collection. This synthetic prompt contains no private information."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl --fail-with-body --max-time 30 -H 'Content-Type: application/json' \
  --data-binary "$CMB_GPU_CHAT" http://127.0.0.1:8000/v1/chat/completions/render
curl --fail-with-body --max-time 30 -H 'Content-Type: application/json' \
  --data-binary "$CMB_GPU_CHAT" http://127.0.0.1:8001/v1/chat/completions/render
curl --fail-with-body --max-time 60 -H 'Content-Type: application/json' \
  --data-binary "$CMB_GPU_CHAT" http://127.0.0.1:8000/v1/chat/completions
curl --fail-with-body --max-time 60 -H 'Content-Type: application/json' \
  --data-binary "$CMB_GPU_CHAT" http://127.0.0.1:3001/v1/chat/completions
```

The first two calls are explicit validation oracles, not production per-request
rendering. For acceptance, preserve raw requests/responses and Router/Worker log
offsets; HTTP success alone is insufficient. Complete this finite matrix, with
at most one request per listed warm/direction/response-mode cell and bounded
event waits (no unlimited retries):

1. Compare complete local facade IDs with both Worker public-render results for
   Completion text/IDs, Chat thinking on/off, ordered tools/history reasoning,
   text parts/unicode and an output constraint. Do not compare only lengths.
2. Warm W0 then route once, and use a fresh public prefix to warm W1 then route
   once; perform JSON and SSE variants. Require a real event-derived positive
   score on the first routed attempt, matching token digest and backend counters.
3. Send a non-null cache-salt request once. Require non-affinity fallback and
   unchanged original generation request, never a guessed cache key.
4. Cancel one queued render request and one active stream. For the stream, require
   a nonterminal first frame, active-before and idle-after observations, and an
   exact request-ID-linked abort counter/log with Worker PID identity. Merely
   observing running=0 is not cancellation proof.

The existing `scripts/kv_aware_cuda_validate.py` binds its evidence to a native
CLI executable/process. It is **not** an unchanged acceptance runner for the
Python-hosted extension: adapt/review its source-and-process identity checks
before claiming its matrix for this backend. The steps above are a finite plan,
not a claim that a new full GPU harness or result already exists.

## Draft PR proposal for #295 / #294

Proposed title: **Add optional bounded in-process vLLM text preprocessing for
Python-launched KV-aware routing**.

This independently reviewable increment wires existing KV configuration through
the Python/PyO3 entrypoint and adds one optional vLLM 0.29 preprocessing backend.
It preserves native defaults and original generation bytes, prepares exact text
tokens once per verified render contract, and reuses them across compatible
retries. A single dedicated execution thread provides bounded admission,
cancellation/deadline handling and explicit interpreter-lifetime protection.

The request-context/contract types here are internal, provisional integration
choices for discussion under #294/#295, not an assertion that maintainers have
accepted a new shared API. No event/hash/scorer/layout/Worker changes are proposed.
Unsupported cache identity retains the existing safe fair fallback; invalidated
deployment identity does not. Repeated Worker preprocessing remains.

Before publication, attach current-candidate validation evidence, actual artifact
hashes and the remaining CPU/GPU NOT_RUN reasons; review raw-byte preservation,
ServingRender-versus-generation equivalence, contract invalidation, bounded
shutdown and attribution/licensing. Human review and publication approval remain
pending. No PR has been created by this document.

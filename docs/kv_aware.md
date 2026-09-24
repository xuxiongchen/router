# KV-event routing for two independent text workers

`kv_aware` uses KV blocks reported by each worker to route a request to a worker
with its longest cached prefix. This initial implementation supports the native
`vllm-router` CLI, static Regular HTTP routing, and independently addressable
DP=1 workers serving **Qwen/Qwen3-0.6B** at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. It does not require a worker patch.

The supported inputs are a Completion string, a single Completion token-ID
array, and the pinned model's text Chat template. Chat permits an optional
initial system message, alternating user/assistant string messages, and a final
user message. Generation prompting is enabled; `enable_thinking` may be true
or false. Tools, multimodal content, reasoning history, adapters, cache salt,
prompt truncation, batched prompts, and unknown input transformations do not
receive an approximate KV score. They use the policy's non-affinity fallback
while the original request is forwarded.

Only complete blocks count. Worker lifecycle changes, publisher discontinuities,
and cache-clear events invalidate ownership. A retained subscriber runtime is
owned by the Router. Missing positive evidence uses fair fallback among eligible
workers. This is a correctness feature, not a multi-GPU performance claim.

KV Events are asynchronous PUB/SUB observations, not a synchronized cache
snapshot. This slice does not replay missed history. Observed gaps or disconnects
purge ownership; a lost final event cannot be detected until a subsequent
sequence or connection event reveals the discontinuity. Affinity never changes
the request's inference semantics, and a selected worker may still need to
recompute an evicted prefix.

## Runtime contract

Use an isolated environment with vLLM **0.29.0**. Both workers must use the same
model/tokenizer revision, unmodified Chat template, block size **16**, hash
algorithm **sha256_cbor**, and **PYTHONHASHSEED=0**. Set
`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` so events carry full hash bytes. The
Router verifies the local `tokenizer.json` SHA-256:

```text
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

The pinned public assets are available from
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).
The Router does not download assets at startup. A matching file does not by
itself establish the workers' configuration: validate the running workers too.

These example commands are for a user-provisioned host and an explicitly
authorized GPU. Choose memory limits that fit two copies on that GPU. Launch
each worker as its own process; two URLs for one worker are not sufficient.
The validation script will check separate HTTP and EngineCore processes.

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_SERVER_DEV_MODE=1 \
vllm serve Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --served-model-name Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 \
  --data-parallel-size 1 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --enforce-eager \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:5557","topic":"kv"}'
```

Launch the second process with HTTP port **8001** and event endpoint
**tcp://127.0.0.1:5558**; retain the other model/hash settings. Development mode
is used only for the isolated real-cache-clear test. Keep these services bound
to loopback. Do not enable development endpoints on a public production server.

Start the candidate native Router after both workers are ready:

```sh
/absolute/path/to/candidate/vllm-router \
  --host 127.0.0.1 --port 3001 --backend vllm \
  --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --policy kv_aware --kv-model Qwen/Qwen3-0.6B \
  --kv-tokenizer-path /absolute/path/to/pinned/tokenizer.json \
  --kv-hash-algo sha256_cbor --kv-block-size 16 --kv-hash-seed 0 \
  --kv-events-topic-filter kv --kv-events-port 5557 \
  --kv-events-endpoint http://127.0.0.1:8000=tcp://127.0.0.1:5557 \
  --kv-events-endpoint http://127.0.0.1:8001=tcp://127.0.0.1:5558 \
  --log-level debug
```

Capture this process's stdout/stderr in a new task-owned log. Explicit endpoint
mapping takes precedence over `--kv-events-port`, which remains a host/port
fallback for unmapped workers. A shared endpoint and `base_port + dp_rank` are
not supported.

## Reproducible validation

Run the focused Rust tests, formatting, check, and Clippy against the actual
candidate checkout on Linux. Run the full applicable regression once the
candidate is stable. The local development policy may reserve `cargo test` and
`cargo build` for the user; in that case record them as **NOT RUN** until the
user supplies results. Do not substitute another checkout's build or historical
downstream GPU results.

The standard-library harness offers an offline `self-check` plus two GPU-host
commands. `python3 scripts/kv_aware_cuda_validate.py self-check` tests evidence
parsing without contacting any server; it does not establish CUDA correctness.
`build` compiles a
clean, committed candidate and records its commit, tree, command, toolchain,
architecture, native path and SHA-256. Run it only after tests pass. Use a fresh
evidence directory outside the source tree:

```sh
python3 scripts/kv_aware_cuda_validate.py build \
  --source /absolute/path/to/clean/candidate \
  --candidate FULL_CANDIDATE_COMMIT_SHA \
  --output /absolute/path/to/new/build-evidence
```

Start the Router using the resulting
`build-evidence/target/release/vllm-router`. The Linux GPU build is specific to
that host architecture; an ARM64 development artifact is not an x86_64 artifact.
After launch, run the finite acceptance matrix with the actual task-owned PIDs:

```sh
python3 scripts/kv_aware_cuda_validate.py validate \
  --source /absolute/path/to/clean/candidate \
  --candidate FULL_CANDIDATE_COMMIT_SHA \
  --native /absolute/path/to/build-evidence/target/release/vllm-router \
  --build-manifest /absolute/path/to/build-evidence/build.json \
  --tokenizer /absolute/path/to/pinned/tokenizer.json \
  --router-log /absolute/path/to/new/router.log \
  --router-pid ROUTER_PID \
  --worker0-pid HTTP0_PID --engine0-pid ENGINE0_PID \
  --worker1-pid HTTP1_PID --engine1-pid ENGINE1_PID \
  --allow-cache-reset \
  --output /absolute/path/to/new/cuda-evidence
```

This validation is deliberately sequential and requires exclusive use of the
two workers. For each input form it warms only W0 directly, then sends the first
Router request for that prefix; a fresh prefix verifies the reverse direction.
`/tokenize` does not warm the KV cache. PASS requires all of the following:

- Both workers return identical exact tokens; their token digest matches the
  Router's decision. The digest is SHA-256 of concatenated big-endian u32 IDs.
- The selected worker has a positive prefix score, and the other has zero.
- Worker request-counter deltas independently prove the actual backend.
- The first direction returns JSON; the reverse direction returns valid SSE.
- A fresh cold prefix has zero scores. A real worker cache reset removes the
  warmed ownership. vLLM queues its clear event until a scheduler step, so the
  harness sends a fresh, nonmatching one-token request directly to W0 after
  reset, verifies that backend, then tests the old prefix through the Router.
  Backend errors and an observed aborted stream release load.
- The live Router `/proc/PID/exe` hash matches the candidate build manifest;
  EngineCore PIDs have vLLM's DP=1 EngineCore process title and belong to the
  corresponding HTTP process. All five processes retain their original
  PID/start-time, command and executable identities.
- Each running worker's `/version` response reports exactly `0.29.0`. These
  responses are saved separately from the harness Python environment's package
  metadata. Missing endpoints, version mismatches and unreachable workers fail
  preflight; an installed local package cannot establish a worker's version.

The script never provisions hardware, uses SSH, starts/stops servers, injects
KV events, or pushes code. `--allow-cache-reset` authorizes resetting the owned
W0 prefix cache for the clear-event case. Without it that case is NOT RUN and
the overall result is INCOMPLETE. A missing counter, missing decision log,
ambiguous backend delta, token mismatch, or missing abort evidence cannot pass.
The default request counter is `vllm:request_success_total`; an override must
name an equivalent completed-request counter, never an approximate cache metric.

`summary.json` is the acceptance result. `worker_versions.json` retains both
live version responses, including failed preflight observations. Per-case JSON files and log excerpts
contain only this run's synthetic requests and observations. Keep full build
logs and machine metadata outside the public patch. Review the evidence before
sharing it. Worker restart/new generation, late events, sequence gaps, and
compatibility with the disabled feature require the corresponding deterministic
CPU tests; a real cache-clear case alone does not prove those behaviors.

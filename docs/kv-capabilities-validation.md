# Finite validation and remaining gates

Read [the capability contract](kv-capabilities.md) first. Nothing here authorizes deployment or installs a Worker patch automatically. No existing GPU/SSH authorization is reused. Only explicitly owned processes may be stopped; never use broad `pkill`, unrelated cache deletion or inherited instance credentials.

## CPU reproduction

Use an isolated Linux environment with Rust 1.95, Python 3.11 for the recorded CPU run, vLLM 0.29.0+cpu, Torch 2.13.0+cpu and Transformers 5.17.0. The new task container is `cmb-kv-capabilities-dev-x86`, source `/kv-capabilities`, evidence `/capability-evidence`, target `/capability-target`. It does not use the original `/workspace`; the ARM dev container remains stopped. No macOS packages are installed.

```sh
export PATH=/usr/local/cargo/bin:$PATH
export CARGO_TARGET_DIR=/capability-target
export PYO3_PYTHON=/opt/render-bridge-venv/bin/python
cd /kv-capabilities
cargo fmt --all -- --check
cargo check --locked --lib --tests --offline
cargo clippy --locked --lib --tests --offline

$PYO3_PYTHON -B -m unittest discover -s py_test -p test_kv_dense_source_oracle.py -v
$PYO3_PYTHON -B -m unittest discover -s py_test -p test_kv_capabilities.py -v
PYTHONPATH=/kv-capabilities/py_src $PYO3_PYTHON -B -m unittest discover \
  -s py_test/unit -p test_kv_render_entrypoint.py -v
VLLM_KV_PROPOSAL_SOURCE=/capability-evidence/engine-proposal/modified \
  $PYO3_PYTHON -B -m pytest -q py_test/test_kv_worker_export.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
  CMB_CAPABILITIES_TEST_MODEL_DIR=/capability-evidence/assets/SmolLM2-135M-Instruct \
  $PYO3_PYTHON -B -m unittest py_test.test_kv_capabilities_vllm -v
$PYO3_PYTHON -B scripts/render_bridge_gpu_validate.py --self-check
$PYO3_PYTHON -B scripts/kv_capabilities_performance.py --self-check
```

For another machine, `VLLM_KV_PROPOSAL_SOURCE` is a separately prepared pinned vLLM source tree with the reviewed patch applied, not the installed package. Without the optional source/assets, corresponding tests explicitly skip; a skip is not a PASS. The synthetic fixture never qualifies a real GPU layout. Test asset revision/hashes are in the review evidence; download only configuration/tokenizer files for CPU checks.

### Focused Rust execution/build gate

These reproduction commands operate on the new candidate, not PR1 or the original checkout. The separately authorized dedicated GPU-host run executed the focused Rust tests and debug build successfully; [the hardware report](kv-capabilities-gpu-results.md) records the exact measured candidate/artifact. Local reproduction still follows the operator's environment permissions:

```sh
docker exec cmb-kv-capabilities-dev-x86 bash -c '
  set -e
  export PATH=/usr/local/cargo/bin:$PATH
  export CARGO_TARGET_DIR=/capability-target
  export PYO3_PYTHON=/opt/render-bridge-venv/bin/python
  cd /kv-capabilities
  cargo test --locked --lib kv_capabilit --offline
  cargo test --locked --lib policies::kv_aware::tests --offline
  cargo test --locked --lib prompt_tokens::bridge::tests --offline
  cargo test --locked --lib kv_ --offline
  cargo build --locked --lib --offline
  sha256sum /capability-target/debug/libvllm_router_rs.so
'
```

`cargo check --tests` type-checks test code but does not execute it. The focused set includes real local HTTP + ZMQ subscriber/metadata integration, stale generations, gap/clear/epoch fencing, Dense boundaries, original raw forwarding and render-once/retry protections. No internal `codex_verify.sh` or release build is required here. Broad upstream/native regressions and release/wheel portability remain separate gates.

## Fresh authorization for another GPU run

Request a new SSH endpoint, dedicated directory and finite time budget. Deployment approval must explicitly include reviewing/applying the single Worker patch and fully restarting **only owned test Workers**, optional cache clear/restart cases, the local Rust test/debug build gate, and the two event ports' network isolation. Do not assume a previous instance or authorization still applies.

Suggested budget: up to 2 hours, one GPU, two independent DP=1 Workers. Run Qwen3-0.6B then SmolLM2-135M-Instruct sequentially, never as a mixed cohort. Keep one identical Router native binary across both runs. Pin test model revisions from reachable public ModelScope repositories; no production revision allowlist. If assets cannot be verified or the actual effective manager is unsupported, stop that cell and report it accurately.

### Worker preparation (human-approved only)

In a separate vLLM checkout at `98dff2a81d747d1dba01a47f939f48c3526d4206`, first run `git apply --check` on [the patch](dependencies/vllm-0.29-kv-capabilities.patch), review it, then apply/build/install in the dedicated environment according to that pinned vLLM source's build instructions. Do not copy individual files over a user's installed Worker. Record the resulting checkout commit, package version, patch hash and changed installed-source hashes. A CPU wheel cannot be reused as CUDA runtime evidence.

After approval, launch each owned Worker with these arguments (ports 8100/5557 for the first, 8101/5558 for the second). The explicit wildcard is required by this pinned publisher's bind/connect convention; it must be network-isolated before starting. HTTP stays loopback.

```sh
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_PLUGINS='' \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve "$MODEL_DIR" --host 127.0.0.1 --port 8100 \
  --tokenizer "$MODEL_DIR" --served-model-name model --max-model-len 2048 \
  --generation-config vllm --data-parallel-size 1 --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 --gpu-memory-utilization 0.40 --enforce-eager \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor --block-size 16 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv","enable_capabilities":true}'
```

Use shell/job supervision to retain exact API and EngineCore PIDs and separate stdout/stderr logs. Probe the fixed endpoint and save actual descriptors before traffic. A 404 means the proposal is unavailable, 409 means the effective mechanism is unsupported; neither is a real-layout PASS. The linked export uses no GPU kernel for metadata but must read an actually initialized inference Worker.

### Exact candidate/native binding and finite runner

In the dedicated Router source checkout, verify a clean expected candidate, run the authorized focused tests and `cargo build --locked --lib` under the same Python ABI, and record command/exit logs. Compute SHA-256 of the resulting debug extension. The build manifest supplied to the runner must contain `status: "PASS"`, `candidate_sha` equal to source HEAD, and `native_sha256` of the actual produced file. Do not generate that manifest from an old copied artifact. Also record Cargo.lock, Python/package/compiler versions and linkage. No global LD_PRELOAD workaround is an installation recipe.

With an automatic render config as in the main guide, start the already prepared Workers, then invoke the bounded public runner:

```sh
python -B scripts/render_bridge_gpu_validate.py \
  --automatic-capabilities --worker-vllm-root "$WORKER_VLLM_PACKAGE" \
  --source "$ROUTER_SOURCE" --candidate "$CANDIDATE_SHA" \
  --native "$NATIVE_SO" --build-manifest "$BUILD_MANIFEST" \
  --render-config "$RENDER_CONFIG" --model model --output "$NEW_EVIDENCE_DIR" \
  --worker0 http://127.0.0.1:8100 --worker1 http://127.0.0.1:8101 \
  --worker0-pid "$API0_PID" --worker1-pid "$API1_PID" \
  --engine0-pid "$ENGINE0_PID" --engine1-pid "$ENGINE1_PID" \
  --worker0-log "$WORKER0_LOG" --worker1-log "$WORKER1_LOG" \
  --event0 tcp://127.0.0.1:5557 --event1 tcp://127.0.0.1:5558 \
  --publisher0 'tcp://*:5557' --publisher1 'tcp://*:5558' \
  --budget-seconds 1200
```

All variables are explicit deployment inputs, not guessed PIDs/paths. The runner launches/stops only its own Router child, maps the actual native inode/SHA from `/proc`, checks source identity and saves complete token/counter evidence. It neither launches/stops inference Workers nor installs patches/models. Use separate output directories for the two sequential model runs. The finite loopback runner does not support Worker API keys; authenticated export is covered separately by CPU tests. On-disk Worker source hashes are provenance, not proof of which Python code an already-running process loaded; start owned Workers only after installation/provenance capture.

Acceptance requires:

- Real descriptor inventory/type/units admitted for both Workers; actual boot topics bound to those instances.
- Full input tokens equal among local facade, both Worker `/render` responses and actual generation, for supported Completion/Chat JSON/SSE shapes.
- Direct-only warmup on each Worker, then first Router request chooses the sole positive owner, corroborated by independent request and prefix hit/query counters.
- N=B-1/B/B+1, 2B-1/2B/2B+1 and long exact-block cases; raw matches and reusable tokens recorded separately. Observed-subset predictions may miss unobserved blocks and concurrent eviction remains a race; never report the prediction as an authoritative cache inventory.
- Cold/unsupported requests have zero affinity; four idle salted fallbacks split 2/2. This is a finite fairness check, not a production load-balancing claim.
- No request-level remote rendering; metadata access may increase from documented background refresh/revalidation, not one lookup per request.
- If separately authorized, reset only an owned Worker cache and demonstrate old ownership is cleared. Restart one owned Worker: new epoch must invalidate the old Router contract; restarting Router repeats conformance and begins empty. These mutation cases are manual/exclusive and are not automatically performed by the runner.

Stop only owned task processes on completion or budget expiry. Any cloud fix must be returned to this contribution branch and all final affected tests rerun against the new candidate/native SHA. Every GPU PASS must identify its actual executed candidate and native artifact; a later documentation commit is not a new measured binary.

## Optional finite performance comparison

After correctness succeeds, `scripts/kv_capabilities_performance.py` compares
`round_robin` and `kv_aware` with the same candidate/native and the same two
exclusive DP=1 Workers. Supply the same source/build/model/Worker identity
arguments as above, omit `--automatic-capabilities`, and use a fresh output
directory. A bounded example adds:

```sh
--requests 32 --groups 4 --prompt-format text \
--input-tokens 1024 --prefix-tokens 768 --output-tokens 32 \
--scenarios locality cold --concurrencies 1 4 --budget-seconds 900
```

The default text mode forwards original Completion text and includes the
actual in-process vLLM preparation cost in KV timings. Setup checks full tokens
against both Worker `/render` endpoints outside the timed interval; generation
must return the same full prompt IDs. Token lengths are measured, not assumed
identical across the different phase nonces. The optional token-ID mode is a
narrower diagnostic that does not measure text tokenization cost.

Each phase has fresh first-block namespaces, the same logical trace and
comparable target lengths. It does not claim byte-identical traffic. Direct
warmup/startup are excluded, policy order is counterbalanced, and all-cold
controls must have zero backend prefix hits. No cache reset or development mode
is enabled by default. An optional reset flag requires separate explicit scope
and an already available approved endpoint; it must not be used to enable DEV.

Report TTFT from request send to the first nonempty generated text, end-to-end
latency, throughput, actual backend prefix hit/query token deltas, per-Worker
completed requests, fairness and sampled Router loads. RR bypasses Router
preprocessing while KV includes it, so this is a finite end-to-end product
comparison, not an isolated scoring benchmark. Two Workers on one GPU, a debug
extension, short synthetic traces, client HTTP setup and token-ID response
overhead limit extrapolation to production. Gains, neutral results and
regressions must all be reported; no universal speedup is an acceptance rule.
The ordinary RR path does not maintain the same Router load counter as KV's
owned lease. A zero RR `/workers.load` is not an idle GPU and must not be
compared numerically with the KV counter. Actual Worker completed-request
counts provide the finite request-distribution evidence.

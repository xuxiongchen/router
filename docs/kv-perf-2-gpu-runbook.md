# Perf-2 finite GPU comparison

This is an execution entry point, **not a GPU result**. It needs new SSH/GPU
authorization, a dedicated directory, a deadline, and permission to manage only
the two named test Workers. No old authorization, deployment hook, GPU identity
or native artifact is inherited. Do not run against unrelated processes.

Use the existing capabilities Worker proposal and fixed vLLM 0.29 environment;
this increment adds no Worker patch. Retain the complete prior evidence. Follow
the optimized wheel and native provenance procedure in [Perf-1](kv-perf-1.md).
Bind every run to the actual candidate SHA, installed wheel SHA, mapped native
SHA, Cargo.lock/build overrides, model assets, Worker process identities and
current `nvidia-smi` UUID. A UUID different from an older preflight is a different
device observation, not proof that the older device was retested.

CT/CLT requires the [immutable Worker cohort contract](kv-perf-2-completion-input.md#required-immutable-worker-cohort).
Before the **first** cohort-preparation hook, verify that no previous Router
remains, both exclusive Workers are drained, and no other clients can use them.
Initial identity preflight does not prove those conditions. Between arms the
runner waits for requests and post-window oracles, verifies drained Workers,
stops and waits for the old Router, then invokes the hook and starts a fresh
Router with startup conformance. Do not replace Workers while that Router is
serving or rely on background metadata polling to make token input safe.

Both GPU runners require a clean checkout whose HEAD equals the candidate and
build-manifest candidate, plus the actual native hash. A later documentation-only
HEAD is not the SHA compiled into an earlier wheel. Either deploy the original
clean compiled candidate with its original wheel/manifest and verify its actual
installation, or rebuild at the newly selected exact checkout SHA. Never relabel
an old native artifact as a newly compiled candidate. The finite run is
documented in [GPU results](kv-perf-2-gpu-results.md).

## Arms and artifacts

| Arm | Render | Load guard | Completion forwarding |
| --- | --- | --- | --- |
| product_rr | ordinary product path | ordinary product path | original |
| C0 | actual vLLM | off | original |
| CL | actual vLLM | on | original |
| CT | actual vLLM | off | prepared IDs when eligible |
| CLT | actual vLLM | on | prepared IDs when eligible |

Both public switches default off. Experimental C0/CL/CT/CLT select the existing
`render_kv` mode in **one** optimized `kv-perf` wheel. The true product RR remains
separate. Existing A/B/C and product_kv arms remain available, but are not
required in every matrix. Do not change forwarding, connections or resource
budgets to attribute their differences to pure render cost.

The final confirmation uses a separate optimized **production** wheel built
without `VLLM_ROUTER_BUILD_KV_PERF`. Pass `--production-validation`: the native
handshake must report `enabled=false`, no experimental modes, and no selected
mode. Merely not selecting a mode in a feature-enabled build is rejected. This
flag allows product_rr plus any of C0/CL/CT/CLT, using ordinary production
dispatch and the same public switches. Install and verify the wheel before
running; pointing the harness at an old extension is not a production check.

## Existing runner, bounded inputs

Run from the authorized Linux checkout. Supply these existing identity and
deployment arguments once in a shell array; all values must be resolved from
the new instance, not copied from old evidence:

```bash
perf_common=(
  --source "$PERF2_SOURCE" --candidate "$PERF2_SHA"
  --native "$PERF2_NATIVE" --build-manifest "$PERF2_BUILD_MANIFEST"
  --render-config "$PERF2_RENDER_CONFIG" --worker-vllm-root "$PERF2_VLLM_ROOT"
  --model "$PERF2_MODEL"
  --worker0 http://127.0.0.1:8100 --worker1 http://127.0.0.1:8101
  --worker0-pid "$PERF2_WORKER0_PID" --worker1-pid "$PERF2_WORKER1_PID"
  --engine0-pid "$PERF2_ENGINE0_PID" --engine1-pid "$PERF2_ENGINE1_PID"
  --worker0-log "$PERF2_WORKER0_LOG" --worker1-log "$PERF2_WORKER1_LOG"
  --event0 tcp://127.0.0.1:5557 --event1 tcp://127.0.0.1:5558
  --publisher0 'tcp://*:5557' --publisher1 'tcp://*:5558'
  --cache-state fresh-cohort --cohort-preparation-hook "$PERF2_APPROVED_HOOK"
  --allow-cohort-preparation --cohort-timeout 180
)
python -B scripts/kv_capabilities_performance.py "${perf_common[@]}" \
  --arms product_rr C0 CL CT CLT --requests 32 --rounds 1 \
  --scenarios locality cold --concurrencies 1 4 \
  --input-tokens 1024 --prefix-tokens 768 --output-tokens 32 \
  --seed perf2-exploration --budget-seconds 3600 --output "$PERF2_EXPLORATION_OUT"
```

The hook is supplied/approved separately; the runner does not install packages,
patch/start/kill Workers, enable a reset API, or connect by SSH. It verifies new
HTTP/Engine processes, changed event epochs and unchanged semantic capability
contracts after the hook. Hook failure does not prove cleanup; hand retained
partial evidence to the owner. Two KV event ports require isolation. HTTP stays
loopback-only. Always reduce the run budget to fit the new authorized deadline,
including shutdown and evidence collection; 3,600 seconds is a ceiling, not an
instruction to exceed a shorter grant.

The example is a 20-phase exploration, not guaranteed to finish inside every
GPU grant. Before starting, freeze the requested subset, order, sample count
and wall-clock budget. Split into explicitly named runs if necessary; do not
reduce samples or discard failed phases afterward to manufacture a pass.

After exploration, confirm the chosen combination against product_rr and C0
with rotated/reversed order (`--rounds 3`) and a longer bounded sample
(`--requests 128`, maximum 256), retaining the ~1K owner-burst and cold cases.
Then run the selected production candidate with `--production-validation`
and its **new** production native/build manifest in `perf_common`; never reuse
the ablation manifest. Re-read the current Worker PIDs before every separate
invocation because a previous fresh-cohort run replaced them.

## Functional regression must enable the new production switches

The older functional matrix with both defaults off is baseline evidence only.
Use the new flags in the existing acceptance runner to exercise the production
changes. This runner attaches the already authorized exclusive Workers; it does
not accept the performance runner's cohort-hook arguments. Supply current PIDs:

```bash
functional_common=(
  --source "$PERF2_SOURCE" --candidate "$PERF2_SHA"
  --native "$PERF2_NATIVE" --build-manifest "$PERF2_BUILD_MANIFEST"
  --render-config "$PERF2_RENDER_CONFIG" --worker-vllm-root "$PERF2_VLLM_ROOT"
  --automatic-capabilities --production-validation --model "$PERF2_MODEL"
  --worker0 http://127.0.0.1:8100 --worker1 http://127.0.0.1:8101
  --worker0-pid "$PERF2_WORKER0_PID" --worker1-pid "$PERF2_WORKER1_PID"
  --engine0-pid "$PERF2_ENGINE0_PID" --engine1-pid "$PERF2_ENGINE1_PID"
  --worker0-log "$PERF2_WORKER0_LOG" --worker1-log "$PERF2_WORKER1_LOG"
  --event0 tcp://127.0.0.1:5557 --event1 tcp://127.0.0.1:5558
  --publisher0 'tcp://*:5557' --publisher1 'tcp://*:5558'
)
python -B scripts/render_bridge_gpu_validate.py "${functional_common[@]}" \
  --kv-load-guard --kv-completion-token-input \
  --budget-seconds 1200 --output "$PERF2_FUNCTIONAL_OUT"
```

Use the production native and its matching build manifest, not the ablation
artifact. Absence of an actual native feature handshake is not accepted as
production proof. The runner clears inherited experimental dispatch/stage
environment settings. Its observation of facade results is a functional test
observer, not headline performance instrumentation.

Without either new switch the previous corpus remains unchanged. With either
switch active, it adds bounded single-text Completion original-vs-Router
equivalence cases: JSON/SSE, real stop-triggering output, Unicode/whitespace,
fixed sampling settings, exact complete Worker prompt IDs, core usage and finish
reason. Explicit prepared cases require the prepared counter to increment, so
silent raw fallback cannot pass. Echo/logprobs cases must remain original;
existing token-array/truncation/special-token and Chat cases remain covered by
the old corpus. Original ingress hashes remain checked for every routed case;
no GPU raw-backend-byte claim is made for transformed bodies or old paths.

Invalid JSON and negative output-length requests must retain direct-Worker
error status and make no backend dispatch. The existing active-stream abort
proof now uses an eligible prepared fixture when the token switch is enabled,
checks actual Worker IDs before closing the stream, proves abort rather than
natural completion, and checks the forwarding counter and final zero load.
Any failed Perf-2 functional case stops the matrix with partial evidence saved.

The deterministic comparison excludes request IDs, timestamps, SSE boundaries,
cache-hit details and logprob floating-point values; raw responses are retained
for review. It does not weaken the production API or omit those response fields.
JSON/SSE stop/usage equivalence is an explicit measured assertion, not a claim
that arbitrary stochastic generations match bitwise.

GPU acceptance does not mutate profiles/epochs or stop Workers to simulate
staleness. Keep the focused Rust CPU tests
`kv_prepared_completion_retry_reuses_body_tokens_and_epoch_fence`,
`kv_render_bridge_epoch_change_blocks_retry`,
`kv_load_guard_generation_change_before_dispatch_releases_lease`, and
`kv_load_guard_completion_chat_json_sse_and_timeout_share_lease` as separate
gates, plus `completion_input` header/payload-limit tests and the separate
`render_bridge_native_probe.py` actual-extension transport/lifetime probe. Run the production
functional command again for the approved Smol configuration on the same native
artifact, with fresh recorded model/Worker identities. Do not claim Smol
performance from these functional results.

Two distinct optional windows reuse the same runner:

- `--scenarios natural --concurrencies 4 --natural-warmup-requests 32`:
  interleaved Router-selected burn-in with distinct tails, no direct owner
  warming, then a timed window in the **same** Router/Worker cohort. Each arm
  still starts from its own verified fresh cohort. This is finite natural-cache
  exploration, not proof of stationary production traffic. Burn-in evidence is
  saved separately and is excluded from reported latency/counter windows.
- `--input-tokens 4096 --prefix-tokens 3072`: a bounded longer-input window.
  Token targets are approximate for text. The runner checks full actual IDs and
  the real `/v1/models` context limits on both Workers before generating; it
  refuses prompt-plus-output over the served context. Maximum retained prompt
  length is 8,192 actual IDs. Do not expand the model's context to make a test fit.

Keep diagnostic stage/trace flags off for headline runs. A separate finite
diagnostic can enable `--stage-timing`; detailed trace also needs `--stage-trace
--log-level info`. Do not add nested stage p95 values or infer unmeasured GIL/GPU
kernel timings. Do not introduce pool or affinity changes into these arms.

## Chat regression is separate, never Completion token forwarding

The same runner has a narrow real-Chat endpoint fixture, without another
benchmark framework:

```bash
python -B scripts/kv_capabilities_performance.py "${perf_common[@]}" \
  --request-kind chat --arms product_rr C0 CL --scenarios locality \
  --concurrencies 4 --requests 32 --rounds 1 \
  --input-tokens 1024 --prefix-tokens 768 --output-tokens 32 \
  --seed perf2-chat-regression --budget-seconds 900 --output "$PERF2_CHAT_OUT"
```

This sends a single user text message with `enable_thinking=false`, prepares
using the actual Chat render endpoint, reads Chat SSE `delta.content`, and
checks actual top-level Worker prompt IDs. It removes Completion-only
`add_special_tokens`. It does **not** claim tools/reasoning throughput or Chat
tokens-in/out. Completion forwarding counters must remain zero even if CT/CLT
are explicitly selected in a Chat invocation. Repeat the chosen production
Chat regression with `--production-validation` and the production artifact.

The Chat template can create a common first complete hash block, so the existing
strict zero-hit `cold` fixture is intentionally unavailable for Chat. Do not
call its template-prefix hits an all-cold pass. Completion retains its original
zero-hit cold case. Chat supports locality, shared, or finite natural windows.

For real-vLLM CPU tools/reasoning/JSON/SSE input regression, reuse the public
`RealVllmRenderTests.test_real_cpu_facade_public_http_and_public_fixtures` in
`py_test/test_render_bridge_vllm.py`, with the frozen model configuration files
and `CMB_RENDER_TEST_MODEL_DIRS` in the isolated vLLM environment. Its existing
corpus includes tool calls and thinking/history variations; it is a preprocessing
equivalence test, not a GPU-generation performance test. The existing
`scripts/render_bridge_gpu_validate.py --automatic-capabilities` remains the
bounded actual-generation/token and stream-cancellation matrix. Its generic
capabilities corpus does not by itself prove full tools/reasoning generation;
any unexecuted configuration-specific cases stay explicit release gates.

## Evidence and acceptance

The same client ingress JSON bytes/order and full prepared tokens must match
across strict pairs. CT/CLT intentionally change **backend** bodies; ingress
hashes do not claim byte-identical forwarding. `trace.json` retains the bounded
literal ingress strings and expected arrays. Post-window probes save actual
Worker arrays from both direct original requests and the measured Router path.
These probes are outside all measured counters. They change output length to
one and request ID echo, so they prove prompt-input identity, not deterministic
equivalence of an entire timed response. The affected functional matrix must
separately cover JSON/SSE, sampling/stop/usage, unsupported fallbacks, headers,
limits, retries, cancellation and errors.

The Router oracle has its own forwarding-counter window: enabling token-ID
echo must not silently turn CT/CLT into raw probes. Its prepared/raw attempt
counts are checked separately from the headline window.

Always-on forwarding counters verify that eligible text CT/CLT requests really
use prepared forwarding rather than silently falling back. Raw arms must record
only raw forwarding. Counters distinguish actual backend attempts and total
ingress/backend bytes; the harness disables retries for performance windows.
Byte totals are not a full backend wire-body capture. Missing counters are an
error, not zero. Actual full token arrays remain bounded evidence, not production
logs. Completion-only findings must not be extrapolated to Chat.

The harness requires successful, complete SSE, exact usage/output token counts,
counter consistency, no timed remote `/render`, unchanged processes/native and
drained Workers. Any measured failure or unresolved request stops subsequent
arms and retains partial evidence. RR Router load remains
`UNKNOWN_UNMAINTAINED`; compare actual Worker running/waiting/cache metrics on
the same basis. Counted hits are tokens, not request-hit rates. Report ingress
and derived payload sizes alongside latency, E2E, output tokens/s and errors.

32 requests are exploratory, not stable p95/p99 or an SLO. Preserve negative
cold results, lower hit rates and throughput regressions. No pre-agreed SLO is
invented. One GPU shared by two Workers is not two independent devices. Final
Smol correctness on the same production artifact does not claim Smol performance
unless separately measured. Chat key functionality/performance, production
extension regressions, applicable CI and human correctness/license/dependency
review remain release gates. Functional success, performance benefit and
publication readiness must be reported separately.

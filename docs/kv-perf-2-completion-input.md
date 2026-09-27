# Opt-in Completion prepared input

This is a separate increment from the cache-first load guard. Both switches
default off and can be enabled independently. Nothing changes in the native
tokenizer admission range; no Worker API, Worker scheduler, hash algorithm,
model-specific renderer, render pool or cache-key format is introduced.

Use the Python/PyO3 vLLM input backend with
`--kv-completion-token-input` (or `kv_completion_token_input=True`). The standalone
Rust CLI does not embed the Python renderer and does not offer this switch.
Native-only startup with the option is rejected, not silently ignored. The
automatic Worker capability cohort is mandatory: legacy fixed-profile render
conformance alone cannot detect a remote Worker restart with a new tokenizer,
and routing-only hints must not become actual generation input on that basis.

## Deliberately narrow first subset

Only an exact, cache-eligible, successful vLLM 0.29 render of a single string
Completion can supply the independent proof marker. The request must explicitly
set `add_special_tokens: false`, have one ordinary non-beam output, and contain
only reviewed generation/stream fields. `return_token_ids` is permitted for the
out-of-band oracle. No special-token setting is rewritten.

Missing/true/null special-token settings, explicit truncation, echo, suffix,
logprobs/prompt-logprobs, offsets, batch/multiple outputs, embeddings, adapters,
salt and unreviewed fields stay on the original path. Existing token-array
requests are never decoded/re-encoded. Unsupported optimization is not an
invalid request. Invalid requests cannot acquire the proof marker.

The fixed vLLM implementation checks `prompt_token_ids` before invoking text
tokenization; the array path skips text pre-tokenization but retains post-token
validation. Therefore simply accepting arrays was not enough: public CPU tests
compare complete final IDs and every effective sampling parameter on Qwen and
Smol, plus invalid-request behavior and actual text-tokenizer call counts. They
do not replace real GPU generation/response validation.

## Transport and lifetime contract

Ingress bytes remain immutable. A distinct bounded body replaces only the
top-level string prompt's byte range with the same complete IDs used for routing.
Non-prompt values, whitespace, missing/null distinctions and numeric spellings
are retained verbatim. Duplicate top-level keys conservatively fall back.
This is **derived semantic forwarding**, not byte-identical backend forwarding.

The reviewed header allowlist retains normal Bearer/Basic authentication and
request/trace headers. Unknown or signed-body headers and nonidentity content
encoding cause raw fallback. Derived requests recalculate Content-Length and
transport framing, remove identity Content-Encoding, and respect the configured
final payload bound. A larger token JSON array may cost more than the text.

The derived body is created once outside retries. Every attempt rechecks the
active render contract and binds selection/reservation to the current Worker
generation, including CT with the load guard off. No per-request metadata RPC,
remote render or additional render is introduced. A stale pre-dispatch rejection
is local, not a backend failure. Existing retry policy is retained; a dispatched
token request is never blindly retried as original text. Streaming responses
are not replayed after headers have been returned to the client.

Worker remains responsible for Completion JSON/text/SSE output, stop processing,
usage and necessary validation. Its payload is not KV tensors and token IDs are
not a complete cache key; uncached GPU prefill/decode still run normally.

## Observability and validation

Always-on, low-cardinality counters
`vllm_router_kv_completion_forward_total{mode="raw|prepared"}` and
`vllm_router_kv_completion_payload_bytes_total{kind="ingress|backend"}` identify
actual KV Completion send attempts, including retries. Zero series are registered
at startup; missing metrics must not be read as zero. These are not cache hits.
Optional stage timing measures `completion_backend_body`; main performance
windows keep detailed timing and trace off. Body contents/IDs are not logged.

See [the finite GPU runbook](kv-perf-2-gpu-runbook.md) for C0/CL/CT/CLT and real
product RR, followed by feature-off production wheel confirmation. Keep cold
negative results and report payload size. No performance benefit, all-model
equivalence, GPU validation or publication readiness follows from CPU success.

## Chat and render scope

Chat keeps original forwarding and the existing tools/reasoning path. The load
guard is shared by supported Completion and Chat. Complete Chat tokens-in/out is
not implemented: the pinned vLLM 0.29 streaming derenderer explicitly rejects
configured reasoning/tool parsers. Completing that separate protocol and output
state machine cannot be represented as a safe prompt-field replacement. No
tools/reasoning feature is removed to claim a faster Chat path.

No offload restructuring, tokenizer cache, affinity default or multi-executor
pool is included. First measure these two independent increments; existing
non-saturated/bursty executor evidence does not justify another concurrency
implementation here.

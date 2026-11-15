# HTTP Completion inputs for KV-aware routing

Regular HTTP KV routing uses each Worker's existing input preparation contract.
Text is prepared by that Worker; explicit token IDs use the direct path, with
separate prefixes for separate prompts. Retries reuse prepared input and make a
fresh selection and reservation. Workers receive the validated ingress bytes.
Ordinary HTTP deployments do not need local tokenizer assets.

```sh
vllm-router --host 127.0.0.1 --port 19000 \
  --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --policy kv_aware --intra-node-data-parallel-size 1
```

The Worker metadata/discovery contract supplies the event publishers and cache
identity. Explicit sources use `PolicyConfig.static_kv_event_sources`; this is
the existing configuration API, not a separate environment mapping. Unknown or
incompatible sources remain routing candidates without cache credit.

Worker input and identity contracts govern cache credit. Unknown extensions,
cache salt, adapters and other unmodeled identities remain routable without
cache credit; known input tokens still contribute their full input cost.
Session binding skips optional preparation RPCs, not known token accounting.

The Router does not load Completion tokenizer assets or validate IDs against a
local model vocabulary. The selected Worker retains input validation and serving
errors. Per-worker preparation may use render/tokenize RPCs; direct ID evidence
needs no RPC. Failed preparation removes cache evidence, not the original body
or ordinary forwarding path.

Observed placement estimates are not actual APC hit counts or complete
bootstrap/generation/recovery guarantees. See
[the HTTP consumer RFC](https://github.com/vllm-project/router/issues/405) for the
real-Worker acceptance requirements.

### Optional Completion history

`--kv-aware-completion-history` (off by default) can break a lowest-cost tie
between independently addressed Regular HTTP workers when `--router-temperature 0`
(the Router's selection temperature, not the request's sampling temperature).
A valid session binding wins first. This is a proposed subset of
[#295 §4B](https://github.com/vllm-project/router/issues/295), not an accepted
change to its ordering. If any retry-eligible target has positive event credit
(even below one billable token), the usual scorer decides. Otherwise, use the
longest exact overlap in **complete-block tokens** at each registration's block
size; do not exceed the ordinary pick's work or backlog. Both the scorer cost
and the load components must tie. A history-selected worker can become the
session's binding under the existing affinity rules and its separate TTL.
History never grants KV credit: a known 33-token request with no event credit
still reserves 33 tokens. Preparation uses the existing Direct or per-worker
tokens; enabling history sends no additional preparation RPC.

Only plain, single-prompt requests whose explicit model matches the registered
cache identity qualify. Successful upstream **headers**, not dispatch or body
completion, record the selected worker's prefix. A later body failure does not
undo that observation. An error or cancellation before headers records nothing.
Records expire 60 seconds after a successful observation. Observing the same
retained prefix on the same registration refreshes one entry; looking it up
does not. Each policy keeps at most 128 entries of at most 256 IDs each
(32,768 retained IDs, not a process memory limit). At capacity, an oldest
entry from a most-represented registration is evicted. A block larger than the
prefix cap cannot yield a hint. Replacement or removal fences the old registration.
KV Remove/Clear changes placement evidence, not the fact that the request was
routed. An unobserved same-address engine restart cannot be identified by this
local history. This option does not enable P/D, gRPC, IGW or Program history.

`vllm_router_prefix_routing_decisions_total` retains one observation per actual
stage decision; retries can produce multiple decisions for one user request.
History reasons use the separate `vllm_router_history_decisions_total` counter,
with fixed `phase="regular_history"` and `reason` labels. `eligible_ordinary`
means a qualifying overlap kept the ordinary pick; `override` means it changed
the pick. Neither is an APC hit or proof of current cache residency.

### Sequential Prefill history (P/D)

`--kv-aware-prefill-history` is a separate, default-off option for static,
independently addressed DP1 HTTP Prefill workers using `kv_aware` and sequential
NIXL pull. It accepts plain, single-prompt Completions for an explicitly matched
model. With zero **routing** temperature, no positive event credit, and equal
work and occupancy, the longest unexpired exact prefix can break the Prefill
tie. A hint does not reduce the reservation. Decode selection and affinity do
not use this history.

Prefill headers do not record a hint. The router consumes the bounded 2xx body
and checks the single Completion and its current attempt's NIXL pull metadata
(including request IDs, remote engine, endpoint, a nonempty block group, and
`remote_num_tokens` matching the entire prepared input, not just its first 256 IDs).
The proposed observation is correlated with the selected HTTP P's current
registration, model, rank and exact prepared input before dispatch, then with
that attempt's full response before recording. A static P's `instance_id` is
its Worker URL; the NIXL `remote_engine_id` is a different namespace. They are
not compared. Handoff shape does not independently authenticate a connector
engine, endpoint, boot identity, transferred range or current cache residency.
A hidden remote restart that reuses the live HTTP registration is not detected.

This slice requires explicit `pd_kv_event_sources` enrollment with valid
model/block metadata and a usable source binding for the P. It serves the
window with no applicable positive event credit, not a general deployment
without event sources. Missing enrollment reports `missing_static_source`;
missing source/input evidence keeps ordinary selection and accounting, without
inventing proof of zero credit. The ordinary forwarding path stays usable.
The advisory contract remains proposed under
[#575](https://github.com/vllm-project/router/issues/575).

The proposed first slice covers one block group and TP1/PP1/DCP1, not push, multi-rank,
discovery, EPD, Program, Chat or adapters. It shares the bounded store above.
Known output controls such as stop/seed do not change prompt eligibility;
unknown extensions remain ineligible. P's response is buffered and nonstreaming
even when the client requests a streaming D response. A qualifying P
observation survives a later D-only failure; every new attempt still needs its
own handoff. Local registration replacement invalidates old tickets and history.
The existing 60-second history TTL is advisory, not a remote ownership lease.

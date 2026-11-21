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

For compatibility versus opt-in strict event recovery, see
[`--kv-event-strict-recovery`](../cli_reference.md#workers-and-backend).

### Optional prefix history

Add `--kv-aware-completion-history` to a Regular `--policy kv_aware --router-temperature 0` invocation (off by default). With no usable session binding or positive event credit, a recently observed whole-block token prefix can break a load tie. A hint expires about 60 seconds after the last successful observation. This is a preference, not evidence that KV is still resident.

It requires a registered single-rank worker with a verified model, block size and KV source, and one eligible prompt with backend-effective IDs. HTTP workers support single-prompt Completion and Chat through the existing input preparation; gRPC workers support token-ID Completion. Responses, native generation and gRPC Chat keep ordinary routing without history.

### Optional Prefill history

Prefill history is off by default and can be enabled only through the embedded
Rust configuration API. Start with a static HTTP P/D `ServerConfig` with a
`kv_aware` Prefill policy and zero Router temperature. In its `RouterConfig`,
set `kv_aware_prefill_history` and enroll each Prefill worker by its configured
URL with a named model matching the request and cache registration. For one
configured Prefill worker, the setup looks like this:

```rust
use vllm_router_rs::{
    config::PdKvEventSources,
    server::{self, ServerConfig},
};

async fn start(mut config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    config.router_config.kv_aware_prefill_history = true;
    config.router_config.pd_kv_event_sources.insert(
        "http://127.0.0.1:8000".into(),
        PdKvEventSources {
            endpoints: vec!["tcp://127.0.0.1:5557".into()],
            replay_endpoints: None,
            block_size: 16, // Must match the Prefill worker's block size.
            model: Some("served-model".into()),
        },
    );
    server::startup(config).await
}
```

There is no CLI or Python toggle. URL-derived source identities, including
those supplied by `--pd-kv-events`, do not establish the named identity needed
for history. The feature covers one prompt in HTTP Completion, DP1/TP1, one
block group and sequential NIXL pull, not discovery, E/P/D or Program scheduling.

The Router records an advisory prefix only after consuming the complete bounded
successful P response and correlating its handoff with the selected registration,
request and entire prepared input. A static `instance_id` is the Worker URL,
not the NIXL engine ID: this does not authenticate the remote engine or prove
transfer. A later D failure does not erase a valid P observation; every new
attempt still needs its own handoff.

Only equal-cost P choices with no positive event credit anywhere in the feasible
pool can use the hint. Missing evidence disables history, not normal forwarding.
History never discounts token work: a 32-token observation for a 33-token input
still reserves all 33 tokens. Decode selection and stage cleanup are unchanged.
The existing bounded history reasons distinguish ineligible input, missing
evidence, credit precedence and rejected handoffs; CPU coverage does not
establish APC or latency gains. Design and acceptance remain in
[#575](https://github.com/vllm-project/router/issues/575) and
[#378](https://github.com/vllm-project/router/issues/378).

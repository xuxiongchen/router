# HTTP Completion inputs for KV-aware routing

The Regular HTTP router can prepare one text prompt or one token-ID sequence
before KV-aware selection. Preparation runs once; retries reuse those IDs while
selection and request accounting run for each attempt. The Worker still receives
the original validated JSON bytes and performs its usual Completion serving.
Chat, batches and unsupported input extensions retain their ordinary path.

This is an opt-in contract for a static, independently addressable DP1 cohort
using Normal Full Attention. The operator must deploy immutable local frontend
assets that match the Workers' effective Completion tokenizer and model
vocabulary. A served-model alias is not remote attestation. Compare complete
Worker input token arrays for the deployed configuration before enabling credit;
bounded probes alone do not prove all inputs or authenticate a remote object.

```sh
export VLLM_ROUTER_MODEL=/models/example
export VLLM_ROUTER_COMPLETION_MODEL_ALIAS=example
export VLLM_ROUTER_KV_EVENT_SOURCES='{
  "http://127.0.0.1:8000": {
    "endpoints": ["tcp://127.0.0.1:5557"],
    "replay_endpoints": ["tcp://127.0.0.1:5559"],
    "block_size": 16,
    "model": "example"
  },
  "http://127.0.0.1:8001": {
    "endpoints": ["tcp://127.0.0.1:5558"],
    "block_size": 16,
    "model": "example"
  }
}'
vllm-router --host 127.0.0.1 --port 19000 \
  --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --policy kv_aware --intra-node-data-parallel-size 1
```

`VLLM_ROUTER_MODEL` must name a local directory with `config.json`,
`tokenizer.json` and the frontend's other required configuration. The nonzero
model `vocab_size`, tokenizer and sibling added-token configuration are captured
at startup through the existing frontend cache. If `VLLM_ROUTER_TOKENIZER` is
set, it must resolve to that same directory or a file within it. Asset changes
require a restart; there is no request-time reload or file watcher.

`VLLM_ROUTER_KV_EVENT_SOURCES` is a JSON string, not a file path. Its explicit
model and block size must describe each actual Worker. The static source shape
is adapted from the P/D discovery path; it skips HTTP metadata calls for listed
Workers and uses an empty event topic. Configure the publisher accordingly.
Replay endpoints are optional. Without static entries the existing metadata
discovery remains available; absent or incompatible metadata gives load-only
selection, not fabricated KV credit.

The request model must be absent or equal to the configured alias to prepare
Completion tokens. Every available candidate must also have matching discovered
model identity and rank zero before this vector is used for placement. LoRA,
salt, session extensions, truncation and unproved modifiers receive no credit;
legal requests still reach the Worker without dropping their fields. Known
invalid explicit IDs fail before dispatch, and tokenizer failures are service
errors, not an unsupported-input fallback. Scheduling-only integer `priority`
does not change the input tokens; Worker-side scheduling validation still applies.

Dense ingestion rejects a whole batch before mutation when an extension cannot
be represented by its token-only query identity. It also rejects contradictory
residency metadata rather than converting a remote observation into local Device
credit. Live and replay ingestion use the shared strict wire decoder and validate
complete data topics separately from the empty replay terminal. Valid generic
integer hashes, ranks and storage tiers remain accepted.

This provides observed placement estimates with the existing cost model. It does
not establish complete bootstrap, remote generation authority, durable recovery
or actual APC hit counts. Debug logs distinguish prepared input from the tokens
actually admitted to selection; logs never maintain the safety state.

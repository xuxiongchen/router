# Worker capabilities for the optional vLLM input backend

This is a proposed, opt-in follow-up to Render Bridge `f0f02adb64a26d819b0e6e9e501a37b8a9d71f09`, not a change to PR1 `624d8408b2610046db40e48f6d512b3f0741fc02`. It does not depend on historical PR #130. Publication/base selection remains a human/maintainer decision.

## What changes

`"kv_capabilities": "worker"` in the render configuration replaces the legacy Qwen-family gate **only for the vllm input backend**. The Router obtains effective KV facts from both actual Workers, verifies one compatible static cohort, then uses the existing in-process vLLM 0.29 renderer. A new model does not require a model-specific Router renderer or a handwritten Profile when it uses this already supported mechanism.

This does not enable arbitrary models. Initial admission is one complete ordinary full-attention group, Normal execution, DP/TP/PP/DCP/PCP=1, local GPU prefix cache, exact full-width `sha256_cbor`, numeric u32 seed, no quantization/LoRA/multimodal/speculation/connector/offload. The effective manager/spec/coordinator must be the audited built-in implementations. Model family labels do not establish those facts. Unknown required semantics fail closed; harmless optional descriptor extensions are ignored.

Native tokenizer/template support is unchanged. The legacy manual `cache_layout` vllm configuration also retains its existing Qwen gate. The ordinary KVEventPool, event decoder, block hashing algorithm, DP workers, retries and original request forwarding are unchanged. No tokens-in/out, scorer/history/PD features or renderer executor pool are added.

## Accurate dependency and proposed transport

The inspected Python package is `vllm 0.29.0+cpu`; source identifies `98dff2a81d747d1dba01a47f939f48c3526d4206`. Cargo.lock independently pins the Rust vLLM dependencies at that commit, and PyO3 at 0.26.0. A Cargo pin alone does not prove the Python installation identity.

Stock 0.29 has `EngineCore.get_kv_cache_group_metadata()` and an internal utility-RPC chain, but no sufficient public capability endpoint. `/server_info` is development-only configuration information, not this effective-runtime export. The separately delivered [linked engine proposal](dependencies/vllm-0.29-kv-capabilities.patch) adds one fixed read-only proposed `GET /v1/kv-cache/capabilities`, via a fixed AsyncLLM method and the existing utility transport. The existing `/v1` bearer middleware protects it when Worker API keys are configured. No arbitrary RPC, private socket discovery or DEV_MODE is required.

**Affected Workers need this reviewed patch and a full restart. Router startup never installs it.** Enable it explicitly in `--kv-events-config` with `"enable_capabilities": true`. Stock/default behavior is unchanged when disabled. The proposed API/type names are not maintainer-approved; coordination remains open with [#294](https://github.com/vllm-project/router/issues/294) and [#295](https://github.com/vllm-project/router/issues/295). This is one linked change, not the old multi-patch chain.

| Required fact | Actual source / object | Unit / validity | Exposure gap addressed |
|---|---|---|---|
| Complete initialized groups | EngineCore initialized worker cache configs before scheduler normalization; scheduler `kv_cache_config.kv_cache_groups` | All required layer/spec groups after initialization | Existing group metadata alone can lose inner spec distinctions during normalization |
| Effective mechanism | `KVCacheManager.coordinator`, `single_type_managers`, actual group specs | Exact audited class identity, not class names/MRO/model labels | Proposed exporter verifies the full initialized inventory and effective path |
| Block units | spec `block_size`, manager effective `block_size`, scheduler/hash pool size, coordinator alignment | Tokens; all equal in this first mechanism | Distinguishes allocation, hashing and reuse units |
| Hash namespace | actual request hasher, `get_none_hash_seed()`, initialized `NONE_HASH`, full-byte event setting | Canonical CBOR / 32 bytes / actual seed root | Requested CLI/env values alone are insufficient |
| Execution restrictions | effective scheduler/config, CP sizes, speculation/lookahead, connectors/offload | Normal, all parallel degrees 1 | Not inferred from the Router CPU renderer |
| Serving/cache namespace | actual model config, served alias, revision/dtype/cache dtype, weight version | One static compatible cohort | Not a mandatory model-weight checksum/catalogue |
| Publisher identity | actual stock ZmqEventPublisher, rank, effective topic, cached LAST_ENDPOINT | Per-publisher boot epoch and exact topic | Stock sequence resets and has no epoch |
| Dense reuse | scheduler -> manager -> unitary coordinator -> FullAttentionManager | Prepared input N, logical token hits | Distinguishes stored matches from reusable prefix |

The example wire shape is [a synthetic public fixture](../tests/fixtures/kv_capabilities/descriptor.json), **not a Profile to fill in or deploy**. Wire schema and mechanism semantic versions are separate. Worker URL is the configured owner; DP rank is zero. In this static single-engine case, the engine-owned publisher boot UUID binds that owner to its event source; it is not the unrelated handshake instance_id.

The new publisher topic is `<configured-topic>.<boot-uuid>`. The payload remains the same three-frame vLLM event format. The exporter reports a next-wire-sequence watermark without incrementing it. Resolved endpoint information is captured during socket setup, not by accessing a ZMQ socket from a foreign thread. The Router connects only to its configured Worker-to-event mapping, checks exported ports, and compares the entire topic (ZMQ subscriptions by themselves are prefix matches).

## Configuration (no manual layout)

```json
{
  "serving_args": [
    "--model", "/models/current-model", "--tokenizer", "/models/current-model",
    "--served-model-name", "model", "--max-model-len", "2048",
    "--generation-config", "vllm"
  ],
  "worker_urls": ["http://127.0.0.1:8100", "http://127.0.0.1:8101"],
  "kv_capabilities": "worker"
}
```

Use `--policy kv_aware --kv-input-backend vllm --kv-render-config <file>` and the existing two `--kv-events-endpoint HTTP_URL=tcp://HOST:PORT` mappings. Do not pass a topic override: the effective epoch topic is observed. `cache_layout` and automatic discovery are mutually exclusive. Omitted hash/block/seed options resolve to verified facts; explicit conflicts report requested/effective values. Native omitted defaults remain unchanged. Optional `worker_api_key_env` names an environment variable; credentials are not stored in the descriptor. Local preprocessing assets remain explicit, deterministic and offline. Worker layouts are never inferred from those assets.

## Ownership lifecycle and limitations

Registration reads metadata, existing startup probes compare actual local/Worker render tokens, and a second metadata read fences replacement during conformance. The Rust subscriber then independently fetches and validates the descriptor before accepting future events. HTTP reads/body sizes and subscriber resources are bounded.

Metadata is not a block inventory. Initial/reconnected views are empty observed subsets; old cached prefixes may be missed. Same-boot gaps, reconnects, health transitions, malformed events and clear events purge state before revalidation. Clear drops same-batch stores conservatively. Event callbacks and metadata replies are fenced by the existing ownership generation; a late callback cannot repopulate a retired generation. Successful metadata refresh never restores entries. Full 32-byte hashes are preserved.

A changed boot/topic/source or incompatible namespace/mechanism invalidates the fixed render contract. **Restart the Router to repeat automatic conformance before using the replacement Worker.** Merely matching a model path does not prove replacement tokenizer/template equality. This increment does not claim transparent reboot recovery, snapshots, replay, complete-cache knowledge or exactly-once delivery. Online weight/config mutation is outside the static contract and requires full Worker and Router restart.

Unavailable metadata in an already verified same-boot deployment revokes affinity, while independently valid raw-forwarding fallback remains available. Unsupported effective mechanisms or changed input contracts are not silently converted into exact token/cache support. Polling is control-plane only: registration/revalidation plus a 30-second background refresh, never per request. A request invokes the existing bounded render executor once and retries reuse its result. Timeout does not mean synchronous Python computation has stopped.

## Stored blocks versus reusable tokens

Pinned source uses fresh prepared `Request.num_tokens` (including BOS/template tokens), then `max_cache_hit_length = N - 1`, then full-attention block alignment. For **this verified Normal mechanism only**:

```
reusable_tokens = B * min(contiguous_matched_blocks, floor(max(N - 1, 0) / B))
```

The preserved GPU observation N=464, B=16, matched=29 therefore allows 448 reusable tokens. It is not a bad prior result: 29 is stored coverage, not 464 tokens of saved work. The new path ranks reusable tokens, so 28 and 29 matches tie in this example; existing least-load/fair ties apply. `prefix_blocks` retains raw matches; `reusable_prefix_tokens`, `query_tokens` and `score_kind` identify the new score. Legacy scoring remains advisory raw coverage. Predictions are observed-subset hints, not guarantees against concurrent eviction or subsequent scheduling/preemption.

## Public validation and status

| Area | Status / exact scope |
|---|---|
| Dense source oracle | CPU source-executed scheduler/manager/coordinator methods against inert read-only pools; 18 explicit boundary/hole/removal/clear fixtures |
| Worker proposal | CPU real initialized manager/spec objects, controlled scheduler/engine configuration, real FastAPI auth route and loopback ZMQ; utility transport is controlled, not a GPU EngineCore |
| Python discovery/entrypoint | CPU HTTP bounds/errors/auth/counters, automatic defaults/conflicts, no request-time discovery/render RPC |
| Non-Qwen preprocessing | Real vLLM 0.29 HfRenderer and official HTTP renderer; SmolLM2-135M-Instruct, 10 input shapes, full token equality; descriptor is explicitly synthetic |
| Rust descriptor/subscriber/selection | Public focused tests added and type-checked; execution remains a user-run gate under the current repository permission |
| Real Worker layout/generation/KV hits with this candidate | **Not yet GPU-verified**; requires fresh authorization, reviewed Worker update/restart and exact new native build |
| Release ABI / production TTFT | Unvalidated, separate release gates; no global LD_PRELOAD recipe |

SmolLM2 test assets are config/tokenizer-only, from ModelScope `HuggingFaceTB/SmolLM2-135M-Instruct` commit `c134cb42e51e0d1f29041149173377c623c99b25`; no weights were needed for CPU preprocessing. This revision pin is test provenance, not a production allowlist. Qwen3-0.6B remains the subsequent hardware baseline; use the **same new Router binary** for both model runs, sequentially with two independent copies of one model at a time.

Commands and outstanding acceptance gates: [validation guide](kv-capabilities-validation.md). Full results must bind the exact candidate/tree and actual native artifact. The old Render Bridge native SHA/results must not be relabeled as this candidate's evidence.

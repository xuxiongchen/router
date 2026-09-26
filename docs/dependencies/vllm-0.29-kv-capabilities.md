# Linked vLLM capability proposal (not a stock API)

Base: vLLM `0.29.0`, source revision
`98dff2a81d747d1dba01a47f939f48c3526d4206`.
The actual audited CPU wheel is `0.29.0+cpu`; its `_version.py` reports
`g98dff2a81`. All six modified original files and ten supporting runtime source
files were byte-compared with that pinned Git checkout and matched. The patch
was generated from preserved original/modified source trees, not installed
Worker edits. See the adjacent provenance JSON for exact original and proposed
file hashes; these are reproducibility evidence, not a production allowlist.

`vllm-0.29-kv-capabilities.patch` is an independent engine proposal, not the old
six-patch stack, a published PR, or a maintainer-approved interface. Apply it
only to a separately reviewed source checkout, after explicit deployment
authorization. Affected Workers **must be updated and fully restarted**.
Installing or enabling the Router alone does not add this HTTP API.

## Interface and deployment

Proposed fixed `GET /v1/kv-cache/capabilities`, with existing `/v1` bearer
authentication when Worker API keys are configured. Disabled returns 404;
unsupported initialized mechanisms return 409; utility deadline returns 503.
No arbitrary RPC method or arguments are accepted. No DEV_MODE is required.
The timeout abandons the HTTP response; it does not claim synchronous engine
computation stopped.

The API calls the fixed AsyncLLM method, then the existing
`engine_core.call_utility_async("get_kv_cache_capabilities")`. EngineCore exports
effective initialized objects. The original `get_kv_cache_group_metadata()` is
used and cross-checked; its stock fields alone do not establish support.
In particular, `generate_scheduler_kv_cache_config()` normalizes an original
`UniformTypeKVCacheSpecs` group to its first member. The opt-in patch retains
the complete original initialized Worker group/spec inventory before that
normalization, and verifies every leaf and layer name. An ordinary uniform
group with differing head dimensions is accepted only when all leaf mechanisms
and token units match; a hidden window/custom spec is rejected.

Explicit opt-in (together with existing prefix-cache/event flags):

```sh
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 vllm serve MODEL \
  --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:*","topic":"kv","enable_capabilities":true}'
```

The descriptor records both configured and resolved endpoints. Production
Router endpoint mapping remains an independently configured trust boundary;
the Router never connects to an arbitrary host supplied by metadata. Prefer a
reviewed fixed reachable event port for ordinary deployments. The loopback
wildcard-port example is for source/finite local tests.

Wire schema v1 and mechanism semantic v1 are distinct. Endpoint naming, shared
types, and the experimental opt-in remain #294 coordination topics. This patch
does not claim endorsement or backward compatibility with future semantics.

## Recognized subset

Only exact audited built-in Scheduler/AsyncScheduler, KVCacheManager,
UnitaryKVCacheCoordinator, FullAttentionManager and FullAttentionSpec identities
grant the capability. Empty inventory, custom subclasses, windows/chunks,
non-causal attention, speculation, CP, DP/TP/PP other than one, transfer/offload,
LoRA, multimodal and online weight-transfer configurations are unsupported.
No model-family string affects the decision. Router v1 currently additionally
rejects non-null quantization; the exporter reports its actual configured value.
The actual stock request-hasher
closure, hash block units, resolved root hash, and bytes/full width are checked.
The Router's current seed representation is uint32; nonnumeric/default vLLM
seed strings and out-of-range values are explicitly unsupported, never mapped
to a different seed. Source hashes are evidence, not a production allowlist.

`hash.extra_keys="none"` defines the supported request subset; it does not
claim vLLM never supports extra keys. Router request gates must still exclude
salt/adapter/multimodal/prompt-embedding and cache-read-disabled requests.

Online model/config/weight mutation is outside this static deployment contract
and requires full Worker **and Router** restart. The boot epoch is not a live
configuration-change detector or weight-file attestation.

## Event binding

Opt-in stock ZMQ publisher generates a UUID before the publishing thread starts
and appends `.` plus that UUID to the configured topic. The descriptor includes
this exact effective topic, publisher epoch, rank, and next wire sequence.
The UUID identifies the one publisher owned by this static EngineCore; in this
restricted one-engine/one-publisher deployment it supplies the engine-incarnation
binding. It is not the older handshake's `instance_id`, nor a cross-process
attestation for unsupported custom publishers or layouts.
`LAST_ENDPOINT` is captured in socket setup, not read cross-thread by metadata.
The publishing thread increments the watermark under a short lock before send;
metadata snapshots never increment it. A snapshot may conservatively fence out
an in-flight send. The original three-frame event payload remains unchanged.
With opt-in disabled, the original topic and sequence path are retained; the
new config flag is removed before calling legacy custom publisher constructors.

Consumers must verify exact topic equality, not merely ZMQ prefix subscription.
On bootstrap/reconnect/gap, begin an empty observed subset and reject wire
sequences below the observed watermark. A descriptor is not a block inventory;
no replay, complete-cache knowledge, reboot transparency, or exactly-once
delivery is claimed. Successful metadata refresh cannot revive old entries.
This Router increment requires Router restart on **any** publisher epoch,
topic, or source change, even if the descriptor's remaining fields match, so
the replacement Worker's actual input tokens are verified again. Same-boot
gap/clear recovery can bootstrap an empty observed subset. Metadata, hash, or
model incompatibility also requires a new Router contract. Missing events or
already-cached prefixes can cause safe false negatives after a reconnect; no
global cache reset is required.

## Source-executed tests

From the new isolated Router worktree, with the pinned vLLM dependencies:

```sh
VLLM_KV_PROPOSAL_SOURCE=/path/to/patched-vllm-checkout \
  python -m pytest -q py_test/test_kv_worker_export.py
```

The tests load only proposal modules into the test interpreter. They instantiate
real KVCacheManager/coordinator/manager/spec objects with a CPU block pool,
without model weights or a GPU. Scheduler attributes and EngineCore-facing
configuration are controlled test objects; the actual selected EngineCore and
AsyncLLM methods are extracted from proposal source. HTTP tests exercise the
real FastAPI route and existing auth middleware over a fake utility transport.
ZMQ tests exercise the actual proposed publisher on loopback. This proves the
adapter/transport slice, **not** full GPU EngineCore initialization. Hardware
acceptance remains separately authorized work.

Original source-audit result: **39 passed** (CPU, 24.12 seconds). Ruff checks for
new exporter/API/test code and patch applicability passed. That original audit
and its provenance JSON remain CPU-only. The later separately authorized
[finite GPU deployment report](../kv-capabilities-gpu-results.md) records actual
Worker initialization, the same patch hash, candidate/native binding and the
remaining production-installation limitations; it does not relabel this audit.

## Compact field provenance (original pinned source)

| Fact | Actual source | Meaning / external availability before proposal |
|---|---|---|
| Group metadata | `v1/engine/core.py:433` | Initialized scheduler inventory; private utility only |
| Complete original groups | `core.py:319–332`, `core/kv_cache_utils.py:1988` | Worker specs precede scheduler first-leaf normalization; missing from old export |
| Effective manager | `core/kv_cache_coordinator.py:139`, `:510` | Actual manager objects and effective block tokens, not model names |
| Allocation / hash / alignment | `single_type_kv_cache_manager.py:76`, `kv_cache_coordinator.py:510`, `sched/scheduler.py:286` | Separate token units; all equal only in accepted Dense subset |
| Terminal recompute | `kv_cache_manager.py:258`, `single_type_kv_cache_manager.py:746` | At most `N-1`, block aligned, first hole terminates prefix |
| Actual N / read exclusions | `v1/request.py:288`, `kv_cache_manager.py:213` | Prepared request token count; cache-read-disabled requests excluded |
| Hash root / representation | `kv_cache_utils.py:83`, `:134`, `:621`; `core.py:224` | Resolved seed/root, full bytes, chained tuple with extra keys |
| Speculation / CP / connectors | `sched/scheduler.py:138–185`, `:257–299` | Effective initialized objects/config, not renderer assumptions |
| Event group annotation | `kv_cache_manager.py:666` | Group kind/window added to actual stored events |
| Publisher sequence | `distributed/kv_events.py:453` | Stock monotonic per-boot sequence, but no wire boot epoch |
| Fixed internal RPC | `v1/engine/core_client.py:1129` | Existing utility chain, not an existing public capabilities endpoint |
| Existing HTTP auth | `entrypoints/serve/middleware/authenticate.py:12` | `/v1` is guarded when API keys are configured |

There is no stock public HTTP endpoint with these complete facts. The DEV_MODE
`/server_info` endpoint is not an alternative: it is configuration-oriented,
lacks initialized manager/epoch evidence, and requires broad development mode.

# PR #517 on Ascend 910C

On two independent workers, Router credit matched backend prefix reuse for Qwen3.5-4B and for Qwen3.8-27B-W8A8 with MTP disabled. The DeepSeek Flash-derived mini checkpoint exposed eight Ascend cache groups that our layout exporter could not describe to the Router. We verified physical KV reads for that checkpoint in 2P1D after omitting the rejected declaration. Its hybrid credit remains unverified.

## Test setup

| Component | Tested value |
|---|---|
| Router | Full HEAD of [vllm-project/router#517](https://github.com/vllm-project/router/pull/517), `e1ee84a0f9680367a935e4946878a37511504539` |
| Router executable | Linux aarch64 standalone Rust CLI, SHA256 `ceaedcbbfc0198d4b3ac2bf4c9f6da0dd0958aa99a65e1e31c3f8c9615f57c8b` |
| Serving stack | Ascend 910C, vLLM `0.32.0+empty`, [vllm-ascend `28e9e97ba`](https://github.com/vllm-project/vllm-ascend/tree/28e9e97ba254b076a88275e8a275556cb9a11664), Mooncake transfer engine for NPU `0.3.14rc3` |
| Topology | Regular had two independent workers. PD had two independent Prefill engines (P0/P1) and one Decode engine (D0). Each engine used one logical device, TP=1 and DP=1. |
| Routing | `kv_aware`, static HTTP workers. The Router's `--kv-connector nixl` non-push option forwarded the complete Prefill response's `kv_transfer_params`. The engine on both ends used `MooncakeHybridConnector`. |

Each formal scenario exercised JSON and SSE on both `/v1/chat/completions` and `/v1/completions`. A small workload used two warmups, followed by 20 streaming requests each at concurrency 1 and 4, with 511 measured input tokens per request and a 256-token output budget. All 240 measured requests in the six formal scenarios completed at the HTTP/client level. The final, separate no-MTP Regular diagnostic added 40. Several PD workloads produced degenerate text, so these counts measure request completion and do not establish usable inference throughput.

## Results

| Model and mode | Generated text | Hybrid routing and backend reuse | Physical KV transfer |
|---|---|---|---|
| Qwen3.5-4B, Regular | Coherent reasoning; all four Router smoke requests hit the 1,024-token budget. Direct short-answer probes finished. | Warm credit/cached tokens **1,280/1,280** on 1,281 input tokens; divergent prefix **0/0**. | N/A |
| Qwen3.5-4B, 2P1D | One chat response answered the prompt; the other three smoke responses contained 1,024 exclamation marks each. | Warm request selected P0, credit **1,280**. Producer lookup candidate **640**; producer-reported cached tokens **0**. Divergent credit **0**. | **48/48** matched transfers had positive external tokens and a successful Decode read (P0 41, P1 7). |
| Qwen3.8-27B-W8A8, Regular, MTP on | All four smoke requests returned relevant answers and stopped. Both workers drafted and accepted tokens. | Runtime `speculative_tokens=3`, `hash_lookahead_tokens=1`; hybrid declaration omitted under the zero-value admission rule. | N/A |
| Qwen3.8-27B-W8A8, 2P1D, MTP on | Decode drafted and accepted tokens. Three smoke responses contained only exclamation marks; the fourth produced unrelated reasoning. | Prefill reported the same 3/1 values. No hybrid credit established. | **0/46** successful reads. All 46 Prefill payloads arrived intact at Decode (P0 40, P1 6); every read attempt failed during block pairing. |
| DeepSeek V4 Flash-derived four-layer W8A8 mini checkpoint, Regular | Anomalous text on both direct workers and through the Router. | Both ranks rejected the unknown group kind and used load-only scoring. | N/A |
| Same DeepSeek mini checkpoint, 2P1D | Responses contained unrelated text and repeated special markers. | PD rejected the declared layout. Requests ran with the hybrid declaration omitted. | **46/46** matched transfers had positive external tokens and a successful Decode read (P0 39, P1 7). |
| Qwen3.8-27B-W8A8, Regular, **MTP off diagnostic** | All four smoke requests returned relevant answers and stopped. | Both runtime fields were zero. Warm credit/cached tokens **1,792/1,792** on 1,793 input tokens; divergent prefix **0/0**. | N/A; no no-MTP PD run was performed. |

For the two positive Regular tests, we directly warmed one worker, correlated its fresh KV event with the Router's applied sequence, and sent the same prefix once through the Router. The selected worker's response supplied the cached-token count. These were **explicit token-ID inputs**. On the text path the worker render endpoints returned HTTP 403. vLLM 0.32 documents the `--enable-scale-out` requirement for those endpoints; the tested workers did not enable it. Text-input cache affinity still needs validation. [vLLM endpoint documentation](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/usage/security.md#L159-L165)

### Two DeepSeek transfer samples

For each sample, the Prefill response and Decode request had equal typed `kv_transfer_params`, including the nested block IDs. We joined the Decode allocation and read log by the remote request ID. The read times below describe individual requests.

| Prefill source | External tokens scheduled by D0 | Successful D0 read |
|---|---:|---:|
| P0 | 92 | 35.61 ms |
| P1 | 510 | 42.43 ms |

## Small workload observations

Each pair below gives concurrency 1 / concurrency 4. These are descriptive measurements of the Regular Qwen runs. The Qwen3.8 batches at concurrency 4 produced different output-token totals and different numbers of length-capped responses, which limits comparisons across MTP settings.

| Regular configuration | Output tokens/s | Total output tokens per batch | Responses ending at the output cap (of 20) |
|---|---:|---:|---:|
| Qwen3.5-4B | 22.48 / 75.76 | 5,120 / 5,120 | 20 / 20 |
| Qwen3.8-27B-W8A8, MTP on | 23.72 / 74.61 | 5,120 / 2,658 | 20 / 0 |
| Qwen3.8-27B-W8A8, MTP off diagnostic | 10.89 / 38.07 | 5,120 / 4,888 | 20 / 2 |

In each Qwen PD run, 37 of 40 measured responses contained only exclamation marks. The remaining three were also repetitive. We exclude those runs and the anomalous DeepSeek outputs from performance comparisons.

## Failure analysis

**DeepSeek layout declaration.** The checkpoint had two sparse MLA groups with logical block sizes 512 and 16,384 and compression factors 4 and 128. Six sliding groups had (block size, window) pairs of (128, 128) × 4, (8, 8) and (32, 128). Hash granularity was 8 tokens. Our exporter matched exact Python class names and emitted `kind=unknown` for `AscendMLAAttentionSpec` and `AscendSlidingWindowMLASpec`. Regular logged `unsupported group kind` for both workers; PD rejected the declaration at startup. vLLM's own classifier recognizes the base classes through subclass checks. The next step is to verify their page, compression and window semantics and compare a canonical declaration with live events. The harness also calculated a 32,769-token span for its conservative affinity probe and stopped before sending it because the preserved context limit was 10,240. That limit is a second issue. No observation here establishes that Ascend failed to publish KV events. [Ascend cache specs](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/core/kv_cache_interface.py) · [vLLM classifier](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/kv_cache_interface.py#L1373-L1378) · [Router layout admission](https://github.com/KaisennHu/router/blob/e1ee84a0f9680367a935e4946878a37511504539/src/kv_index/hybrid/layout.rs#L403-L453)

**Qwen3.8 MTP and PD transfer.** The Router requires zero speculative tokens and zero hash lookahead for hybrid credit. Independently, the Decode engine's `MooncakeHybridConnector` failed before issuing `batch_transfer_sync_read`. It passed remote and local lists to `group_concurrent_contiguous`; NumPy reported `ValueError: operands could not be broadcast together with shapes (0,) (3,)`. Those are the lengths after taking adjacent differences, which implies one remote block ID and four local IDs for the affected group. Three extra Mamba speculative slots could explain the mismatch; the trace does not identify the group. The connector logs the exception and still runs its completion and cleanup path. HTTP 200 therefore did not establish a successful read. A useful follow-up is a single deterministic request with MTP retained and per-group allocation tracing, then a same-version `MooncakeConnectorV1` control. The Qwen3.5-family A3 example uses that alternative connector. Applicability to this TP=1 Qwen3.8 checkpoint remains to be tested. [Router admission rule](https://github.com/KaisennHu/router/blob/e1ee84a0f9680367a935e4946878a37511504539/src/kv_index/hybrid/layout.rs#L346-L347) · [exception handling](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py#L684-L704) · [block pairing and read](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py#L790-L822) · [Qwen-family A3 example](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/docs/source/tutorials/models/Qwen3.5-397B-A17B.md#L425-L427)

**Qwen3.5 producer credit and generation.** On the 1,281-token warm request the Mamba producer removed the last prompt token, and vLLM's local lookup reserved another token for logits. That gives a cap of 1,279, or 640 at the worker's 640-token block boundary. This accounts for the initial lookup candidate. The producer's two request-level cache fields reported zero; per-group hits and their final common boundary were not logged. The 1,280 external tokens read by Decode measure a different operation. To distinguish the boundaries, repeat with 1,281 and 1,282 tokens and capture each group's lookup result. Separately, both Qwen PD configurations produced degenerate responses. Successful KV reads on Qwen3.5 coexist with bad output, and direct workers already produced anomalous text on DeepSeek. Same-input direct-worker and engine-proxy controls are needed before attributing generation quality to the Router or a specific backend component. [Producer truncation](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py#L1425-L1449) · [vLLM local lookup](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_manager.py#L289-L298)

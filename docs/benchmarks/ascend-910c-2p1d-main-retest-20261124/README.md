# Ascend 910C 2P1D validation of vLLM Router main

[中文报告](README.zh-CN.md) · [Redacted request-level evidence](evidence.json)

On November 24, 2026 (UTC), we retested three models on a single Ascend 910C host using the Router main commit containing [vllm-project/router#493](https://github.com/vllm-project/router/pull/493). All three models loaded, served streaming and non-streaming requests, admitted their hybrid cache layouts, reused warmed prefixes on both Prefill workers, and completed real KV reads from both Prefills. Qwen3.8 also used MTP on Decode. Full PD acceptance remains open for the two Qwen boundary requests, and Router overestimates one reproducible cache boundary for DeepSeek mini4.

## Tested setup

- Router main `d226cf43a459a91a65affff6d67d3887b5eb5605`. Standalone ARM64 binary SHA-256 `ac5147c4a82ca5fc48ab0949889e44e390c615e23dc2ab9623d19cb50ca3e9b9`. [Tested Router commit](https://github.com/vllm-project/router/commit/d226cf43a459a91a65affff6d67d3887b5eb5605)
- vLLM `0.32.0+empty`, vLLM Ascend `0.19.1rc2.dev3401+g28e9e97ba`, CANN 9.2.1. P0, P1, and D0 were independent engines on three separate logical devices, each TP=1 and DP=1. [Tested Ascend revision](https://github.com/vllm-project/vllm-ascend/commit/28e9e97ba254b076a88275e8a275556cb9a11664)
- Prefill used `kv_aware` throughout. Router `--kv-connector nixl` selected the non-push handoff path, which passed through the engine's `kv_transfer_params`. The engines used `MooncakeConnectorV1` for Qwen and `MooncakeHybridConnector` for DeepSeek. Qwen3.8 used `qwen3_5_mtp` with three speculative tokens on all three engines. Speculation was off for the other models.
- KaisennHu's reference PD matrix used 1P1D on Ascend A2 (910B1), Router `3ff5e6d`, and vLLM Ascend `0.32.0rc1`. The 910C mini4 checkpoint and the A2 four-layer checkpoint have different published identifiers. [A2 validation and reproduction](https://github.com/KaisennHu/router/blob/477ea8d30a44b61e3c87fef4d6d490f243c43f09/docs/benchmarks/a2-kv-aware-validation.md#pd-disaggregation)

## Results

Cache counts are shown as **Router credit / actual Prefill hit**, in tokens. Each model passed all four API smoke requests across `/v1/chat/completions` and `/v1/completions`, with JSON and SSE for each endpoint. All three models received and applied replayed KV events for both Prefills. Throughput, latency benchmarking, and colocated serving were outside this retest.

| Model | Warm P0 and P1, each | Short boundary | Unrelated prefix | Equal P→D payloads | Observed D reads, P0 + P1 | Strict transfer result |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Qwen3.5-4B | 1,920 / 1,920 | 1,281 input, 0 / 0 | 0 / 0 | 8/8 | 7/8, 4 + 3 | BLOCKED, 7/8 requests passed |
| Qwen3.8-27B W8A8 + MTP | 2,688 / 2,688 | 1,793 input, 0 / 0 | 0 / 0 | 25/25 | 24/25, 15 + 9 | BLOCKED, all 25 lacked the required positive per-request allocation log |
| DeepSeek-V4-Flash mini4 W8A8 | 16,384 / 16,384 | **16,385 input, 16,384 / 0** | 0 / 0 | 10/10 | 10/10, 5 + 5 | PASS, 10/10 requests passed |

Read counts here use all Router requests in each standard run. The A2 report counts requests requiring a connector read. Our strict gate also required a positive request-linked external allocation record alongside the matching payload, sender and read completion. Both Qwen cache rows match the A2 warm, boundary and unrelated-prefix results. DeepSeek matches A2 on warm and unrelated prefixes, while A2 reports 0 / 0 for its PD boundary. [Request-level evidence](evidence.json) · [A2 PD matrix](https://github.com/KaisennHu/router/blob/477ea8d30a44b61e3c87fef4d6d490f243c43f09/docs/benchmarks/a2-kv-aware-validation.md#pd-disaggregation)

### Qwen3.5-4B

In the standard run, the 1,281-token boundary showed an initial connector calculation of 1,280 and zero external allocation. The matching D lookup returned zero cached tokens and D executed a 1,280-token batch. Together these observations support local recomputation. In a separate cold-D run, the first request at the same boundary read 1,280 external tokens in 83.43 ms. After that request and the two warm-prefix probes, the repeated boundary had zero external allocation and D ran one-token batches, consistent with local reuse in that separate run. [Boundary and cold-D records](evidence.json)

### Qwen3.8-27B W8A8 with MTP

Both Prefills admitted the layout with the producer no-tail-drop path verified. The D-side MTP probe processed 915 input and 48 output tokens with a request-linked Mooncake read and 914 external tokens recorded by D. The before/after speculative counters increased by **16 drafts, 48 draft tokens, and 32 accepted tokens**. Sixteen of the 25 requests were additional PD coverage probes, all with completed reads. The separate 1,793-token boundary had zero D-local lookup hits and no matching remote-read completion. The aggregate D counters are consistent with recomputation at that boundary, with the per-request execution detail still missing. [MTP metrics, boundary lookup, and read records](evidence.json)

### DeepSeek-V4-Flash mini4 W8A8

The same warmed P0 returned zero local hits when Router received 16,385 input tokens and its lookup saw 16,384. Adding one token to the same prefix gave P0 a 16,385-token lookup and a 16,384-token hit. Router gave 16,384 credit on both requests. D completed the remote read for the shorter request even though Prefill had to recompute its prefix. The 16,384-token diagnostic went to P1 and returned 0 / 0 there. [Adjacent probes, Prefill lookup, and Decode read](evidence.json)

The observed boundary follows from two operations in the tested path. Ascend MooncakeHybrid trims one token on a compressed-cache Prefill request. The vLLM 0.32 cache lookup reserves one more for computing logits. The resulting lookup cap for a 16,385-token request is 16,383, below the 16,384-token cache boundary. The tested Router's `nixl` credit path subtracts one token for this layout. The next input length, 16,386, is sufficient under both calculations. [Router credit cap](https://github.com/vllm-project/router/blob/d226cf43a459a91a65affff6d67d3887b5eb5605/src/kv_index/source/binding.rs#L209-L235) · [Ascend producer trim](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py#L1425-L1471) · [vLLM cache hit cap](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_manager.py#L289-L299)

The 910C checkpoint, `DeepSeek-V4-Flash-0731-mini4-dspark-w8a8`, was used for compatibility testing. Its four HTTP smoke responses contained repetitive output and exhausted the 1,024-token budget. The evidence preserves the harness generation check as `FAIL`. Generation quality was outside compatibility acceptance for this checkpoint. The cache probes supplied fixed token IDs, and the Prefill lookup occurred before generation.

## Reproduction

1. Start P0, P1, and D0 with the tested engine connectors and options. Enable prefix caching on each P, and on D for both Qwen models. Set the mini4 context to 32,768, use `--prefix-cache-retention-interval 16384` on both Prefills, and disable prefix caching on its D. Have each P publish full SHA-256 CBOR KV block hashes on its own live and replay endpoints and topic.
2. Warm P0 and P1 directly with two prompts. For each prompt, initialize Python's `random.Random(seed)` once and generate IDs using `randrange(700, 1200)`. Use seeds 7900 and 7901 for P0 and P1, respectively, and seed 8900 for the unrelated control. Use warm lengths of 1,927 for Qwen3.5, 2,695 for Qwen3.8, and 16,401 for mini4. Set the direct warm output budget to two tokens. Start Router and wait for the two rank-0 replay sequences to be applied.
3. Send both warm probes, the shorter slice of P0's prompt, and the unrelated control through Router in sequence. The shorter slices are 1,281, 1,793, and 16,385 tokens respectively. For mini4, also send the 16,384- and 16,386-token slices and record which P gets each. Set the routed affinity probe output budget to eight tokens. Use `/v1/completions` with explicit token IDs, `temperature=0`, and `vllm_xargs.kv_cache_report_mode=incremental`. Join the Router admission, Prefill lookup and reply, Decode request, and connector completion by request identity.
4. For the Qwen3.5 cold-D control, submit the boundary as the first request to a fresh D0. Repeat it after the two routed warm probes. For Qwen3.8, isolate the MTP probe with seed 9700, 915 input tokens, a 48-token output budget and `ignore_eos=true`. Collect D0 counters immediately before and after it. [A2 reproduction](https://github.com/KaisennHu/router/blob/477ea8d30a44b61e3c87fef4d6d490f243c43f09/docs/benchmarks/a2-kv-aware-validation.md#reproduce-the-matrix)

## Questions for maintainers

- Should the Router's pass-through PD path account for the compressed-cache producer trim separately from the vLLM lookup cap? The 16,385/16,386 pair gives a candidate regression test for the credit estimate.
- For the Qwen boundary requests, which Decode-side record best identifies a planned external allocation that is skipped or completed after a local-cache lookup? That would help distinguish expected reuse from recomputation in a follow-up.

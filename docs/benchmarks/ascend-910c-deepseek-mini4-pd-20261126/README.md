# DeepSeek Mini4 PD routing on Ascend 910C

[中文报告](README.zh-CN.md) · [Engine and Router setup](router-method.md) · [Recreate the workload](workload.md) · [Aggregate evidence](evidence.json)

On two Ascend 910C cards, vLLM Router dispatched requests between separate prompt-processing (Prefill, P) and generation (Decode, D) engines for two DeepSeek Mini4 checkpoints. We verified real KV reads and matched the Router’s cache credits to engine reuse for both models. V4.1 required a local engine-side fix for a DP rank mismatch. With Decode fixed to round robin, `kv_aware` reduced P95 time to first token (TTFT) by **22.33% on V4 and 26.73% on V4.1**, while aggregate output tokens/s fell **1.60% and 10.72%**, respectively, versus round robin.

**Output quality limits the throughput claim.** Both checkpoints have four text-backbone layers and failed our answer-quality checks. All 144 measured streams emitted reasoning, reached the 128-token cap, and produced no final answer content. Output tokens/s below counts those generated tokens. A model that answers correctly can have different answer lengths, stopping behavior and speculative-token acceptance, all of which can affect throughput. We did not run a full-quality checkpoint under the same conditions, so we cannot quantify the size or direction of that effect. These results measure a fixed protocol replay. Throughput of useful answers and production capacity remain unmeasured.

## Hardware and deployment

Single host, two 910C cards, four independently addressable logical devices (`Ascend910_9362`, 61.27–61.28 GiB of visible HBM each). The models ran sequentially. The workload was synthetic and no production traffic was replayed. Software was [Router main `d71b621b`](https://github.com/vllm-project/router/commit/d71b621b619a769880aac8d33155298dfdbedfbc), vLLM `0.32.0+empty` and [vllm-ascend `28e9e97b`](https://github.com/vllm-project/vllm-ascend/commit/28e9e97ba254b076a88275e8a275556cb9a11664), with the V4.1 local modification described below.

| Tested checkpoint | Deployment | Settings that affect the result |
|---|---|---|
| DeepSeek-V4-Flash-0731-mini4-dspark-w8a8 | 2P1D. Two independent P engines and one D engine, each TP=1 / DP=1, on logical devices 0/1/2. | 32,768-token context, Prefill retention interval 16,384, with no explicit speculative decoding. |
| DeepSeek-V4.1-Flash-1017-HW-W8A8-mini4-dspark | 1P1D with two data-parallel ranks **per role**, each TP=1. Four independently addressed HTTP endpoints. P ranks 0/1 on devices 0/1, D ranks 0/1 on 2/3. | 10,240-token context, DSpark with 5 speculative tokens on both roles. P enabled prefix caching and bounded sliding-window replay. D disabled both. |

For functional checks we used `--prefill-policy kv_aware --decode-policy round_robin`. For performance we changed only the Prefill policy and kept the global `--policy` argument aligned with it. The engines used `MooncakeHybridConnector`; the Router used `--kv-connector nixl` to pass the complete Prefill transfer object to Decode. Mooncake performed the physical read. The setup page includes the HTTP rank syntax, per-P layout input and native event discovery.

## What passed on hardware

| Check | V4 | V4.1 after the local fix |
|---|---|---|
| Model / HTTP | 3/3 endpoints served requests. Chat and text completion, streaming and non-streaming, all passed (4/4). | 4/4 endpoints served requests. All four API forms passed. |
| Physical KV transfer | 12/12 functional requests across both P engines. | 10/10 across all four P→D pairings. |
| Events, cache estimate and actual reuse | Native event discovery on both P endpoints. Prompts of 16,385 and 16,386 tokens produced Router credit / actual reuse of 0/0 and 16,384/16,384. | Native rank 0/1 discovery. On each P, a 2,048-token probe yielded 1,920 raw hit tokens. The engine replayed 128, leaving 1,792 effectively reused, matching the Router’s 1,792. An unrelated prefix gave 0/0. |
| Speculative decoding | Not enabled. | Each D rank produced 240 draft tokens during the focused probe and accepted 0. Execution was observed. This probe demonstrated no accepted-token benefit. |
| Answer quality | Four repetitive outputs out of four smoke requests. | Two truncated chat responses and two incorrect short completions out of four smoke requests. |

**V4.1 before and after the fix.** The Router and direct probes sent global `X-data-parallel-rank: 1` to an external-LB endpoint exposing a single local rank. The engine reported `data_parallel_rank 1 is out of range [0, 1).` None of the four API forms passed the first smoke test. Only 6/26 functional attempts had a verified read, all P0→D0, and all three benchmark arms were blocked. The deployment team reported mapping the rank to local 0 inside `input_processor.py` for the single-local-rank case. The rerun retained the four physical endpoints and the same Router binary. It removed answer quality from the mandatory **replay** gate, kept the quality failure visible, and measured three 24-request arms. The original unmodified combination has no passing V4.1 result here.

The team also observed incorrect repetitive output in a separate DSpark-disabled diagnostic. Its raw run and the hotfix diff were not included in the two reviewed archives. The four-layer text configuration and the DSpark-enabled failures are independently recorded. The diagnostic supports a checkpoint-quality limitation. Its exact cause remains unverified.

## Three-policy replay

Each model used one round, 24 requests per policy, six sessions with four turns each, up to four active sessions and one outstanding request per session. Each arm restarted the engines and Router. Within a model all arms received the same prerecorded conversation histories and produced the same output-token total (3,072). All **144/144** measured requests completed the stream and had a correlated successful, positive-length Mooncake read at the selected D. All **144/144** also stopped at the length cap with empty final-answer content.

**Because both Mini4 checkpoints failed the answer-quality checks, TPOT and E2E measured on these truncated responses are unreliable estimates of latency when generating correct answers.**

| Model | Prefill policy | Requests completed | Output tokens/s ↑ | TTFT mean / P95 (s) ↓ | Mean TPOT (ms/token) ↓ | Mean E2E (s) ↓ | E2E P95 (s) ↓ | P cache hit | P tokens recomputed ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 | `round_robin` | 24/24 | 118.58 | 0.72 / 2.32 | 20.46 | 3.32 | 4.95 | 64.24% | 164,180 |
| V4 | `cache_aware` | 24/24 | 118.43 | 0.69 / 1.90 | 20.55 | 3.30 | 4.48 | 74.94% | 115,028 |
| V4 | `kv_aware` | 24/24 | 116.69 | 0.71 / 1.80 | 20.67 | 3.34 | 4.43 | 64.24% | 164,180 |
| V4.1 | `round_robin` | 24/24 | 61.59 | 1.47 / 8.50 | 41.23 | 6.71 | 15.09 | 72.89% | 26,180 |
| V4.1 | `cache_aware` | 24/24 | 61.53 | 1.37 / 7.12 | 41.98 | 6.70 | 15.19 | 85.09% | 14,404 |
| V4.1 | `kv_aware` | 24/24 | 54.99 | 1.24 / 6.22 | 50.27 | 7.63 | 14.22 | 72.89% | 26,180 |

Session completion time (JCT) spans four turns, from the first request dispatch to the end of the fourth response. Across six sessions per policy, JCT P95 was 14.01, 13.75, and 13.92 s on V4 and 30.73, 30.68, and 35.69 s on V4.1, in the policy order shown above.

| Relative to round robin on the same model | Output tokens/s | TTFT P95 | P cache hit (percentage points) | P tokens recomputed |
|---|---:|---:|---:|---:|
| V4 · cache_aware | −0.12% | −17.95% | +10.71 | −29.94% |
| V4 · kv_aware | −1.60% | −22.33% | 0.00 | 0.00% |
| V4.1 · cache_aware | −0.10% | −16.15% | +12.19 | −44.98% |
| V4.1 · kv_aware | −10.72% | −26.73% | 0.00 | 0.00% |

The `kv_aware` arms recorded positive credit on 18/24 requests on each model, with no policy fallback. Aggregate P hit rate was the same as round robin. Relative to `cache_aware`, `kv_aware` shortened TTFT P95 by 5.34% on V4 and 12.62% on V4.1. Output tokens/s fell 1.47% and 10.63%. On V4.1, switching from round robin to `kv_aware` reduced request E2E P95 from 15.09 to 14.22 s. Mean request E2E rose 13.68% and four-turn JCT P95 rose 16.15%. A single round does not establish how stable those differences are.

## Relation to the H20 result

[KaisennHu’s SWE-bench Pro report](https://github.com/KaisennHu/router/blob/06bfc5236ca42a0109288add0e1c5e7b75c45822/tests/python/e2e/perf/router_policy/pd_policy_repro/results/swebench_4p4d.md) measured 4P4D on eight H20 GPUs with Qwen3-4B-Instruct-2507, 247 requests per policy per round, two rounds, and **both** P and D policies changed. Over those rounds, `kv_aware` versus round robin increased output tokens/s by 24.8% and hit rate by 9.97 percentage points, and reduced TTFT P95 by 16.2%. Our within-model 910C runs also showed lower TTFT P95. They did not show additional aggregate hits or output throughput over round robin. Hardware, models, traffic, topology and policy scope all differed. No cross-hardware speedup can be derived from this pair of experiments.

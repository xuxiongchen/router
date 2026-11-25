# vLLM Router 主线在 Ascend 910C 上的 2P1D 复测

[English report](README.md) · [脱敏逐请求证据](evidence.json)

2026 年 11 月 24 日（UTC），我们在单机 Ascend 910C 上，使用已包含 PR #493 的 Router 主线提交 `d226cf43` 复测了三个模型。三个模型都完成了加载、四种 HTTP 请求、双 Prefill 热前缀路由和来自两个 Prefill 的真实 KV 传输。Qwen3.8 全程使用 `kv_aware`，Decode 实际接受了 MTP 草稿 token。两个 Qwen 各有一条边界请求未通过严格传输验收，DeepSeek mini4 有一处可复现的缓存命中高估。当前结论是这个固定版本在所测配置下已实现基础互通，完整验收仍有上述未决项。[目标 PR](https://github.com/vllm-project/router/pull/493)

## 环境与对照

- Router 主线提交为 `d226cf43a459a91a65affff6d67d3887b5eb5605`，ARM64 独立可执行文件的 SHA-256 为 `ac5147c4a82ca5fc48ab0949889e44e390c615e23dc2ab9623d19cb50ca3e9b9`。同一交付清单的 SHA-256 见证据文件。[实测 Router 提交](https://github.com/vllm-project/router/commit/d226cf43a459a91a65affff6d67d3887b5eb5605)
- vLLM 为 `0.32.0+empty`，vLLM Ascend 为 `0.19.1rc2.dev3401+g28e9e97ba`，CANN 为 9.2.1。P0、P1、D0 为三个独立引擎，各占一个逻辑设备，每个引擎均为 TP=1、DP=1。[实测 Ascend 提交](https://github.com/vllm-project/vllm-ascend/commit/28e9e97ba254b076a88275e8a275556cb9a11664)
- Router 的 Prefill 策略为 `kv_aware`。`--kv-connector nixl` 选择非 push 参数透传路径，将 P 返回的 `kv_transfer_params` 传给 D。引擎实际使用的连接器是 Qwen 的 `MooncakeConnectorV1` 和 mini4 的 `MooncakeHybridConnector`。Qwen3.8 的三个引擎均配置 `qwen3_5_mtp`、3 个 speculative tokens，其他两个模型关闭推测解码。
- KaisennHu 的对照表使用 A2（910B1）1P1D、Router `3ff5e6d`、vLLM Ascend `0.32.0rc1`。其 mini4 是四层测试 checkpoint，与本轮所用 checkpoint 的标识不同。下文比较的是相同方法下的可观测行为。本轮未测混部、吞吐或延迟。[A2 验证与复现报告](https://github.com/KaisennHu/router/blob/477ea8d30a44b61e3c87fef4d6d490f243c43f09/docs/benchmarks/a2-kv-aware-validation.md#pd-disaggregation)

## 结果

三模型的 Chat Completions 和 Completions 各通过 JSON、SSE 两种请求，每个模型共 4/4。Router 接纳了三个场景的 hybrid layout，并应用了两个 P 各自的 replay 事件。下表「预测 / 实际」均指 Router 给出的 prefix credit 和**被选中 Prefill** 的实际命中 token 数。

| 模型 | 分别预热 P0、P1，每个 P 的预测 / 实际 | P0 短边界 | 不同前缀 | P/D 参数相等 | D 实际远端读取（来自 P0 + P1） | 严格传输验收 |
|---|---:|---|---:|---:|---|---|
| Qwen3.5-4B | 1,920 / 1,920 | 输入 1,281，0 / 0 | 0 / 0 | 8/8 | 7/8（4 + 3） | BLOCKED，7/8 条通过 |
| Qwen3.8-27B W8A8 + MTP | 2,688 / 2,688 | 输入 1,793，0 / 0 | 0 / 0 | 25/25 | 24/25（15 + 9） | BLOCKED，25 条均缺少按请求记录的正外部分配日志 |
| DeepSeek-V4-Flash mini4 W8A8 | 16,384 / 16,384 | **输入 16,385，16,384 / 0** | 0 / 0 | 10/10 | 10/10（5 + 5） | PASS，10/10 条通过 |

这里的读取分母取每个正式场景的全部路由请求。A2 的分母只计需要远端读的请求。行内严格验收还要求匹配请求的外部 KV 分配量为正、传输参数和发送端一致、有读取完成证据。两个 Qwen 的热前缀、短边界和不同前缀的 **Prefill 命中数**均与 A2 一致。mini4 的热前缀与不同前缀也一致，但 A2 的 PD 短边界为 0 / 0。[逐请求记录](evidence.json) · [A2 PD 对照表](https://github.com/KaisennHu/router/blob/477ea8d30a44b61e3c87fef4d6d490f243c43f09/docs/benchmarks/a2-kv-aware-validation.md#pd-disaggregation)

### Qwen3.5-4B

正式场景的 1,281-token 请求中，D 连接器最初报告 1,280，外部 KV 分配量为 0。同一请求的 D 缓存 lookup 实际为 0，执行窗口内跑了 1,280-token 批次，支持 D 本地重算的判断。独立冷 D 对照中，同样的边界作为首个请求时成功远端读取 1,280 tokens，耗时 83.43 ms。此后经过两个热前缀请求，再发该边界时，外部分配为 0，D 只跑了单 token 批次，符合此时已能本地复用的表现。两个实验的缓存历史不同。[正式与冷 D 两组记录](evidence.json)

### Qwen3.8-27B W8A8 + MTP

P0 和 P1 均验证了 producer 保留尾块的实际路径，通过了布局准入。独立 MTP 探针输入 915 tokens、输出 48 tokens，D 有同请求的 Mooncake 读取完成记录，外部命中增加 914 tokens。D 在探针前后新增 **16 次草稿尝试、48 个草稿 token、32 个被接受的 token**。25 条请求中包含 16 条额外的 PD 覆盖探针，这 16 条全部完成远端读取。缺少读取完成记录的是另一条 1,793-token 边界请求，其 D 本地 lookup 为 0。汇总计数与该边界重算吻合，缺少该请求独立的执行量记录，因此保留未决。[MTP 前后计数与边界证据](evidence.json)

### DeepSeek-V4-Flash mini4 W8A8

同一预热过的 P0 上，输入 16,385 时，P 的 lookup 输入为 16,384，实际命中 0。给同一前缀再加一个 token，P 的 lookup 输入变成 16,385，实际命中 16,384。Router 两次均给出 16,384 credit。较短请求仍有 D 从 P 读取 KV 的完成记录。额外的 16,384-token 探针选中了 P1，在该 P 上为 0 / 0。[相邻长度探针和 D 端证据](evidence.json)

固定版本的源码能够解释这个边界。在启用压缩缓存的 producer 路径上，Ascend HybridConnector 会裁掉一个输入 token，vLLM 查缓存时又为计算 logits 留出一个 token。对于 16,385 输入，最大可命中长度是 16,383，距离该布局的 16,384 恢复边界差一个 token。本次 Router 的 `nixl` 透传配置在该布局下只扣一个 token。输入延长到 16,386 后，两侧均达到恢复边界。[Router 命中上限](https://github.com/vllm-project/router/blob/d226cf43a459a91a65affff6d67d3887b5eb5605/src/kv_index/source/binding.rs#L209-L235) · [Ascend producer 裁剪](https://github.com/vllm-project/vllm-ascend/blob/28e9e97ba254b076a88275e8a275556cb9a11664/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py#L1425-L1471) · [vLLM 查找上限](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_manager.py#L289-L299)

行内的 `DeepSeek-V4-Flash-0731-mini4-dspark-w8a8` 仅用于兼容性验证。四条 HTTP 烟测都观察到重复输出，并用满 1,024-token 预算。证据文件保留工具的生成检查原值 `FAIL`，文本质量不纳入这个 checkpoint 的兼容性验收。缓存探针使用固定输入 token ID，Prefill 在生成前查缓存，可以独立判断上述预测偏差。

## 复现要点

1. 按上述角色和连接器启动三个独立引擎。两个 P 均启用 prefix caching，Qwen 的 D 启用，mini4 的 D 关闭。mini4 上下文设为 32,768，两个 P 均配置 `--prefix-cache-retention-interval 16384`。每个 P 使用独立的事件、replay 端点和 topic，发布完整的 SHA-256 CBOR block hash。
2. 先直连 P0、P1 分别预热。每个输入以 Python `random.Random(seed)` 初始化一次，然后用 `randrange(700, 1200)` 生成 token ID。P0、P1 的 seed 分别为 7900、7901，不同前缀用 8900。三个模型的预热输入长度依次为 1,927、2,695、16,401，预热输出预算为 2。启动 Router，等待它应用两个 rank 0 的 replay。
3. 经 Router 依次请求两个热前缀、P0 预热前缀的短切片和不同前缀。三个短切片的长度依次为 1,281、1,793、16,385。mini4 另测 16,384 和 16,386，记录实际所选 P。使用 `/v1/completions` 的显式 token ID，输出预算 8，`temperature=0`，并设 `vllm_xargs.kv_cache_report_mode=incremental`。按请求身份串联 Router 选择、P lookup 和响应、D 请求及连接器读取完成日志。
4. Qwen3.5 冷 D 对照先向全新的 D 发送边界请求，两个热前缀请求完成后再发一次。Qwen3.8 的 MTP 探针单独执行，seed 为 9700，输入 915，输出预算 48，`ignore_eos=true`，紧邻探针前后分别采集 D 指标。[A2 复现步骤](https://github.com/KaisennHu/router/blob/477ea8d30a44b61e3c87fef4d6d490f243c43f09/docs/benchmarks/a2-kv-aware-validation.md#reproduce-the-matrix)

## 希望维护者协助确认

- 对于透传给 MooncakeHybrid 的请求，Router 估算 Prefill 命中长度时是否应单独计入 producer 的裁剪？16,385 和 16,386 这对相邻输入可以用于验证修复。
- 对两个 Qwen 的边界请求，D 端哪项逐请求记录最适合确认本地 lookup 后，原计划的远端 KV 分配是被跳过还是已完成？这会帮助后续复测区分本地复用和重算。

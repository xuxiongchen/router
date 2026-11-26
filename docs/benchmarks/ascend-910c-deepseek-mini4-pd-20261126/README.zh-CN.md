# 昇腾 910C 上的 DeepSeek Mini4 P/D 兼容性与路由性能

[English](README.md) · [部署与 Router 启动方法](router-method.md) · [生成相同压测请求](workload.md) · [脱敏验证数据](evidence.json)

在两张昇腾 910C 上，DeepSeek V4 Mini4 验证了预填充与解码分离（P/D）、实际跨设备 KV 读取及前缀复用。DeepSeek V4.1 Flash Mini4 在临时修复请求的 DP rank 校验后，也通过了这些技术验证。三策略回放中，`kv_aware` 缩短了首个 token 的等待时间（TTFT P95），但输出吞吐低于轮询。本次测试对应下述设备和拓扑，生产流量、正确答案的吞吐及上线容量尚待验收。

**模型精度与吞吐限制。** 两份 Mini4 权重的文本主干均只保留四层。V4 的四条功能请求均出现重复输出，V4.1 热修后有两条输出截断、两条短答错误。两模型合计 144 条计量响应全部在 128-token 上限结束，流式响应只有推理片段，最终回答字段均为空。表中的输出吞吐计入这些推理 token。裁剪后的生成行为会影响输出长度和终止时机，V4.1 的投机解码草稿接受情况也会影响实际计算量。现有数据只能量化这批固定负载的生成速度，正常权重下的吞吐及变化幅度仍需对照。

## 设备和部署

测试使用两张昇腾 910C（A3），共四个可独立分配的逻辑 NPU。每个逻辑设备报告 61.27–61.28 GiB 显存。Router 使用上游 main 的 standalone，提交为 `d71b621b619a769880aac8d33155298dfdbedfbc`。推理引擎为 `vLLM 0.32.0+empty`，vLLM Ascend 提交为 `28e9e97ba254b076a88275e8a275556cb9a11664`。 [Router 提交](https://github.com/vllm-project/router/commit/d71b621b619a769880aac8d33155298dfdbedfbc) · [vLLM Ascend 提交](https://github.com/vllm-project/vllm-ascend/commit/28e9e97ba254b076a88275e8a275556cb9a11664)

| 测试权重 | 部署拓扑 | 逻辑设备 | 最大上下文 |
|---|---|---:|---:|
| DeepSeek V4 Mini4 | 2P1D。两个独立 Prefill 引擎、一个 Decode 引擎，每个引擎 TP=1、DP=1 | 3 | 32,768 |
| DeepSeek V4.1 Flash Mini4 | 1P1D。Prefill、Decode 各一组，每组 TP=1、DP=2，每个 DP rank 独立 HTTP 端口 | 4 | 10,240 |

功能验证使用 Prefill `kv_aware`、Decode `round_robin`。性能回放逐一切换 Prefill 的 `round_robin`、`cache_aware`、`kv_aware`，Decode 始终轮询。引擎侧用 `MooncakeHybridConnector`，Router 侧用 `--kv-connector nixl` 选择 P/D 参数透传路径，实际同步读取由 Mooncake 执行。两个通过的场景均通过 Prefill 原生 HTTP 接口发现 KV 事件源。 [Router 命令与布局要点](router-method.md)

## 功能验证

功能请求覆盖聊天与文本补全，各测普通 JSON 和流式 SSE。按请求关联 Prefill 返回参数、Decode 收到的嵌套参数、实际执行的 rank、正外部 token 分配、同步读取和完成日志，并核对关联错误。缓存验证先给两个 Prefill 单独预热，再检查 Router 对已知前缀的估算和实际复用，同时使用陌生前缀作负对照。

| 场景 | API | 服务实例覆盖 | 逐请求非零 KV 读取 | 布局、缓存估算及实际复用 | 文本质量 |
|---|---:|---:|---:|---|---|
| V4 Mini4 | 4/4 | 3/3 | 12/12 | 通过 | 4/4 失败，重复输出 |
| V4.1 Flash Mini4，首跑 | 0/4 | 3/4 | 6/26，仅 P0→D0 | 验证受阻 | 验证受阻 |
| V4.1 Flash Mini4，热修后 | 4/4 | 4/4 | 10/10，四种 P→D 配对均覆盖 | 通过 | 4/4 失败，两条截断、两条短答错误 |

- **V4 的缓存边界。** 对 16,385-token 输入，Router 估算与引擎实际复用均为 0。对 16,386-token 输入，两项均为 16,384。
- **V4.1 的缓存边界。** 两路 Prefill 的 2,048-token 正例均原始命中 1,920 个 token。扣除滑动窗口要求重算的 128 个 token 后，实际复用 1,792，与 Router 估算一致。258-token 边界探针原始命中 256，扣除重算后为 128，与 Router 一致。陌生前缀的估算和实际复用均归零。
- **V4.1 的 rank 兼容性。** 首跑采用 external LB，每个 API 进程仅服务一个本地 rank。发往全局 rank 1 的请求携带 `X-data-parallel-rank: 1`，引擎的本地校验报 `data_parallel_rank 1 is out of range [0, 1)`。现场在引擎 `input_processor.py` 将该场景的 rank 映射到本地 0，外部 URL 与全局 rank 的绑定以及 Router 提交保持一致。复测四个端口均被请求覆盖。该结果适用于热修后的组合，原版组合仍需正式修复并复验。
- **V4.1 的 DSpark。** 两个 Decode rank 各观察到 240 个草稿 token，接受量均为 0。测试确认草稿路径执行，未观察到草稿带来的速度收益。现场关闭 DSpark 的诊断仍观察到异常文本，该诊断由现场记录提供。归档中的独立证据覆盖启用 DSpark 的运行和四层权重配置。

## 三策略回放结果

每模型每策略运行一轮，6 个固定合成会话，每会话 4 个请求，最多 4 个会话并行。各策略在对应模型上使用相同输入和输出预算，各轮重新启动引擎与 Router。V4 每组输入 459,116 token，V4.1 每组输入 96,604 token。每组均输出 3,072 token。每组 24 个请求均通过协议检查和逐请求非零 KV 读取验证。

**性能口径。** V4.1 首跑三个策略均受阻。热修后，复测保留 `generation=FAIL`，放宽生成质量门控后完成三组技术回放。两模型的 144 条计量响应均只有推理片段、没有最终答案，并在 128 token 上限停止。输出吞吐计入这些推理 token。

每组按 24 个请求统计延迟。**TTFT** 从发出请求计到收到首个非空文本或推理片段。**TPOT** 是首个 token 之后每个输出 token 的平均耗时，先按请求计算，再对请求取均值。**E2E** 从发出单个请求计到该请求的响应流结束。E2E P95 是这 24 个请求端到端耗时的第 95 百分位数。

| 模型 | Prefill 策略 | 请求成功 | 输出 token/s | TTFT 均值 / P95（秒） | TPOT 均值（毫秒/token） | E2E 均值（秒） | E2E P95（秒） | P 输入命中率 | P 新算 token |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 | `round_robin` | 24/24 | 118.58 | 0.72 / 2.32 | 20.46 | 3.32 | 4.95 | 64.24% | 164,180 |
| V4 | `cache_aware` | 24/24 | 118.43 | 0.69 / 1.90 | 20.55 | 3.30 | 4.48 | 74.94% | 115,028 |
| V4 | `kv_aware` | 24/24 | 116.69 | 0.71 / 1.80 | 20.67 | 3.34 | 4.43 | 64.24% | 164,180 |
| V4.1 | `round_robin` | 24/24 | 61.59 | 1.47 / 8.50 | 41.23 | 6.71 | 15.09 | 72.89% | 26,180 |
| V4.1 | `cache_aware` | 24/24 | 61.53 | 1.37 / 7.12 | 41.98 | 6.70 | 15.19 | 85.09% | 14,404 |
| V4.1 | `kv_aware` | 24/24 | 54.99 | 1.24 / 6.22 | 50.27 | 7.63 | 14.22 | 72.89% | 26,180 |

四轮会话的完成时间（JCT）从第一轮请求发出计到第四轮响应结束，每组有 6 个会话。按表中三种策略的顺序，JCT P95 在 V4 上分别为 14.01、13.75、13.92 秒，在 V4.1 上分别为 30.73、30.68、35.69 秒。

- **`cache_aware` 相对轮询。** V4、V4.1 的 Prefill 命中率分别增加 **10.71**、**12.19 个百分点**，新算 token 分别减少 **29.94%**、**44.98%**。输出吞吐分别下降 0.12%、0.10%，TTFT P95 分别缩短 17.95%、16.15%。
- **`kv_aware` 相对轮询。** V4、V4.1 的 TTFT P95 分别缩短 **22.33%**、**26.73%**，输出吞吐分别下降 **1.60%**、**10.72%**。两模型各有 18/24 个请求记录了正缓存估算，计量请求均未发生策略降级，整批 Prefill 命中率和新算量与轮询持平。V4.1 的单请求 E2E 均值增加 13.68%，E2E P95 从 15.09 秒降至 14.22 秒，四轮会话的 JCT P95 增加 16.15%。
- **两种缓存策略之间。** `kv_aware` 相对 `cache_aware`，V4、V4.1 的 TTFT P95 分别缩短 5.34%、12.62%，输出吞吐分别下降 1.47%、10.63%。每个模型仅测一轮，每策略只有 24 个请求，尾部指标会受少数请求影响。

## 与 H20 公开结果对照

KaisennHu 使用 8 张 H20、Qwen3-4B-Instruct-2507 和 4P4D 回放 SWE-bench Pro，每策略每轮 247 个请求，共两轮，Prefill 与 Decode 同时切换策略。公开报告的 `kv_aware` 相对轮询，输出吞吐提高 **24.8%**，TTFT P95 缩短 **16.2%**，Prefill 命中率增加 **9.97 个百分点**。本次 910C 测试也观察到 TTFT P95 缩短，输出吞吐和整批 Prefill 复用的变化与 H20 不同。两组实验的模型、负载、硬件和策略切换范围均不同，吞吐差异无法单独归因于硬件。 [H20 原报告](https://github.com/KaisennHu/router/blob/06bfc5236ca42a0109288add0e1c5e7b75c45822/tests/python/e2e/perf/router_policy/pd_policy_repro/results/swebench_4p4d.md) · [H20 实验配置](https://github.com/KaisennHu/router/blob/06bfc5236ca42a0109288add0e1c5e7b75c45822/tests/python/e2e/perf/router_policy/pd_policy_repro/README.md)

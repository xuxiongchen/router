# Running the Ascend 910C DeepSeek PD tests

[Results in English](README.md) · [Replay workload and metric definitions](workload.md)

Both Mini4 checkpoints failed the text-quality checks. The 144 measured responses stopped at 128 generated tokens and exposed reasoning fragments without a final answer. The experiment measures route and transfer behavior and output-token rate under that cap. The effect of this quality limitation on useful-answer throughput needs a matched run with reliable checkpoints.

## Software and placement

The field run used the Rust `vllm-router` binary from vLLM Router main at `d71b621b619a769880aac8d33155298dfdbedfbc`, vLLM `0.32.0+empty`, and vLLM Ascend at `28e9e97ba254b076a88275e8a275556cb9a11664`. vLLM used the V2 Model Runner, Ascend quantization, and `MooncakeHybridConnector` on both stages. The deployment ran on one test machine with four selectable Ascend 910C logical NPU devices. The V4 profile used three logical devices and the V4.1 profile used four. [Router source](https://github.com/vllm-project/router/tree/d71b621b619a769880aac8d33155298dfdbedfbc) · [vLLM Ascend source](https://github.com/vllm-project/vllm-ascend/tree/28e9e97ba254b076a88275e8a275556cb9a11664)

| Profile | Prefill | Decode | Tensor and data parallelism |
| --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 W8A8 Mini4 | P0 and P1, HTTP ports 8000 and 8001 | D0, HTTP port 8002 | Three independent engines. Each uses TP=1 and DP=1. |
| DeepSeek V4.1 Flash 1017 W8A8 Mini4 DSpark | One DP group with ranks 0 and 1, ports 8000 and 8001 | One DP group with ranks 0 and 1, ports 8002 and 8003 | TP=1, DP=2 and local DP size 1 per HTTP process. P and D have separate DP rendezvous ports. |

V4 uses a 32,768-token context. V4.1 uses a 10,240-token context and five DSpark speculative tokens on both stages. Both use Prefill context parallelism of 1. Prefix caching is enabled on Prefill, and Decode uses transferred state with local prefix caching disabled. V4.1 enables sliding-window bounded replay on Prefill and disables it on Decode.

### V4.1 external load-balancing requirement

Router registers each endpoint with its global DP rank (`URL@0` or `URL@1`). The tested Router sent that rank in the `X-data-parallel-rank` HTTP header. The test engine in external load-balancing mode had one local engine behind each HTTP port. It rejected rank 1 at request processing with `data_parallel_rank 1 is out of range [0, 1).` The first run failed its API smoke checks.

The lab operator reported that the successful rerun used a lab-only change in `vllm/v1/engine/input_processor.py` that selected local slot 0 when that process exposed a single local engine. The worker URL, global rank and DP coordination still identified the external replica. Reproducing the successful V4.1 path requires equivalent handling of global endpoint identity and local dispatch. Confirm this on both rank-1 ports using real generation requests. Model-list responses alone can succeed before the rank mismatch is exposed.

## Start workers

This Bash example mirrors the service-relevant launch flags with sample ports and user-provided checkpoint and fabric settings. Set `PROFILE=v4` or `PROFILE=v41`, `CHECKPOINT` to your checkpoint directory, `FABRIC_IP` to the local address reachable by the connector, and `FABRIC_IF` to its interface. Use a fresh `RUN_DIR` for each run. Define the function and start the background workers in the same Bash shell so they share these settings. Each example assumes logical device IDs 0–3, following the table. The lab's request and runtime observers only collected evidence. Equivalent evidence can be collected with local instrumentation at the HTTP, scheduler and Mooncake layers.

```bash
export PROFILE="${PROFILE:-v4}"
export CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the model directory}"
# Set and export FABRIC_IP and FABRIC_IF for the test host before starting.
export RUN_DIR="$(mktemp -d)"

start_worker() {
  local role=$1 dev=$2 api=$3 rank=$4 live=$5 replay=$6 kv_port=$7 rpc=$8
  local dp=1 served=dsv4-mini mode=deepseek_v4 context=32768 utilization=0.85
  local kv_role=kv_producer topic events transfer
  local -a dp_args=() model_args=() role_args=() metrics_env=()
  topic=$(python3 -c 'from uuid import uuid4; print(uuid4().hex)')
  if [[ "$role" == p ]]; then
    role_args=(--enable-prefix-caching)
  else
    kv_role=kv_consumer
    role_args=(--no-enable-prefix-caching)
  fi

  case "$PROFILE" in
    v4)
      model_args=(--attention_config.indexer_kv_dtype int8)
      if [[ "$role" == p ]]; then
        role_args+=(--prefix-cache-retention-interval 16384)
      fi
      ;;
    v41)
      dp=2 served=dsv41-flash-mini4 mode=deepseek_v41 context=10240 utilization=0.95
      dp_args=(--data-parallel-size-local 1 --data-parallel-rank "$rank"
               --data-parallel-address "${FABRIC_IP:?set FABRIC_IP}" --data-parallel-rpc-port "$rpc")
      model_args=(--limit-mm-per-prompt '{"image":0}'
                  --speculative-config '{"method":"dspark","num_speculative_tokens":5}')
      if [[ "$role" == p ]]; then
        role_args+=(--swa-bounded-replay)
      else
        role_args+=(--no-swa-bounded-replay)
      fi
      mkdir -p "$RUN_DIR/metrics-$role"
      metrics_env=("PROMETHEUS_MULTIPROC_DIR=$RUN_DIR/metrics-$role")
      ;;
    *) printf 'Unsupported profile: %s\n' "$PROFILE" >&2; return 2 ;;
  esac

  events=$(jq -nc --arg live "tcp://*:$live" --arg replay "tcp://*:$replay" --arg topic "$topic" '
    {enable_kv_cache_events:true,publisher:"zmq",endpoint:$live,replay_endpoint:$replay,
     topic:$topic,buffer_steps:10000,hwm:100000,max_queue_size:100000}')
  transfer=$(jq -nc --arg role "$kv_role" --arg id "$topic" --argjson port "$kv_port" --argjson dp "$dp" '
    {kv_connector:"MooncakeHybridConnector",kv_buffer_device:"npu",kv_role:$role,kv_port:$port,
     engine_id:$id,kv_load_failure_policy:"fail",
     kv_connector_extra_config:{prefill:{dp_size:$dp,tp_size:1,pp_size:1},
                                decode:{dp_size:$dp,tp_size:1,pp_size:1}}}')
  env "${metrics_env[@]}" ASCEND_RT_VISIBLE_DEVICES="$dev" \
    VLLM_HOST_IP="${FABRIC_IP:?set FABRIC_IP}" HCCL_IF_IP="$FABRIC_IP" \
    HCCL_SOCKET_IFNAME="${FABRIC_IF:?set FABRIC_IF}" GLOO_SOCKET_IFNAME="$FABRIC_IF" TP_SOCKET_IFNAME="$FABRIC_IF" \
    PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_SERVER_DEV_MODE=1 VLLM_ENGINE_READY_TIMEOUT_S=1200 VLLM_LOGGING_LEVEL=DEBUG \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
    vllm serve "${CHECKPOINT:?set CHECKPOINT}" --host 127.0.0.1 --port "$api" --served-model-name "$served" \
    --tensor-parallel-size 1 --data-parallel-size "$dp" --prefill-context-parallel-size 1 "${dp_args[@]}" \
    --max-model-len "$context" --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --gpu-memory-utilization "$utilization" --prefix-caching-hash-algo sha256_cbor \
    --no-disable-hybrid-kv-cache-manager --trust-remote-code --enforce-eager \
    --kv-events-config "$events" --enable-prompt-tokens-details --enable-scale-out --skip-mm-profiling \
    --quantization ascend --block-size 128 --tokenizer-mode "$mode" --tool-call-parser "$mode" \
    --enable-auto-tool-choice --reasoning-parser "$mode" "${model_args[@]}" "${role_args[@]}" \
    --kv-transfer-config "$transfer"
}
```

Start the listed worker commands for one profile in parallel. For V4.1 the two ranks in a stage need to rendezvous. The event and transfer arguments are **base** ports supplied to the workers. vLLM offsets the event publication and replay addresses for global DP rank 1. Each V4 engine has global rank 0 in its own group and receives a distinct port.

```bash
# Arguments: role device HTTP global-rank event replay Mooncake DP-rendezvous
case "$PROFILE" in
  v4)
    start_worker p 0 8000 0 5557 5657 22000 0 >"$RUN_DIR/p0.log" 2>&1 &
    start_worker p 1 8001 0 5558 5658 22001 0 >"$RUN_DIR/p1.log" 2>&1 &
    start_worker d 2 8002 0 5560 5660 22200 0 >"$RUN_DIR/d0.log" 2>&1 &
    ;;
  v41)
    start_worker p 0 8000 0 5557 5657 22000 23100 >"$RUN_DIR/p0.log" 2>&1 &
    start_worker p 1 8001 1 5557 5657 22000 23100 >"$RUN_DIR/p1.log" 2>&1 &
    start_worker d 2 8002 0 5560 5660 22200 23200 >"$RUN_DIR/d0.log" 2>&1 &
    start_worker d 3 8003 1 5560 5660 22200 23200 >"$RUN_DIR/d1.log" 2>&1 &
    ;;
  *) printf 'Unsupported profile: %s\n' "$PROFILE" >&2; exit 2 ;;
esac
```

Share the fresh multiprocess Prometheus directory between the two V4.1 processes within each stage. Keep the P and D directories distinct. In this setup, each stage's `/metrics` exposed the real engine labels 0 and 1, which Router used to validate explicit rank pins. Confirm both labels before registration. During measurement, attribute counters to the actual rank and count each rank once even when both endpoints export them. In the passing runs both Prefill endpoints served `/kv_event_sources` natively. The collection middleware could supply a verified adapter on HTTP 404/405 in other deployments. These discovery receipts recorded native HTTP and did not activate that adapter. Check that the returned rank, endpoint, replay address and topic match each process.

## Generate the Prefill layout declaration

The test passed the actual Prefill layout to `--kv-hybrid-layout`. The generator below describes the runtime groups, hashing and replay geometry observed for these two trimmed checkpoints. Run it only after verifying the same group order, block sizes and replay contract on your workers. For a different checkpoint or engine, populate the layout from its actual runtime groups and check its observed KV events. The file keys are base worker URLs. The rank lives in the respective `ranks` entry. The worker registers a fresh publisher topic on each start and Router discovers it through HTTP. [Upstream layout file format](https://github.com/vllm-project/router/blob/d71b621b619a769880aac8d33155298dfdbedfbc/docs/load_balancing/kv_aware.md#layout-files)

```bash
export LAYOUT_FILE="$RUN_DIR/prefill-layout.json"

python3 - "$PROFILE" "$CHECKPOINT" "$LAYOUT_FILE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

profile, checkpoint, filename = sys.argv[1:]
directory = Path(checkpoint)
files = ("config.json", "model.safetensors.index.json")
metadata = {
    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
    for name in files if (directory / name).is_file()
}
if "config.json" not in metadata:
    raise SystemExit("Expected a checkpoint with config.json")
weight_version = "config-index-sha256:" + hashlib.sha256(
    json.dumps(metadata, sort_keys=True).encode()
).hexdigest()

def group(kind, spec, block, cacheable=True, ratio="1", replay=0, **extra):
    return dict(kind=kind, spec_type=spec, block_size=block, prefix_cacheable=cacheable,
                tokens_per_state=ratio, prefix_replay_tokens=replay, **extra)

def sliding(block, window, cacheable=True, replay=0):
    return group("sliding_window_mla", "SlidingWindowMLASpec", block, cacheable,
                 replay=replay, sliding_window=window)

if profile == "v4":
    model, unit, speculative, ranks = "dsv4-mini", 8, 0, [0, 0]
    groups = [
        group("mla_attention", "MLAAttentionSpec", 512, ratio="4", cache_role="sparse"),
        group("mla_attention", "MLAAttentionSpec", 16384, ratio="128", cache_role="sparse"),
        *[sliding(128, 128) for _ in range(4)],
        sliding(8, 8), sliding(32, 128),
    ]
elif profile == "v41":
    model, unit, speculative, ranks = "dsv41-flash-mini4", 128, 5, [0, 1]
    groups = [
        group("mla_attention", "MLAAttentionSpec", 128, ratio="2", cache_role="sparse"),
        group("scratch", "CircularBufferSpec", 32, cacheable=False),
        *[sliding(128, 128, cacheable=False, replay=128) for _ in range(3)],
    ]
else:
    raise SystemExit(f"Unknown profile {profile}")

layout = {}
for port, rank in zip((8000, 8001), ranks):
    cap = dict(rank=rank, model=model, weight_version=weight_version,
               declared_groups=len(groups), hash_algorithm="sha256_cbor", full_hashes=True,
               hash_seed="0", hash_block_size=unit, partial_hash_hits=False, prefix_caching=True,
               speculative_tokens=speculative, hash_lookahead_tokens=0,
               groups=[dict(group_idx=i, **g) for i, g in enumerate(groups)])
    if speculative:
        cap["tail_drop_groups"] = []
    layout[f"http://127.0.0.1:{port}"] = dict(schema_version=1, ranks=[cap])
Path(filename).write_text(json.dumps(layout, indent=2) + "\n")
PY
```

The root hash in the field declaration was confirmed against the engine's complete SHA-256 event hashes. With the tested engine and `PYTHONHASHSEED=0`, the example uses Router's equivalent `hash_seed` field. The metadata digest labels the config and weight index.

## Start Router

Run the following after the workers are ready. Point `ROUTER_BIN` to the pinned Rust executable if it is outside your PATH. Set `PREFILL_POLICY=round_robin`, `cache_aware`, or `kv_aware` for the selected arm. All three use `round_robin` for Decode. `--kv-connector nixl` selects the Router HTTP pull handoff that forwards `kv_transfer_params` from Prefill to Decode. The worker connector performing the transfer is `MooncakeHybridConnector`. [Router HTTP PD implementation](https://github.com/vllm-project/router/blob/d71b621b619a769880aac8d33155298dfdbedfbc/src/routers/http/vllm_pd_router.rs)

```bash
PREFILL_POLICY=${PREFILL_POLICY:-kv_aware}
LAYOUT_ARGS=()
if [[ "$PREFILL_POLICY" == kv_aware ]]; then
  LAYOUT_ARGS=(--kv-hybrid-layout "${LAYOUT_FILE:?set LAYOUT_FILE}")
fi
case "$PROFILE" in
  v4)
    DP=1
    WORKERS=(--prefill http://127.0.0.1:8000 --prefill http://127.0.0.1:8001
             --decode http://127.0.0.1:8002)
    ;;
  v41)
    DP=2
    WORKERS=(--prefill http://127.0.0.1:8000@0 --prefill http://127.0.0.1:8001@1
             --decode http://127.0.0.1:8002@0 --decode http://127.0.0.1:8003@1)
    ;;
  *) printf 'Unsupported profile: %s\n' "$PROFILE" >&2; exit 2 ;;
esac

BYPASS_PROXY="${FABRIC_IP:?set FABRIC_IP},127.0.0.1,localhost"
RUST_LOG=info,vllm_router_rs=debug HF_HUB_OFFLINE=1 NO_PROXY="$BYPASS_PROXY" no_proxy="$BYPASS_PROXY" \
  "${ROUTER_BIN:-vllm-router}" --host 127.0.0.1 --port 8100 --prometheus-host 127.0.0.1 --prometheus-port 29000 \
  --intra-node-data-parallel-size "$DP" --disable-retries \
  --worker-startup-check-interval 1 --worker-startup-timeout-secs 1200 \
  --vllm-pd-disaggregation --kv-connector nixl \
  --policy "$PREFILL_POLICY" --prefill-policy "$PREFILL_POLICY" --decode-policy round_robin \
  --router-temperature 0 --cache-threshold 0.3 --balance-abs-threshold 2 --balance-rel-threshold 1.5 \
  "${WORKERS[@]}" "${LAYOUT_ARGS[@]}"
```

## Verify the route and measure each policy

1. Check that `/v1/models` exposes the expected model ID. Send chat completions and text completions through Router, both with and without streaming. Check the response body, streamed completion marker and all worker identities.
2. For each PD request, follow the same remote request ID through the Prefill response, Decode input, positive external-token allocation, successful nonzero-length synchronous Mooncake read, and completion log. Check all P/D combinations in the V4.1 topology.
3. Warm each Prefill worker directly with its own distinct token-ID prefix. Wait for the corresponding KV event to be applied in Router, then replay a matching request and a different prefix. Compare Router credit and the selected Prefill worker's observed reuse after required recomputation. For V4, probe input lengths 16,385 and 16,386 after warming at least 16,401 tokens. Mooncake's compressed PD path trims the last input token at Prefill and the engine also retains a position for local computation. Credit and observed reuse were 0 at 16,385 and 16,384 at 16,386. For V4.1, a 2,048-token input replay had 1,920 raw cached tokens and a 128-token bounded replay window. Both Prefill ranks gave 1,792 effective reused tokens, matching Router credit. An unrelated prefix produced zero credit and zero reuse.
4. Start fresh engines and Router for each policy, use the same model-specific workload and dedicated preflight, and keep Decode at `round_robin`. Save request timings, successful completion counts, token usage and rank-specific Prefill counter snapshots. The measured runs used 24 requests per arm. All measured responses hit the 128-token limit and contained only reasoning fragments. Interpret output-token throughput as the observed rate for that capped replay. [Run the deterministic workload and see metric definitions](workload.md)

# Recreating the measured request set

The two request sets were generated offline. Each pair of adjacent sessions shares a long system prefix. Different pairs have different prefixes. The assistant history is fixed in advance so that an incorrect or repetitive model answer cannot change the next prompt. No production prompts or model responses are required.

## Generate the exact inputs

Run this Python 3 snippet in a new directory. It creates two JSON files there and verifies their bytes against the recorded workload SHA-256 values. The synthetic English text below is the complete template.

```python
import hashlib
import json
from pathlib import Path
import random

words = (
    "bridge cabinet engine filter frame garden gauge handle harbor hinge inlet ledger "
    "marker motor panel pipe plate relay rotor screen sensor signal socket spring "
    "station switch tank tester thread valve vessel wheel"
).split()
workloads = {
    "v4": (880, "1196420ea77edfd71e406d8d55405f701a3fdda9716e178e347f1f3b26c3dca6"),
    "v41": (180, "dc5d1c71815164b1aef2ff8ca3d0ca4602526b4bbe0c82fcedd5d31138ba1d3b"),
}
for name, (units, expected_hash) in workloads.items():
    rng = random.Random(517)
    sessions = []
    for index in range(6):
        if index % 2 == 0:
            source = random.Random(rng.getrandbits(64))
            rows = [" ".join(source.choices(words, k=20)) for _ in range(units)]
            prefix = (
                f"Engineering record group {index // 2}. Review the following inventory before answering.\n"
                + "\n".join(rows)
            )
        messages = [{"role": "system", "content": prefix}]
        turns = []
        for turn in range(4):
            messages.append({"role": "user", "content": (
                f"Case {index}, step {turn}. Select three relevant maintenance checks for this inventory. "
                "Explain the purpose and evidence of each check in one or two sentences."
            )})
            turns.append({"messages": list(messages), "max_tokens": 128})
            messages.append({"role": "assistant", "content":
                "I will review sensor readings, check the repair ledger, and inspect the seals."})
        sessions.append({"id": f"case-{index:02d}", "turns": turns})
    data = {"version": 1, "kind": "synthetic", "seed": 517, "prefix_units": units,
            "concurrency": 4, "sessions": sessions}
    raw = (json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"{name}: unexpected workload digest {digest}")
    with Path(f"{name}-workload.json").open("xb") as f:
        f.write(raw)
    print(name, digest)
```

## Replay and measure

1. Start each policy arm with fresh engines and Router. Run the same per-Prefill calibration and cache-event probes outside the measurement window. For `kv_aware`, wait for rank-specific replay to be applied on both P endpoints. The measured systems had a prepared index and warmed engines, rather than a first-ever cold request. Use the actual checkpoint tokenizer and chat template to check the prompt plus output budget against the model context. The V4 workload also requires at least 16,384 shared token IDs between successive turns of a session.
2. Start the first four sessions. Send one streaming `POST /v1/chat/completions` per turn with the stored messages, the actual served model name, `temperature=0`, `n=1`, `max_tokens=128` and `stream_options={"include_usage":true}`. After a turn completes, dispatch the next turn of that session immediately. When a session ends, start the next session in file order. Use a fresh request ID per turn and correlate it to the Router decision and the P/D transfer. Keep D on round robin for every arm.
3. Expect 24 completed streams and 24 successful, nonzero remote reads per arm. Verify a matching P response and D request, positive external tokens, correct P/D identities, and a successful synchronous read at D. Inspect policy decisions for fallback. Separately record finish reasons and whether any final-answer content was emitted.
4. Compute output tokens/s as the sum of server-reported completion tokens divided by the wall-clock span of the entire arm. Measure time to first token (TTFT) from dispatch to the first nonempty text or reasoning fragment. Measure end-to-end latency (E2E) from the same dispatch to the end of the response stream. Compute time per output token (TPOT) as `(E2E − TTFT) / (output tokens − 1)` for each request with at least two output tokens, then average those values. E2E P95 uses the 24 request durations. Session completion time (JCT) runs from the first dispatch to the last completion across four turns. JCT P95 uses the six session durations. The reported percentiles use linear interpolation at `(n − 1) × percentile`.
5. Snapshot each P engine’s metrics before and after measurement, allowing the counters to settle. Use local prefix-hit tokens divided by prefix-query tokens. Read the change in `vllm:prompt_tokens_by_source_total{source="local_compute"}` for newly computed tokens. For V4.1 filter shared per-role metrics to each physical DP rank and count it once. D metrics are excluded from P hit rates.

All six arms returned 3,072 output tokens apiece. Those tokens were reasoning output only, and every stream reached the cap. A throughput measurement for correct final answers requires a quality-qualified checkpoint, evaluation of final answers and representative stopping behavior.

# RFC295-B1 finite GPU acceptance — 2026-09-28

Status: **PASS for the bounded zero-physical-KV history fixture**, not a
performance comparison or a new Chat/model/layout acceptance matrix.

## Exact identities

- Harness candidate: `807f1bfd13934a0c3ea8623dd2dedaf03605c005`.
- Product/native compilation candidate: `3238b56d1bb265b8ea0d1f381e3f3f20ffd46099`.
- Native SHA256: `f2613d588ffc8a551c7b7166c6dd900837306767643adc6e2a366e42fa13f898`.
- Wheel SHA256: `e59cc9e4870cd0f6a14902c02d315900092a21fce110125baee6040c2b50d553`.
- Successful summary SHA256: `5d7c72ead155e9b2c6b03b6630a0b1ff93ca6140f671aec8798236a8beac3c75`.
- Downloaded evidence archive SHA256: `dfa4026e9e558b712c10cdd818f8ba55c2dec01e054275dd1176d955e3e95531`.

This is the **same** optimized production wheel tested on CPU, installed into
an isolated Python 3.12 environment; no cloud native rebuild is claimed.
The harness verifies actual native mappings/inode/hash and production features
disabled. Installed Router Python files match the product candidate. Only
documentation, the public GPU harness and its CPU fixture differ between the
two source SHAs. All cloud-side harness corrections were committed locally.

## Deployment

One NVIDIA GeForce RTX 4080 SUPER, 32,760 MiB, driver 595.71.05; two independently
addressable DP/TP/PP=1 Workers on that card. Qwen3-0.6B uses the prior fixed
ModelScope snapshot `09b42cad3d112e832108974449ccb5e8e0f5b5d1`; all model
files were rehashed against its manifest. Context 4096, block size 16,
`sha256_cbor`, seed 0, prefix caching enabled, eager execution and memory
utilization 0.40 per Worker. HTTP is loopback-only; event ports 5557/5558 were
isolated under the fresh user authorization.

vLLM 0.29.0 at `98dff2a81d747d1dba01a47f939f48c3526d4206`, torch distribution
2.13.0 (CUDA installation), transformers 5.17.0, tokenizers 0.23.2,
Python 3.12.3, pydantic 2.13.5 and pyzmq 27.2.0. The previously installed,
separately proposed capability-export snapshot was reused read-only. Its
patch/source hashes were checked, and Workers were freshly started under it.
No new Worker patch was installed, and this is **not stock zero-engine-change**
automatic discovery. Actual process identities and epochs were checked before
and after the fixture; on-disk source provenance is not claimed as in-process
Python-module attestation for every Worker module.

The Conda C++ runtime lacked `GLIBCXX_3.4.30`. Router-only
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` selects the existing system
library; its hash/mapping is recorded. Worker launch explicitly unsets that
override. No system library was replaced. Only orjson 3.12.0 was added to the
new Router venv; final `pip check` passed.

## Six-request result

Two distinct synthetic three-token Completion inputs, each repeated three
times with one output token: four JSON responses and two SSE responses.
These are supported token-ID ingress, not a new text-template or Chat claim.

| Input / repetition | Worker | Stage | History matched tokens | Physical blocks / reusable tokens / actual hit tokens |
|---|---|---|---|---|
| Prefix 0 / first | 0 | least_load | 0 | 0 / 0 / 0 |
| Prefix 0 / repeat 1 | 0 | exact_history | 3 | 0 / 0 / 0 |
| Prefix 0 / repeat 2, SSE | 0 | exact_history | 3 | 0 / 0 / 0 |
| Prefix 1 / first | 1 | least_load | 0 | 0 / 0 / 0 |
| Prefix 1 / repeat 1 | 1 | exact_history | 3 | 0 / 0 / 0 |
| Prefix 1 / repeat 2, SSE | 1 | exact_history | 3 | 0 / 0 / 0 |

All six returned HTTP 200, matched the actual Worker generation input IDs,
reserved and successfully committed history, and produced exactly one Worker
completion-counter increment each: three per Worker **within this fixture**.
Both Workers' physical cache-hit deltas stayed zero. History therefore supplied
advisory affinity without manufacturing reusable tokens or relying on HBM hits.
CL was enabled, but these serial requests remained within its load slack;
this is not a GPU overload/race test. Those lifecycle races remain CPU coverage.
Prepared Completion was disabled; forwarding counters confirm the raw path.

Metadata counters were `[8, 8]` before and after the measured six-request slice;
no metadata HTTP request was added within it. This short observation does not
replace the CPU proof about request-path RPC or a long-duration refresh test.
Router startup conformance/Worker render-oracle requests are outside those six
generation requests and are retained in access logs.

## Retained failures and correction

1. Concurrent initialization on the shared card interfered with memory
   profiling: one Worker estimated only 0.33 GiB KV space versus 0.44 GiB needed.
   After the other Worker completed initialization, the failed Worker was
   started sequentially with unchanged limits. Initial logs remain intact.
2. The first fixture stopped after its first successful generation: ANSI color
   escapes split the literal `committed=true` log field. The actual commit was
   present. Commit `a430add` reuses the existing ANSI stripper; `807f1bf` formats
   its regression cases. Plain/colored true, false, missing-commit and fake-reuse
   fixtures are covered. All 37 public harness tests and self-check passed in
   the GPU environment; affected test-file Black/Ruff checks passed separately.
   The original failed run remains `gpu-history`; the corrected run is
   `gpu-history-ansi`. No retry-to-success generation loop or weakened assertion.

Workers were not restarted between the failed parser attempt and corrected
fixture; the Router was new. The inputs remain below one physical block, and
each corrected request independently verifies zero physical hit deltas. Thus
this is neither a claimed cold-cache performance run nor a hidden cache reset.

## Closeout and remaining gates

All tracked Worker, EngineCore and Router processes were stopped by 20:55
Asia/Shanghai, before the 22:30 deadline. Ports 8100/8101/5557/5558/19000/19001
were released; GPU memory returned to 1 MiB. No models, caches or evidence were
deleted, no unrelated process was stopped, and no push/PR was performed.

Evidence: local `cmb-rfc295-b1-review/gpu-30241/remote-final/evidence/` contains
deployment identity, both runs, per-request tokens/counters/logs, process
records and `closeout.json`; the checksum-verified archive is retained beside it.

The [CPU report](rfc295-b1-cpu-results.md) remains the source for product
regressions and outstanding CI issues. Maintainer integration/interface and
capability-export agreement, human correctness/authorship/license review,
CI handling and final publication approval remain open. No new model family,
prepared Chat, Hybrid/MTP/PD or throughput/TTFT improvement is claimed.

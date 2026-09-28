# RFC295-B1 CPU closeout — 2026-09-28

This CPU snapshot is preserved. A later fresh authorization completed the
[finite GPU history fixture](rfc295-b1-gpu-results.md), reusing these exact
wheel/native bytes with a test-only ANSI-log parser correction.

## Candidate and provenance

- Branch: `cmb_rfc295_b1`; stacked base `fdabd202bfa518df2d684d76e86f1e0d97f9542f`.
- Tested runtime: `3238b56d1bb265b8ea0d1f381e3f3f20ffd46099`.
- Runtime tree: `01189c4bbb77d79b2a49dfb5c818c79d0b06d8f4`.
- Cargo.lock SHA256: `000665f280cab3ba36fd6392afd5e9c39f72fcb10d35521e8161ded0d1b4d614`.
- Linux amd64, Rust/Cargo 1.95.0, Python 3.11.2; isolated contribution mount,
  4 CPUs and 12 GiB memory. Original source and accepted branches untouched.

Logical commits: `fcf67c7` (store and lifecycle), `c1973fe` (public probes and
docs), `e313959` (typed ingress namespace), `3238b56` (preserve Python/PyO3
positional compatibility). Later documentation-only commits are not new tested
runtime identities. No push, PR, Worker installation or GPU access was performed.

## Results

| Check | Result |
|---|---|
| Frozen-source Rust release filters: history / KV / policies / bridge / CLI | PASS 30 / 76 / 88 / 13 / 1; overlapping, not summed |
| Python entrypoint / renderer boundary / performance-hardware harness | PASS 17 / 40 / 37 |
| `cargo check --all-targets --all-features`, fmt | PASS |
| Default-profile `cargo clippy --all-targets --all-features -- -D warnings` | PASS |
| Installed optimized production wheel + `pip check` | PASS |
| Actual native: history off / on / on+CL / on+CL+CT | PASS 12 cases total |
| GPU entry `--self-check` | PASS; fixtures only, no hardware |
| Changed Python files, Ruff 0.16.0 | PASS |
| Whole Python Black 26.5.1 | FAIL inherited: same 11 files as correctly configured base |
| Whole Python Ruff 0.16.0 | FAIL inherited: unused `SimpleNamespace`, unchanged `py_test/test_kv_dense_source_oracle.py` |
| Codespell 2.4.1 workflow options | FAIL: inherited Unicode-escaped fixture at `src/prompt_tokens/completion_input.rs:209`; one new false positive on the standard serialization-trait module path at `src/routers/http/router.rs:4338` |
| GPU / all historical tests / hosted CodeQL and Trivy | NOT RUN |

Python formatting checks used 3.11.2; upstream workflow selects 3.12. The new
spelling false positive is in the typed-ingress failure test, not an inherited
warning; a narrow spelling exception remains for human review. No broad ignore
or runtime change was made after artifact freeze. No unrelated baseline
formatting/code was changed to manufacture a clean full-CI status.
Frozen-source replay checked the exact runtime tree before and after the tests.

Native probes exercise actual PyO3, GIL responsiveness, token consistency,
JSON/SSE transport, deadlines, disconnects, late synchronous work and shutdown.
The facade and HTTP Workers are synthetic. History-enabled cases verify
affinity/commit logs without physical ownership. This does **not** establish
real vLLM rendering, GPU cache residency/hit rates or a performance improvement.
One mock GET handler logged a closed-client `BrokenPipeError`; all asserted
cases and child exits passed, and that warning is retained in raw evidence.

## Production artifact

Wheel: `vllm_router-0.1.15-cp38-abi3-linux_x86_64.whl`.

| Artifact | SHA256 |
|---|---|
| Wheel | `e59cc9e4870cd0f6a14902c02d315900092a21fce110125baee6040c2b50d553` |
| Installed native, 59,268,584 bytes | `f2613d588ffc8a551c7b7166c6dd900837306767643adc6e2a366e42fa13f898` |
| Rust library test executable | `ab9342ed36446bd9179219f75a81e80176a87c19695d774bc385a09657093a87` |
| Rust CLI test executable | `9dceb7cf767d98850ab1403db2d93b9797281b7c36b3106a058633d22910fb39` |

The root native crate was compiled at the tested SHA, with effective rustc
`opt-level=3`, thin LTO, one codegen unit, no incremental compilation and
`extension-module,abi3-py38`. Actual loaded mappings, installed Python files,
wheel/native bytes and all four probe hashes were checked.
`kv_perf_capabilities()` reports disabled, no modes, no selected mode.

Installed CPU environment: vLLM `0.29.0+cpu`, torch `2.13.0+cpu`, transformers
`5.17.0`, tokenizers `0.23.2`, setuptools `77.0.3`, setuptools-rust `1.13.0`,
wheel `0.48.0`. Cargo.lock pins PyO3 `0.26` and Tokio `1.53.1`; inherited vLLM
Rust dependency revision is `98dff2a81d747d1dba01a47f939f48c3526d4206`.
This environment identity does not broaden the supported model/layout matrix.

## Evidence and remaining gates

Local delivery bundle: `cmb-rfc295-b1-review` beside the contribution checkout.
It retains `frozen-3238b56-checks.log`, `clippy-3238b56-final.log`,
`production-release-final/build-manifest.json`, wheel, build log, four
`native-*/summary.json` files and event logs, `final-evidence.json`, saved Rust
test executables and development failures. Earlier candidate evidence is not
relabeled as the final candidate. A docs-only closeout does not require or
claim another native compilation.

Human review priorities: advisory-versus-physical separation, namespace scope,
selection/reservation atomicity, bridge→generation→store lock order, retry and
late-commit fencing, successful-header commit semantics, TTL/capacity and
cleaner lifetime, source authorship/license, and positional API compatibility.

The [support matrix and finite hardware command](rfc295-b1.md) require fresh
one-GPU/two-independent-Worker authority. B1 adds no Worker patch; automatic
generic Dense still depends on the existing separate capability-export
proposal, not a stock API. Resolve that proposal and the maintainer integration
base/interfaces, outstanding CI handling and human publication approval before
publishing the [Draft](rfc295-b1-draft.md). No issue is closed by this slice.

# Draft: opt-in bounded exact-token history for static Regular KV routing

**Draft for human review; not published.** B1 CPU and actual optimized
production-native validation passed at
`3238b56d1bb265b8ea0d1f381e3f3f20ffd46099`. Hardware, outstanding CI handling,
maintainer integration agreement and release approval remain open.
No PR or RFC edit has been made.

Refs #295, https://github.com/vllm-project/router/issues/295#issuecomment-5791829565,
and #294. Implements the proposed B slice, not the parent RFC or PD deliverable.

This increment consumes existing request-scoped exact tokens, contract identity,
generation fences and a single load lease. It ports bounded prefix/session
history and attempt reservation/commit/rollback from the CMB downstream store.
Real physical KV selection remains primary within existing load protection;
history is advisory and never inserts physical blocks or adds reusable tokens.

Regular commits at successful backend response headers. Pre-commit failure,
cancellation and retries roll back; clear/restart/retirement and render-contract
invalidation fence late commits. Default behavior is unchanged. Rust config,
Python/PyO3 and both CLIs expose the same opt-in boundary. Public tests cover
store bounds, selection, lifecycle, namespace isolation and actual native
JSON/SSE forwarding; the finite hardware entry uses real zero-block inputs.

Scope: static Regular, independently addressable DP=1, existing Normal Full
Attention inputs. No renderer/parser rewrite, prepared Chat, Hybrid/MTP/PD,
multi-tier residency or new cost model. Existing standalone cache_aware, CL,
exact-history and physical ownership remain separately documented.

Dependency: the inherited generic vLLM backend still requires the separate
0.29 capability-export proposal. Stock `/server_info` is not an equivalent
runtime/epoch export. B1 adds no Worker patch and does not claim the current
automatic path meets #294's zero-engine-change goal. Restricted native/advisory
alternatives and specific discovery fields are documented for M1/M3/M5/M7/M8
co-review. Integration base/interfaces remain subject to maintainer agreement.

Validation: Rust history/KV/policies/bridge/CLI filters passed (30/76/88/13/1;
overlapping filters), as did Python entrypoint/boundary/harness (17/40/37),
strict default-profile Clippy, check, fmt and pip dependency checks. One actual
production native passed off/on/on+CL/on+CL+CT, 12 synthetic transport cases,
including JSON/SSE, deadline and disconnect. Native SHA256:
`f2613d588ffc8a551c7b7166c6dd900837306767643adc6e2a366e42fa13f898`.
Full Black/Ruff checks retain baseline failures. Codespell flags one inherited
fixture and one new test-only false positive; GPU and hosted security checks
are unrun. [Exact results, wheel identity and limitations](rfc295-b1-cpu-results.md).

Before posting: resolve the integration base/interface assignment, handle
outstanding CI gates, and complete human correctness, authorship/license and
release review. Hardware acceptance is separately authorized and unclaimed.
Reference, do not close, #295 or #294.

Suggested issue/comment update (human-approved publication only): record B as
the current local implementation, keep A/B/C tracking separate, update the old
Chat boundary with the finite accepted text/one-tool results, and distinguish
the current single capability proposal from the historical six-patch PD stack.
Do not claim approved ownership, a merge base or a complete parent trust state.

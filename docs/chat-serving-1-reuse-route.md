# Official reuse audit and prepared Chat decision

Audit date: 2026-09-27. Fixed dependency and current upstream are different
sources. This is a design decision accompanying executed raw-Chat tests, not a
new Worker API or a claim that an RFC was approved.

## Actual source map

Pinned runtime source: `98dff2a81d747d1dba01a47f939f48c3526d4206`, vLLM 0.29.0.
Read the installed source actually exercised by CPU tests:

| Source | Relevant behavior |
| --- | --- |
| `entrypoints/scale_out/render/serving.py` | ServingRender wraps official request validation/rendering and prepares GenerateRequest |
| `renderers/online_renderer.py` | Input template/tokenizer and request/parser adjustments; already reused by Router |
| `entrypoints/openai/chat_completion/serving.py` | Model/admission checks, rendering, effective sampling, engine generation, ordinary JSON/SSE response generators |
| `parser/abstract_parser.py`, `tool_parsers/hermes_tool_parser.py` | Stateful official tools/reasoning extraction; Hermes structural tags change required/named output grammar |
| `entrypoints/scale_out/token_in_token_out/{protocol,serving}.py` | Prepared generate protocol and serving checks; not interchangeable with full Chat serving |
| `renderers/online_derenderer.py` | Batch parsed output exists; streaming explicitly raises NotImplementedError when a tool/reasoning parser is configured |
| `entrypoints/scale_out/derender/serving.py` | Outer validation/bounds and public response construction still required around inner derender |

The last limitation has an executed CPU regression. Do not remove parser
configuration to force streaming success. Official ordinary Chat output replay
passes independently; lack of prepared derender is not lack of Chat support.

Current upstream snapshot inspected:
`924707f1bf94ff583d89bff7522ee12ff032c286` (2026-09-27T12:01:31Z).
[OnlineDerenderer source](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/renderers/online_derenderer.py)
adds parsed streaming and executor offload, but reconstructs a parser by replaying
previous output chunks. Calling it in-process removes HTTP overhead, not replay.
The current [GenerateRequest protocol](https://github.com/vllm-project/vllm/blob/924707f1bf94ff583d89bff7522ee12ff032c286/vllm/entrypoints/scale_out/token_in_token_out/protocol.py)
is not evidence of receiving-engine input-contract enforcement.

- [RFC #47161](https://github.com/vllm-project/vllm/issues/47161): open; describes
  streaming state/replay tradeoffs. Do not borrow its reported benchmarks as ours.
- [#50550](https://github.com/vllm-project/vllm/pull/50550): merged September 18,
  merge `70164bd`; parsed streaming exists upstream, not in our installed fixed source.
- [#57350](https://github.com/vllm-project/vllm/issues/57350): closed by
  [#57528](https://github.com/vllm-project/vllm/pull/57528), merged September 24,
  `5747d4500a085496b874bd03c5cafb01f51ace56`. Executor offload does not eliminate
  repeated parser work.
- [#57572](https://github.com/vllm-project/vllm/issues/57572): shared per-choice
  processor remains open at audit time.
- [#58588](https://github.com/vllm-project/vllm/pull/58588): draft request-level
  output-mode work; text output is not proof of complete tools/reasoning parity.
- Router [#294](https://github.com/vllm-project/router/issues/294) and
  [#295](https://github.com/vllm-project/router/issues/295) are open coordination
  proposals, not an approved base/API or permission to add the broader cost,
  history, PD or multi-tier work here.

## Decision

Ship the raw-Chat semantic increment. **Prefer Route 2 for a separately agreed
prepared-input follow-up:** retain Worker OpenAIServingChat and its output
parsers; replace only redundant input work at its checked serving boundary.
Do not implement Route 1 in parallel, upgrade the Worker silently, copy its
output stack into Router, or recommend the newer replay path without measuring
its actual long-output behavior. This increment measures ordinary pinned
serving output only; newer derender timing is explicitly NOT RUN.

No production prepared Chat code or Worker diff is installed or advertised.
Below is a minimal related integration proposal, not an existing HTTP endpoint
or approved wire type. An executable patch must wait for agreement on this
admission boundary; injecting an unchecked `prompt_token_ids` field is unsafe.

## Route 2: required receiving-side contract

1. Retain original typed Chat request and its raw ingress identity. Complete
   schema/model/auth/admission, tool/reasoning compatibility, effective sampling,
   context-length and special-token checks still run. Rendering currently also
   performs request adjustments: skipping its body wholesale is not safe.
2. An explicitly trusted frontend may provide bounded typed prepared input and
   normalized conversation/required serving context, referencing an input-contract
   ID and engine incarnation. Public client bodies/headers cannot grant trust;
   request hashes do not attest tokenizer equivalence or trusted origin.
3. At **actual receiving engine admission**, compare the expected contract and
   incarnation against the engine selected for that request. Retain a stable
   engine/contract handle across admission, or validate within EngineCore so
   replacement between frontend check and enqueue cannot slip through.
4. A mismatch must reject before generation or use a specifically defined raw
   path before any dispatch. No retry after ambiguous dispatch or emitted SSE.
   Faster metadata polling and local generation fencing do not solve this race.
5. Use existing initialized serving/renderer objects for parser adjustment and
   sampling/output. Thread prepared inputs into their normal execution once;
   prove template/tokenizer call counts are zero on accepted prepared input.
   Preserve original request for tools, reasoning, usage, errors and cancellation.
6. Keep input-contract identity separate from physical KV layout and cache
   residency epoch. Unknown cache-key extras, adapters, salt, embeddings or
   multimodal transforms cannot be silently discarded.
7. Test missing/forged/obsolete contracts, same-URL engine and tokenizer changes
   between check and dispatch, no-generation rejections, timeout/cancel/late
   completion, streaming errors and lease cleanup. A generic endpoint alias or
   an old conformance receipt is not sufficient evidence.

These gates also apply to closing Completion CT's existing check/use limit.
Its default-off immutable-cohort restriction is unchanged. There is no claim of
safe rolling replacement, zero overhead, universal model support or a completed
prepared Chat implementation.

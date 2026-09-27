# Draft: source-backed text Chat / Agent acceptance on the raw serving path

**Draft preparation only; not submitted. GPU/model evidence is pending.**

Base: Perf-2 `98c024e81b67a0eba85e7312cc07ad0c6b248566`.
Do not close Router RFC #294 or #295 with this bounded increment.

## What changes

- Public deterministic CPU replay through vLLM 0.29's official input,
  incremental detokenization, tool/reasoning and JSON/SSE output implementations.
- Bounded synthetic tool execution and actual result/ID history; single/multiple
  calls, selected required/named/auto forms, reasoning separation, output schemas,
  usage, malformed output rejection, cancelled generator and recovery.
- Actual production native + OpenAI SDK wire replay proving raw Chat bodies and
  ordinary SDK/auth headers survive, with observed dispatch and transport counts.
- Existing GPU runner extended with direct-Worker versus Router semantic cases,
  finite real tool loops, Chat abort/recovery and meaningful first-output timing.
- Separate harness/native source identities for test-only follow-ups; product
  changes cannot be attributed to an older compiled native.

Chat serving and in-process render already existed. This is not a new renderer,
Chat tokens-in/out, model-specific Rust parser, scheduling policy, or Worker patch.
No production Rust/Python module, dependency lock or default changes.

## Validation

CPU: 14 new test methods; 31 existing harness tests; official renderer regression
with its 27 named cases and negative cases; three actual native lifecycle scenarios;
helper self-checks. These are overlapping evidence layers, not a combined unique
model-coverage count. Changed-file Ruff and newly added Python Black checks pass.
Inherited repository-wide Python formatting issues remain; full CI is not green
by implication and is not represented as complete.

Production native source is still `2a0d179ec5413f5486dd9dbd2cf8cb6bb22a9110`;
SHA-256 `f17ccc17cc343b5cea68145a3e73d6442cbd38e2413f6f9a75606a23928e683b`.
No rebuild was necessary for test/document-only changes.

Real model tool competence, generation grammar compliance, actual GPU Worker
input/abort metrics and any new performance claim: **NOT RUN**, pending permission.

## Prepared input decision

Keep raw Chat useful and complete within its reviewed text scope. Prefer a
separately coordinated Worker serving-boundary integration over copying output
processing into Router. Installed 0.29 rejects parsed streaming derender; newer
upstream support has different dependencies and parser-history replay cost.
Any future prepared route needs receiving-engine contract enforcement at admission,
not faster polling. Existing Completion CT remains default-off and restricted to
immutable cohorts. No rolling replacement guarantee is added.

Review focus: synthetic tool bounds, official parser fixture fidelity, error and
stream semantics, meaningful TTFT definition, raw forwarding evidence, artifact
provenance and strict separation of replay from real-model results.

References: [support/results/runbook](chat-serving-1.md),
[official-source audit and route decision](chat-serving-1-reuse-route.md).
Human correctness, authorship/license and publication approval remain required.

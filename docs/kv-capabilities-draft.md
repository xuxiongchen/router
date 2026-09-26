# Draft: Discover Worker KV capabilities for the vLLM input backend

## Summary

Add an opt-in Worker-derived capability path for static Regular, independent DP=1 deployments. It recognizes one audited Normal full-attention mechanism rather than a Qwen model-family label. Reuse the existing in-process vLLM 0.29 renderer and startup token conformance; keep original request forwarding, render-once/retry behavior and native tokenizer restrictions.

Separate stored prefix coverage from reusable logical tokens, including the 464-token / 29-block / 448-hit boundary. Add generation-fenced control-plane discovery and an epoch-bound observed-subset event consumer; metadata never populates ownership.

## Dependencies and coordination

- Based on the locally verified Render Bridge `f0f02adb64a26d819b0e6e9e501a37b8a9d71f09`; PR1 unchanged, historical #130 not a dependency. This does not imply maintainer approval of the publication base.
- Requires the separately reviewed linked vLLM 0.29 capability-export/epoch-topic proposal and full Worker restart. The proposed HTTP endpoint is not stock or maintainer-approved. Coordinate endpoint/shared semantics with #294/#295 before publishing.
- A changed Worker boot invalidates the fixed input contract; restart Router to repeat automatic conformance. No transparent reboot/replay claim.

## Evidence / limitations

Public CPU tests cover initialized manager/spec traversal, descriptor/auth transport, source-executed reuse boundaries and Python entrypoint/automatic defaults. Real vLLM CPU preprocessing for non-Qwen SmolLM2-135M-Instruct matches official render full tokens across 10 input shapes. That preprocessing test uses a controlled descriptor and does not prove actual GPU KV layout.

The authorized isolated Linux run passes 55 KV-related Rust tests, 13 bridge tests, fmt/check/Clippy and a debug native build. Two independent DP=1 Workers pass 27/27 finite GPU cases for Qwen3-0.6B and 27/27 for non-Qwen SmolLM2-135M-Instruct, with the same Router binary and no new family-specific production code or handwritten Profile. Full tokens match local preparation, both Worker render endpoints and actual generation. N=464 has 29 stored matches, 448 predicted reusable tokens and 448 actual hit tokens. An actual owned SmolLM2 Worker restart verifies changed-epoch contract rejection, mandatory Router re-conformance and empty observed-subset bootstrap; real cache clear was not run.

[Exact candidate/native bindings and finite measurements](kv-capabilities-gpu-results.md) distinguish measured code from subsequent documentation-only commits. The finite real-text Qwen and SmolLM2 comparisons increase prefix token hits but worsen TTFT and reduce throughput; neither establishes a performance speedup. Each model completed all 256 timed requests, with 16/16 actual Worker request counts per phase. This is finite request-distribution evidence, not universal load balance. Production TTFT and release ABI portability remain unvalidated. No salt/adapter/MM, Hybrid/MTP/PD, cost model, new renderer pool or tokens-in/out support is claimed.

## Review focus

1. Complete original Worker inventory versus scheduler-normalized specs; exact manager/spec/hasher identities and block units.
2. Publisher boot identity, sequence watermark, socket-thread ownership, stale reply/event generation fencing and clear/gap behavior.
3. Input contract lifetime across Worker restart; raw fallback versus changed-contract failure.
4. Dense terminal recompute formula and distinction between prediction and actual backend counters.
5. Human authorship/license review, endpoint/base coordination, runtime validation and final release approval.

## Publication (not executed)

Inspect the new branch and obtain the approved stack/base choice first. Do not append this work to PR1 or push the linked engine change as if already deployed.

```sh
git switch codex/cmb-kv-capabilities-1
git status --short
git log --oneline f0f02adb64a26d819b0e6e9e501a37b8a9d71f09..HEAD
test -n "$APPROVED_BASE"
git push --set-upstream origin codex/cmb-kv-capabilities-1
gh pr create --repo vllm-project/router --draft \
  --head xuxiongchen:codex/cmb-kv-capabilities-1 --base "$APPROVED_BASE" \
  --title 'Discover Worker KV capabilities for the vLLM input backend' \
  --body-file docs/kv-capabilities-draft.md
```

`APPROVED_BASE` is intentionally not invented. Remove this publication instruction section from the posted PR body if desired. Do not run these commands until publication is explicitly approved.

# Draft: Discover Worker KV capabilities for the vLLM input backend

## Summary

Add an opt-in Worker-derived capability path for static Regular, independent DP=1 deployments. It recognizes one audited Normal full-attention mechanism rather than a Qwen model-family label. Reuse the existing in-process vLLM 0.29 renderer and startup token conformance; keep original request forwarding, render-once/retry behavior and native tokenizer restrictions.

Separate stored prefix coverage from reusable logical tokens, including the 464-token / 29-block / 448-hit boundary. Add generation-fenced control-plane discovery and an epoch-bound observed-subset event consumer; metadata never populates ownership.

## Dependencies and coordination

- Based on accepted Render Bridge `f0f02adb64a26d819b0e6e9e501a37b8a9d71f09`; PR1 unchanged, historical #130 not a dependency.
- Requires the separately reviewed linked vLLM 0.29 capability-export/epoch-topic proposal and full Worker restart. The proposed HTTP endpoint is not stock or maintainer-approved. Coordinate endpoint/shared semantics with #294/#295 before publishing.
- A changed Worker boot invalidates the fixed input contract; restart Router to repeat automatic conformance. No transparent reboot/replay claim.

## Evidence / limitations

Public CPU tests cover initialized manager/spec traversal, descriptor/auth transport, source-executed reuse boundaries and Python entrypoint/automatic defaults. Real vLLM CPU preprocessing for non-Qwen SmolLM2-135M-Instruct matches official render full tokens across 10 input shapes. That preprocessing test uses a controlled descriptor and does not prove actual GPU KV layout.

Rust test code is type-checked; execution and the new native artifact build are outstanding user-run gates. Real two-Worker Qwen/non-Qwen GPU acceptance is pending fresh authorization. Production TTFT and release ABI portability are not established. No salt/adapter/MM, Hybrid/MTP/PD, cost model, new renderer pool or tokens-in/out support is claimed.

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

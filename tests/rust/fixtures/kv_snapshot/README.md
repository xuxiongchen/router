# Synthetic snapshot consumer inputs

These files are local CPU capture evidence, not active remote leases or Worker
identity attestations. They contain only synthetic token ranges and UUIDs.
Tests read them from this directory; no environment variables, tokenizer assets,
weights or running external services are required.

## Snapshot captures

Five files were emitted from the candidate catalogue on official vLLM 0.30.0,
base `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, producer overlay tree
`d56eb6f72a0df64c2364452373d6e233857e3034`. The real `Request`, block hasher,
`BlockPool`, event classes and msgspec encoder generated the payloads; the capture
uses the overlay's `cold_fixture` and `structure_fixture` helpers, not hand-written
ideal event metadata. JSON whitespace is compacted here; payload bytes and all
parsed fields are unchanged. This does not claim full EngineCore/Worker boot.

The profile is local full-attention, rank/group 0, block size 16, lookahead 0.
The target request has 33 tokens and two cached full blocks. Its Store is seq0,
already outside the two-batch replay ring. Unrelated tokens 1000..1032 advance
the history; a positive suffix cannot reconstruct the target without snapshot.

| File | Target range | Capture purpose |
|---|---|---|
| `regular_snapshot.json` | 100..132 | W=2, H=3; suffix re-reports only unrelated Q. |
| `removed_snapshot.json` | 100..132 | Same S(W), followed by actual eviction of both target blocks. |
| `replacement_snapshot.json` | 300..332 | Distinct cache/stream incarnation and prefix Y. |
| `healthy_b_snapshot.json` | 500..532 | Independent source B. |
| `structure_snapshot.json` | 100..132 | First block nonresident; resident descendant still needs its structural ancestor. |

Descriptor digests cover resident flag, u64-BE payload length and original wire
bytes. They are checked by the consumer tests along with counts and W+1..H.
Capture sessions were released; the Router tests create fresh bounded local
attempts through the real registration lease. Never treat the archived lease ID
as an active remote retention contract.

## Official legacy wire

`official-wire.json` comes from unmodified official event/publisher modules:

- vLLM 0.30.0: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, module blob
  `96ddd7ce950e3c7c4535bb9ce1f50f9170c4ad14`.
- vLLM 0.33.0: `4ba787ec9959e566d3b587a796dff07673324da5`, module blob
  `95fc8548bff6c20b0afb7e4c829777b3d97918e7`.

A real inproc ZMQ publisher/replay probe encoded Store, Remove and Clear,
retained seq1/2 and returned the official empty-topic/-1/empty-payload terminal.
Within each stream the replay bytes equal the original encoded records. Plain
cross-version semantics match; legal MessagePack header differences need not
produce equal bytes. The 0.33 lookahead payload is a negative semantic control,
not a claim of lookahead, Hybrid or snapshot activation compatibility.

## Integrity

SHA256 of the compact repository files:

```text
f7c72bd69685f4e2828bfe9f3b0df1d19b08b6b03f9547b042ee39fd8dbe4c4c  healthy_b_snapshot.json
ffae1b7f1c7c2bebcb7da50932172e28e494543f9dba553276ad3496fceec978  official-wire.json
e08d94f928ffb2658b5d126134b2a07e83aee9581a35baeeb75c26311e080fa5  regular_snapshot.json
bf0d62ac783543cadadb9787ffb0dd2ead2ab18a6159a43ba1bfa1278dcdb322  removed_snapshot.json
ba67e211d06a00c02078065c8fe5ebe500d77f51c59000285e4ffa7a2495ab0a  replacement_snapshot.json
0ebcdcf97bb3bc74c036b0d37f1df8d48f14176199e8d05afebed0947638916b  structure_snapshot.json
```

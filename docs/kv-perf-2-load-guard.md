# Opt-in cache-first load guard

This independent increment starts at `6b17f5a` (the production runtime there is
unchanged from `9a9d968`). It does not change cache hashing, ownership or the
Normal Dense reusable-prefix formula, and adds no Worker patch.

`--kv-load-guard` / Python `kv_load_guard=True` is **off by default**. It applies
only to the existing static Regular KV-aware deployment. No performance benefit
or universally optimal threshold is claimed before measurement.

The initial experiment retains cache preference within **one excess in-flight
request** of the least-loaded eligible Worker. If the cache-best candidate is
outside that band, choose the greatest real reusable prefix inside the band;
retain the existing least-load and fair tie-break rules. All-cold requests are
still fair least-load choices, not fabricated cache hits. Busy workers are not
held in a new queue: existing bounded Router admission remains in charge.

Selection and the existing per-attempt load lease reservation share a short
Router critical section. No rendering or network await occurs in it. Completion
and supported Chat use the same lease, retained through response headers, JSON
buffering or the client-owned SSE body; cancellation/drop releases it. Retries
release the prior attempt before reserving another one. Retired generation
candidates are not reintroduced by the guard.

The feature does not change the original request payload. The separate
Completion token-input increment has its own switch and review. The intended
comparison is C0 (both off), CL (guard only), CT (token input only), CLT (both),
plus genuine product round robin. A new, explicitly authorized GPU window is
required; prior results are baseline evidence, not tests of this change.

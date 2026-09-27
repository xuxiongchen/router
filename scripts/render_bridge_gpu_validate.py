#!/usr/bin/env python3
"""Finite real-worker acceptance runner for the Python-hosted render bridge.

Run only inside a freshly authorized GPU instance. Attaches two already-running
exclusive DP=1 workers; starts/stops ONLY its own Router child. It never changes
vLLM, clears caches, installs packages, downloads models, or launches engines.
The child observes actual facade return values, without changing them, so exact
rendering cannot be confused with a successful non-affinity fallback. All saved
request bodies are synthetic public fixtures. The observer is test-only, not a
production telemetry API. Worker generation does not export original raw bytes:
raw forwarding is separately covered by the CPU integration suite, not claimed
as byte-for-byte GPU proof here.

--automatic-capabilities adds actual linked-export discovery and a family-neutral
corpus plus exact Dense block boundaries. It requires opted-in, fully restarted
Workers; it never installs the linked patch. Metadata counters allow bounded
background refresh rather than asserting zero total metadata HTTP traffic.
Perf-2 switches opt into load protection and narrow prepared Completion input.
Their production regression additionally requires --production-validation.
"""

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
import urllib.parse
import uuid

import kv_aware_cuda_validate as prior


ROOT = Path(__file__).resolve().parents[1]
BASE = "13b04aa2c3e811b9937abb3dbb3bd60f49a118c0"
CAPABILITIES_BASE = "f0f02adb64a26d819b0e6e9e501a37b8a9d71f09"
CAPABILITY_REFRESH_SECONDS = 30
PATCHED_WORKER_FILES = (
    "config/kv_events.py", "distributed/kv_events.py", "engine/protocol.py",
    "v1/engine/core.py", "v1/engine/async_llm.py", "v1/engine/kv_capabilities.py",
    "entrypoints/serve/__init__.py", "entrypoints/serve/kv_capabilities/__init__.py",
    "entrypoints/serve/kv_capabilities/api_router.py",
)


def require(condition, message):
    prior.require(condition, message)


def save(path, value):
    prior.save(path, value)


def source_identity(source, candidate, automatic_capabilities=False):
    source = Path(source).resolve()
    require(source == ROOT, "runner must belong to the candidate source")
    require(prior.command(["git", "rev-parse", "HEAD"], source) == candidate,
            "candidate SHA does not match source HEAD")
    require(not prior.command(["git", "status", "--porcelain"], source), "candidate source is dirty")
    base = CAPABILITIES_BASE if automatic_capabilities else BASE
    prior.command(["git", "merge-base", "--is-ancestor", base, candidate], source)
    return {"candidate_sha": candidate, "tree_sha": prior.command(
        ["git", "rev-parse", "HEAD^{tree}"], source), "source": str(source)}


def loopback_url(value):
    parsed = urllib.parse.urlsplit(value)
    require(parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
            and parsed.port and parsed.path in ("", "/") and not parsed.username
            and not parsed.query and not parsed.fragment,
            "this finite fixture only accepts loopback HTTP endpoints")
    return value.rstrip("/")


def raw_request(base, route, raw, timeout=60):
    parsed = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request("POST", route, raw, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def read_observations(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def mapped_native(pid, native):
    native = Path(native).resolve()
    entries = []
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(native):
            require(int(fields[4]) == native.stat().st_ino, "mapped artifact inode differs")
            entries.append(line)
    require(entries and any("x" in item.split()[1] for item in entries),
            "Router PID does not execute the candidate native extension")
    return {"path": str(native), "sha256": prior.sha256(native),
            "inode": native.stat().st_ino, "maps": entries}


def validate_native_mode(info, production_validation):
    require(isinstance(info, dict) and info.get("selected_mode") is None,
            "functional regression must not select an experimental dispatch mode")
    if production_validation:
        require(info.get("enabled") is False and info.get("modes") == [],
                "production regression requires the actual feature-off native artifact")
    return info


def child(manifest_path):
    config = json.loads(Path(manifest_path).read_text())
    native_path = Path(config["native"])
    require(prior.sha256(native_path) == config["native_sha256"], "native artifact changed")
    spec = importlib.util.spec_from_file_location("vllm_router_rs", native_path)
    native = importlib.util.module_from_spec(spec)
    sys.modules["vllm_router_rs"] = native
    spec.loader.exec_module(native)
    require(not config.get("production_validation", False)
            or callable(getattr(native, "kv_perf_capabilities", None)),
            "production regression requires an actual native capability handshake")
    capability = (native.kv_perf_capabilities() if hasattr(native, "kv_perf_capabilities")
                  else {"enabled": False, "modes": [], "selected_mode": None})
    save(config["native_mode_identity"], validate_native_mode(
        capability, config.get("production_validation", False)))
    sys.path.insert(0, str(ROOT / "py_src"))
    from vllm_router.router import Router
    from vllm_router.router_args import RouterArgs

    args = RouterArgs(
        host="127.0.0.1", port=config["router_port"], worker_urls=config["workers"],
        policy="kv_aware", kv_input_backend="vllm", kv_render_config=config["render_config"],
        kv_load_guard=config.get("kv_load_guard", False),
        kv_completion_token_input=config.get("kv_completion_token_input", False),
        # Automatic mode must obtain hash/unit defaults from actual Workers,
        # and each verified subscriber obtains its epoch-bound exact topic.
        **({"kv_events_topic_filter": ""} if config["automatic_capabilities"] else {
            "kv_hash_algo": "sha256_cbor", "kv_hash_seed": 0, "kv_events_topic_filter": "kv"}),
        kv_events_endpoints=[worker + "=" + endpoint for worker, endpoint in
                             zip(config["workers"], config["event_endpoints"])],
        worker_startup_timeout_secs=180, worker_startup_check_interval=1,
        request_timeout_secs=90, health_check_interval_secs=60, log_level="debug",
        prometheus_host="127.0.0.1", prometheus_port=config["metrics_port"],
        max_concurrent_requests=16, queue_size=16,
    )
    router = Router.from_args(args)
    facade = router._render_facade
    require(facade is not None, "actual vLLM facade was not constructed")
    original_render, original_startup = facade.render, facade.startup
    with Path(config["observations"]).open("a", buffering=1) as observations:
        def observed_render(kind, raw):
            started = time.monotonic_ns()
            result = original_render(kind, raw)
            record = {"kind": kind, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                      "result": result, "elapsed_ns": time.monotonic_ns() - started,
                      "pid": os.getpid(), "at_unix": time.time()}
            observations.write(json.dumps(record, ensure_ascii=False) + "\n")
            observations.flush()
            return result

        def observed_startup():
            try:
                return original_startup()
            finally:
                save(config["facade_identity"], {
                    "contract_id": facade.contract_id, "effective_config": facade.effective_config,
                    "startup_failure": facade.startup_failure,
                    "conformance": getattr(facade, "conformance", None),
                    "capability_cohort": getattr(facade, "capability_cohort", None),
                    "python_module": str(Path(sys.modules[facade.__class__.__module__].__file__).resolve()),
                })

        facade.render, facade.startup = observed_render, observed_startup
        router.start()


@contextmanager
def owned_router(config, out):
    manifest = out / "router-child.json"
    save(manifest, config)
    environment = os.environ.copy()
    for key in ("VLLM_ROUTER_KV_PERF_MODE", "VLLM_ROUTER_KV_STAGE_TIMING", "VLLM_ROUTER_KV_STAGE_TRACE"):
        environment.pop(key, None)
    with (out / "router.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                    "--child", str(manifest)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=environment)
        try:
            deadline = time.monotonic() + 200
            while time.monotonic() < deadline:
                require(process.poll() is None, "owned Router exited during startup; inspect router.log")
                try:
                    status, _, _ = prior.request(f"http://127.0.0.1:{config['router_port']}",
                                                 "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("Router startup exceeded finite deadline")
            validate_native_mode(json.loads(Path(config["native_mode_identity"]).read_text()),
                                 config.get("production_validation", False))
            yield process
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    process.kill()  # Only this runner's exact owned child.
                    process.wait(timeout=5)
            save(out / "router-exit.json", {"pid": process.pid, "returncode": process.returncode,
                                            "finished_at_unix": time.time()})


def render_access_count(paths):
    # Stock access logs are independent corroboration, not a guarantee of an
    # arbitrary deployment's logging coverage. Record rather than invent proof.
    return [sum('POST /v1/' in line and '/render HTTP/' in line
                for line in Path(path).read_text(errors="replace").splitlines()) for path in paths]


def metadata_access_count(paths):
    return [sum('GET /v1/kv-cache/capabilities HTTP/' in line
                for line in Path(path).read_text(errors="replace").splitlines()) for path in paths]


def worker_capabilities(workers, model):
    """Two bounded control-plane snapshots; never called by routed()."""
    spec = importlib.util.spec_from_file_location(
        "gpu_capability_bridge", ROOT / "py_src/vllm_router/render_bridge.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    descriptors = {url: bridge._remote_capabilities(url, 10, None) for url in workers}
    contracts = [bridge._capability_contract(value, model) for value in descriptors.values()]
    require(contracts[0] == contracts[1], "Workers have incompatible actual capability contracts")
    require(len({value["events"]["epoch"] for value in descriptors.values()}) == 2,
            "independent Workers must have distinct publisher/engine epochs")
    return descriptors


def worker_source_evidence(root):
    root = Path(root).resolve()
    files = {name: prior.sha256(root / name) for name in PATCHED_WORKER_FILES}
    return {"package_directory": str(root), "files_sha256": files,
            "linked_proposal_sha256": prior.sha256(
                ROOT / "docs/dependencies/vllm-0.29-kv-capabilities.patch"),
            "classification": "ON_DISK_SOURCE_PROVENANCE_NOT_LOADED_PYTHON_MODULE_ATTESTATION",
            "requires": "Record Worker interpreter/package locations at fresh authorized deployment; fully restart after patching. No installed-source allowlist is imposed."}


def verify_dense_decision(decision, token_count, block_size, unsupported=False):
    require(decision.get("score_kind") == "reusable_prefix_tokens",
            "automatic capability policy did not enable Dense reuse scoring")
    require(decision.get("query_tokens") == (0 if unsupported else token_count),
            "decision N differs from actual prepared generation input")
    for score in [decision, *decision["scores"]]:
        matched = score.get("prefix_blocks")
        reused = score.get("reusable_prefix_tokens")
        require(type(matched) is int and matched >= 0 and type(reused) is int,
                "missing separate stored/reusable score evidence")
        expected = 0 if unsupported else block_size * min(
            matched, max(token_count - 1, 0) // block_size)
        require(reused == expected, "Dense reusable prediction differs from terminal-recompute rule")


def boundary_lengths(block_size):
    require(type(block_size) is int and 1 < block_size <= 512,
            "finite boundary fixture supports block sizes 2..512")
    return sorted({block_size - 1, block_size, block_size + 1,
                   2 * block_size - 1, 2 * block_size, 2 * block_size + 1,
                   464, 29 * block_size})


def valid_fixture_vocabulary(serving_args):
    option = "--tokenizer" if "--tokenizer" in serving_args else "--model"
    root = Path(serving_args[serving_args.index(option) + 1])
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    vocab = tokenizer["model"]["vocab"]
    values = vocab.values() if isinstance(vocab, dict) else range(len(vocab))
    special = {item["id"] for item in tokenizer.get("added_tokens", []) if item.get("special")}
    values = sorted({value for value in values if type(value) is int and value >= 0} - special)
    require(len(values) >= 256, "finite synthetic boundary corpus requires at least 256 ordinary token IDs")
    return values


def prefix_metrics(values):
    return {name: sum(value for (key, _), value in values.items() if key == name)
            for name, _ in values if "prefix_cache_" in name}


def generation_tokens(body, stream, chat):
    if stream:
        events = [json.loads(line[5:].strip()) for line in body.splitlines()
                  if line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]")]
    else:
        events = [json.loads(body)]
    lists = []
    for event in events:
        require(not event.get("error"), "generation response contains an error")
        candidates = [event] if chat else event.get("choices", [])
        for item in candidates:
            if item.get("prompt_token_ids") is not None:
                lists.append(item["prompt_token_ids"])
    require(lists and all(value == lists[0] for value in lists),
            "stock generation response did not expose consistent prompt_token_ids")
    return lists[0]


def completion_semantics(body, stream):
    """Compare generated text/finish/usage, not IDs, timestamps or SSE boundaries."""
    if stream:
        require(b"data: [DONE]" in body, "semantic oracle SSE is incomplete")
        events = [json.loads(line[5:].strip()) for line in body.splitlines()
                  if line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]")]
    else:
        events = [json.loads(body)]
    choices, usage = {}, None
    for event in events:
        require(isinstance(event, dict) and not event.get("error"), "semantic oracle contains an error")
        if event.get("usage") is not None:
            value = event["usage"]
            usage = {key: value.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        for choice in event.get("choices", []):
            index, text = choice.get("index"), choice.get("text")
            require(type(index) is int and index == 0 and isinstance(text, str),
                    "semantic fixture requires one ordinary text Completion")
            current = choices.setdefault(index, {"text": "", "finish_reason": None})
            current["text"] += text
            if choice.get("finish_reason") is not None:
                require(current["finish_reason"] is None, "duplicate terminal Completion choice")
                current["finish_reason"] = choice["finish_reason"]
    require(list(choices) == [0] and choices[0]["finish_reason"] in ("stop", "length"),
            "semantic oracle did not complete one successful choice")
    require(usage is not None and all(type(value) is int and value >= 0 for value in usage.values())
            and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "semantic oracle lacks consistent real usage")
    return {"choices": [choices[0]], "usage": usage}


def public_render_tokens(body, chat):
    rendered = json.loads(body)
    if not chat:
        require(isinstance(rendered, list) and len(rendered) == 1,
                "Completion public render must return exactly one input")
        rendered = rendered[0]
    require(isinstance(rendered, dict), "public render result is not a token-input object")
    tokens = rendered.get("token_ids")
    require(isinstance(tokens, list) and tokens
            and all(type(t) is int and 0 <= t < 2**32 for t in tokens),
            "worker public render did not return one exact text token list")
    return tokens


def positive_payload(model, name, kind, stream=False, *, nonce=None):
    """Make the warmed prefix unique before any complete shared template block.

    An explicit system message avoids a tokenizer's injected default system
    prefix becoming cached on both Workers during earlier Chat shape cases.
    This is a public test fixture, not a model-specific Router renderer.
    """
    nonce = nonce or uuid.uuid4().hex
    text = (f"{nonce} Public GPU render bridge case {name}. " * 12
            + "Explain the water cycle in ordinary words.")
    payload = {"model": model, "max_tokens": 8, "temperature": 0, "stream": stream,
               "return_token_ids": True}
    if kind == "chat":
        payload.update(messages=[
            {"role": "system", "content": f"{nonce} Public isolated prefix fixture."},
            {"role": "user", "content": text},
        ], chat_template_kwargs={"enable_thinking": False})
    else:
        payload.update(prompt=text, add_special_tokens=True)
    return payload


class Validation(prior.Validation):
    def __init__(self, args, out):
        super().__init__(args, out)
        self.observations = out / "facade-observations.jsonl"
        self.model = args.model
        self.worker_logs = [args.worker0_log, args.worker1_log]
        self.automatic_capabilities = args.automatic_capabilities
        self.perf2 = args.kv_load_guard or args.kv_completion_token_input

    def case(self, name, callback):
        require(time.monotonic() < self.args.deadline, "finite GPU matrix budget exhausted")
        result = super().case(name, callback)
        if self.perf2:
            require(self.results[-1]["status"] == "PASS", "Perf-2 functional case failed; stop the matrix")
        return result

    def forwarding_snapshot(self):
        if not self.perf2:
            return None
        # The existing performance runner owns the exact metric parser. This
        # local import is after module initialization (that runner imports us).
        from kv_capabilities_performance import metric_snapshot
        return metric_snapshot(f"http://127.0.0.1:{self.args.metrics_port}")[0]

    def forwarding_evidence(self, before, prepared, completion=True, dispatched=True):
        if before is None:
            return None
        from kv_capabilities_performance import forwarding_window
        return forwarding_window(before, self.forwarding_snapshot(), int(completion and dispatched),
                                 prepared, completion=completion and dispatched)

    def oracle(self, payload, name):
        route = "/v1/chat/completions" if "messages" in payload else "/v1/completions"
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        values = []
        for index, worker in enumerate(self.workers):
            status, headers, body = raw_request(worker, route + "/render", raw)
            save(self.out / f"{name}.worker{index}.render.json", {
                "http_status": status, "headers": headers, "response": body.decode(),
                "raw_request_sha256": hashlib.sha256(raw).hexdigest()})
            require(status == 200, f"worker{index} public render HTTP {status}: {body[:300]!r}")
            tokens = public_render_tokens(body, route == "/v1/chat/completions")
            values.append(tokens)
        require(values[0] == values[1], "real Workers disagree on complete token IDs")
        return raw, values[0], route

    def routed(self, name, payload, expected=None, unsupported=False, cold=False,
               expect_prepared=None):
        self.idle()
        payload = {**payload, "return_token_ids": True}
        raw, tokens, route = self.oracle(payload, name)
        (self.out / f"{name}.request.json").write_bytes(raw)
        before = self.counters()
        before_prefix = [prefix_metrics(prior.metrics(worker)) for worker in self.workers]
        before_observation = len(read_observations(self.observations))
        before_access = render_access_count(self.worker_logs)
        before_metadata = metadata_access_count(self.worker_logs)
        forwarding_before = self.forwarding_snapshot()
        request_started = time.monotonic()
        offset = Path(self.args.router_log).stat().st_size
        status, headers, body = raw_request(self.args.router, route, raw)
        (self.out / f"{name}.response.bin").write_bytes(body)
        require(status == 200, f"Router HTTP {status}: {body[:500]!r}")
        if payload.get("stream"):
            require("text/event-stream" in headers.get("content-type", headers.get("Content-Type", ""))
                    and b"data: [DONE]" in body, "incomplete SSE generation response")
        else:
            require(json.loads(body).get("choices"), "generation returned no choices")
        actual_generation_tokens = generation_tokens(body, bool(payload.get("stream")), "messages" in payload)
        require(actual_generation_tokens == tokens,
                "actual Worker generation input tokens differ from public render output")
        actual, delta = self.observed_backend(before)
        self.idle()
        tail = self.log_tail(offset)
        (self.out / f"{name}.router.log").write_text(tail)
        decisions = prior.parse_decisions(tail)
        require(len(decisions) == 1, "expected exactly one first-attempt routing decision")
        decision = decisions[0]
        require(decision["worker"].rstrip("/") == self.workers[actual], "decision/counter backend mismatch")
        observations = read_observations(self.observations)[before_observation:]
        require(len(observations) == 1, "request was not rendered exactly once")
        observed = observations[0]
        require(observed["raw_sha256"] == hashlib.sha256(raw).hexdigest(), "facade did not receive original bytes")
        prepared = (self.args.kv_completion_token_input and route == "/v1/completions"
                    and observed["result"].get("completion_token_input_eligible") is True)
        if expect_prepared is not None:
            require(prepared == expect_prepared, "explicit Completion proof fixture eligibility changed")
        forwarding = self.forwarding_evidence(forwarding_before, prepared,
                                              completion=route == "/v1/completions")
        scores = {item["worker"].rstrip("/"): item["prefix_blocks"] for item in decision["scores"]}
        require(set(scores) == set(self.workers), "decision lacks both worker scores")
        if self.automatic_capabilities:
            verify_dense_decision(decision, len(tokens), self.args.block_size, unsupported)
        if unsupported:
            require(observed["result"]["status"] == "unsupported", "cache salt was not rejected from exact path")
            require(decision["token_ids_sha256"] is None and all(score == 0 for score in scores.values()),
                    "unsupported cache identity influenced affinity")
        else:
            require(observed["result"]["status"] == "exact"
                    and observed["result"]["token_ids"] == tokens,
                    "actual in-process facade tokens differ from Worker full render IDs")
            require(decision["token_ids_sha256"] == prior.token_digest(tokens), "Rust token digest differs")
        if expected is not None:
            require(actual == expected and scores[self.workers[expected]] > 0
                    and scores[self.workers[1 - expected]] == 0,
                    "real-event positive ownership did not select the uniquely warmed Worker")
            if self.automatic_capabilities:
                require(decision["reusable_prefix_tokens"] > 0,
                        "warmed-owner proof has stored coverage but zero reusable prefix")
        after_access = render_access_count(self.worker_logs)
        require(after_access == before_access, "generation unexpectedly used request-level Worker /render HTTP")
        after_prefix = [prefix_metrics(prior.metrics(worker)) for worker in self.workers]
        prefix_delta = [{key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}
                        for before, after in zip(before_prefix, after_prefix)]
        hits = prefix_delta[actual].get("vllm:prefix_cache_hits_total", 0)
        queries = prefix_delta[actual].get("vllm:prefix_cache_queries_total", 0)
        if self.automatic_capabilities and not unsupported:
            require(queries == len(tokens) and 0 <= hits <= queries,
                    "exclusive Worker hit/query metric delta does not match prepared input N")
            require(decision["reusable_prefix_tokens"] <= hits,
                    "Router reusable prediction exceeds actual backend cache reuse")
        if cold:
            require(all(score == 0 for score in scores.values()) and hits == 0,
                    "fresh cold boundary unexpectedly had cache ownership or backend hits")
        if expected is not None:
            hits = prefix_delta[expected].get("vllm:prefix_cache_hits_total", 0)
            queries = prefix_delta[expected].get("vllm:prefix_cache_queries_total", 0)
            require(0 < hits <= queries, "positive routing lacks a valid backend prefix-cache token hit/query delta")
        after_metadata = metadata_access_count(self.worker_logs)
        elapsed = time.monotonic() - request_started
        if self.automatic_capabilities:
            # Periodic/revalidation control-plane reads may coincide with a
            # request; do not misreport every observed access as per-request RPC.
            budget = 2 + math.ceil(elapsed / CAPABILITY_REFRESH_SECONDS)
            require(all(0 <= after - before <= budget for before, after in
                        zip(before_metadata, after_metadata)),
                    "unexpected metadata access burst during stable finite request")
        result = {"name": name, "status": "PASS", "actual_backend": actual,
                  "request": payload, "http_status": status, "response_sha256": hashlib.sha256(body).hexdigest(),
                  "decision": decision, "completed_request_deltas": delta,
                  "worker_token_count": len(tokens), "worker_token_ids_sha256": prior.token_digest(tokens),
                  "actual_generation_token_ids": actual_generation_tokens,
                  "completion_forwarding": forwarding,
                  "forwarding_scope": ("derived prepared token body; original facade ingress remains unchanged"
                                       if prepared else "original backend path; exact bytes covered by CPU wire tests"),
                  "facade_observation": observed, "prefix_cache_metric_deltas": prefix_delta,
                  "prefix_cache_token_hit_ratios": [
                      value.get("vllm:prefix_cache_hits_total", 0) / value["vllm:prefix_cache_queries_total"]
                      if value.get("vllm:prefix_cache_queries_total", 0) > 0 else None
                      for value in prefix_delta],
                  "worker_render_access_counts_before": before_access,
                  "worker_render_access_counts_after": after_access,
                  "worker_metadata_access_counts_before": before_metadata,
                  "worker_metadata_access_counts_after": after_metadata,
                  "metadata_access_scope": "Includes allowed background control-plane refresh; CPU source/integration tests establish no request-path metadata call.",
                  "reusable_prediction_scope": "Observed-subset lower bound, not complete cached inventory or guaranteed future work savings."}
        save(self.out / f"{name}.json", result)
        return result

    def completion_equivalence(self, name, stream, extra=None, stop_probe=False):
        """Untimed real original-vs-Router generation; no Worker API/patch change."""
        payload = {"model": self.model,
                   "prompt": f"{uuid.uuid4().hex}\n  café 中文 🙂\t\nComplete this sentence: Water flows",
                   "add_special_tokens": False, "n": 1, "temperature": 0,
                   "top_p": 0.95, "seed": 7, "max_tokens": 16,
                   "stream": stream, "return_token_ids": True, **(extra or {})}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if stop_probe:
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.idle()
            status, _, body = raw_request(self.workers[0], "/v1/completions", raw)
            (self.out / f"{name}.stop-probe.response.bin").write_bytes(body)
            require(status == 200, "stop-discovery generation failed")
            generated = completion_semantics(body, stream)["choices"][0]["text"]
            require(generated, "stop fixture needs actual nonempty deterministic reference text")
            payload["stop"] = generated[:4]
        self.idle()
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        (self.out / f"{name}.direct-original.request.json").write_bytes(raw)
        status, headers, direct = raw_request(self.workers[0], "/v1/completions", raw)
        (self.out / f"{name}.direct-original.response.bin").write_bytes(direct)
        require(status == 200, "direct original Completion failed")
        reference = completion_semantics(direct, stream)
        expected_prepared = self.args.kv_completion_token_input and not extra
        routed = self.routed(name, payload, expect_prepared=expected_prepared)
        backend = (self.out / f"{name}.response.bin").read_bytes()
        actual = completion_semantics(backend, stream)
        require(actual == reference, "deterministic Completion text/finish/usage differs from original path")
        require(generation_tokens(direct, stream, False) == routed["actual_generation_token_ids"],
                "original-vs-Router actual Worker prompt IDs differ")
        if stop_probe:
            require(actual["choices"][0]["finish_reason"] == "stop", "stop fixture did not actually stop")
        result = {"name": name, "status": "PASS", "direct_original_headers": headers,
                  "original_semantics": reference, "router_semantics": actual,
                  "actual_generation_token_ids": routed["actual_generation_token_ids"],
                  "completion_forwarding": routed["completion_forwarding"],
                  "scope": "temperature=0, fixed seed; exact text, finish and core token usage. "
                           "IDs/timestamps/chunk boundaries/cache-usage details and logprob float values are not compared; raw responses retained."}
        save(self.out / f"{name}.equivalence.json", result)
        return result

    def invalid_completion(self, name, raw):
        self.idle()
        observed = []
        before = self.forwarding_snapshot()
        for label, endpoint in (("direct-original", self.workers[0]), ("router", self.args.router)):
            status, headers, body = raw_request(endpoint, "/v1/completions", raw)
            (self.out / f"{name}.{label}.response.bin").write_bytes(body)
            observed.append({"path": label, "http_status": status, "headers": headers})
            require(400 <= status < 500, "invalid request was converted into a successful generation")
        require(observed[0]["http_status"] == observed[1]["http_status"],
                "invalid Completion Router status differs from direct Worker")
        self.idle()
        result = {"name": name, "status": "PASS", "request_utf8": raw.decode(), "responses": observed,
                  "completion_forwarding": self.forwarding_evidence(before, False, dispatched=False),
                  "scope": "Invalid schema/JSON remains invalid and produces no backend dispatch; error text need not be byte-identical."}
        save(self.out / f"{name}.json", result)
        return result

    def cancel_payload(self):
        return getattr(self, "_cancel_payload_override", None) or super().cancel_payload()

    def cancel_cleanup(self):
        if not self.perf2:
            return super().cancel_cleanup()
        payload = super().cancel_payload()
        payload.update(add_special_tokens=False, return_token_ids=True)
        self._cancel_payload_override = payload
        try:
            _raw, tokens, _route = self.oracle(payload, "stream_cancel_cleanup")
            before = self.forwarding_snapshot()
            result = super().cancel_cleanup()
            actual = generation_tokens(json.dumps(result["first_nonterminal_event"]).encode(), False, False)
            require(actual == tokens, "cancelled request actual Worker prompt IDs differ")
            result.update(actual_generation_token_ids=actual,
                          completion_forwarding=self.forwarding_evidence(
                              before, self.args.kv_completion_token_input))
            save(self.out / "stream_cancel_cleanup.json", result)
            return result
        finally:
            self._cancel_payload_override = None

    def positive(self, name, kind, target, stream=False):
        payload = positive_payload(self.model, name, kind, stream)
        raw, tokens, route = self.oracle(payload, name + "-warm")
        require(len(tokens) >= 48, "positive fixture is too short for several cache blocks")
        self.idle()
        before = self.counters()
        status, _, body = raw_request(self.workers[target], route, raw)
        require(status == 200, f"direct warm HTTP {status}: {body[:300]!r}")
        require(generation_tokens(body, stream, kind == "chat") == tokens,
                "direct warm generation tokens differ from Worker render")
        actual, delta = self.observed_backend(before)
        require(actual == target, "direct warm reached the wrong Worker")
        time.sleep(self.args.event_wait)
        result = self.routed(name, payload, expected=target)
        result["direct_warm_completed_request_deltas"] = delta
        save(self.out / f"{name}.json", result)
        return result

    def salt_fairness(self):
        observed = []
        for index in range(4):
            result = self.routed(f"salt-fallback-{index}", {
                "model": self.model, "prompt": f"Public salted fixture {uuid.uuid4().hex}. " * 8,
                "cache_salt": "public-render-bridge-salt", "max_tokens": 1, "temperature": 0},
                unsupported=True)
            observed.append(result["actual_backend"])
        require(observed.count(0) == observed.count(1) == 2, "idle non-affinity fallback did not split four requests evenly")
        return {"name": "salt_fairness", "status": "PASS", "backends": observed,
                "scope": "four sequential idle unsupported requests; not general load-balancing performance"}

    def dense_boundary(self, token_count, target):
        """Separate fresh cold and direct-only-warmed prefixes; never reset KV."""
        name = f"boundary-n{token_count}-w{target}"
        rng = random.Random(uuid.uuid4().hex)
        vocabulary = self.args.fixture_vocabulary
        cold_ids = rng.choices(vocabulary, k=token_count)
        warm_ids = rng.choices(vocabulary, k=token_count)
        require(cold_ids != warm_ids, "boundary fixture prefixes must be independent")
        base = {"model": self.model, "max_tokens": 1, "temperature": 0,
                "add_special_tokens": False, "return_token_ids": True}
        cold = self.routed(name + "-cold", {**base, "prompt": cold_ids}, cold=True)
        payload = {**base, "prompt": warm_ids}
        raw, tokens, route = self.oracle(payload, name + "-direct-warm")
        require(tokens == warm_ids and len(tokens) == token_count,
                "actual public preprocessing changed the exact boundary N")
        self.idle()
        before = self.counters()
        status, _, body = raw_request(self.workers[target], route, raw)
        require(status == 200 and generation_tokens(body, False, False) == warm_ids,
                "direct boundary warm generation did not preserve exact input IDs")
        actual, warm_delta = self.observed_backend(before)
        require(actual == target, "boundary direct warm went to the wrong Worker")
        time.sleep(self.args.event_wait)
        reusable = self.args.block_size * ((token_count - 1) // self.args.block_size)
        warm = self.routed(name + "-warm", payload, expected=target if reusable else None)
        require(warm["actual_generation_token_ids"] == warm_ids,
                "routed generation changed boundary input IDs")
        raw_scores = {entry["worker"].rstrip("/"): entry["prefix_blocks"]
                      for entry in warm["decision"]["scores"]}
        require(raw_scores[self.workers[target]] == token_count // self.args.block_size
                and raw_scores[self.workers[1 - target]] == 0,
                "boundary needs the direct-warm complete stored coverage observed in real events")
        require(warm["decision"]["reusable_prefix_tokens"] == reusable,
                "boundary terminal recompute prediction mismatch")
        hits = warm["prefix_cache_metric_deltas"][warm["actual_backend"]].get(
            "vllm:prefix_cache_hits_total", 0)
        require(hits == reusable, "controlled fully-warmed boundary hit count differs from prediction")
        result = {"name": name, "status": "PASS", "query_tokens": token_count,
                  "block_tokens": self.args.block_size, "direct_warm_target": target,
                  "raw_matched_blocks_on_warm_owner": raw_scores[self.workers[target]],
                  "predicted_reusable_tokens": reusable, "actual_hit_tokens": hits,
                  "cold_result": cold["name"], "warm_result": warm["name"],
                  "direct_warm_completed_request_deltas": warm_delta,
                  "scope": "Cold and warm use separate fresh prefixes: a completed cold probe would itself populate cache. Warm prefix reaches only its explicit Worker before first Router dispatch."}
        save(self.out / f"{name}.json", result)
        return result


def verify_workers(args, config):
    processes = [prior.process(pid) for pid in (args.worker0_pid, args.worker1_pid,
                                               args.engine0_pid, args.engine1_pid)]
    require(len({item["pid"] for item in processes}) == 4, "four independent HTTP/EngineCore PIDs required")
    for index in range(2):
        worker, engine = processes[index], processes[index + 2]
        require(engine["engine_core_title"] and prior.descendant(engine["pid"], worker["pid"]),
                "EngineCore is not an independently addressable DP=1 worker child")
        flags, env = worker["selected_arguments"], worker["selected_environment"]
        require(flags.get("--data-parallel-size") == ["1"]
                and flags.get("--tensor-parallel-size") == ["1"]
                and flags.get("--pipeline-parallel-size") == ["1"], "expected explicit DP/TP/PP=1")
        descriptor = config.get("capabilities", {}).get(config["workers"][index])
        block_size = descriptor["hash"]["block_tokens"] if descriptor else 16
        seed = descriptor["hash"]["seed"] if descriptor else 0
        require(flags.get("--block-size") == [str(block_size)]
                and flags.get("--prefix-caching-hash-algo") == ["sha256_cbor"], "Worker hash contract mismatch")
        require(env.get("PYTHONHASHSEED") == str(seed) and env.get("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") == "0",
                "Worker hash seed/full-byte events mismatch")
        require(env.get("VLLM_SERVER_DEV_MODE") in (None, "0"), "development mode is not allowed")
        kv = json.loads(flags["--kv-events-config"][0])
        require(kv.get("enable_kv_cache_events") and kv.get("publisher") == "zmq"
                and kv.get("endpoint") == config["publisher_endpoints"][index],
                "Worker publisher does not match declared Router event endpoint")
        if descriptor:
            events = descriptor["events"]
            require(kv.get("enable_capabilities") is True,
                    "Worker was not explicitly restarted with enable_capabilities=true")
            require(events["configured_endpoint"] == kv["endpoint"]
                    and events["topic"] == kv.get("topic", "") + "." + events["epoch"],
                    "effective descriptor is not bound to configured publisher/epoch topic")
            require(urllib.parse.urlsplit(events["resolved_endpoint"]).port
                    == urllib.parse.urlsplit(config["event_endpoints"][index]).port,
                    "resolved Worker publisher port differs from subscriber endpoint")
        else:
            require(kv.get("topic") == "kv", "legacy Worker topic must be kv")
        require(prior.vllm_publisher_mode(kv["endpoint"]) == "bind", "Worker publisher must bind")
        options = config["serving_args"]
        model_path = str(Path(options[options.index("--model") + 1]).resolve())
        require(worker["serve_model_path"] == model_path, "Worker and facade model asset directory differ")
        argv = [value.decode() for value in Path(f"/proc/{worker['pid']}/cmdline").read_bytes().split(b"\0") if value]
        # Read only the reviewed preprocess flags, never serialize arbitrary
        # worker arguments or environment (which may contain unrelated secrets).
        selected = {}
        cursor = 0
        bool_flags = {"--enable-auto-tool-choice", "--exclude-tools-when-tool-choice-none"}
        while cursor < len(options):
            key = options[cursor]
            value = True if key in bool_flags else options[cursor + 1]
            cursor += 1 if value is True else 2
            if key == "--model":
                continue  # Already checked the official serve positional model.
            if value is True:
                require(key in argv, f"Worker preprocessing flag differs: {key}")
            else:
                found = []
                for index_arg, argument in enumerate(argv):
                    flag, separator, inline = argument.partition("=")
                    if flag == key:
                        found.append(inline if separator else argv[index_arg + 1])
                require(found == [value], f"Worker preprocessing argument differs: {key}")
            selected[key] = value
        require("--enable-prefix-caching" in argv, "Worker prefix caching is not explicitly enabled")
        worker["verified_preprocessing_arguments"] = selected
        log = Path((args.worker0_log, args.worker1_log)[index]).stat()
        fds = [Path(f"/proc/{worker['pid']}/fd/{fd}").stat() for fd in (1, 2)]
        require(any((fd.st_dev, fd.st_ino) == (log.st_dev, log.st_ino) for fd in fds),
                "supplied Worker log is not that API process stdout/stderr")
    return processes


def run(args):
    # The shared helper includes MODEL in its filtered process identity and
    # cancellation request. Bind it before the FIRST identity capture, not only
    # before cancellation; otherwise a non-Qwen model changes that evidence.
    prior.MODEL = args.model
    out = prior.output_directory(args.output, args.source)
    report = {"status": "RUNNING", "started_at_unix": time.time(), "command": sys.argv,
              "limitations": ["Chat remains original forwarding. Eligible Completion tokens-in is opt-in; Worker still validates and owns text output.",
                              "Queued cancellation uses existing CPU synthetic lifecycle proof; no artificial GPU delay is injected.",
                              "Raw facade ingress is recorded; stock Worker does not export raw generation bytes."]}
    save(out / "summary.json", report)
    args.deadline = time.monotonic() + args.budget_seconds
    try:
        require(platform.system() == "Linux", "run in the authorized GPU process /proc namespace")
        identity = source_identity(args.source, args.candidate, args.automatic_capabilities)
        native_hash = prior.sha256(args.native)
        build = json.loads(Path(args.build_manifest).read_text())
        require(build.get("status") == "PASS" and build.get("candidate_sha") == args.candidate
                and build.get("native_sha256") == native_hash,
                "native build manifest must bind exact candidate and actual .so SHA")
        deployment = json.loads(Path(args.render_config).read_text())
        require((deployment.get("kv_capabilities") == "worker") == args.automatic_capabilities,
                "--automatic-capabilities must agree with render configuration")
        require(not deployment.get("worker_api_key_env"),
                "finite GPU harness requires unauthenticated exclusive loopback Workers; authenticated route is CPU-tested separately")
        workers = [loopback_url(args.worker0), loopback_url(args.worker1)]
        require(deployment["worker_urls"] == workers and len(set(workers)) == 2, "deployment Worker URLs differ")
        config = {"native": str(Path(args.native).resolve()), "native_sha256": native_hash,
                  "workers": workers, "event_endpoints": [args.event0, args.event1],
                  "publisher_endpoints": [args.publisher0 or args.event0, args.publisher1 or args.event1],
                  "router_port": args.router_port, "metrics_port": args.metrics_port,
                  "render_config": str(Path(args.render_config).resolve()),
                  "serving_args": deployment["serving_args"],
                  "automatic_capabilities": args.automatic_capabilities,
                  "kv_load_guard": args.kv_load_guard,
                  "kv_completion_token_input": args.kv_completion_token_input,
                  "production_validation": args.production_validation,
                  "native_mode_identity": str(out / "native-mode-identity.json"),
                  "observations": str(out / "facade-observations.jsonl"),
                  "facade_identity": str(out / "facade-identity.json")}
        if args.automatic_capabilities:
            require(args.worker_vllm_root, "automatic evidence requires --worker-vllm-root")
            for endpoint in config["event_endpoints"]:
                parsed = urllib.parse.urlsplit(endpoint)
                require(parsed.scheme == "tcp" and parsed.hostname == "127.0.0.1"
                        and parsed.port and not parsed.path and not parsed.username
                        and not parsed.query and not parsed.fragment,
                        "automatic finite fixture requires resolved loopback-only KV subscriber endpoints")
            config["capabilities"] = worker_capabilities(workers, args.model)
            args.block_size = next(iter(config["capabilities"].values()))["hash"]["block_tokens"]
            args.fixture_vocabulary = valid_fixture_vocabulary(config["serving_args"])
            boundary_lengths(args.block_size)  # Reject unbounded fixture units before Router launch.
            report["capabilities_before"] = config["capabilities"]
            report["worker_source_provenance"] = worker_source_evidence(args.worker_vllm_root)
        worker_processes = verify_workers(args, config)
        report.update(identity, native_sha256=native_hash, native=str(Path(args.native).resolve()),
                      build_manifest_sha256=prior.sha256(args.build_manifest),
                      render_config_sha256=prior.sha256(args.render_config), workers=worker_processes,
                      kv_load_guard=args.kv_load_guard,
                      kv_completion_token_input=args.kv_completion_token_input,
                      production_validation=args.production_validation,
                      python=sys.version, platform=platform.platform(),
                      installed_versions={name: importlib.metadata.version(name) for name in
                                          ("vllm", "torch", "transformers", "tokenizers", "pydantic")})
        prior.capture_worker_versions(workers, out / "worker_versions.json")
        report["gpu"] = prior.command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid",
                                       "--format=csv,noheader"])
        args.router = f"http://127.0.0.1:{args.router_port}"
        args.router_log = str(out / "router.log")
        save(out / "summary.json", report)
        with owned_router(config, out) as process:
            report["router_process"] = prior.process(process.pid)
            report["mapped_native"] = mapped_native(process.pid, args.native)
            report["native_mode"] = json.loads(Path(config["native_mode_identity"]).read_text())
            save(out / "summary.json", report)
            validation = Validation(args, out)
            report["cases"] = validation.results
            metadata_before = metadata_access_count(validation.worker_logs)
            if args.automatic_capabilities:
                require(all(value > 0 for value in metadata_before),
                        "Worker access logs must expose actual startup metadata reads")
            matrix_started = time.monotonic()
            # Reuse the public, independently reviewed shape corpus. This imports
            # definitions only; it never launches the CPU/mock render service.
            sys.path.insert(0, str(ROOT / "py_test"))
            if args.automatic_capabilities:
                sys.path.insert(0, str(ROOT))
                from test_kv_capabilities_vllm import request_cases
                shape_cases = request_cases(args.model)
            else:
                from test_render_bridge_vllm import actual_cases
                shape_cases = actual_cases(args.model)
            for name, _, payload in shape_cases:
                validation.case("tokens-" + name, lambda name=name, payload=payload:
                                validation.routed("tokens-" + name, payload))
            for kind in ("completion", "chat"):
                for target in (0, 1):
                    for stream in (False, True):
                        name = f"positive-{kind}-w{target}-{'sse' if stream else 'json'}"
                        validation.case(name, lambda name=name, kind=kind, target=target, stream=stream:
                                        validation.positive(name, kind, target, stream))
            if args.automatic_capabilities:
                for index, token_count in enumerate(boundary_lengths(args.block_size)):
                    validation.case(f"boundary-n{token_count}",
                                    lambda token_count=token_count, target=index % 2:
                                    validation.dense_boundary(token_count, target))
            validation.case("salt_fairness", validation.salt_fairness)
            if validation.perf2:
                for stream in (False, True):
                    kind = "sse" if stream else "json"
                    for stopping in (False, True):
                        name = f"perf2-completion-{kind}-{'stop' if stopping else 'ordinary'}"
                        validation.case(name, lambda name=name, stream=stream, stopping=stopping:
                                        validation.completion_equivalence(name, stream, stop_probe=stopping))
                for name, extra in (("echo", {"echo": True}), ("logprobs", {"logprobs": 1})):
                    name = "perf2-original-fallback-" + name
                    validation.case(name, lambda name=name, extra=extra:
                                    validation.completion_equivalence(name, False, extra=extra))
                invalid = json.dumps({"model": args.model, "prompt": "invalid fixture",
                                      "add_special_tokens": False, "max_tokens": -1}).encode()
                validation.case("perf2-invalid-negative-length", lambda:
                                validation.invalid_completion("perf2-invalid-negative-length", invalid))
                validation.case("perf2-invalid-json", lambda:
                                validation.invalid_completion("perf2-invalid-json", b'{"prompt":'))
            # Existing helper has full first-record/active-before-close checks,
            # log/PID/ID correlation and no natural completion masquerading as abort.
            validation.case("stream_cancel_cleanup", validation.cancel_cleanup)
            validation.idle()
            if args.automatic_capabilities:
                metadata_after = metadata_access_count(validation.worker_logs)
                elapsed = time.monotonic() - matrix_started
                allowed = 2 + math.ceil(elapsed / CAPABILITY_REFRESH_SECONDS)
                require(all(0 <= after - before <= allowed for before, after in
                            zip(metadata_before, metadata_after)),
                        "metadata calls exceeded stable finite background-refresh budget")
                report["metadata_access_audit"] = {
                    "before": metadata_before, "after": metadata_after,
                    "elapsed_seconds": elapsed, "allowed_delta_per_worker": allowed,
                    "scope": "Observed HTTP access logs, allowing 30-second control-plane refresh and two revalidation calls. Not a zero-total-RPC assertion; no request-path call is established separately by CPU tests."}
                after_capabilities = worker_capabilities(workers, args.model)
                for url, before_descriptor in config["capabilities"].items():
                    after_descriptor = after_capabilities[url]
                    # Watermarks may advance as this matrix publishes events;
                    # a replacement/contract change is not accepted mid-run.
                    before_copy, after_copy = json.loads(json.dumps(before_descriptor)), json.loads(json.dumps(after_descriptor))
                    before_copy["events"].pop("next_sequence")
                    after_copy["events"].pop("next_sequence")
                    require(before_copy == after_copy, "Worker capability/epoch changed during finite matrix")
                    require(after_descriptor["events"]["next_sequence"] >= before_descriptor["events"]["next_sequence"],
                            "Worker publisher watermark regressed")
                report["capabilities_after"] = after_capabilities
                require(worker_source_evidence(args.worker_vllm_root) == report["worker_source_provenance"],
                        "on-disk Worker source changed during matrix")
            require(mapped_native(process.pid, args.native) == report["mapped_native"],
                    "mapped extension changed during validation")
            for before in worker_processes:
                expected = {key: value for key, value in before.items()
                            if key != "verified_preprocessing_arguments"}
                require(prior.process(before["pid"]) == expected, "Worker process identity changed during matrix")
            report["cases"] = validation.results
        require(source_identity(args.source, args.candidate, args.automatic_capabilities) == identity,
                "source changed during validation")
        require(prior.sha256(args.native) == native_hash, "native artifact changed during validation")
        report["status"] = "PASS" if all(case["status"] == "PASS" for case in report["cases"]) else "FAIL"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    report["finished_at_unix"] = time.time()
    save(out / "summary.json", report)
    print(f"GPU render bridge {report['status']}: {out / 'summary.json'}", flush=True)
    return int(report["status"] != "PASS")


def self_check():
    """Dependency-free parser checks; no HTTP, GPU, subprocess or file mutation."""
    tokens = [1, 2, 151644]
    chat = {"prompt_token_ids": tokens, "choices": [{"finish_reason": "length"}]}
    completion = {"choices": [{"prompt_token_ids": tokens, "finish_reason": "length"}]}
    for is_chat, event in ((True, chat), (False, completion)):
        require(generation_tokens(json.dumps(event).encode(), False, is_chat) == tokens,
                "generation JSON token parser failed")
        sse = b"data: " + json.dumps(event).encode() + b"\r\n\r\ndata: [DONE]\n\n"
        require(generation_tokens(sse, True, is_chat) == tokens, "generation SSE token parser failed")
        rendered = {"token_ids": tokens} if is_chat else [{"token_ids": tokens}]
        require(public_render_tokens(json.dumps(rendered).encode(), is_chat) == tokens,
                "public render token parser failed")
    invalid = [lambda: public_render_tokens(b'[{"token_ids":[true]}]', False),
               lambda: public_render_tokens(b'[{"token_ids":[-1]}]', False),
               lambda: public_render_tokens(b'[]', False),
               lambda: generation_tokens(b'{"choices":[]}', False, False),
               lambda: generation_tokens(b'data: [DONE]\n\n', True, True)]
    for callback in invalid:
        try:
            callback()
        except RuntimeError:
            continue
        raise RuntimeError("invalid token evidence was accepted")
    dense = {"score_kind": "reusable_prefix_tokens", "query_tokens": 464,
             "prefix_blocks": 29, "reusable_prefix_tokens": 448,
             "scores": [{"prefix_blocks": 28, "reusable_prefix_tokens": 448},
                        {"prefix_blocks": 29, "reusable_prefix_tokens": 448}]}
    verify_dense_decision(dense, 464, 16)
    short = {"score_kind": "reusable_prefix_tokens", "query_tokens": 16,
             "prefix_blocks": 1, "reusable_prefix_tokens": 0,
             "scores": [{"prefix_blocks": 1, "reusable_prefix_tokens": 0}]}
    verify_dense_decision(short, 16, 16)
    require(boundary_lengths(16) == [15, 16, 17, 31, 32, 33, 464],
            "Dense finite boundary corpus lost required lengths")
    for key, value in (("reusable_prefix_tokens", 464), ("query_tokens", 465),
                       ("score_kind", "stored_prefix_blocks"), ("prefix_blocks", True)):
        invalid_dense = {**dense, key: value}
        try:
            verify_dense_decision(invalid_dense, 464, 16)
        except RuntimeError:
            continue
        raise RuntimeError("invalid Dense score evidence was accepted")
    nonce = "0123456789abcdef0123456789abcdef"
    fixture = positive_payload("public-non-qwen", "self-check", "chat", True, nonce=nonce)
    require(fixture["messages"][0]["role"] == "system"
            and fixture["messages"][0]["content"].startswith(nonce + " ")
            and fixture["messages"][1]["role"] == "user"
            and fixture["messages"][1]["content"].startswith(nonce + " ")
            and fixture["model"] == "public-non-qwen" and fixture["stream"] is True,
            "positive Chat fixture lost its early unique system prefix")
    completion_fixture = positive_payload("public-non-qwen", "self-check", "completion", nonce=nonce)
    require("messages" not in completion_fixture
            and completion_fixture["prompt"].startswith(nonce + " ")
            and completion_fixture["add_special_tokens"] is True,
            "positive Completion fixture changed unexpectedly")
    ordinary = {"enabled": False, "modes": [], "selected_mode": None}
    validate_native_mode(ordinary, True)
    validate_native_mode({**ordinary, "enabled": True}, False)
    for invalid_mode in ({**ordinary, "enabled": True}, {**ordinary, "selected_mode": "render_kv"}):
        try:
            validate_native_mode(invalid_mode, True)
        except RuntimeError:
            continue
        raise RuntimeError("experimental native accepted as production functional proof")
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    original = json.dumps({"id": "original", "usage": usage,
                           "choices": [{"index": 0, "text": "onetwo", "finish_reason": "length"}]}).encode()
    streamed = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in (
        {"id": "different", "choices": [{"index": 0, "text": "one", "finish_reason": None}]},
        {"id": "different", "choices": [{"index": 0, "text": "two", "finish_reason": "length"}]},
        {"choices": [], "usage": usage},
    )) + b"data: [DONE]\n\n"
    require(completion_semantics(original, False) == completion_semantics(streamed, True),
            "semantic comparison incorrectly depends on IDs or stream chunk boundaries")
    for invalid_semantics, stream in ((streamed.replace(b"data: [DONE]", b""), True),
                                     (b'{"choices":[]}', False)):
        try:
            completion_semantics(invalid_semantics, stream)
        except RuntimeError:
            continue
        raise RuntimeError("incomplete semantic response was accepted")
    legacy_cancel = object.__new__(prior.Validation).cancel_payload()
    require("add_special_tokens" not in legacy_cancel and "return_token_ids" not in legacy_cancel,
            "legacy cancellation fixture changed")
    overridden = object.__new__(Validation)
    overridden._cancel_payload_override = {**legacy_cancel, "add_special_tokens": False, "return_token_ids": True}
    require(overridden.cancel_payload() is overridden._cancel_payload_override,
            "prepared cancellation did not reuse the exact oracle fixture")
    # Exercise the real run entry ordering, stopping before its first filesystem
    # operation. No process, network call or output directory is created.
    from types import SimpleNamespace
    class ModelProbeDone(Exception):
        pass

    def stop_before_output(*_args):
        require(prior.MODEL == "public-non-qwen", "model bound after initial process evidence")
        raise ModelProbeDone

    original_model, original_output = prior.MODEL, prior.output_directory
    try:
        prior.MODEL = "unrelated-initial-model"
        prior.output_directory = stop_before_output
        try:
            run(SimpleNamespace(model="public-non-qwen", output=None, source=None))
        except ModelProbeDone:
            pass
        else:
            raise RuntimeError("model ordering probe did not stop before filesystem access")
    finally:
        prior.MODEL, prior.output_directory = original_model, original_output
    print("PASS render bridge GPU parsers/fixtures/bootstrap/Perf-2 semantic contracts (no hardware)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--source", default=str(ROOT))
    parser.add_argument("--candidate")
    parser.add_argument("--native")
    parser.add_argument("--build-manifest")
    parser.add_argument("--render-config")
    parser.add_argument("--automatic-capabilities", action="store_true",
                        help="Require proposed Worker capabilities, generic shapes and Dense boundary evidence")
    parser.add_argument("--kv-load-guard", action="store_true",
                        help="Exercise the public cache-first load-protection switch")
    parser.add_argument("--kv-completion-token-input", action="store_true",
                        help="Exercise proven Completion token forwarding, not Chat tokens-in/out")
    parser.add_argument("--production-validation", action="store_true",
                        help="Require actual native extension built without the experimental kv-perf feature")
    parser.add_argument("--worker-vllm-root",
                        help="Explicit shared Worker vllm package directory for on-disk source hashes, not loaded-module attestation")
    parser.add_argument("--output")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--worker0", default="http://127.0.0.1:8100")
    parser.add_argument("--worker1", default="http://127.0.0.1:8101")
    for name in ("worker0-pid", "worker1-pid", "engine0-pid", "engine1-pid"):
        parser.add_argument("--" + name, type=int)
    for name in ("worker0-log", "worker1-log", "event0", "event1", "publisher0", "publisher1"):
        parser.add_argument("--" + name)
    parser.add_argument("--router-port", type=int, default=3101)
    parser.add_argument("--metrics-port", type=int, default=29101)
    parser.add_argument("--event-wait", type=float, default=2)
    parser.add_argument("--budget-seconds", type=int, default=1200)
    parser.add_argument("--request-counter", default="vllm:request_success_total")
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return 0
    if args.self_check:
        return self_check()
    for name in ("candidate", "native", "build_manifest", "render_config", "output", "worker0_pid",
                 "worker1_pid", "engine0_pid", "engine1_pid", "worker0_log", "worker1_log", "event0", "event1"):
        require(getattr(args, name) is not None, "missing --" + name.replace("_", "-"))
    require(0 <= args.event_wait <= 10, "event wait must be finite and at most ten seconds")
    require(not (args.kv_load_guard or args.kv_completion_token_input) or args.automatic_capabilities,
            "Perf-2 acceptance requires verified automatic Worker capabilities")
    require(60 <= args.budget_seconds <= 1800, "matrix budget must be 60..1800 seconds")
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("supervised matrix interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    # Reserve the existing bounded child shutdown window inside the total
    # budget. A hung request cannot defeat the between-case deadline checks.
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds - 45)
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    sys.exit(main())

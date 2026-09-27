#!/usr/bin/env python3
"""Finite product comparison and explicitly test-only KV performance ablations.

Starts/stops only its own Router child. Two exclusive DP=1 Workers must already
be running in a newly authorized environment. No packages, patches or models
are installed. There is no built-in Worker/cloud management: only a separately
authorized user-provided fresh-cohort hook may replace the specified Workers.
Hook failure/timeout does not prove its Workers were cleaned up; the saved
evidence must be handed to the user or their owning supervisor. Default trace mode
uses fresh first-block namespaces per phase, preventing cache carry-over without
clearing any Worker cache. Logical trace/order and target lengths are identical,
NOT byte-identical prompts. Actual text token lengths can differ and are recorded.
Explicit --allow-test-worker-cache-reset instead requires an
already-authorized working reset endpoint and reuses byte-identical traces.
The runner never enables development mode or falls back after a failed reset.

Product RR remains a separate baseline. Test-only A/B/C require the kv-perf
native feature and respectively measure shared forwarding without render + RR,
real render + RR, and real render + KV. C-B includes changed Worker placement,
not just scorer time. The ordinary product_kv arm needs no experimental mode.
Perf-2 C0/CL/CT/CLT all perform real render+KV, varying only load protection
and prepared Completion forwarding. --production-validation requires the
feature-off native artifact and exercises their ordinary production paths.

Timed requests use warmed per-client keep-alive connections, omit prompt-ID
echoes, and require actual usage and a complete SSE response. The full token
oracle runs after the timing/counter window, never warming its measured cache.
Headers, first SSE, first reasoning, first nonempty text and completion are
separate durations. TTFT means first nonempty text, not response headers.

Fresh-cohort hook contract (not supplied by this repository): the caller must
obtain fresh authority for replacing only these test Workers, then pass an
absolute executable with --cache-state fresh-cohort --cohort-preparation-hook
/absolute/user-approved-script --allow-cohort-preparation. No shell is used.
The script receives CMB_KV_PERF_WORKER_URLS as a JSON array of the two unchanged
loopback URLs, and CMB_KV_PERF_PHASE_DIR as the evidence directory. It must print
one JSON object, e.g. {"status":"PASS","state":"fresh_empty_cache",
"worker0_pid":101,"worker1_pid":102,"engine0_pid":103,"engine1_pid":104}.
PIDs above are schema examples, not real processes. The harness independently
checks live PID/start identity, changed event epochs, semantic compatibility,
and identical full prepared tokens; it never trusts that declaration alone.
The hook may not change models/configuration, patch packages or reuse Engines.

Public local check: python scripts/kv_capabilities_performance.py --self-check
For an already bound GPU invocation, a final single-cell experiment adds:
--arms product_rr A B C --rounds 3 --scenarios locality --concurrencies 4
--max-seconds 3600 plus the explicitly authorized fresh-cohort options above.
Always choose a smaller bound if the fresh GPU authorization has less time.
The default namespaced mode is NON-STRICT exploration; unavailable stock reset
is not enabled automatically or presented as a verified alternative.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
from functools import lru_cache
import http.client
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid

import kv_aware_cuda_validate as prior
import render_bridge_gpu_validate as acceptance


ROOT = Path(__file__).resolve().parents[1]
require, save = prior.require, prior.save
BENCHMARK_ENV = "VLLM_ROUTER_KV_PERF_MODE"
ARMS = {"product_rr": ("round_robin", None), "product_kv": ("kv_aware", None),
        "A": ("kv_aware", "shared_rr"), "B": ("kv_aware", "render_rr"),
        "C": ("kv_aware", "render_kv"),
        **{arm: ("kv_aware", "render_kv") for arm in ("C0", "CL", "CT", "CLT")}}
PERF2_FLAGS = {"C0": (False, False), "CL": (True, False),
               "CT": (False, True), "CLT": (True, True)}


def arm_configuration(arm, production_validation=False):
    policy, mode = ARMS[arm]
    if production_validation:
        require(arm == "product_rr" or arm in PERF2_FLAGS,
                "production validation supports product_rr and C0/CL/CT/CLT only")
        mode = None
    load_guard, token_input = PERF2_FLAGS.get(arm, (False, False))
    return {"policy": policy, "benchmark_mode": mode, "kv_load_guard": load_guard,
            "kv_completion_token_input": token_input,
            "production_validation": production_validation}


def percentiles(values):
    ordered = sorted(values)
    return {f"p{p}": ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)]
            if ordered else None for p in (50, 95)}


def fairness(values):
    total = sum(values)
    return total * total / (len(values) * sum(value * value for value in values)) if total else None


def sse_events(response, clock=time.perf_counter):
    """Yield whole bounded SSE records with their observed arrival timestamp."""
    data, consumed = [], 0
    while consumed < 8 * 1024 * 1024:
        line = response.readline(65537)
        require(line, "SSE ended before [DONE]")
        require(len(line) <= 65536, "SSE line exceeded bounded fixture size")
        consumed += len(line)
        require(not line.startswith(b"event: error"), "SSE error event")
        if line.startswith(b"data:"):
            data.append(line[5:].strip())
        if line in (b"\n", b"\r\n") and data:
            raw = b"\n".join(data)
            data = []
            if raw == b"[DONE]":
                yield None, clock()
                return
            event = json.loads(raw)
            require(isinstance(event, dict) and not event.get("error"), "invalid/error SSE response")
            yield event, clock()
    raise RuntimeError("SSE response exceeded finite byte budget")


def streamed_request(url, payload, request_id, timeout=60, expected_tokens=None, measure_ttft=True,
                     connection=None, require_token_ids=True, require_keepalive=False,
                     retain_token_ids=False):
    parsed = urllib.parse.urlsplit(url)
    chat = "messages" in payload
    owned_connection = connection is None
    if owned_connection:
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    correlation_id = "cmb-perf-" + uuid.uuid4().hex
    row = {"request_id": request_id, "correlation_id": correlation_id, "status": "ERROR",
           "request_sha256": hashlib.sha256(raw).hexdigest(),
           "request_hash_scope": "client ingress bytes; not Router-derived backend bytes"}
    started = time.perf_counter()
    row.update(started_monotonic=started, connection_reused=connection.sock is not None)
    first_text, first_event, first_reasoning = None, None, None
    usage, seen_ids, done, response_id, finish_reason = None, None, False, None, None
    active_socket = [connection.sock]
    expired = threading.Event()

    def expire():
        # HTTPResponse.readline and chunk framing can otherwise be kept alive
        # by a slow drip below each socket timeout. Shutdown interrupts those
        # blocking reads at the absolute deadline, including tail draining.
        expired.set()
        value = active_socket[0]
        if value is not None:
            try:
                value.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    watchdog = threading.Timer(max(0, started + timeout - time.perf_counter()), expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        require(not require_keepalive or connection.sock is not None,
                "timed keep-alive connection was closed; automatic reconnect is not allowed")
        if connection.sock is not None:
            connection.sock.settimeout(timeout)
        connection.request("POST", "/v1/chat/completions" if chat else "/v1/completions", raw,
                           {"Content-Type": "application/json", "X-Request-Id": correlation_id})
        active_socket[0] = connection.sock
        require(not expired.is_set(), "absolute request deadline expired while dispatching")
        response = connection.getresponse()
        row.update(http_status=response.status, headers_ms=(time.perf_counter() - started) * 1000)
        if response.status != 200:
            raise RuntimeError(f"generation HTTP {response.status}: {response.read(400)!r}")
        require("text/event-stream" in response.getheader("Content-Type", ""), "response is not SSE")
        remaining = timeout - (time.perf_counter() - started)
        require(remaining > 0, "request deadline expired before SSE body")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        for event, arrived in sse_events(response):
            require(arrived - started <= timeout, "absolute request deadline expired")
            if event is None:
                done = True
                break
            if first_event is None:
                first_event = arrived
            if response_id is None:
                response_id = event.get("id")
            if event.get("usage"):
                usage = event["usage"]
            if chat and event.get("prompt_token_ids") is not None:
                ids = event["prompt_token_ids"]
                require(seen_ids is None or seen_ids == ids, "generation prompt IDs changed within SSE")
                seen_ids = ids
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    require(choice["finish_reason"] in ("stop", "length"), "unsuccessful generation finish reason")
                    finish_reason = choice["finish_reason"]
                content = choice.get("delta", {}) if chat else choice
                reasoning = content.get("reasoning") or content.get("reasoning_content")
                if first_reasoning is None and isinstance(reasoning, str) and reasoning:
                    first_reasoning = arrived
                generated = content.get("content") if chat else content.get("text")
                if first_text is None and isinstance(generated, str) and generated:
                    first_text = arrived
                ids = choice.get("prompt_token_ids") if not chat else None
                if ids is not None:
                    require(seen_ids is None or seen_ids == ids, "generation prompt IDs changed within SSE")
                    seen_ids = ids
        require(done, "SSE never produced completion")
        require(finish_reason is not None, "SSE completed without a successful generation finish reason")
        # Consume the HTTP framing after [DONE], so the same socket can serve
        # another request. Never silently replay a failed/reconnected request.
        tail = response.read(65537)
        require(len(tail) <= 65536 and not tail.strip(), "unexpected response bytes after SSE completion")
        require(not expired.is_set() and time.perf_counter() - started <= timeout,
                "absolute request deadline expired")
        if measure_ttft:
            require(first_text is not None, "SSE never produced nonempty generated text")
        expected = payload["prompt"] if isinstance(payload.get("prompt"), list) else expected_tokens
        require(expected, "exact prepared trace is missing")
        if require_token_ids:
            require(seen_ids == expected, "actual Worker prompt IDs differ from exact prepared trace")
        else:
            require(not payload.get("return_token_ids") and seen_ids is None,
                    "timed response unexpectedly included diagnostic prompt IDs")
        require(usage is not None and usage.get("prompt_tokens") == len(expected),
                "actual generation usage missing or prompt token length differs")
        require(usage.get("completion_tokens") == payload["max_tokens"],
                "fixed-output trace did not generate the requested token count")
        row.update(status="PASS", ttft_ms=(first_text - started) * 1000 if first_text is not None else None,
                   first_sse_ms=(first_event - started) * 1000 if first_event is not None else None,
                   first_reasoning_ms=(first_reasoning - started) * 1000 if first_reasoning is not None else None,
                   end_to_end_ms=(time.perf_counter() - started) * 1000,
                   prompt_tokens=len(expected), output_tokens=usage["completion_tokens"],
                   response_id=response_id, finish_reason=finish_reason)
    except Exception as error:
        row.update(error=f"{type(error).__name__}: {error}", end_to_end_ms=(time.perf_counter() - started) * 1000)
    finally:
        if retain_token_ids:
            # Finite fixtures only; preserve actual arrays even on mismatch.
            # Timed requests never enable ID echo or retain this diagnostic.
            row["actual_worker_prompt_token_ids"] = seen_ids
            row["expected_prompt_token_ids"] = expected_tokens
            row["ingress_body_utf8"] = raw.decode("utf-8")
        watchdog.cancel()
        watchdog.join(timeout=1)
        if watchdog.is_alive():
            row.update(status="ERROR", error="deadline watchdog did not quiesce before connection return")
        row["deadline_exceeded"] = expired.is_set()
        if owned_connection or row["status"] != "PASS":
            connection.close()
    return row


class KeepAliveClients:
    """One exclusive warmed HTTP/1.1 connection per concurrent client slot."""

    def __init__(self, url, concurrency):
        self.url, self.connections = url, []
        self.available = queue.Queue()
        parsed = urllib.parse.urlsplit(url)
        try:
            for index in range(concurrency):
                connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
                self.connections.append(connection)
                connection.request("GET", "/health")
                response = connection.getresponse()
                response.read()
                require(response.status == 200 and connection.sock is not None,
                        "Router did not preserve a warmed keep-alive connection")
                self.available.put((index, connection))
        except BaseException:
            self.close()
            raise

    def request(self, payload, request_id, expected_tokens):
        index, connection = self.available.get(timeout=60)
        try:
            row = streamed_request(self.url, payload, request_id, expected_tokens=expected_tokens,
                                   connection=connection, require_token_ids=False, require_keepalive=True)
            row["client_slot"] = index
            return row
        finally:
            self.available.put((index, connection))

    def close(self):
        for connection in self.connections:
            connection.close()


def validate_benchmark_capabilities(info, arm, production_validation=False):
    mode = arm_configuration(arm, production_validation)["benchmark_mode"]
    require(isinstance(info, dict), "native benchmark capability response is not an object")
    require(info.get("selected_mode") == mode, "native did not confirm the requested experimental mode")
    if mode is not None:
        require(info.get("enabled") is True and info.get("loopback_only") is True
                and info.get("environment_variable") == BENCHMARK_ENV and mode in info.get("modes", []),
                "experimental arms require an explicitly enabled loopback-only kv-perf build")
    if production_validation:
        require(info.get("enabled") is False and info.get("modes") == [],
                "production validation requires a feature-off native artifact, not an idle ablation build")
    return info


def native_benchmark_capabilities(native, production_validation=False):
    handshake = getattr(native, "kv_perf_capabilities", None)
    require(not production_validation or callable(handshake),
            "production validation requires an actual native capability handshake")
    return (handshake() if callable(handshake)
            else {"enabled": False, "modes": [], "selected_mode": None})


def counters(workers):
    names = ("vllm:request_success_total", "vllm:prefix_cache_hits_total",
             "vllm:prefix_cache_queries_total", "vllm:num_requests_running")
    values = [prior.metrics(worker) for worker in workers]
    return [{name: prior.count(value, name) for name in names} for value in values]


def idle(workers, router=None, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        values = [prior.metrics(worker) for worker in workers]
        running = [prior.count(value, "vllm:num_requests_running") for value in values]
        waiting = [prior.count(value, "vllm:num_requests_waiting") for value in values]
        loads = [0, 0]
        if router:
            values = prior.json_request(router, "/workers")["workers"]
            selected = {item["url"].rstrip("/"): item for item in values if item["url"].rstrip("/") in workers}
            require(set(selected) == set(workers), "Router does not have exactly both fixture Workers")
            loads = [selected[worker]["load"] for worker in workers]
        if running == [0, 0] and waiting == [0, 0] and loads == [0, 0]:
            return
        time.sleep(0.1)
    raise RuntimeError("exclusive fixture Workers did not return to idle")


def child(path):
    config = json.loads(Path(path).read_text())
    native_path = Path(config["native"])
    require(prior.sha256(native_path) == config["native_sha256"], "native artifact changed")
    spec = importlib.util.spec_from_file_location("vllm_router_rs", native_path)
    native = importlib.util.module_from_spec(spec)
    sys.modules["vllm_router_rs"] = native
    spec.loader.exec_module(native)
    capability = native_benchmark_capabilities(native, config.get("production_validation", False))
    save(config["benchmark_identity"], validate_benchmark_capabilities(
        capability, config["arm"], config.get("production_validation", False)))
    sys.path.insert(0, str(ROOT / "py_src"))
    from vllm_router.router import Router
    from vllm_router.router_args import RouterArgs
    options = {"host": "127.0.0.1", "port": config["router_port"],
               "worker_urls": config["workers"], "policy": config["policy"],
               "worker_startup_timeout_secs": 150, "worker_startup_check_interval": 1,
               "request_timeout_secs": 60, "health_check_interval_secs": 60,
               "disable_retries": True, "log_level": config["log_level"],
               "prometheus_host": "127.0.0.1", "prometheus_port": config["metrics_port"],
               # Client concurrency remains 1/4. Avoid measuring an implicit
               # small token-bucket refill limit or an artificial server queue.
               "max_concurrent_requests": 128, "rate_limit_tokens_per_second": 100000,
               "queue_size": 0}
    if config["policy"] == "kv_aware":
        options.update(kv_input_backend="vllm", kv_render_config=config["render_config"],
                       kv_load_guard=config.get("kv_load_guard", False),
                       kv_completion_token_input=config.get("kv_completion_token_input", False),
                       kv_events_topic_filter="", kv_events_endpoints=[worker + "=" + endpoint
                       for worker, endpoint in zip(config["workers"], config["event_endpoints"])])
    router = Router.from_args(RouterArgs(**options))
    # No per-request token/trace observer: benchmark the actual production path.
    facade = router._render_facade
    if facade is not None:
        startup = facade.startup
        def record_startup():
            try:
                return startup()
            finally:
                save(config["facade_identity"], {"contract_id": facade.contract_id,
                     "effective_config": facade.effective_config, "startup_failure": facade.startup_failure,
                     "conformance": facade.conformance, "capability_cohort": facade.capability_cohort})
        facade.startup = record_startup
    router.start()


@contextmanager
def owned_router(config, directory):
    manifest = directory / "router-child.json"
    save(manifest, config)
    environment = os.environ.copy()
    for key in (BENCHMARK_ENV, "VLLM_ROUTER_KV_STAGE_TIMING", "VLLM_ROUTER_KV_STAGE_TRACE"):
        environment.pop(key, None)
    mode = arm_configuration(config["arm"], config.get("production_validation", False))["benchmark_mode"]
    if mode is not None:
        environment[BENCHMARK_ENV] = mode
    if config["stage_timing"]:
        environment["VLLM_ROUTER_KV_STAGE_TIMING"] = "1"
    if config["stage_trace"]:
        environment["VLLM_ROUTER_KV_STAGE_TRACE"] = "1"
    with (directory / "router.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--child", str(manifest)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=environment)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                require(process.poll() is None, "owned Router exited during startup")
                try:
                    if prior.request(config["router"], "/health", timeout=1)[0] == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("owned Router startup timeout")
            validate_benchmark_capabilities(json.loads(Path(config["benchmark_identity"]).read_text()),
                                            config["arm"], config.get("production_validation", False))
            yield process
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            save(directory / "router-exit.json", {"pid": process.pid, "returncode": process.returncode})


def make_trace(args, vocabulary, scenario, namespace, trace_seed):
    """Same logical trace/target lengths; actual text lengths are independently recorded."""
    rng = random.Random(trace_seed)
    has_locality = scenario in ("locality", "shared", "natural")
    prefix_count = args.groups if has_locality else args.requests
    prefixes = [rng.choices(vocabulary, k=args.prefix_tokens) for _ in range(prefix_count)]
    namespace_rng = random.Random(namespace)
    for prefix in prefixes:
        prefix[:args.block_size] = namespace_rng.choices(vocabulary, k=args.block_size)
    suffixes = [rng.choices(vocabulary, k=args.input_tokens - args.prefix_tokens) for _ in range(args.requests)]
    # Fixed bursts deliberately include an owner/locality structure, not an
    # adversarial request-order search. Each locality prefix occurs equally.
    order = ([i // (args.requests // args.groups) for i in range(args.requests)]
             if has_locality else list(range(args.requests)))
    if has_locality and (scenario == "natural" or getattr(args, "trace_order", "burst") == "interleaved"):
        order = [i % args.groups for i in range(args.requests)]
    base = {"model": args.model, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": args.output_tokens, "ignore_eos": True, "temperature": 0,
            "add_special_tokens": False, "return_token_ids": False}
    trace = [{**base, "prompt": prefixes[group] + suffixes[i]} for i, group in enumerate(order)]
    warm = []
    if has_locality and scenario != "natural":
        for group, prefix in enumerate(prefixes):
            # Distinct tail means warmup covers the intended reusable prefix,
            # not the entire timed query. Same full prompt length in both arms.
            request = {**base, "max_tokens": 1,
                       "prompt": prefix + rng.choices(vocabulary, k=args.input_tokens - args.prefix_tokens)}
            for owner in ((0, 1) if scenario == "shared" else (group % 2,)):
                warm.append((owner, dict(request)))
    if getattr(args, "prompt_format", "token_ids") == "text":
        words = ("forest river water city cloud light garden energy stone morning school "
                 "ocean paper music winter summer green blue animal people earth food").split()
        text_rng = random.Random(trace_seed)
        # Fresh content starts at byte zero, before any shared phrase. Token
        # targets are approximate in text mode; real Worker preparation below
        # records complete IDs and actual lengths outside the timed interval.
        text_prefixes = [hashlib.sha256(f"{namespace}:{group}".encode()).hexdigest() + " "
                         + " ".join(text_rng.choices(words, k=max(1, args.prefix_tokens - 40)))
                         for group in range(prefix_count)]
        tail_words = max(1, args.input_tokens - args.prefix_tokens)
        for index, request in enumerate(trace):
            request["prompt"] = (text_prefixes[order[index]] + "\n"
                                 + " ".join(text_rng.choices(words, k=tail_words))
                                 + "\nBriefly summarize the themes of these words.")
        for index, (_owner, request) in enumerate(warm):
            group = index // 2 if scenario == "shared" else index
            request["prompt"] = (text_prefixes[group] + "\n"
                                 + " ".join(text_rng.choices(words, k=tail_words))
                                 + "\nBriefly summarize the themes of these words.")
    if getattr(args, "request_kind", "completion") == "chat":
        require(getattr(args, "prompt_format", "token_ids") == "text", "Chat fixture requires real text")
        for request in trace + [row for _owner, row in warm]:
            text = request.pop("prompt")
            request.pop("add_special_tokens", None)
            request.update(messages=[{"role": "user", "content": text}],
                           chat_template_kwargs={"enable_thinking": False})
    return trace, warm, order


def prepare_trace(workers, trace):
    """Out-of-band measurement setup only, never in a timed request path."""
    expected = []
    for request in trace:
        chat = "messages" in request
        if isinstance(request.get("prompt"), list):
            expected.append(request["prompt"])
            continue
        raw = json.dumps(request, separators=(",", ":")).encode()
        per_worker = []
        for worker in workers:
            route = "/v1/chat/completions/render" if chat else "/v1/completions/render"
            status, _, body = acceptance.raw_request(worker, route, raw, timeout=20)
            require(status == 200, "text trace preparation failed at real Worker /render")
            per_worker.append(acceptance.public_render_tokens(body, chat))
        require(per_worker[0] == per_worker[1], "Workers disagree on real text trace preprocessing")
        expected.append(per_worker[0])
    return expected


def natural_warmup_trace(args, trace, order, trace_seed):
    """Router-selected interleaved burn-in; distinct tails, no owner injection."""
    rng = random.Random(trace_seed + ":natural-burn-in")
    first = {group: order.index(group) for group in range(args.groups)}
    rows = []
    for index in range(args.natural_warmup_requests):
        request = dict(trace[first[index % args.groups]])
        chat = "messages" in request
        content = request["messages"][0]["content"] if chat else request["prompt"]
        if isinstance(content, str):
            prefix = content.split("\n", 1)[0]
            tail = " ".join(rng.choices(("river", "garden", "cloud", "paper", "green"),
                                       k=max(1, args.input_tokens - args.prefix_tokens)))
            text = prefix + "\n" + tail + "\nDescribe these warmup themes."
            if chat:
                request["messages"] = [{"role": "user", "content": text}]
            else:
                request["prompt"] = text
        else:
            request["prompt"] = (request["prompt"][:args.prefix_tokens]
                                 + rng.choices(args.vocabulary, k=args.input_tokens - args.prefix_tokens))
        rows.append(request)
    return rows


def verify_context_budget(workers, model, traces):
    """Use actual served context limit, never infer it from token-length targets."""
    limits = []
    for worker in workers:
        models = prior.json_request(worker, "/v1/models")
        matches = [value for value in models.get("data", []) if value.get("id") == model]
        require(len(matches) == 1 and type(matches[0].get("max_model_len")) is int,
                "actual Worker model/context limit is unavailable")
        limits.append(matches[0]["max_model_len"])
    maximum = max(len(tokens) + request["max_tokens"]
                  for requests, expected in traces for request, tokens in zip(requests, expected))
    require(maximum <= min(limits), "actual prompt plus output exceeds a Worker context limit")
    return {"worker_context_limits": limits, "maximum_requested_sequence_tokens": maximum}


def run_natural_warmup(config, trace, expected, concurrency, evidence_path):
    """Bounded, untimed Router burn-in, preserving the cohort for measurement."""
    rows, futures = [], []
    clients = KeepAliveClients(config["router"], concurrency)
    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="owned-natural-burn-in")
    try:
        futures = [executor.submit(clients.request, request, index, expected[index])
                   for index, request in enumerate(trace)]
        for future in as_completed(futures):
            rows.append(future.result())
        require(len(rows) == len(trace) and all(row["status"] == "PASS" for row in rows),
                "natural burn-in failed; no measured window is allowed")
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        clients.close()
        save(evidence_path, {"requests": sorted(rows, key=lambda row: row["request_id"]),
             "complete": len(rows) == len(trace) and all(row["status"] == "PASS" for row in rows),
             "scope": "Untimed Router-selected interleaved burn-in; no direct owner warmup. "
                      "Same Router/Workers continue into the measured window. Not proof of production steady state."})


class LoadSampler:
    def __init__(self, router, workers, router_load_known=True, interval=0.5, router_pid=None):
        self.router, self.workers = router, workers
        self.router_load_known, self.interval, self.router_pid = router_load_known, interval, router_pid
        self.stop = threading.Event()
        self.samples, self.errors = [], []
        self.thread = threading.Thread(target=self.run, name="owned-performance-load-sampler", daemon=True)

    def run(self):
        while self.interval > 0 and not self.stop.is_set():
            try:
                status, _, body = prior.request(self.router, "/workers", timeout=2)
                require(status == 200, "load snapshot unavailable")
                values = json.loads(body)["workers"]
                by_url = {item["url"].rstrip("/"): item for item in values}
                worker_values = [metric_snapshot(worker)[1] for worker in self.workers]
                self.samples.append({"at_monotonic": time.monotonic(),
                    "router_loads": ([by_url[worker]["load"] for worker in self.workers]
                                     if self.router_load_known else None),
                    "worker_running": [optional_count(values, "vllm:num_requests_running")
                                       for values in worker_values],
                    "worker_waiting": [optional_count(values, "vllm:num_requests_waiting")
                                       for values in worker_values],
                    "router_process": process_resources(self.router_pid)})
            except Exception as error:
                self.errors.append(type(error).__name__)
            self.stop.wait(self.interval)

    def close(self):
        self.stop.set()
        self.thread.join(8)
        require(not self.thread.is_alive(), "load sampler failed to stop")


def optional_count(values, name):
    selected = [value for (metric, _labels), value in values.items() if metric == name]
    return sum(selected) if selected else None


def metric_snapshot(base):
    """Preserve raw histogram windows; do not subtract or add quantiles."""
    status, _, raw = prior.request(base, "/metrics", timeout=2)
    require(status == 200, "metrics snapshot unavailable")
    values = {}
    for line in raw.splitlines():
        match = re.fullmatch(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^ ]+)(?:\s+.*)?", line)
        if match:
            name, labels, value = match.groups()
            try:
                values[(name, labels or "")] = float(value)
            except ValueError:
                pass
    return raw, values


def process_resources(pid):
    if pid is None:
        return None
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        selected = {key: int(value.split()[0]) for key, value in
                    (line.split(":", 1) for line in status.splitlines())
                    if key in ("VmRSS", "VmHWM", "Threads")}
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        selected.update(user_cpu_ticks=int(stat[11]), system_cpu_ticks=int(stat[12]),
                        clock_ticks_per_second=os.sysconf("SC_CLK_TCK"))
        return selected
    except (OSError, ValueError, IndexError) as error:
        return {"unavailable": type(error).__name__}


def window_snapshot(config, pid):
    workers = [metric_snapshot(worker) for worker in config["workers"]]
    router_raw, _ = metric_snapshot(config["metrics"])
    names = ("vllm:request_success_total", "vllm:prefix_cache_hits_total",
             "vllm:prefix_cache_queries_total", "vllm:num_requests_running")
    return {"at_monotonic": time.monotonic(), "worker_raw_prometheus": [raw for raw, _ in workers],
            "router_raw_prometheus": router_raw, "router_process": process_resources(pid),
            "worker_counters": [{name: prior.count(values, name) for name in names}
                                for _, values in workers],
            "histogram_scope": "Aggregate window only; not correlated to individual request IDs. "
                               "Nested stage durations must not be summed or p95-subtracted."}


def forwarding_window(before, after, expected_requests, prepared, completion=True):
    """Always-on production counters prove a token arm did not silently fall back."""
    def value(raw, name, key, label):
        found = []
        for line in raw.splitlines():
            match = re.fullmatch(re.escape(name) + r"\{([^}]*)\}\s+([^ ]+)(?:\s+.*)?", line)
            if match and re.search(r'(?:^|,)' + re.escape(key) + r'="' + re.escape(label) + r'"(?:,|$)',
                                   match.group(1)):
                found.append(float(match.group(2)))
        require(found and all(math.isfinite(item) for item in found),
                f"missing/nonfinite production forwarding metric: {name} {label}")
        return sum(found)

    result = {}
    for name, key, labels in (
        ("vllm_router_kv_completion_forward_total", "mode", ("prepared", "raw")),
        ("vllm_router_kv_completion_payload_bytes_total", "kind", ("ingress", "backend")),
    ):
        for label in labels:
            delta = value(after, name, key, label) - value(before, name, key, label)
            require(delta >= 0, "production forwarding counter decreased")
            result[label] = delta
    require(result["prepared"] == (expected_requests if prepared else 0)
            and result["raw"] == (0 if prepared else expected_requests),
            "prepared forwarding eligibility/count differs from the declared arm")
    if completion:
        require(result["ingress"] > 0 and result["backend"] > 0, "forwarded byte counters are empty")
    else:
        require(expected_requests == 0 and result["ingress"] == result["backend"] == 0,
                "Chat unexpectedly entered Completion forwarding counters")
    result["scope"] = ("Per actual backend attempt; failures/retries are not hidden. "
                       "Backend byte totals are not a full backend-body capture or byte-equivalence proof.")
    return result


def token_oracle_after_timing(config, trace, expected, evidence_path):
    """Persist full real Worker IDs, including Router-forwarded probes, after timing."""
    rows = []
    try:
        observed_forwarding = ("metrics" in config and "router" in config
                               and config["arm"] in PERF2_FLAGS)
        forwarding_before = metric_snapshot(config["metrics"])[0] if observed_forwarding else None
        for index, (request, tokens) in enumerate(zip(trace, expected)):
            probe = {**request, "max_tokens": 1, "return_token_ids": True}
            for owner, worker in enumerate(config["workers"]):
                row = streamed_request(worker, probe, index, expected_tokens=tokens, measure_ttft=False,
                                       retain_token_ids=True)
                rows.append({"path": "direct_original", "worker_index": owner, **row})
                require(row["status"] == "PASS", f"post-window actual Worker token oracle failed: {row}")
            if "router" in config and config["arm"] in PERF2_FLAGS:
                row = streamed_request(config["router"], probe, index, expected_tokens=tokens,
                                       measure_ttft=False, retain_token_ids=True)
                rows.append({"path": "through_router", "worker_index": None, **row})
                require(row["status"] == "PASS", f"post-window Router token oracle failed: {row}")
        if observed_forwarding:
            completion = all("messages" not in row for row in trace)
            observation = forwarding_window(
                forwarding_before, metric_snapshot(config["metrics"])[0], len(trace) if completion else 0,
                completion and config["kv_completion_token_input"]
                and all(isinstance(row["prompt"], str) for row in trace), completion=completion)
            save(evidence_path.with_name(evidence_path.stem + "-forwarding-window.json"), observation)
        return rows
    finally:
        save(evidence_path, rows)


def arm_order(arms, round_index, pair_index):
    offset = (round_index + pair_index) % len(arms)
    ordered = arms[offset:] + arms[:offset]
    return ordered if round_index % 2 == 0 else list(reversed(ordered))


def validate_cohort_options(args):
    selected = args.cache_state == "fresh-cohort"
    require(selected == bool(args.cohort_preparation_hook) == args.allow_cohort_preparation,
            "fresh-cohort requires both an explicit user hook and current --allow-cohort-preparation authority")
    if selected:
        hook = Path(args.cohort_preparation_hook)
        require(hook.is_absolute() and hook.is_file() and os.access(hook, os.X_OK),
                "cohort hook must be an explicit existing executable file")
    require(1 <= args.cohort_timeout <= 300, "cohort hook timeout must be bounded to at most 300 seconds")


@lru_cache(maxsize=1)
def capability_contract_function():
    spec = importlib.util.spec_from_file_location("_cmb_perf_render_contract",
                                                  ROOT / "py_src/vllm_router/render_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._capability_contract


def validate_fresh_cohort(previous_descriptors, descriptors, previous_processes, processes):
    require(set(previous_descriptors) == set(descriptors), "cohort hook changed Worker URLs")
    contract = capability_contract_function()
    for worker, descriptor in descriptors.items():
        old_epoch = previous_descriptors[worker]["events"]["epoch"]
        require(descriptor["events"]["epoch"] != old_epoch,
                "cohort hook reused an old Worker event epoch")
        model = previous_descriptors[worker]["namespace"]["served_model_names"][0]
        require(contract(previous_descriptors[worker], model) == contract(descriptor, model),
                "fresh cohort changed the semantic Worker capability contract")
    previous = {(value["pid"], value["start_ticks"]) for value in previous_processes}
    require(len(processes) == 4 and all((value["pid"], value["start_ticks"]) not in previous
                                      for value in processes),
            "cohort hook reused an old HTTP or EngineCore process; empty-cache claim is unproven")


def prepare_fresh_cohort(args, common, directory):
    """Execute only a caller-provided, explicitly authorized local script.

    No built-in Worker deployment or kill/reset commands. The executable must
    print one JSON object: status=PASS, state=fresh_empty_cache, worker0_pid,
    worker1_pid, engine0_pid, engine1_pid. It receives the existing task URLs
    and output directory via CMB_KV_PERF_WORKER_URLS / CMB_KV_PERF_PHASE_DIR.
    A new Router is constructed only after actual epochs/processes are checked.
    """
    validate_cohort_options(args)
    hook = Path(args.cohort_preparation_hook).resolve()
    hook_sha = prior.sha256(hook)
    observation = {"command": [str(hook)], "script_sha256": hook_sha,
                   "status": "RUNNING", "authority": "explicit invocation flag; no inherited SSH authority",
                   "previous_descriptors": common["capabilities"],
                   "previous_processes": common["worker_processes"]}
    save(directory / "cohort-preparation.json", observation)
    try:
        environment = os.environ.copy()
        environment.update(CMB_KV_PERF_WORKER_URLS=json.dumps(common["workers"]),
                           CMB_KV_PERF_PHASE_DIR=str(directory))
        result = subprocess.run([str(hook)], capture_output=True, text=True, timeout=args.cohort_timeout,
                                check=False, cwd=ROOT, env=environment)
        observation.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        require(result.returncode == 0, "authorized cohort preparation hook failed")
        require(prior.sha256(hook) == hook_sha, "cohort hook source changed while running")
        declared = json.loads(result.stdout)
        observation["hook_result"] = declared
        require(isinstance(declared, dict) and declared.get("status") == "PASS"
                and declared.get("state") == "fresh_empty_cache", "hook did not declare a fresh empty cohort")
        for key in ("worker0_pid", "worker1_pid", "engine0_pid", "engine1_pid"):
            require(type(declared.get(key)) is int and declared[key] > 0, "invalid hook process manifest")
            setattr(args, key, declared[key])
        descriptors = acceptance.worker_capabilities(common["workers"], args.model)
        observation["observed_descriptors"] = descriptors
        fresh_config = {**common, "capabilities": descriptors}
        processes = acceptance.verify_workers(args, fresh_config)
        observation["observed_processes"] = processes
        validate_fresh_cohort(common["capabilities"], descriptors, common["worker_processes"], processes)
        observation.update(status="PASS", hook_result=declared, verified_descriptors=descriptors,
                           verified_processes=processes)
        common.update(capabilities=descriptors, worker_processes=processes)
        return observation
    except BaseException as error:
        observation.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(directory / "cohort-preparation.json", observation)


def validate_reset_response(decoded):
    # Pinned vLLM 0.29's optional dev/cache API returns {"success": bool}.
    # A 200 status alone does not prove a successful reset of held blocks.
    require(isinstance(decoded, dict) and decoded.get("success") is True
            and not decoded.get("error"), "Worker cache reset did not explicitly succeed")


def reset_exact_test_workers(args, config, directory):
    require(args.allow_test_worker_cache_reset, "cache reset is not authorized for this run")
    idle(config["workers"])
    results = []
    for worker in config["workers"]:
        status, _, response = prior.request(worker, "/reset_prefix_cache", {}, timeout=10)
        results.append({"worker": worker, "http_status": status, "response": response})
        require(status == 200, "authorized reset endpoint is unavailable; no DEV_MODE enabling/fallback is allowed")
        decoded = json.loads(response) if response.strip() else None
        validate_reset_response(decoded)
    save(directory / "authorized-cache-resets.json", results)


def phase(args, common, arm, scenario, concurrency, pair_seed, out, round_index=0):
    arm_options = arm_configuration(arm, args.production_validation)
    policy, mode = arm_options["policy"], arm_options["benchmark_mode"]
    name = f"{scenario}-c{concurrency}-r{round_index + 1}-{arm}"
    directory = out / name
    directory.mkdir()
    if args.cache_state == "fresh-cohort":
        prepare_fresh_cohort(args, common, directory)
    config = {**common, **arm_options, "arm": arm,
              "facade_identity": str(directory / "facade-identity.json"),
              "benchmark_identity": str(directory / "native-benchmark.json")}
    strict_pair = args.cache_state in ("reset", "fresh-cohort")
    namespace = pair_seed if strict_pair else pair_seed + ":" + arm
    trace, warm, order = make_trace(args, args.vocabulary, scenario, namespace, pair_seed)
    expected_trace = prepare_trace(config["workers"], trace)
    expected_warm = prepare_trace(config["workers"], [request for _owner, request in warm])
    natural_trace = natural_warmup_trace(args, trace, order, pair_seed) if scenario == "natural" else []
    expected_natural = prepare_trace(config["workers"], natural_trace)
    require(all(len(tokens) <= 8192 for tokens in expected_trace + expected_warm + expected_natural),
            "actual fixture tokens exceed the bounded full-ID evidence limit")
    context = verify_context_budget(config["workers"], args.model,
                                   [(trace, expected_trace), ([request for _owner, request in warm], expected_warm),
                                    (natural_trace, expected_natural)])
    if scenario == "cold":
        require(all(len(ids) >= args.block_size for ids in expected_trace)
                and len({tuple(ids[:args.block_size]) for ids in expected_trace}) == len(expected_trace),
                "actual prepared cold prompts do not have distinct complete first hash blocks")
    save(directory / "trace.json", {"logical_order": order, "requests": trace,
         "ingress_request_bodies_utf8": [json.dumps(request, separators=(",", ":")) for request in trace],
         "namespace": namespace, "expected_prompt_token_ids": expected_trace,
         "direct_warm": [{"worker_index": owner, "request": request,
                          "expected_prompt_token_ids": expected_warm[index]}
                         for index, (owner, request) in enumerate(warm)],
         "natural_router_burn_in": [{"request": request, "expected_prompt_token_ids": tokens}
                                    for request, tokens in zip(natural_trace, expected_natural)]})
    if args.cache_state == "reset":
        # Clear before creating the Router, so no old event/index generation is
        # carried across phases. This never enables an unavailable reset API.
        reset_exact_test_workers(args, config, directory)
    with owned_router(config, directory) as process:
        process_before = prior.process(process.pid)
        native_before = acceptance.mapped_native(process.pid, args.native)
        idle(config["workers"], config["router"])
        # Exclude subscription startup and direct warmup from measured latency.
        time.sleep(args.event_wait)
        warm_rows = []
        for index, (owner, request) in enumerate(warm):
            result = streamed_request(config["workers"][owner], {**request, "return_token_ids": True}, f"warm-{index}",
                                      expected_tokens=expected_warm[index], measure_ttft=False)
            require(result["status"] == "PASS", f"direct warm failed: {result}")
            warm_rows.append({"worker_index": owner, **result})
        save(directory / "warmup-results.json", warm_rows)
        if natural_trace:
            run_natural_warmup(config, natural_trace, expected_natural, concurrency,
                               directory / "natural-burn-in-results.json")
        idle(config["workers"], config["router"])
        time.sleep(args.event_wait)
        before_window = window_snapshot(config, process.pid)
        save(directory / "metrics-before.json", before_window)
        before = before_window["worker_counters"]
        metadata_before = acceptance.metadata_access_count(config["worker_logs"])
        render_before = acceptance.render_access_count(config["worker_logs"])
        sampler = LoadSampler(config["router"], config["workers"], arm != "product_rr",
                              args.sample_interval, process.pid)
        rows = []
        clients = KeepAliveClients(config["router"], concurrency)
        sampler.thread.start()
        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="owned-performance-client")
        futures = []
        try:
            futures = [executor.submit(clients.request, request, i, expected_trace[i])
                       for i, request in enumerate(trace)]
            for future in as_completed(futures):
                rows.append(future.result())
        finally:
            elapsed = time.perf_counter() - started
            # On deadline, do not execute queued work or block on an executor
            # context manager. At most concurrency active sockets finish within
            # their 60-second timeout; this is inside the cleanup reservation.
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            clients.close()
            sampler.close()
            recorded = {row["request_id"] for row in rows}
            unresolved = []
            for index, future in enumerate(futures):
                if index in recorded:
                    continue
                if future.cancelled():
                    unresolved.append({"request_id": index, "state": "CANCELLED_BEFORE_START"})
                elif future.done():
                    try:
                        rows.append(future.result())
                    except Exception as error:
                        rows.append({"request_id": index, "status": "ERROR",
                                     "error": f"{type(error).__name__}: {error}"})
                else:
                    unresolved.append({"request_id": index, "state": "IN_FLIGHT_AT_INTERRUPTION"})
            save(directory / "request-finalization.json", {"unresolved": unresolved,
                 "scope": "Unresolved requests are not counted as successful or proved cancelled at the Worker."})
            # Retain partial/error records even when the bounded run is interrupted.
            save(directory / "requests.json", sorted(rows, key=lambda row: row["request_id"]))
            save(directory / "load-samples.json", sampler.samples)
        idle(config["workers"], config["router"])
        after_window = window_snapshot(config, process.pid)
        save(directory / "metrics-after.json", after_window)
        after = after_window["worker_counters"]
        forwarding = (forwarding_window(before_window["router_raw_prometheus"],
                                        after_window["router_raw_prometheus"],
                                        len(rows) if args.request_kind == "completion" else 0,
                                        config["kv_completion_token_input"] and args.prompt_format == "text"
                                        and args.request_kind == "completion",
                                        completion=args.request_kind == "completion")
                      if arm in PERF2_FLAGS else None)
        deltas = [{key: current[key] - previous[key] for key in current} for previous, current in zip(before, after)]
        passed = [row for row in rows if row["status"] == "PASS"]
        require(len(passed) == args.requests and len(rows) == args.requests,
                "measured request failed or is unresolved; stop instead of continuing other arms")
        request_counts = [row["vllm:request_success_total"] for row in deltas]
        prefix_hits = [row["vllm:prefix_cache_hits_total"] for row in deltas]
        prefix_queries = [row["vllm:prefix_cache_queries_total"] for row in deltas]
        load_values = [sample["router_loads"] for sample in sampler.samples if sample["router_loads"] is not None]
        save(directory / "requests.json", sorted(rows, key=lambda row: row["request_id"]))
        save(directory / "load-samples.json", sampler.samples)
        require(sum(request_counts) == len(passed), "Worker counter deltas disagree with successful requests; Workers must be exclusive")
        require(all(0 <= hit <= query for hit, query in zip(prefix_hits, prefix_queries)), "invalid backend prefix metrics")
        require(sum(prefix_queries) == sum(map(len, expected_trace)),
                "backend cache-query tokens differ from actual prepared trace")
        if scenario == "cold":
            require(sum(prefix_hits) == 0, "fresh all-cold control unexpectedly reused cached tokens")
        require(acceptance.render_access_count(config["worker_logs"]) == render_before,
                "timed generation issued unexpected remote /render calls")
        require(acceptance.mapped_native(process.pid, args.native) == native_before, "mapped native changed during phase")
        metadata_after = acceptance.metadata_access_count(config["worker_logs"])
        render_after = acceptance.render_access_count(config["worker_logs"])
        token_oracle_after_timing(config, trace, expected_trace, directory / "post-timing-token-oracle.json")
        idle(config["workers"], config["router"])
        for before_process in common["worker_processes"]:
            expected_process = {key: value for key, value in before_process.items()
                                if key != "verified_preprocessing_arguments"}
            require(prior.process(before_process["pid"]) == expected_process,
                    "Worker identity changed during a measured phase or its post-window oracle")
        result = {"name": name, "status": "PASS" if len(passed) == args.requests else "FAIL",
                  "policy": policy, "arm": arm, "benchmark_mode": mode, "round": round_index + 1,
                  "production_validation": args.production_validation,
                  "kv_load_guard": config["kv_load_guard"],
                  "kv_completion_token_input": config["kv_completion_token_input"],
                  "completion_forwarding_window": forwarding,
                  "scenario": scenario, "concurrency": concurrency,
                  "requested": args.requests, "successful": len(passed), "errors": len(rows) - len(passed),
                  "prompt_format": args.prompt_format,
                  "request_kind": args.request_kind,
                  "chat_fixture_scope": "single user text, thinking disabled" if args.request_kind == "chat" else None,
                  "actual_prompt_tokens_min": min(map(len, expected_trace)),
                  "actual_prompt_tokens_max": max(map(len, expected_trace)),
                  "actual_prompt_tokens_mean": sum(map(len, expected_trace)) / len(expected_trace),
                  "context_budget": context,
                  "elapsed_seconds": elapsed, "requests_per_second": len(passed) / elapsed,
                  "output_tokens_per_second": sum(row["output_tokens"] for row in passed) / elapsed,
                  "ttft_ms": percentiles([row["ttft_ms"] for row in passed]),
                  "headers_ms": percentiles([row["headers_ms"] for row in passed]),
                  "first_sse_ms": percentiles([row["first_sse_ms"] for row in passed]),
                  "first_reasoning_ms": percentiles([row["first_reasoning_ms"] for row in passed
                                                     if row["first_reasoning_ms"] is not None]),
                  "end_to_end_ms": percentiles([row["end_to_end_ms"] for row in passed]),
                  "per_worker_completed_requests": request_counts,
                  "request_jain_fairness": fairness(request_counts),
                  "prefix_hit_tokens": prefix_hits, "prefix_query_tokens": prefix_queries,
                  "prefix_token_hit_ratio": sum(prefix_hits) / sum(prefix_queries) if sum(prefix_queries) else None,
                  "per_worker_prefix_hit_ratio": [hit / query if query else None for hit, query in zip(prefix_hits, prefix_queries)],
                  "per_worker_mean_sampled_router_load": [sum(row[i] for row in load_values) / len(load_values) for i in range(2)] if load_values else None,
                  "per_worker_max_sampled_router_load": [max(row[i] for row in load_values) for i in range(2)] if load_values else None,
                  "router_load_status": "UNKNOWN_UNMAINTAINED" if arm == "product_rr" else "ROUTER_INFLIGHT_NOT_GPU_QUEUE",
                  "load_samples": len(sampler.samples), "load_sampling_errors": sampler.errors,
                  "sample_interval_seconds": args.sample_interval,
                  "metadata_access_before": metadata_before,
                  "metadata_access_after": metadata_after,
                  "remote_render_access_before": render_before,
                  "remote_render_access_after": render_after,
                  "router_process": process_before, "mapped_native": native_before,
                  "physical_trace_sha256": prior.sha256(directory / "trace.json"),
                  "request_bodies_sha256": hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest(),
                  "request_bodies_hash_scope": "immutable client ingress; never a derived backend-body hash",
                  "prepared_tokens_sha256": hashlib.sha256(json.dumps(expected_trace, separators=(",", ":")).encode()).hexdigest(),
                  "client_connections": concurrency,
                  "requests_using_existing_connection": sum(row.get("connection_reused", False) for row in rows),
                  "initial_cache_state": args.cache_state,
                  "strict_byte_identical_pair": strict_pair,
                  "strict_byte_identical_scope": "client ingress and full prepared token IDs; backend body may be derived",
                  "measured_cache_preparation": ("router_selected_natural_burn_in" if scenario == "natural"
                                                 else "direct_owner_warmup" if warm else "unwarmed_cold"),
                  "natural_burn_in_requests": len(natural_trace),
                  "worker_processes": common["worker_processes"], "worker_descriptors": common["capabilities"],
                  "stage_timing": args.stage_timing, "stage_trace": args.stage_trace,
                  "scope": "Closed-loop warmed keep-alive clients; no prompt token-ID echo during timing. "
                           "Full actual Worker prompt-ID oracle runs after all timed counters. "
                           "Trace submission order is fixed; concurrent wire arrival interleavings are not serialized. "
                           "C-B includes placement effects, not just scorer time. Router load is not GPU queue. "
                           "One Timer thread per active request provides an absolute deadline; its CPU/thread cost is equal-arm client overhead. "
                           "Strict identical-body claims require explicit successful reset or a freshly verified cohort hook."}
        save(directory / "result.json", result)
    return result


def run(args):
    prior.MODEL = args.model  # Stable process snapshot semantics for non-Qwen aliases.
    out = prior.output_directory(args.output, args.source)
    report = {"status": "RUNNING", "started_at_unix": time.time(), "phases": [],
              "limitations": ["Finite synthetic product comparison; no universal improvement or production TTFT claim.",
                "Product RR and shared-forward A are distinct baselines. C-B includes changed Worker placement.",
                "Prefix-hit counters are tokens, not request hit rates. Router load is not GPU utilization or engine queue.",
                "Without explicitly authorized supported reset, namespaces/initial caches differ and comparisons are not strict.",
                "Closed-loop throughput is not production capacity; no p99 or SLO PASS is inferred.",
                "Identical hashes bind ingress and prepared IDs, not derived token-forward backend bytes.",
                "Natural is a finite Router-selected burn-in followed by a persistent-cohort window, not verified production steady state."]}
    save(out / "summary.json", report)
    try:
        identity = acceptance.source_identity(args.source, args.candidate, True)
        native_hash = prior.sha256(args.native)
        build = json.loads(Path(args.build_manifest).read_text())
        require(build.get("status") == "PASS" and build.get("candidate_sha") == args.candidate
                and build.get("native_sha256") == native_hash, "candidate/native build manifest mismatch")
        deployment = json.loads(Path(args.render_config).read_text())
        workers = [acceptance.loopback_url(args.worker0), acceptance.loopback_url(args.worker1)]
        require(deployment.get("kv_capabilities") == "worker" and not deployment.get("worker_api_key_env")
                and deployment["worker_urls"] == workers, "requires matching unauthenticated automatic-capability loopback deployment")
        descriptors = acceptance.worker_capabilities(workers, args.model)
        args.block_size = next(iter(descriptors.values()))["hash"]["block_tokens"]
        require(args.prefix_tokens % args.block_size == 0 and args.prefix_tokens >= 2 * args.block_size,
                "locality prefix must contain complete verified hash blocks")
        args.vocabulary = acceptance.valid_fixture_vocabulary(deployment["serving_args"])
        common = {"native": str(Path(args.native).resolve()), "native_sha256": native_hash,
                  "render_config": str(Path(args.render_config).resolve()), "serving_args": deployment["serving_args"],
                  "workers": workers, "router_port": args.router_port, "metrics_port": args.metrics_port,
                  "router": f"http://127.0.0.1:{args.router_port}", "automatic_capabilities": True,
                  "metrics": f"http://127.0.0.1:{args.metrics_port}", "log_level": args.log_level,
                  "stage_timing": args.stage_timing, "stage_trace": args.stage_trace,
                  "event_endpoints": [args.event0, args.event1],
                  "publisher_endpoints": [args.publisher0, args.publisher1], "capabilities": descriptors,
                  "worker_logs": [args.worker0_log, args.worker1_log]}
        for endpoint in common["event_endpoints"]:
            parsed = urllib.parse.urlsplit(endpoint)
            require(parsed.scheme == "tcp" and parsed.hostname == "127.0.0.1" and parsed.port
                    and not parsed.path and not parsed.username and not parsed.query and not parsed.fragment,
                    "finite performance fixture requires resolved loopback-only KV subscriber endpoints")
        processes = acceptance.verify_workers(args, common)
        common["worker_processes"] = processes
        prior.capture_worker_versions(workers, out / "worker-versions.json")
        report.update(identity, native_sha256=native_hash, native=str(Path(args.native).resolve()),
                      build_manifest_sha256=prior.sha256(args.build_manifest),
                      render_config_sha256=prior.sha256(args.render_config),
                      worker_processes=processes, capabilities=descriptors,
                      worker_source_provenance=acceptance.worker_source_evidence(args.worker_vllm_root),
                      installed_versions={name: importlib.metadata.version(name) for name in
                        ("vllm", "torch", "transformers", "tokenizers", "pydantic")},
                      input_token_target=args.input_tokens, prefix_token_target=args.prefix_tokens,
                      output_tokens=args.output_tokens, prompt_format=args.prompt_format,
                      request_kind=args.request_kind,
                      requests_per_phase=args.requests, groups=args.groups,
                      rounds=args.rounds, arms=args.arms, trace_order=args.trace_order,
                      production_validation=args.production_validation,
                      natural_warmup_requests=args.natural_warmup_requests,
                      stage_timing=args.stage_timing, stage_trace=args.stage_trace,
                      sample_interval_seconds=args.sample_interval, build_manifest=build,
                      comparison_classification=("IDENTICAL_INGRESS_AND_PREPARED_IDS_EXPLICIT_" + args.cache_state.upper()
                                                 if args.cache_state != "namespaced"
                                                 else "EXPLORATORY_NAMESPACE_AND_INITIAL_STATE_DIFFER"),
                      trace_mode=("ingress_byte_identical_with_explicit_" + args.cache_state if args.cache_state != "namespaced"
                                  else "same_logical_trace_fresh_first_block_namespaces"),
                      timing_definitions={"headers_ms": "client-observed response headers, not Rust upstream headers",
                          "first_sse_ms": "first complete non-DONE SSE JSON record including empty-text records",
                          "ttft_ms": "first complete SSE record with nonempty generated text",
                          "first_reasoning_ms": "first nonempty reasoning field when present, otherwise null",
                          "end_to_end_ms": "HTTP response fully drained after SSE DONE",
                          "clock": "all client durations share perf_counter; no Rust/Python clocks subtracted"},
                      gpu=prior.command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid", "--format=csv,noheader"]))
        seed = args.seed or uuid.uuid4().hex
        report["trace_seed"] = seed
        for round_index in range(args.rounds):
            for pair, (scenario, concurrency) in enumerate((s, c) for s in args.scenarios for c in args.concurrencies):
                pair_seed = f"{seed}:{scenario}:c{concurrency}:r{round_index + 1}"
                first_pair_identity = None
                for arm in arm_order(args.arms, round_index, pair):
                    result = phase(args, common, arm, scenario, concurrency, pair_seed, out, round_index)
                    if args.cache_state != "namespaced":
                        pair_identity = (result["request_bodies_sha256"], result["prepared_tokens_sha256"])
                        first_pair_identity = first_pair_identity or pair_identity
                        require(pair_identity == first_pair_identity,
                                "paired experiment changed immutable ingress bytes/order or actual prepared tokens")
                    report["phases"].append(result)
                    save(out / "summary.json", report)
        for before in common["worker_processes"]:
            expected = {key: value for key, value in before.items() if key != "verified_preprocessing_arguments"}
            require(prior.process(before["pid"]) == expected, "Worker identity changed during performance comparison")
        require(acceptance.source_identity(args.source, args.candidate, True) == identity, "candidate source changed")
        require(prior.sha256(args.native) == native_hash, "native artifact changed")
        report["status"] = "PASS" if all(item["status"] == "PASS" for item in report["phases"]) else "FAIL"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    report["finished_at_unix"] = time.time()
    save(out / "summary.json", report)
    print(f"Finite performance {report['status']}: {out / 'summary.json'}", flush=True)
    return int(report["status"] != "PASS")


def self_check():
    from io import BytesIO
    from types import SimpleNamespace
    from unittest.mock import patch
    events = list(sse_events(BytesIO(
        b'data: {"choices":[{"text":""}]}\n\n'
        b'data: {"choices":[{"text":"A"}]}\n\n'
        b'data: {"usage":{"completion_tokens":32}}\n\n'
        b'data: [DONE]\n\n'), clock=lambda: 5.0))
    require(len(events) == 4 and events[-1][0] is None, "SSE complete-record parser failed")
    require(percentiles([1, 2, 3, 4, 5]) == {"p50": 3, "p95": 5}, "percentile calculation failed")
    require(fairness([8, 8]) == 1.0 and fairness([16, 0]) == 0.5, "fairness calculation failed")
    args = SimpleNamespace(groups=4, requests=16, prefix_tokens=48, input_tokens=64,
                           block_size=16, model="public-test", output_tokens=32)
    first, warm, order = make_trace(args, list(range(256)), "locality", "phase-a", "fixed")
    second, _, order2 = make_trace(args, list(range(256)), "locality", "phase-b", "fixed")
    same, _, _ = make_trace(args, list(range(256)), "locality", "phase-a", "fixed")
    require(first == same and order == order2, "trace is not reproducible")
    require(len(warm) == 4 and all(len(row["prompt"]) == 64 for row in first), "trace dimensions changed")
    require(all(a["prompt"][16:] == b["prompt"][16:] and a["prompt"][:16] != b["prompt"][:16]
                for a, b in zip(first, second)), "phase isolation changed more than first hash block")
    cold, warm, _ = make_trace(args, list(range(256)), "cold", "phase-c", "fixed")
    require(not warm and len({tuple(row["prompt"][:16]) for row in cold}) == 16, "cold trace contains shared prefixes")
    args.prompt_format = "text"
    texts, _, _ = make_trace(args, list(range(256)), "locality", "phase-a", "fixed")
    require(all(isinstance(row["prompt"], str) for row in texts), "text trace was not preserved as raw text")
    payload = {"prompt": "public test", "max_tokens": 2}
    stream = (b'data: {"id":"test","choices":[{"text":"","prompt_token_ids":[1,2,3]}]}\n\n'
              b'data: {"id":"test","choices":[{"text":"A","finish_reason":"length"}]}\n\n'
              b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
              b'data: [DONE]\n\n')
    class FakeResponse(BytesIO):
        status = 200
        def getheader(self, _name, _default):
            return "text/event-stream"
    class FakeConnection:
        sock = None
        def request(self, *_args, **_kwargs):
            pass
        def getresponse(self):
            return FakeResponse(stream)
        def close(self):
            pass
    with patch.object(http.client, "HTTPConnection", return_value=FakeConnection()):
        row = streamed_request("http://127.0.0.1:1", payload, 0, expected_tokens=[1, 2, 3])
    require(row["status"] == "PASS" and row["ttft_ms"] >= 0 and row["prompt_tokens"] == 3,
            "end-to-end SSE reader consumed headers as output or lost token evidence")
    # A one-token warmup can decode to no text (e.g. a special token). It is not
    # timed, but still must finish with exact prompt IDs and generation usage.
    payload = {"prompt": "public test", "max_tokens": 1}
    stream = (b'data: {"id":"warm","choices":[{"text":"","prompt_token_ids":[1,2,3],"finish_reason":"length"}]}\n\n'
              b'data: {"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
              b'data: [DONE]\n\n')
    with patch.object(http.client, "HTTPConnection", return_value=FakeConnection()):
        warm_row = streamed_request("http://127.0.0.1:1", payload, "warm", expected_tokens=[1, 2, 3],
                                    measure_ttft=False)
        timed_row = streamed_request("http://127.0.0.1:1", payload, "timed", expected_tokens=[1, 2, 3])
    require(warm_row["status"] == "PASS" and warm_row["ttft_ms"] is None
            and warm_row["prompt_tokens"] == 3 and warm_row["output_tokens"] == 1,
            "valid empty decoded warmup did not retain exact token evidence")
    require(timed_row["status"] == "ERROR" and "nonempty generated text" in timed_row["error"],
            "timed empty decoded response was accepted without TTFT")
    for raw in (b'data: [DONE]', b'data: {"error":"x"}\n\n', b'event: error\n\n'):
        try:
            list(sse_events(BytesIO(raw)))
        except RuntimeError:
            continue
        raise RuntimeError("invalid/incomplete SSE accepted")
    validate_reset_response({"success": True})
    for value in ({"success": False}, {"success": 1}, {}):
        try:
            validate_reset_response(value)
        except RuntimeError:
            continue
        raise RuntimeError("unsuccessful reset response was accepted")
    print("PASS finite performance self-check (18 checks; no HTTP/GPU/processes)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--source", default=str(ROOT))
    for name in ("candidate", "native", "build-manifest", "render-config", "output", "worker-vllm-root",
                 "worker0-log", "worker1-log", "event0", "event1", "publisher0", "publisher1"):
        parser.add_argument("--" + name)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--worker0", default="http://127.0.0.1:8100")
    parser.add_argument("--worker1", default="http://127.0.0.1:8101")
    for name in ("worker0-pid", "worker1-pid", "engine0-pid", "engine1-pid"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--router-port", type=int, default=3102)
    parser.add_argument("--metrics-port", type=int, default=29102)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--prefix-tokens", type=int, default=768)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--prompt-format", choices=("text", "token_ids"), default="text",
                        help="Default real text exercises actual tokenization; token_ids is an explicitly narrower diagnostic")
    parser.add_argument("--request-kind", choices=("completion", "chat"), default="completion",
                        help="Chat is a single pure-text user message with thinking disabled; never prepared Completion forwarding")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=["product_rr", "C0", "CL", "CT", "CLT"],
                        help="C0/CL/CT/CLT are the Perf-2 2x2; existing A/B/C remain available. Include product_rr")
    parser.add_argument("--production-validation", action="store_true",
                        help="Require feature-off native artifact; C0/CL/CT/CLT use ordinary production dispatch, not an experimental mode")
    parser.add_argument("--rounds", type=int, default=3, help="Rotated/reversed arm order; never select only the best round")
    parser.add_argument("--seed", help="Recorded deterministic trace seed; omitted generates a fresh run namespace")
    parser.add_argument("--trace-order", choices=("burst", "interleaved"), default="burst")
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--scenarios", nargs="+", choices=("locality", "cold", "shared", "natural"), default=["locality", "cold"])
    parser.add_argument("--natural-warmup-requests", type=int, default=32,
                        help="Finite Router-selected interleaved burn-in before natural window; no direct owner warmup")
    parser.add_argument("--budget-seconds", "--max-seconds", dest="budget_seconds", type=int, default=900,
                        help="Default 900; explicit maximum 3600, always within the caller's fresh authorized remaining budget")
    parser.add_argument("--event-wait", type=float, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.5,
                        help="Same Worker running/waiting sampling in every arm; 0 disables sampling for overhead controls")
    parser.add_argument("--log-level", choices=("warn", "info"), default="warn")
    parser.add_argument("--stage-timing", action="store_true", help="Diagnostic low-cardinality stage histograms, off for headlines")
    parser.add_argument("--stage-trace", action="store_true", help="Bounded correlated detailed trace; requires stage timing and info logging")
    parser.add_argument("--cache-state", choices=("namespaced", "reset", "fresh-cohort"), default="namespaced",
                        help="Default is NON-STRICT exploration. Fresh authorized hook or supported reset is required for controlled identical-byte pairs")
    parser.add_argument("--cohort-preparation-hook", help="Absolute executable provided by the user; no bundled Worker deployment code")
    parser.add_argument("--allow-cohort-preparation", action="store_true",
                        help="Fresh authority to run that user hook before each arm; never inferred from old SSH access")
    parser.add_argument("--cohort-timeout", type=int, default=180)
    parser.add_argument("--allow-test-worker-cache-reset", action="store_true",
                        help="Explicit fresh authority for clearing only these two idle Workers; requires approved available endpoint, never enables DEV_MODE")
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return 0
    if args.self_check:
        return self_check()
    for key in ("candidate", "native", "build_manifest", "render_config", "output", "worker_vllm_root",
                "worker0_log", "worker1_log", "event0", "event1", "publisher0", "publisher1",
                "worker0_pid", "worker1_pid", "engine0_pid", "engine1_pid"):
        require(getattr(args, key) is not None, "missing --" + key.replace("_", "-"))
    require(sys.platform == "linux", "GPU performance run requires the authorized Linux /proc namespace")
    require(8 <= args.requests <= 256 and 2 <= args.groups <= 16 and args.requests % args.groups == 0,
            "finite trace requires 8..256 requests divisible by 2..16 groups")
    require(256 <= args.input_tokens <= 8192 and 32 <= args.prefix_tokens < args.input_tokens
            and 1 <= args.output_tokens <= 64, "finite token dimensions exceeded")
    require(8 <= args.natural_warmup_requests <= 128 and args.natural_warmup_requests % args.groups == 0,
            "natural burn-in requires 8..128 requests divisible by groups")
    require(args.concurrencies and len(args.concurrencies) == len(set(args.concurrencies))
            and set(args.concurrencies) <= {1, 4}, "concurrency must be 1 and/or 4, once each")
    require(len(args.scenarios) == len(set(args.scenarios)), "duplicate scenarios")
    require(args.request_kind != "chat" or (args.prompt_format == "text" and "cold" not in args.scenarios),
            "Chat fixture needs --prompt-format text and locality/shared/natural; its shared template precludes the zero-hit all-cold control")
    require(args.arms and len(args.arms) == len(set(args.arms)) and "product_rr" in args.arms,
            "include the true product_rr baseline exactly once; A is not its replacement")
    for arm in args.arms:
        arm_configuration(arm, args.production_validation)
    require(1 <= args.rounds <= 3, "bounded experiment supports 1..3 rounds")
    require(args.sample_interval == 0 or 0.1 <= args.sample_interval <= 5, "invalid bounded sample interval")
    require(not args.stage_trace or (args.stage_timing and args.log_level == "info"),
            "detailed trace requires --stage-timing --log-level info, and is not a headline window")
    require((args.cache_state == "reset") == args.allow_test_worker_cache_reset,
            "--cache-state reset and explicit --allow-test-worker-cache-reset must be supplied together")
    validate_cohort_options(args)
    require(120 <= args.budget_seconds <= 3600 and 0 <= args.event_wait <= 5, "finite budget exceeded")
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("finite performance deadline/interruption")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    # Request sockets timeout in <=60s; owned Router shutdown <=35s. Reserve
    # that cleanup window inside the finite wall-clock budget.
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds - 110)
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())

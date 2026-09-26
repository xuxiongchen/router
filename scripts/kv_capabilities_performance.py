#!/usr/bin/env python3
"""Finite, opt-in GPU comparison: round_robin versus kv_aware + vLLM input.

Starts/stops only its own Router child. Two exclusive DP=1 Workers must already
be running in a newly authorized environment. No packages, patches, models,
Workers or cloud resources are installed/launched/changed. Default trace mode
uses fresh first-block namespaces per phase, preventing cache carry-over without
clearing any Worker cache. Logical trace/order and target lengths are identical,
NOT byte-identical prompts. Actual text token lengths can differ and are recorded.
Explicit --allow-test-worker-cache-reset instead requires an
already-authorized working reset endpoint and reuses byte-identical traces.
The runner never enables development mode or falls back after a failed reset.

TTFT is request-send start to the first complete SSE record containing nonempty
generated text, not response headers. Router start and direct warmup are excluded
from timings. Round-robin bypasses Router token preparation while KV includes it:
this is an end-to-end product comparison, not an isolated scorer microbenchmark.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
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


def streamed_request(url, payload, request_id, timeout=60, expected_tokens=None, measure_ttft=True):
    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    row = {"request_id": request_id, "status": "ERROR", "request_sha256": hashlib.sha256(raw).hexdigest()}
    started = time.perf_counter()
    first_text, usage, seen_ids, done, response_id = None, None, None, False, None
    try:
        connection.request("POST", "/v1/completions", raw, {"Content-Type": "application/json"})
        response = connection.getresponse()
        row["http_status"] = response.status
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
            if response_id is None:
                response_id = event.get("id")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if first_text is None and isinstance(choice.get("text"), str) and choice["text"]:
                    first_text = arrived
                ids = choice.get("prompt_token_ids")
                if ids is not None:
                    require(seen_ids is None or seen_ids == ids, "generation prompt IDs changed within SSE")
                    seen_ids = ids
        require(done, "SSE never produced completion")
        if measure_ttft:
            require(first_text is not None, "SSE never produced nonempty generated text")
        expected = payload["prompt"] if isinstance(payload["prompt"], list) else expected_tokens
        require(expected and seen_ids == expected, "actual Worker prompt IDs differ from exact prepared trace")
        require(usage is not None and usage.get("prompt_tokens") == len(expected),
                "actual generation usage missing or prompt token length differs")
        require(usage.get("completion_tokens") == payload["max_tokens"],
                "fixed-output trace did not generate the requested token count")
        row.update(status="PASS", ttft_ms=(first_text - started) * 1000 if first_text is not None else None,
                   end_to_end_ms=(time.perf_counter() - started) * 1000,
                   prompt_tokens=len(seen_ids), output_tokens=usage["completion_tokens"], response_id=response_id)
    except Exception as error:
        row.update(error=f"{type(error).__name__}: {error}", end_to_end_ms=(time.perf_counter() - started) * 1000)
    finally:
        connection.close()
    return row


def counters(workers):
    names = ("vllm:request_success_total", "vllm:prefix_cache_hits_total",
             "vllm:prefix_cache_queries_total", "vllm:num_requests_running")
    values = [prior.metrics(worker) for worker in workers]
    return [{name: prior.count(value, name) for name in names} for value in values]


def idle(workers, router=None, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        running = [prior.count(prior.metrics(worker), "vllm:num_requests_running") for worker in workers]
        loads = [0, 0]
        if router:
            values = prior.json_request(router, "/workers")["workers"]
            selected = {item["url"].rstrip("/"): item for item in values if item["url"].rstrip("/") in workers}
            require(set(selected) == set(workers), "Router does not have exactly both fixture Workers")
            loads = [selected[worker]["load"] for worker in workers]
        if running == [0, 0] and loads == [0, 0]:
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
    sys.path.insert(0, str(ROOT / "py_src"))
    from vllm_router.router import Router
    from vllm_router.router_args import RouterArgs
    options = {"host": "127.0.0.1", "port": config["router_port"],
               "worker_urls": config["workers"], "policy": config["policy"],
               "worker_startup_timeout_secs": 150, "worker_startup_check_interval": 1,
               "request_timeout_secs": 60, "health_check_interval_secs": 60,
               "disable_retries": True, "log_level": "warn",
               "prometheus_host": "127.0.0.1", "prometheus_port": config["metrics_port"],
               # Client concurrency remains 1/4. Avoid measuring an implicit
               # small token-bucket refill limit or an artificial server queue.
               "max_concurrent_requests": 128, "rate_limit_tokens_per_second": 100000,
               "queue_size": 0}
    if config["policy"] == "kv_aware":
        options.update(kv_input_backend="vllm", kv_render_config=config["render_config"],
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
    with (directory / "router.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--child", str(manifest)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
    prefix_count = args.groups if scenario == "locality" else args.requests
    prefixes = [rng.choices(vocabulary, k=args.prefix_tokens) for _ in range(prefix_count)]
    namespace_rng = random.Random(namespace)
    for prefix in prefixes:
        prefix[:args.block_size] = namespace_rng.choices(vocabulary, k=args.block_size)
    suffixes = [rng.choices(vocabulary, k=args.input_tokens - args.prefix_tokens) for _ in range(args.requests)]
    # Fixed bursts deliberately include an owner/locality structure, not an
    # adversarial request-order search. Each locality prefix occurs equally.
    order = ([i // (args.requests // args.groups) for i in range(args.requests)]
             if scenario == "locality" else list(range(args.requests)))
    base = {"model": args.model, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": args.output_tokens, "ignore_eos": True, "temperature": 0,
            "add_special_tokens": False, "return_token_ids": True}
    trace = [{**base, "prompt": prefixes[group] + suffixes[i]} for i, group in enumerate(order)]
    warm = []
    if scenario == "locality":
        for group, prefix in enumerate(prefixes):
            # Distinct tail means warmup covers the intended reusable prefix,
            # not the entire timed query. Same full prompt length in both arms.
            warm.append((group % 2, {**base, "max_tokens": 1,
                         "prompt": prefix + rng.choices(vocabulary, k=args.input_tokens - args.prefix_tokens)}))
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
        for group, (_owner, request) in enumerate(warm):
            request["prompt"] = (text_prefixes[group] + "\n"
                                 + " ".join(text_rng.choices(words, k=tail_words))
                                 + "\nBriefly summarize the themes of these words.")
    return trace, warm, order


def prepare_trace(workers, trace):
    """Out-of-band measurement setup only, never in a timed request path."""
    expected = []
    for request in trace:
        if isinstance(request["prompt"], list):
            expected.append(request["prompt"])
            continue
        raw = json.dumps(request, separators=(",", ":")).encode()
        per_worker = []
        for worker in workers:
            status, _, body = acceptance.raw_request(worker, "/v1/completions/render", raw, timeout=20)
            require(status == 200, "text trace preparation failed at real Worker /render")
            per_worker.append(acceptance.public_render_tokens(body, False))
        require(per_worker[0] == per_worker[1], "Workers disagree on real text trace preprocessing")
        expected.append(per_worker[0])
    return expected


class LoadSampler:
    def __init__(self, router, workers):
        self.router, self.workers = router, workers
        self.stop = threading.Event()
        self.samples, self.errors = [], []
        self.thread = threading.Thread(target=self.run, name="owned-performance-load-sampler", daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                status, _, body = prior.request(self.router, "/workers", timeout=2)
                require(status == 200, "load snapshot unavailable")
                values = json.loads(body)["workers"]
                by_url = {item["url"].rstrip("/"): item for item in values}
                self.samples.append({"at_monotonic": time.monotonic(),
                                     "router_loads": [by_url[worker]["load"] for worker in self.workers]})
            except Exception as error:
                self.errors.append(type(error).__name__)
            self.stop.wait(0.25)

    def close(self):
        self.stop.set()
        self.thread.join(5)
        require(not self.thread.is_alive(), "load sampler failed to stop")


def validate_reset_response(decoded):
    # Pinned vLLM 0.29's optional dev/cache API returns {"success": bool}.
    # A 200 status alone does not prove a successful reset of held blocks.
    require(isinstance(decoded, dict) and decoded.get("success") is True
            and not decoded.get("error"), "Worker cache reset did not explicitly succeed")


def reset_exact_test_workers(args, config, directory):
    require(args.allow_test_worker_cache_reset, "cache reset is not authorized for this run")
    idle(config["workers"], config["router"])
    results = []
    for worker in config["workers"]:
        status, _, response = prior.request(worker, "/reset_prefix_cache", {}, timeout=10)
        results.append({"worker": worker, "http_status": status, "response": response})
        require(status == 200, "authorized reset endpoint is unavailable; no DEV_MODE enabling/fallback is allowed")
        decoded = json.loads(response) if response.strip() else None
        validate_reset_response(decoded)
    save(directory / "authorized-cache-resets.json", results)


def phase(args, common, policy, scenario, concurrency, pair_seed, out):
    name = f"{scenario}-c{concurrency}-{policy}"
    directory = out / name
    directory.mkdir()
    config = {**common, "policy": policy, "facade_identity": str(directory / "facade-identity.json")}
    namespace = pair_seed if args.allow_test_worker_cache_reset else pair_seed + ":" + policy
    trace, warm, order = make_trace(args, args.vocabulary, scenario, namespace, pair_seed)
    expected_trace = prepare_trace(config["workers"], trace)
    expected_warm = prepare_trace(config["workers"], [request for _owner, request in warm])
    if scenario == "cold":
        require(all(len(ids) >= args.block_size for ids in expected_trace)
                and len({tuple(ids[:args.block_size]) for ids in expected_trace}) == len(expected_trace),
                "actual prepared cold prompts do not have distinct complete first hash blocks")
    save(directory / "trace.json", {"logical_order": order, "requests": trace,
         "namespace": namespace, "expected_prompt_token_ids": expected_trace,
         "direct_warm": [{"worker_index": owner, "request": request,
                          "expected_prompt_token_ids": expected_warm[index]}
                         for index, (owner, request) in enumerate(warm)]})
    with owned_router(config, directory) as process:
        process_before = prior.process(process.pid)
        native_before = acceptance.mapped_native(process.pid, args.native)
        idle(config["workers"], config["router"])
        if args.allow_test_worker_cache_reset:
            reset_exact_test_workers(args, config, directory)
        # Exclude subscription startup and direct warmup from measured latency.
        time.sleep(args.event_wait)
        warm_rows = []
        for index, (owner, request) in enumerate(warm):
            result = streamed_request(config["workers"][owner], request, f"warm-{index}",
                                      expected_tokens=expected_warm[index], measure_ttft=False)
            require(result["status"] == "PASS", f"direct warm failed: {result}")
            warm_rows.append({"worker_index": owner, **result})
        save(directory / "warmup-results.json", warm_rows)
        idle(config["workers"], config["router"])
        time.sleep(args.event_wait)
        before = counters(config["workers"])
        metadata_before = acceptance.metadata_access_count(config["worker_logs"])
        render_before = acceptance.render_access_count(config["worker_logs"])
        sampler = LoadSampler(config["router"], config["workers"])
        rows = []
        sampler.thread.start()
        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="owned-performance-client")
        futures = []
        try:
            futures = [executor.submit(streamed_request, config["router"], request, i,
                                       expected_tokens=expected_trace[i])
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
            sampler.close()
        idle(config["workers"], config["router"])
        after = counters(config["workers"])
        deltas = [{key: current[key] - previous[key] for key in current} for previous, current in zip(before, after)]
        passed = [row for row in rows if row["status"] == "PASS"]
        request_counts = [row["vllm:request_success_total"] for row in deltas]
        prefix_hits = [row["vllm:prefix_cache_hits_total"] for row in deltas]
        prefix_queries = [row["vllm:prefix_cache_queries_total"] for row in deltas]
        load_values = [sample["router_loads"] for sample in sampler.samples]
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
        result = {"name": name, "status": "PASS" if len(passed) == args.requests else "FAIL",
                  "policy": policy, "scenario": scenario, "concurrency": concurrency,
                  "requested": args.requests, "successful": len(passed), "errors": len(rows) - len(passed),
                  "prompt_format": args.prompt_format,
                  "actual_prompt_tokens_min": min(map(len, expected_trace)),
                  "actual_prompt_tokens_max": max(map(len, expected_trace)),
                  "actual_prompt_tokens_mean": sum(map(len, expected_trace)) / len(expected_trace),
                  "elapsed_seconds": elapsed, "requests_per_second": len(passed) / elapsed,
                  "output_tokens_per_second": sum(row["output_tokens"] for row in passed) / elapsed,
                  "ttft_ms": percentiles([row["ttft_ms"] for row in passed]),
                  "end_to_end_ms": percentiles([row["end_to_end_ms"] for row in passed]),
                  "per_worker_completed_requests": request_counts,
                  "request_jain_fairness": fairness(request_counts),
                  "prefix_hit_tokens": prefix_hits, "prefix_query_tokens": prefix_queries,
                  "prefix_token_hit_ratio": sum(prefix_hits) / sum(prefix_queries) if sum(prefix_queries) else None,
                  "per_worker_prefix_hit_ratio": [hit / query if query else None for hit, query in zip(prefix_hits, prefix_queries)],
                  "per_worker_mean_sampled_router_load": [sum(row[i] for row in load_values) / len(load_values) for i in range(2)] if load_values else None,
                  "per_worker_max_sampled_router_load": [max(row[i] for row in load_values) for i in range(2)] if load_values else None,
                  "load_samples": len(load_values), "load_sampling_errors": sampler.errors,
                  "metadata_access_before": metadata_before,
                  "metadata_access_after": acceptance.metadata_access_count(config["worker_logs"]),
                  "remote_render_access_before": render_before,
                  "remote_render_access_after": acceptance.render_access_count(config["worker_logs"]),
                  "router_process": process_before, "mapped_native": native_before,
                  "physical_trace_sha256": prior.sha256(directory / "trace.json"),
                  "scope": "Warmup/startup excluded; fresh client HTTP connection per request included. Closed-loop client concurrency 1/4; Router burst128/refill100000 per second and queue0 avoid artificial server limiting. Return-token-IDs response overhead is equal in both policies. Text token counts may vary with fresh namespaces and are recorded."}
        save(directory / "result.json", result)
    return result


def run(args):
    prior.MODEL = args.model  # Stable process snapshot semantics for non-Qwen aliases.
    out = prior.output_directory(args.output, args.source)
    report = {"status": "RUNNING", "started_at_unix": time.time(), "phases": [],
              "limitations": ["Finite synthetic product comparison; no universal improvement or production TTFT claim.",
                "Round-robin bypasses CPU render; kv_aware includes its single render executor, queue and exact-token processing.",
                "Prefix-hit counters are tokens, not request hit rates. Sampled load is not continuous GPU utilization.",
                "No cache reset without explicit flag; default uses matched logical traces with different first-block namespaces.",
                "No release portability or steady-state long-duration throughput claim."]}
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
                  "event_endpoints": [args.event0, args.event1],
                  "publisher_endpoints": [args.publisher0, args.publisher1], "capabilities": descriptors,
                  "worker_logs": [args.worker0_log, args.worker1_log]}
        for endpoint in common["event_endpoints"]:
            parsed = urllib.parse.urlsplit(endpoint)
            require(parsed.scheme == "tcp" and parsed.hostname == "127.0.0.1" and parsed.port
                    and not parsed.path and not parsed.username and not parsed.query and not parsed.fragment,
                    "finite performance fixture requires resolved loopback-only KV subscriber endpoints")
        processes = acceptance.verify_workers(args, common)
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
                      requests_per_phase=args.requests, groups=args.groups,
                      trace_mode="byte_identical_with_explicit_cache_reset" if args.allow_test_worker_cache_reset else "same_logical_trace_fresh_first_block_namespaces",
                      gpu=prior.command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid", "--format=csv,noheader"]))
        seed = uuid.uuid4().hex
        report["trace_seed"] = seed
        for pair, (scenario, concurrency) in enumerate((s, c) for s in args.scenarios for c in args.concurrencies):
            # Counterbalance coarse order across pairs; no selection of best run.
            policies = ["round_robin", "kv_aware"] if pair % 2 == 0 else ["kv_aware", "round_robin"]
            pair_seed = f"{seed}:{scenario}:c{concurrency}"
            for policy in policies:
                result = phase(args, common, policy, scenario, concurrency, pair_seed, out)
                report["phases"].append(result)
                save(out / "summary.json", report)
        for before in processes:
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
              b'data: {"id":"test","choices":[{"text":"A"}]}\n\n'
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
    stream = (b'data: {"id":"warm","choices":[{"text":"","prompt_token_ids":[1,2,3]}]}\n\n'
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
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--prefix-tokens", type=int, default=768)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--prompt-format", choices=("text", "token_ids"), default="text",
                        help="Default real text exercises actual tokenization; token_ids is an explicitly narrower diagnostic")
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--scenarios", nargs="+", choices=("locality", "cold"), default=["locality", "cold"])
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument("--event-wait", type=float, default=2)
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
    require(8 <= args.requests <= 64 and 2 <= args.groups <= 16 and args.requests % args.groups == 0,
            "finite trace requires 8..64 requests divisible by 2..16 groups")
    require(256 <= args.input_tokens <= 2048 and 32 <= args.prefix_tokens < args.input_tokens
            and 1 <= args.output_tokens <= 64, "finite token dimensions exceeded")
    require(args.concurrencies and len(args.concurrencies) == len(set(args.concurrencies))
            and set(args.concurrencies) <= {1, 4}, "concurrency must be 1 and/or 4, once each")
    require(len(args.scenarios) == len(set(args.scenarios)), "duplicate scenarios")
    require(120 <= args.budget_seconds <= 900 and 0 <= args.event_wait <= 5, "finite budget exceeded")
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("finite performance deadline/interruption")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    # Request sockets timeout in <=60s; owned Router shutdown <=35s. Reserve
    # that cleanup window inside the finite wall-clock budget.
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds - 100)
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())

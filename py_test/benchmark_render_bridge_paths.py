"""Finite opt-in CPU three-path benchmark using the real compiled Rust extension.

Example (isolated Linux only; no packages installed by this script):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS='' \
  python py_test/benchmark_render_bridge_paths.py \
    --native-library /render-target/debug/libvllm_router_rs.so \
    --model-directory /oracle-assets --output /render-evidence/cpu-three-paths

Runs only test-owned loopback services/processes and terminates only its children.
Requires vLLM's optional dependencies, psutil and pyzmq. No GPU/model weights.
Router measurements include client HTTP + exact-input preparation + routing +
worker HTTP + mock generation response. Official render HTTP is a different,
smaller envelope and is labeled accordingly: do NOT subtract it to claim a pure
Python conversion cost, inference speedup or TTFT. Serial Python-only timings
are available separately in test_render_bridge_vllm.py.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time

from test_render_bridge_vllm import (
    GOLDEN_CHAT_TEMPLATE_SHA256,
    ROOT,
    actual_cases,
    bridge,
    official_http_renderer,
    _percentiles,
)

_ALLOCATED_PORTS = set()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).decode().split("\0")
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "files": {p: sha256(ROOT / p) for p in sorted(set(paths)) if p and (ROOT / p).is_file()},
    }


def free_port():
    # Released immediately before an owned child binds; startup detects any race.
    for _ in range(32):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in _ALLOCATED_PORTS:
            _ALLOCATED_PORTS.add(port)
            return port
    raise RuntimeError("could_not_allocate_distinct_test_port")


def http_json(port, path="/", timeout=5):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError("probe_http_status_" + str(response.status))
        if path == "/health":
            # Regular Router health is deliberately plain text, unlike telemetry.
            return {}
        return json.loads(payload) if payload else {}
    finally:
        connection.close()


def _child(manifest_path):
    """Real Router process with optional test-only observation of its facade."""
    config = json.loads(Path(manifest_path).read_text())
    library = Path(config["native_library"])
    if sha256(library) != config["native_sha256"]:
        raise RuntimeError("native_artifact_changed")
    spec = importlib.util.spec_from_file_location("vllm_router_rs", library)
    native = importlib.util.module_from_spec(spec)
    sys.modules["vllm_router_rs"] = native
    spec.loader.exec_module(native)
    sys.path.insert(0, str(ROOT / "py_src"))
    from vllm_router.router import Router
    from vllm_router.router_args import RouterArgs

    stats_lock = threading.Lock()
    stats = {"calls": 0, "statuses": Counter(), "python_render_ms": [], "token_lengths": Counter()}
    args = RouterArgs(
        worker_urls=[config["worker_url"]], host="127.0.0.1", port=config["router_port"],
        policy="kv_aware", kv_input_backend=config["backend"],
        kv_render_config=config["render_config"] if config["backend"] == "vllm" else None,
        kv_tokenizer_path=str(Path(config["model_directory"]) / "tokenizer.json"),
        kv_model=config["model"], kv_hash_algo="sha256_cbor", kv_hash_seed=0, kv_block_size=16,
        kv_events_endpoints=[config["worker_url"] + "=" + config["event_endpoint"]],
        worker_startup_timeout_secs=20, worker_startup_check_interval=1,
        request_timeout_secs=30, health_check_interval_secs=60, log_level="error",
        prometheus_host="127.0.0.1", prometheus_port=config["metrics_port"],
        max_concurrent_requests=16, queue_size=16,
    )
    router = Router.from_args(args)
    if router._render_facade is not None:
        facade = router._render_facade
        original = facade.render
        def observed_render(kind, raw):
            begin = time.perf_counter_ns()
            result = original(kind, raw)
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            with stats_lock:
                stats["calls"] += 1
                stats["statuses"][result["status"]] += 1
                stats["token_lengths"][len(result.get("token_ids", []))] += 1
                if len(stats["python_render_ms"]) < 4096:
                    stats["python_render_ms"].append(elapsed)
            return result
        facade.render = observed_render

    class Telemetry(BaseHTTPRequestHandler):
        def do_GET(self):
            with stats_lock:
                payload = json.dumps(stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    telemetry = ThreadingHTTPServer(("127.0.0.1", config["telemetry_port"]), Telemetry)
    thread = threading.Thread(target=telemetry.serve_forever, daemon=True)
    thread.start()
    try:
        router.start()
    finally:
        telemetry.shutdown()
        telemetry.server_close()
        thread.join(5)


@contextmanager
def router_process(config, output):
    manifest = output / (config["backend"] + "-child.json")
    manifest.write_text(json.dumps(config, indent=2))
    log_path = output / (config["backend"] + "-router.log")
    with log_path.open("wb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", str(manifest)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("owned_router_startup_failed_see_" + str(log_path))
                try:
                    http_json(config["router_port"], "/health", timeout=1)
                    http_json(config["telemetry_port"], timeout=1)
                    break
                except (OSError, ValueError, RuntimeError, http.client.HTTPException):
                    time.sleep(0.1)
            else:
                raise RuntimeError("owned_router_startup_deadline")
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def resources(processes):
    import psutil
    entries = []
    for process in [psutil.Process(), *[psutil.Process(p.pid) for p in processes]]:
        cpu = process.cpu_times()
        entries.append({"pid": process.pid, "cpu_seconds": cpu.user + cpu.system,
                        "rss_bytes": process.memory_info().rss})
    return entries


def measure_http(mode, port, route, raw, expected_ids, concurrency, iterations,
                 processes, telemetry_port=None):
    local = threading.local()
    connections = []
    connections_lock = threading.Lock()
    expected_digest = hashlib.sha256(raw).hexdigest()

    def one():
        if not hasattr(local, "connection"):
            local.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            with connections_lock:
                connections.append(local.connection)
        begin = time.perf_counter_ns()
        try:
            local.connection.request("POST", route, raw, {"Content-Type": "application/json"})
            response = local.connection.getresponse()
            payload = response.read()
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            if response.status != 200:
                return {"status": "http_" + str(response.status), "ms": elapsed}
            body = json.loads(payload)
            if mode == "official_http_render":
                if body.get("token_ids") != expected_ids:
                    return {"status": "token_mismatch", "ms": elapsed}
            elif (body.get("cmb_mock_generation") is not True
                  or body.get("cmb_request_sha256") != expected_digest):
                return {"status": "forwarding_mismatch", "ms": elapsed}
            return {"status": "ok", "ms": elapsed}
        except (TimeoutError, socket.timeout):
            local.connection.close()
            return {"status": "timeout", "ms": (time.perf_counter_ns() - begin) / 1e6}
        except (OSError, ValueError, http.client.HTTPException):
            local.connection.close()
            return {"status": "transport_error", "ms": (time.perf_counter_ns() - begin) / 1e6}

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Establish persistent per-client connections and warm the target
            # outside the measurement window, at the actual concurrency level.
            warm = list(pool.map(lambda _: one(), range(concurrency * 2)))
            if any(item["status"] != "ok" for item in warm):
                raise RuntimeError("benchmark_warmup_failed_" + mode + "_" + str(warm))
            before_stats = http_json(telemetry_port) if telemetry_port else None
            before_resources = resources(processes)
            started = time.perf_counter()
            samples = list(pool.map(lambda _: one(), range(iterations)))
            elapsed = time.perf_counter() - started
            after_resources = resources(processes)
            after_stats = http_json(telemetry_port) if telemetry_port else None
    finally:
        for connection in connections:
            connection.close()

    counts = Counter(item["status"] for item in samples)
    latencies = [item["ms"] for item in samples if item["status"] == "ok"]
    result = {"mode": mode, "concurrency": concurrency, "submitted_samples": iterations,
              "successful_samples": len(latencies), "statuses": dict(counts),
              "wall_seconds": elapsed, "successful_requests_per_second": len(latencies) / elapsed,
              "cpu_seconds_all_test_processes": sum(p["cpu_seconds"] for p in after_resources)
                  - sum(p["cpu_seconds"] for p in before_resources),
              "resources_before": before_resources, "resources_after": after_resources,
              "timeouts": counts["timeout"], "cancelled": 0,
              "tail_percentiles": "descriptive only: 10-20 observations give low-confidence tails",
              "percentile_method": "nearest rank",
              "warmup_requests": len(warm), "status": "PASS" if counts["ok"] == iterations else "FAIL"}
    if latencies:
        result.update(_percentiles(latencies))
    if telemetry_port:
        calls = after_stats["calls"] - before_stats["calls"]
        delta = {key: value - before_stats["statuses"].get(key, 0)
                 for key, value in after_stats["statuses"].items()}
        render_ms = after_stats["python_render_ms"][before_stats["calls"]:after_stats["calls"]]
        result["actual_facade_calls"] = calls
        result["actual_facade_outcomes"] = delta
        result["python_render_only_ms"] = _percentiles(render_ms) if render_ms else None
        result["render_measurement_scope"] = "test observer around real facade.render; excludes Rust queue/list conversion"
        if calls != iterations or delta.get("exact") != iterations:
            result["status"] = "FAIL_PROVIDER_FALLBACK_OR_RETRY"
    return result


def run(args):
    import psutil  # noqa: F401 - require prerequisites before starting resources
    import zmq

    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise RuntimeError("benchmark_output_must_be_outside_source_repository")
    output.mkdir(parents=True, exist_ok=False)
    model_dir = Path(args.model_directory).resolve()
    native_source = Path(args.native_library).resolve()
    # Bind results to an immutable copy of the native artifact actually loaded.
    artifact_digest = sha256(native_source)
    native_copy = output / "vllm_router_rs.so"
    shutil.copy2(native_source, native_copy)
    if sha256(native_copy) != artifact_digest or sha256(native_source) != artifact_digest:
        raise RuntimeError("native_artifact_changed_while_copying")
    manifest = source_manifest()
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (output / "source-manifest.json").write_bytes(manifest_bytes)
    report = {
        "source_head": manifest["head"], "source_dirty": manifest["dirty"],
        "platform": platform.platform(), "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("vllm", "torch", "transformers", "tokenizers", "pydantic", "jinja2")},
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "native_artifact": str(native_copy), "native_sha256": artifact_digest,
        "native_build_profile": "caller supplied; inspect build evidence, do not assume release",
        "model_directory": str(model_dir), "iterations_per_cell": args.iterations,
        "model_asset_sha256": {name: sha256(model_dir / name) for name in
                               ("config.json", "tokenizer_config.json", "tokenizer.json")},
        "layout": "one static DP=1 cold worker; empty test-owned event publisher",
        "not_measured": ["GPU inference", "TTFT", "live KV ownership", "Rust list conversion in isolation",
                         "native Rust token IDs via public endpoint (none exists)", "cancellation stress"],
        "native_equivalence_basis": "canonical native-profile assets and separately passing native golden tests required",
        "measurement_scopes": {
            "native_router": "HTTP client -> actual Rust native exact input -> routing -> HTTP mock generation -> response",
            "vllm_router": "HTTP client -> actual Rust/Python bridge -> routing -> HTTP mock generation -> response",
            "official_http_render": "HTTP client -> official vLLM render API -> full token JSON response",
        }, "cells": [], "run_status": "IN_PROGRESS",
    }
    model = "cpu-render-benchmark"
    argv = ["--model", str(model_dir), "--tokenizer", str(model_dir), "--served-model-name", model,
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"]
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    event_port = publisher.bind_to_random_port("tcp://127.0.0.1")
    try:
        with official_http_renderer(argv, generation_probe=True) as worker, ExitStack() as stack:
            render_config = output / "render-config.json"
            render_config.write_text(json.dumps({"serving_args": argv, "worker_urls": [worker],
                "cache_layout": {"kind": "qwen3_dense_full_attention", "block_size": 16,
                                 "hash_algorithm": "sha256_cbor", "hash_seed": 0},
                "bridge_limits": {"max_pending_jobs": 8, "max_input_bytes": 4 * 1024 * 1024,
                                  "max_tokens_per_request": 131072, "max_reserved_tokens": 8 * 131072,
                                  "queue_timeout_ms": 20000, "execution_timeout_ms": 20000}}))
            # Out-of-band, untimed validation; this local facade is closed before
            # measurements and never used as a substitute for the Router bridge.
            reference = bridge.create_facade(render_config)
            try:
                reference.startup()
                template = reference._runtime.renderer.tokenizer.get_chat_template()
                template_sha = hashlib.sha256(template.encode()).hexdigest()
                if template_sha != GOLDEN_CHAT_TEMPLATE_SHA256:
                    raise RuntimeError("three_path_common_chat_requires_canonical_native_template")
                max_length = reference.effective_config["max_model_len"]
                report["effective_config"] = reference.effective_config
                report["effective_template_sha256"] = template_sha
            finally:
                reference.close()
            configs = {}
            processes = []
            for backend in ("native", "vllm"):
                config = {"backend": backend, "native_library": str(native_copy),
                          "native_sha256": artifact_digest, "model_directory": str(model_dir),
                          "model": model, "worker_url": worker, "render_config": str(render_config),
                          "router_port": free_port(), "telemetry_port": free_port(),
                          "metrics_port": free_port(),
                          "event_endpoint": "tcp://127.0.0.1:" + str(event_port)}
                configs[backend] = config
                processes.append(stack.enter_context(router_process(config, output)))
            for length in (32, 4096, 32768, 65536):
                if length + 128 >= max_length:
                    report["cells"].append({"target_tokens": length, "status": "NOT_RUN",
                                            "reason": "model_length_limit", "max_model_len": max_length})
                    continue
                raw = json.dumps({"model": model, "messages": [{"role": "user", "content": " x" * length}],
                                  "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}},
                                 separators=(",", ":")).encode()
                expected_ids = bridge._remote_render(worker, "chat", raw, 20, None)
                for concurrency in (1, 2, 4):
                    for mode in ("native_router", "vllm_router", "official_http_render"):
                        backend = mode.removesuffix("_router")
                        port = (int(worker.rsplit(":", 1)[1]) if mode == "official_http_render"
                                else configs[backend]["router_port"])
                        result = measure_http(
                            mode, port, "/v1/chat/completions/render" if mode == "official_http_render" else "/v1/chat/completions",
                            raw, expected_ids, concurrency, args.iterations, processes,
                            configs["vllm"]["telemetry_port"] if mode == "vllm_router" else None,
                        )
                        result.update(coverage="common_native_supported", target_tokens=length,
                                      actual_tokens=len(expected_ids), request_bytes=len(raw))
                        report["cells"].append(result)
                        (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
                        print(json.dumps(result, sort_keys=True), flush=True)
                        if result["status"] != "PASS":
                            raise RuntimeError("benchmark_cell_failed_stopping_bounded_run")
            # New coverage is deliberately not timed through native fallback:
            # that would compare different work while appearing faster.
            for name, kind, request in actual_cases(model):
                if name not in ("tool_order_False", "text_parts"):
                    continue
                raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
                expected_ids = bridge._remote_render(worker, kind, raw, 20, None)
                report["cells"].append({"coverage": "newly_covered", "case": name,
                                        "mode": "native_router", "status": "NOT_RUN",
                                        "reason": "native_exact_profile_does_not_support_this_request"})
                for concurrency in (1, 2, 4):
                    for mode in ("vllm_router", "official_http_render"):
                        port = (configs["vllm"]["router_port"] if mode == "vllm_router"
                                else int(worker.rsplit(":", 1)[1]))
                        result = measure_http(
                            mode, port, "/v1/chat/completions/render" if mode == "official_http_render" else "/v1/chat/completions",
                            raw, expected_ids, concurrency, args.iterations, processes,
                            configs["vllm"]["telemetry_port"] if mode == "vllm_router" else None,
                        )
                        result.update(coverage="newly_covered", case=name,
                                      actual_tokens=len(expected_ids), request_bytes=len(raw))
                        report["cells"].append(result)
                        (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
                        print(json.dumps(result, sort_keys=True), flush=True)
                        if result["status"] != "PASS":
                            raise RuntimeError("benchmark_cell_failed_stopping_bounded_run")
        final_manifest = json.dumps(source_manifest(), sort_keys=True, separators=(",", ":")).encode()
        report["source_stable_during_run"] = final_manifest == manifest_bytes
        if not report["source_stable_during_run"]:
            raise RuntimeError("source_changed_during_benchmark")
        report["run_status"] = ("FAIL" if any(cell["status"].startswith("FAIL") for cell in report["cells"])
                                else "PASS")
    except BaseException as exc:
        report["run_status"] = "FAILED"
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        publisher.close(linger=0)
        context.term()
        (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if any(cell["status"].startswith("FAIL") for cell in report["cells"]):
        raise RuntimeError("benchmark_cells_failed_see_results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--native-library")
    parser.add_argument("--model-directory")
    parser.add_argument("--output", help="New task-owned directory; must not exist")
    parser.add_argument("--iterations", type=int, default=10, choices=range(10, 21))
    options = parser.parse_args()
    if options.child:
        _child(options.child)
    elif all((options.native_library, options.model_directory, options.output)):
        run(options)
    else:
        parser.error("--native-library, --model-directory and --output are required")

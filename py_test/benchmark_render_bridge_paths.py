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

--facade-only pre-screens independent real facade instances, not a production
Router pool. It has no Rust admission/PyO3 path and cannot predict GPU TTFT.
Its startup oracle is collected once, outside all measured windows; each
instance checks the complete token arrays against that same CPU HTTP oracle.
"""

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
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
from queue import Queue
import random
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


def _stage_metrics(port):
    if port is None:
        return {}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        text = response.read().decode()
        if response.status != 200:
            raise RuntimeError("stage_metrics_unavailable")
    finally:
        connection.close()
    prefixes = ("vllm_router_kv_stage_duration_seconds", "vllm_router_kv_stage_operations_total",
                "vllm_router_kv_bridge_usage")
    return {key: float(value) for line in text.splitlines() if line.startswith(prefixes)
            for key, value in [line.rsplit(" ", 1)]}


def _metric_window(before, after):
    delta = {key: value - before.get(key, 0) for key, value in after.items()
             if key.split("{", 1)[0].endswith(("_sum", "_count", "_bucket", "_total"))}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("stage_metric_reset_during_window")
    means = {}
    for key, value in delta.items():
        if key.split("{", 1)[0].endswith("_sum"):
            count_key = key.replace("_sum{", "_count{") if "{" in key else key[:-4] + "_count"
            if delta.get(count_key, 0) > 0:
                means[key] = value / delta[count_key]
    return {"scope": "aggregate window, not per-request; nested stages not additive; quantiles not differenced",
            "before": before, "after": after, "cumulative_delta": delta,
            "duration_mean_seconds_from_sum_count": means}


def _native_identity(native):
    path = Path(native.__file__).resolve()
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if line.split()[-1] == str(path)})
    if not mapped:
        raise RuntimeError("native_import_not_mapped")
    return {"native_module": str(path), "native_sha256": sha256(path), "native_mapped_files": mapped}


def _child(manifest_path):
    """Real Router process with optional test-only observation of its facade."""
    config = json.loads(Path(manifest_path).read_text())
    library = Path(config["native_library"])
    if sha256(library) != config["native_sha256"]:
        raise RuntimeError("native_artifact_changed")
    if config.get("installed_wheel"):
        import vllm_router_rs as native
    else:
        spec = importlib.util.spec_from_file_location("vllm_router_rs", library)
        native = importlib.util.module_from_spec(spec)
        sys.modules["vllm_router_rs"] = native
        spec.loader.exec_module(native)
        sys.path.insert(0, str(ROOT / "py_src"))
    native_identity = _native_identity(native)
    if native_identity["native_sha256"] != config["native_sha256"]:
        raise RuntimeError("loaded_native_does_not_match_supplied_artifact")
    from vllm_router.router import Router
    from vllm_router.router_args import RouterArgs

    stats_lock = threading.Lock()
    stats = {"calls": 0, "statuses": Counter(), "python_render_ms": [], "token_lengths": Counter(),
             "native_identity": native_identity}
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
        module = sys.modules[type(facade).__module__]
        stats["render_module"] = module.__file__
        stats["render_module_sha256"] = sha256(module.__file__)
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
                        "rss_bytes": process.memory_info().rss,
                        "threads": process.num_threads()})
    return entries


def _runtime_environment():
    import psutil
    def read_optional(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    return {"logical_cpus": os.cpu_count(), "affinity_cpus": psutil.Process().cpu_affinity(),
            "visible_system_memory_bytes": psutil.virtual_memory().total,
            "cgroup_v1_cpu_quota_us": read_optional("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            "cgroup_v1_cpu_period_us": read_optional("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            "cgroup_v1_memory_limit_bytes": read_optional("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "cgroup_v2_cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
            "cgroup_v2_memory_max": read_optional("/sys/fs/cgroup/memory.max")}


def measure_http(mode, port, route, raw, expected_ids, concurrency, iterations,
                 processes, telemetry_port=None, metrics_port=None):
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
                if route == "/v1/completions/render":
                    body = body[0] if isinstance(body, list) and len(body) == 1 else None
                if not isinstance(body, dict) or body.get("token_ids") != expected_ids:
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
            before_metrics = _stage_metrics(metrics_port)
            before_resources = resources(processes)
            started = time.perf_counter()
            samples = list(pool.map(lambda _: one(), range(iterations)))
            elapsed = time.perf_counter() - started
            after_resources = resources(processes)
            after_stats = http_json(telemetry_port) if telemetry_port else None
            after_metrics = _stage_metrics(metrics_port)
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
    if metrics_port is not None:
        result["stage_metric_window"] = _metric_window(before_metrics, after_metrics)
    if latencies:
        result.update(_percentiles(latencies))
    if telemetry_port:
        calls = after_stats["calls"] - before_stats["calls"]
        delta = {key: value - before_stats["statuses"].get(key, 0)
                 for key, value in after_stats["statuses"].items()}
        capture_complete = (0 <= before_stats["calls"] <= after_stats["calls"]
                            <= len(after_stats["python_render_ms"]))
        render_ms = (after_stats["python_render_ms"][before_stats["calls"]:after_stats["calls"]]
                     if capture_complete else [])
        result["actual_facade_calls"] = calls
        result["actual_facade_outcomes"] = delta
        result["native_identity"] = after_stats["native_identity"]
        result["render_module"] = after_stats.get("render_module")
        result["render_module_sha256"] = after_stats.get("render_module_sha256")
        result["python_render_only_ms"] = _percentiles(render_ms) if render_ms else None
        result["python_render_sample_status"] = ("complete" if capture_complete
                                                  else "unavailable_bounded_4096_capture_exhausted")
        result["render_measurement_scope"] = "test observer around real facade.render; excludes Rust queue/list conversion"
        if calls != iterations or delta.get("exact") != iterations:
            result["status"] = "FAIL_PROVIDER_FALLBACK_OR_RETRY"
    return result


class _FacadePool:
    """Test-only long-lived owner threads; deliberately not a product pool.

    The single bounded queue and at most four closed-loop clients keep this
    pre-screen finite. No cancellation semantics are inferred from this pool.
    Each renderer is constructed, run and closed on its own owner thread.
    """

    def __init__(self, module, config, oracle, count):
        self.module, self.config, self.oracle = module, config, oracle
        self.count = count
        self.jobs = Queue(maxsize=16)
        self.threads, self.ready, self.done = [], [], []

    def _worker(self, ready, done):
        facade = None
        try:
            facade = self.module.create_facade(self.config)
            async def verify_shared_oracle():
                # Benchmark-only startup adapter: no inference Worker claim.
                # Avoid N copies of HTTP probes while still checking every
                # instance against all full, independently collected arrays.
                for case in self.oracle:
                    result = await facade._render(case["kind"], case["raw"])
                    if result.get("status") != "exact" or result.get("token_ids") != case["ids"]:
                        raise RuntimeError("facade_shared_oracle_mismatch")
            facade._verify_workers = verify_shared_oracle
            facade.startup()
            renderer = facade._runtime.renderer
            if renderer.model_config.renderer_num_workers != 1:
                raise RuntimeError("renderer_inner_workers_must_be_one")
            ready.set_result({"contract_id": facade.contract_id, "epoch": facade.epoch,
                              "effective_config": facade.effective_config,
                              "renderer_inner_workers": renderer.model_config.renderer_num_workers,
                              "instance_ids": [id(facade), id(facade._loop), id(renderer),
                                               id(facade._runtime.capture)]})
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                future, kind, raw = job
                try:
                    begin = time.perf_counter_ns()
                    reply = facade.render(kind, raw)
                    elapsed = time.perf_counter_ns() - begin
                    future.set_result((reply, elapsed))
                except BaseException as exc:
                    future.set_exception(exc)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            done.set_exception(exc)
        finally:
            try:
                if facade is not None:
                    facade.close()
            except BaseException as exc:
                if not done.done():
                    done.set_exception(exc)
            if not done.done():
                done.set_result(None)

    def __enter__(self):
        try:
            identities = []
            for index in range(self.count):
                ready, done = Future(), Future()
                thread = threading.Thread(target=self._worker, args=(ready, done),
                                          name=f"facade-prescreen-{index}", daemon=True)
                self.ready.append(ready)
                self.done.append(done)
                self.threads.append(thread)
                thread.start()
                # Startup is outside timing; sequential construction avoids
                # confusing initializer contention with request scalability.
                identities.append(ready.result(timeout=180))
            first = identities[0]
            if any((item["contract_id"], item["epoch"]) != (first["contract_id"], first["epoch"])
                   for item in identities):
                raise RuntimeError("facade_contract_epoch_mismatch")
            if any(len({item["instance_ids"][index] for item in identities}) != self.count
                   for index in range(4)):
                raise RuntimeError("facade_mutable_runtime_was_shared")
            self.identity = {key: value for key, value in first.items() if key != "instance_ids"}
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def submit(self, kind, raw):
        future = Future()
        self.jobs.put((future, kind, raw), timeout=30)
        return future

    def __exit__(self, *_exc):
        for thread in self.threads:
            if thread.is_alive():
                self.jobs.put(None, timeout=30)
        for thread, done in zip(self.threads, self.done):
            thread.join(30)
            if thread.is_alive():
                raise RuntimeError("facade_owner_shutdown_timeout")
            done.result()


def _validate_facade_reply(reply, case, identity):
    if (reply.get("status") != "exact" or reply.get("cache_eligible") is not True
            or reply.get("token_ids") != case["ids"]
            or reply.get("contract_id") != identity["contract_id"]
            or reply.get("epoch") != identity["epoch"]):
        raise RuntimeError("facade_full_token_or_contract_mismatch")


def _mixed_facade_probe(pool, oracle):
    """Untimed concurrent shape coverage against the single shared oracle."""
    rng = random.Random(20260926)
    checked = Counter()
    def one(case):
        reply = pool.submit(case["kind"], case["raw"]).result(timeout=30)[0]
        _validate_facade_reply(reply, case, pool.identity)
        return case["name"]
    with ThreadPoolExecutor(max_workers=4) as clients:
        for _ in range(3):
            mixed = list(oracle)
            rng.shuffle(mixed)
            checked.update(clients.map(one, mixed))
    return {"executors": pool.count, "client_concurrency": 4, "rounds": 3,
            "shuffle_seed": 20260926, "full_array_and_epoch_checks": sum(checked.values()),
            "case_counts": dict(checked), "status": "PASS", "timed": False}


def _facade_cell(pool, case, concurrency, iterations, timing):
    def one():
        start = time.perf_counter_ns()
        reply, executing_ns = pool.submit(case["kind"], case["raw"]).result(timeout=30)
        elapsed_ns = time.perf_counter_ns() - start
        return reply, executing_ns, elapsed_ns

    with ThreadPoolExecutor(max_workers=concurrency) as clients:
        warmup = list(clients.map(lambda _: one(), range(concurrency * 2)))
        for reply, _, _ in warmup:
            _validate_facade_reply(reply, case, pool.identity)
        before = resources([])[0]
        start = time.perf_counter_ns()
        samples = list(clients.map(lambda _: one(), range(iterations)))
        wall_ns = time.perf_counter_ns() - start
        after = resources([])[0]
    # Full-array checks and percentile aggregation are outside timing.
    stages, counters = {}, {}
    for reply, _, _ in samples:
        _validate_facade_reply(reply, case, pool.identity)
        if timing != ("stage_durations_ns" in reply and "stage_counters" in reply):
            raise RuntimeError("facade_observer_mode_mismatch")
        for key, value in reply.get("stage_durations_ns", {}).items():
            if type(value) is not int or value < 0:
                raise RuntimeError("invalid_facade_duration")
            stages.setdefault(key, []).append(value / 1e6)
        for key, value in reply.get("stage_counters", {}).items():
            counters.setdefault(key, set()).add(value)
    return {
        "status": "PASS", "mode": "facade_only_independent_instances",
        "case": case["name"], "kind": case["kind"], "target_tokens": case["target_tokens"],
        "actual_tokens": len(case["ids"]), "request_bytes": len(case["raw"]),
        "token_sha256": hashlib.sha256(json.dumps(case["ids"]).encode()).hexdigest(),
        "full_array_checks": iterations + len(warmup), "contract_epoch_checks": iterations + len(warmup),
        "executors": pool.count, "concurrency": concurrency, "stage_timing": timing,
        "samples": iterations, "warmup_requests": len(warmup), "wall_seconds": wall_ns / 1e9,
        "requests_per_second": iterations * 1e9 / wall_ns,
        "cpu_seconds": after["cpu_seconds"] - before["cpu_seconds"],
        "resources_before": before, "resources_after": after,
        "client_ms": _percentiles([elapsed / 1e6 for _, _, elapsed in samples]),
        "facade_call_ms": _percentiles([executing / 1e6 for _, executing, _ in samples]),
        "stage_ms": {key: _percentiles(values) for key, values in stages.items()},
        "stage_counter_values": {key: sorted(values) for key, values in counters.items()},
        "samples_ns": [{"client": elapsed, "facade_call": executing,
                        "stages": reply.get("stage_durations_ns", {})}
                       for reply, executing, elapsed in samples],
    }


def _installed_wheel_identity():
    # Ordinary installed imports, not importlib.spec_from_file_location.
    import vllm_router.render_bridge as installed_bridge
    import vllm_router_rs
    return installed_bridge, {"render_module": installed_bridge.__file__,
                              "render_module_sha256": sha256(installed_bridge.__file__),
                              **_native_identity(vllm_router_rs),
                              "distribution_version": importlib.metadata.version("vllm-router"),
                              "smoke": "ordinary installed imports and live mapped native verified"}


def run_facade(args):
    import psutil  # noqa: F401 - fail before starting owned resources
    # Set before importing torch/tokenizers/vLLM. No package-wide monkeypatch.
    fixed_threads = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "OPENBLAS_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
                     "TOKENIZERS_PARALLELISM": "false"}
    os.environ.update(fixed_threads)
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise RuntimeError("benchmark_output_must_be_outside_source_repository")
    output.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (output / "source-manifest.json").write_bytes(manifest_bytes)
    active_bridge, wheel = (bridge, None)
    if args.facade_source == "installed":
        active_bridge, wheel = _installed_wheel_identity()
        if args.native_library and sha256(args.native_library) != wheel["native_sha256"]:
            raise RuntimeError("specified_native_does_not_match_installed_import")
    model_dir = Path(args.model_directory).resolve()
    report = {
        "source_head": manifest["head"], "source_dirty": manifest["dirty"],
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "runtime_environment": _runtime_environment(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("vllm", "torch", "transformers", "tokenizers", "pydantic", "jinja2")},
        "render_source": args.facade_source, "render_module": active_bridge.__file__,
        "render_module_sha256": sha256(active_bridge.__file__), "installed_wheel": wheel,
        "native_execution_in_timed_path": False,
        "fixed_inner_thread_environment": fixed_threads, "renderer_inner_workers": 1,
        "model_assets": {str(path): sha256(path) for path in active_bridge._asset_files(model_dir, model_dir)},
        "scope": "real Python facade only; test-owned owner threads/queue, not a production render pool",
        "oracle": "one shared official vLLM CPU HTTP render dataset, no inference engine/Worker proof",
        "not_measured": ["Rust admission/queue", "PyO3/GIL attach", "GPU TTFT", "live Worker cache", "cancellation"],
        "resource_scope": "current process snapshots (not peak RSS); Python caches may persist across cells",
        "intervals": "client includes test queue and Future wakeup; facade excludes them; stages nested, not additive",
        "percentiles": "nearest rank; tails are descriptive at these bounded sample counts",
        "cells": [], "mixed_conformance": [], "run_status": "IN_PROGRESS",
    }
    model = "cpu-facade-benchmark"
    argv = ["--model", str(model_dir), "--tokenizer", str(model_dir), "--served-model-name", model,
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"]
    cases, oracle = [], []
    try:
        with official_http_renderer(argv) as worker:
            for name, kind, request in active_bridge._conformance_requests(model, argv):
                raw = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode()
                oracle.append({"name": name, "kind": kind, "raw": raw,
                               "ids": active_bridge._remote_render(worker, kind, raw, 20, None)})
            for name, kind, request in actual_cases(model):
                raw = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode()
                oracle.append({"name": "actual_" + name, "kind": kind, "raw": raw,
                               "ids": active_bridge._remote_render(worker, kind, raw, 20, None)})
            for kind in ("completion", "chat"):
                for length in (32, 1024, 4096):
                    request = {"model": model, "max_tokens": 1}
                    if kind == "completion":
                        request["prompt"] = " x" * length
                    else:
                        request.update(messages=[{"role": "user", "content": " x" * length}],
                                       chat_template_kwargs={"enable_thinking": False})
                    raw = json.dumps(request, separators=(",", ":")).encode()
                    case = {"name": f"{kind}_{length}", "kind": kind, "raw": raw,
                            "target_tokens": length,
                            "ids": active_bridge._remote_render(worker, kind, raw, 20, None)}
                    cases.append(case)
                    oracle.append(case)
        report["shared_oracle_cases"] = len(oracle)
        config = output / "facade-config.json"
        config.write_text(json.dumps({"serving_args": argv, "worker_urls": [worker],
                          "cache_layout": {"kind": "qwen3_dense_full_attention", "block_size": 16,
                                           "hash_algorithm": "sha256_cbor", "hash_seed": 0}}))
        modes = (False, True) if args.stage_timing in (None, "both") else (args.stage_timing == "on",)
        baseline_identity = None
        for count in args.executors:
            for timing in modes:
                os.environ["VLLM_ROUTER_KV_STAGE_TIMING"] = "1" if timing else "0"
                with _FacadePool(active_bridge, config, oracle, count) as pool:
                    identity = (pool.identity["contract_id"], pool.identity["epoch"])
                    if baseline_identity is not None and identity != baseline_identity:
                        raise RuntimeError("pool_or_observer_changed_contract")
                    baseline_identity = identity
                    report["identity"] = pool.identity
                    # Mixed Unicode, tool/history/thinking and Completion/Chat
                    # inputs catch contamination before homogeneous timing.
                    mixed = _mixed_facade_probe(pool, oracle)
                    report["mixed_conformance"].append({**mixed, "stage_timing": timing})
                    for case in cases:
                        for concurrency in args.concurrency:
                            cell = _facade_cell(pool, case, concurrency, args.iterations, timing)
                            report["cells"].append(cell)
                            (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
                            print(json.dumps({key: value for key, value in cell.items()
                                              if key != "samples_ns"}, sort_keys=True), flush=True)
        report["source_stable_during_run"] = source_manifest() == manifest
        if not report["source_stable_during_run"]:
            raise RuntimeError("source_changed_during_benchmark")
        if sha256(active_bridge.__file__) != report["render_module_sha256"]:
            raise RuntimeError("render_module_changed_during_benchmark")
        report["run_status"] = "PASS"
    except BaseException as exc:
        report.update(run_status="FAILED", failure_type=type(exc).__name__)
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def run(args):
    import psutil  # noqa: F401 - require prerequisites before starting resources
    import zmq

    if args.stage_timing == "both":
        raise RuntimeError("three_path_on_off_requires_separate_runs_and_router_processes")
    os.environ["VLLM_ROUTER_KV_STAGE_TIMING"] = "1" if args.stage_timing == "on" else "0"

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
        "runtime_environment": _runtime_environment(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("vllm", "torch", "transformers", "tokenizers", "pydantic", "jinja2")},
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "native_artifact": str(native_copy), "native_sha256": artifact_digest,
        "native_build_profile": "caller supplied; inspect build evidence, do not assume release",
        "model_directory": str(model_dir), "iterations_per_cell": args.iterations,
        "installed_wheel_import": args.installed_wheel,
        "stage_timing": args.stage_timing == "on",
        "comparison_scope": "existing three paths, not controlled A/B/C forwarding ablation",
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
                          "installed_wheel": args.installed_wheel,
                          "native_sha256": artifact_digest, "model_directory": str(model_dir),
                          "model": model, "worker_url": worker, "render_config": str(render_config),
                          "router_port": free_port(), "telemetry_port": free_port(),
                          "metrics_port": free_port(),
                          "event_endpoint": "tcp://127.0.0.1:" + str(event_port)}
                configs[backend] = config
                processes.append(stack.enter_context(router_process(config, output)))
            common_cases = [(kind, length) for kind in (("completion", "chat") if args.cpu_short_matrix else ("chat",))
                            for length in ((32, 1024, 4096) if args.cpu_short_matrix else (32, 4096, 32768, 65536))]
            for kind, length in common_cases:
                if length + 128 >= max_length:
                    report["cells"].append({"target_tokens": length, "status": "NOT_RUN",
                                            "reason": "model_length_limit", "max_model_len": max_length})
                    continue
                request = {"model": model, "max_tokens": 1}
                if kind == "completion":
                    request["prompt"] = " x" * length
                else:
                    request.update(messages=[{"role": "user", "content": " x" * length}],
                                   chat_template_kwargs={"enable_thinking": False})
                raw = json.dumps(request, separators=(",", ":")).encode()
                expected_ids = bridge._remote_render(worker, kind, raw, 20, None)
                for concurrency in (1, 2, 4):
                    for mode in ("native_router", "vllm_router", "official_http_render"):
                        backend = mode.removesuffix("_router")
                        port = (int(worker.rsplit(":", 1)[1]) if mode == "official_http_render"
                                else configs[backend]["router_port"])
                        result = measure_http(
                            mode, port, ("/v1/chat/completions" if kind == "chat" else "/v1/completions")
                            + ("/render" if mode == "official_http_render" else ""),
                            raw, expected_ids, concurrency, args.iterations, processes,
                            configs["vllm"]["telemetry_port"] if mode == "vllm_router" else None,
                            configs[backend]["metrics_port"] if mode != "official_http_render" else None,
                        )
                        result.update(coverage="common_native_supported", kind=kind, target_tokens=length,
                                      actual_tokens=len(expected_ids), request_bytes=len(raw))
                        report["cells"].append(result)
                        (output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
                        print(json.dumps(result, sort_keys=True), flush=True)
                        if result["status"] != "PASS":
                            raise RuntimeError("benchmark_cell_failed_stopping_bounded_run")
            # New coverage is deliberately not timed through native fallback:
            # that would compare different work while appearing faster.
            for name, kind, request in actual_cases(model):
                if args.cpu_short_matrix:
                    break
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
                            configs["vllm"]["metrics_port"] if mode == "vllm_router" else None,
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
    parser.add_argument("--iterations", type=int, default=10, choices=range(10, 201))
    parser.add_argument("--facade-only", action="store_true", help="Real independent-facade CPU pre-screen only")
    parser.add_argument("--facade-source", choices=("installed", "checkout"), default="installed")
    parser.add_argument("--stage-timing", choices=("off", "on", "both"))
    parser.add_argument("--cpu-short-matrix", action="store_true", help="Completion/Chat 32/1024/4096 tokens")
    parser.add_argument("--installed-wheel", action="store_true", help="Router children use ordinary installed wheel imports")
    parser.add_argument("--executors", type=int, nargs="+", choices=(1, 2, 4), default=(1, 2, 4))
    parser.add_argument("--concurrency", type=int, nargs="+", choices=(1, 2, 4), default=(1, 2, 4))
    options = parser.parse_args()
    if options.child:
        _child(options.child)
    elif options.facade_only and all((options.model_directory, options.output)):
        run_facade(options)
    elif all((options.native_library, options.model_directory, options.output)):
        run(options)
    else:
        parser.error("--native-library, --model-directory and --output are required")

"""Opt-in real-extension/GIL probe with a deliberately synthetic facade.

Run directly, not through pytest collection. This does not install vLLM, load
model assets, emit KV events, or claim model/render equivalence. All listeners
and child processes are owned by this invocation and bind only to loopback.
"""

import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(event_file, event, **fields):
    entry = dict(event=event, monotonic=time.monotonic(), pid=os.getpid(),
                 thread=threading.get_ident(), native_thread=threading.get_native_id())
    entry.update(fields)
    data = (json.dumps(entry, sort_keys=True) + "\n").encode()
    descriptor = os.open(event_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def events(path):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    # A reader may see the final append before it is complete.
    return [json.loads(line) for line in lines if line.endswith("}")]


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("bounded probe wait expired")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def child(config_path):
    config = json.loads(Path(config_path).read_text())
    event_path = config["events"]
    spec = importlib.util.spec_from_file_location("vllm_router_rs", config["extension"])
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    record(event_path, "main", extension_sha256=sha256(config["extension"]))
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(0.02):
            record(event_path, "heartbeat")

    heartbeat_thread = threading.Thread(target=heartbeat, name="probe-heartbeat")

    class Facade:
        active = 0
        closed = False
        callback_native_id = None

        def startup(self):
            self.callback_native_id = threading.get_native_id()
            record(event_path, "startup")

        def render(self, kind, raw):
            self.active += 1
            record(event_path, "render_start", kind=kind, raw_hex=raw.hex(), active=self.active,
                   lifetime_guard_alive=any(thread.name == "cmb-render-lifetime" and thread.is_alive()
                                           and not thread.daemon for thread in threading.enumerate()))
            request_object = json.loads(raw)
            delay = request_object.get("probe_delay", 0)
            if delay:
                time.sleep(delay)
            self.active -= 1
            record(event_path, "render_end", active=self.active)
            return dict(status="exact", token_ids=list(range(1, 33)),
                        contract_id="synthetic-native-probe", epoch=1, cache_eligible=True,
                        # Synthetic transport proof only. Real vLLM equivalence
                        # is independently covered by the optional CPU/GPU tests.
                        completion_token_input_eligible=(kind == "completion"
                            and type(request_object.get("prompt")) is str
                            and request_object.get("add_special_tokens") is False
                            and "echo" not in request_object))

        def close(self):
            record(event_path, "close", active=self.active)
            self.closed = True

    facade = Facade()
    router = extension.Router(
        worker_urls=config["workers"], policy=extension.PolicyType.KvAware,
        host="127.0.0.1", port=config["port"],
        kv_tokenizer_path=str(config_path),  # unused: synthetic bridge replaces native tokenizer
        kv_model="synthetic-probe", kv_hash_algo="sha256_cbor",
        kv_events_endpoints=config["endpoints"], worker_startup_timeout_secs=10,
        worker_startup_check_interval=1, log_level="debug" if config.get("kv_exact_history") else "warn", disable_retries=True,
        health_check_interval_secs=60, prometheus_host="127.0.0.1",
        prometheus_port=config["metrics_port"],
        **({"kv_fallback_policy": "cache_aware", "kv_fallback_history_ttl_secs": 60,
            "max_tree_size": 4096} if config.get("kv_exact_history") else {}),
        **({"kv_load_guard": config["kv_load_guard"],
            "kv_completion_token_input": config["kv_completion_token_input"]}
           if config.get("kv_load_guard") or config.get("kv_completion_token_input") else {}),
    )
    heartbeat_thread.start()
    try:
        router.start(
            render_facade=facade, render_contract_id="synthetic-native-probe",
            render_contract_epoch=1,
            render_limits=dict(max_pending_jobs=1, max_input_bytes=65536,
                               max_tokens_per_request=64, max_reserved_tokens=64,
                               queue_timeout_ms=100, execution_timeout_ms=config["deadline_ms"]),
            kv_capabilities_json=(json.dumps(config["cohort"]) if config.get("cohort") else None),
        )
        wait_until(lambda: not Path(f"/proc/self/task/{facade.callback_native_id}").exists()
                   and not any(thread.name == "cmb-render-lifetime" and thread.is_alive()
                               for thread in threading.enumerate()), 2)
        record(event_path, "start_returned", closed=facade.closed, active=facade.active,
               callback_thread_absent=True, lifetime_guard_absent=True)
    finally:
        stopped.set()
        heartbeat_thread.join(2)
        record(event_path, "heartbeat_stopped", alive=heartbeat_thread.is_alive())


class Workers:
    def __init__(self, event_path, *, capabilities=False, response_provider=None):
        self.servers = []
        self.threads = []
        self.urls = []
        self.endpoints = []
        self.descriptors = {}
        self.capabilities = capabilities
        self.response_provider = response_provider
        for index in range(2):
            self._add(index, event_path)

    def _add(self, index, event_path):
        response_provider = self.response_provider
        event_endpoint = f"tcp://127.0.0.1:{free_port()}"
        descriptor = None
        if self.capabilities:
            # Reuse the reviewed PUBLIC synthetic schema fixture. Serving it
            # exercises actual Rust control-plane validation, not a real vLLM
            # Worker or cache publisher. No fake cache-hit events are emitted.
            fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/kv_capabilities/descriptor.json"
            descriptor = json.loads(fixture.read_text())
            descriptor["namespace"]["served_model_names"] = ["synthetic-probe"]
            epoch = f"{index + 1:032x}"
            descriptor["events"].update(epoch=epoch, topic=f"synthetic.{epoch}",
                configured_endpoint=event_endpoint, resolved_endpoint=event_endpoint)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/v1/kv-cache/capabilities" and descriptor is not None:
                    payload = json.dumps(descriptor).encode()
                    record(event_path, "synthetic_capability_read", worker=index)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                value = json.loads(raw)
                record(event_path, "worker_request", worker=index, path=self.path,
                       raw_hex=raw.hex(), headers=dict(self.headers))
                if response_provider is not None:
                    status, content_type, payload = response_provider(value)
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    try:
                        # Force awkward HTTP byte boundaries, including UTF-8.
                        for start in range(0, len(payload), 7):
                            self.wfile.write(payload[start:start + 7])
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if value.get("stream"):
                    choice = ({"delta": {"content": "ok"}, "finish_reason": None}
                              if self.path == "/v1/chat/completions"
                              else {"text": "ok", "finish_reason": "stop"})
                    payload = ("data: " + json.dumps({"id": "synthetic", "choices": [choice]})
                               + "\n\ndata: [DONE]\n\n").encode()
                    content_type = "text/event-stream"
                else:
                    payload = json.dumps(dict(id="synthetic", worker=index, raw_hex=raw.hex())).encode()
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, name=f"probe-worker-{index}")
        thread.start()
        self.servers.append(server)
        self.threads.append(thread)
        url = f"http://127.0.0.1:{server.server_port}"
        self.urls.append(url)
        self.endpoints.append(f"{url}={event_endpoint}")
        if descriptor is not None:
            self.descriptors[url] = descriptor

    def close(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(2)
            require(not thread.is_alive(), "owned mock worker thread did not close")


def request(port, raw, path="/v1/completions", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("POST", path, body=raw,
                           headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


def ready(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except OSError:
        return False
    finally:
        connection.close()


def completion_metrics(port):
    """Only this probe's four public transport counters; absent is not zero."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        body = response.read().decode()
        require(response.status == 200, "metrics endpoint failed")
    finally:
        connection.close()
    values = {}
    for line in body.splitlines():
        match = re.fullmatch(r'vllm_router_kv_completion_forward_total\{mode="(raw|prepared)"\} (\S+)', line)
        if match:
            values[match.group(1)] = float(match.group(2))
        match = re.fullmatch(r'vllm_router_kv_completion_payload_bytes_total\{kind="(ingress|backend)"\} (\S+)', line)
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def require_complete_metrics(port):
    try:
        values = completion_metrics(port)
    except OSError:
        return None
    return values if set(values) == {"raw", "prepared", "ingress", "backend"} else None


def run_case(name, extension, output, *, kv_load_guard=False, kv_completion_token_input=False, kv_exact_history=False):
    case_dir = output / name
    case_dir.mkdir()
    event_path = case_dir / "events.jsonl"
    workers = Workers(event_path, capabilities=kv_completion_token_input)
    child_process = None
    try:
        config = dict(extension=str(extension), events=str(event_path), workers=workers.urls,
                      port=free_port(), deadline_ms=100 if name == "deadline" else 2000,
                      metrics_port=free_port(),
                      kv_load_guard=kv_load_guard, kv_completion_token_input=kv_completion_token_input,
                      kv_exact_history=kv_exact_history,
                      cohort=({"workers": workers.descriptors, "api_key_env": None}
                              if kv_completion_token_input else None),
                      endpoints=workers.endpoints)
        config_path = case_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        with (case_dir / "router.log").open("wb") as log:
            child_process = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--child", str(config_path)],
                stdout=log, stderr=subprocess.STDOUT,
            )
            def is_ready():
                require(child_process.poll() is None, "Router child exited before readiness; see router.log")
                return ready(config["port"])

            wait_until(is_ready)
            metrics_before = wait_until(lambda: require_complete_metrics(config["metrics_port"]))
            require(all(value == 0 for value in metrics_before.values()),
                    "new Completion counter series were not present at zero before requests")
            started = time.monotonic()
            if name == "json_sse":
                payloads = [
                    ("/v1/completions", b'{ "prompt" : "A\\u4e2d", "model":"synthetic-probe", "unknown_extra":{"a":1} }', {}, False),
                    ("/v1/chat/completions", b'{ "messages":[{"role":"user","content":"hi"}], "model":"synthetic-probe", "stream":true, "chat_template_kwargs":{"x":7} }', {}, False),
                ]
                if kv_completion_token_input:
                    eligible = b'{ "prompt" : "A\\u4e2d", "model":"synthetic-probe", "add_special_tokens":false, "temperature":1.000e-1, "stop":null }'
                    payloads.extend([
                        ("/v1/completions", eligible, {"Authorization": "Bearer synthetic-only", "X-Request-Id": "same-request"}, True),
                        ("/v1/completions", eligible[:-2] + b', "stream":true, "stream_options":{"include_usage":true} }', {}, True),
                        ("/v1/completions", eligible[:-2] + b', "echo":false }', {}, False),
                        ("/v1/completions", eligible, {"Content-Digest": "sha-256=:synthetic-old:"}, False),
                        ("/v1/completions", eligible, {"Content-Encoding": "identity"}, True),
                        ("/v1/completions", b'{"prompt":[1,2,3],"model":"synthetic-probe","add_special_tokens":false}', {}, False),
                    ])
                for path, raw, request_headers, transformed in payloads:
                    status, content_type, body = request(config["port"], raw, path, request_headers)
                    require(status == 200, f"synthetic worker response failed: {status}")
                    if json.loads(raw).get("stream"):
                        require("text/event-stream" in content_type and b"data: [DONE]" in body,
                                "SSE response was not relayed completely")
                    else:
                        worker_raw = bytes.fromhex(json.loads(body)["raw_hex"])
                        if transformed:
                            expected = raw.replace(b'"A\\u4e2d"', json.dumps(list(range(1, 33)), separators=(",", ":")).encode(), 1)
                            require(worker_raw == expected, "derived body changed more than prompt")
                        else:
                            require(worker_raw == raw, "fallback worker bytes changed")
                def heartbeat_progress():
                    current = events(event_path)
                    startup_time = next(e["monotonic"] for e in current if e["event"] == "startup")
                    return len([e for e in current if e["event"] == "heartbeat"
                                and e["monotonic"] > startup_time]) >= 3

                wait_until(heartbeat_progress)
            else:
                raw = b'{"model":"synthetic-probe","prompt":"slow synthetic","probe_delay":0.8}'
                connection = http.client.HTTPConnection("127.0.0.1", config["port"], timeout=5)
                try:
                    connection.request("POST", "/v1/completions", body=raw,
                                       headers={"Content-Type": "application/json"})
                    wait_until(lambda: any(e["event"] == "render_start" for e in events(event_path)))
                    if name == "deadline":
                        response = connection.getresponse()
                        require(response.status == 200, "deadline did not use existing fair fallback")
                        response.read()
                        record(event_path, "deadline_response")
                    else:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                        record(event_path, "client_disconnected")
                    connection.close()
                    raw_busy = b'{"model":"synthetic-probe","prompt":"must remain busy"}'
                    status, _, _ = request(config["port"], raw_busy)
                    require(status == 200, "busy did not use existing fair fallback")
                    current = events(event_path)
                    require(len([e for e in current if e["event"] == "render_start"]) == 1,
                            "timed-out/disconnected active callback released admission early")
                    require(not any(e["event"] == "render_end" for e in current),
                            "slow callback already ended before admission assertion")
                    record(event_path, "busy_response_while_active", elapsed=time.monotonic() - started)
                finally:
                    connection.close()
            metrics_after = completion_metrics(config["metrics_port"])
            if name == "json_sse":
                expected_metrics = dict(raw=0, prepared=0, ingress=0, backend=0)
                for path, raw, _, transformed in payloads:
                    if path != "/v1/completions":
                        continue
                    expected = (raw.replace(b'"A\\u4e2d"', json.dumps(list(range(1, 33)), separators=(",", ":")).encode(), 1)
                                if transformed else raw)
                    expected_metrics["prepared" if transformed else "raw"] += 1
                    expected_metrics["ingress"] += len(raw)
                    expected_metrics["backend"] += len(expected)
                require(metrics_after == expected_metrics,
                        f"public Completion transport counters differ: {metrics_after} vs {expected_metrics}")
            record(event_path, "signal_owned_child", child_pid=child_process.pid)
            child_process.send_signal(signal.SIGTERM)
            require(child_process.wait(timeout=10) == 0, "Router child failed graceful shutdown")
        observed = events(event_path)
        if kv_completion_token_input:
            require({e["worker"] for e in observed if e["event"] == "synthetic_capability_read"} == {0, 1},
                    "prepared-input probe bypassed real capability control-plane validation")
        starts = [e for e in observed if e["event"] == "render_start"]
        ends = [e for e in observed if e["event"] == "render_end"]
        callbacks = [e for e in observed if e["event"] in ("startup", "render_start", "render_end", "close")]
        require(len({e["native_thread"] for e in callbacks}) == 1, "callbacks changed native thread")
        main_thread = next(e["native_thread"] for e in observed if e["event"] == "main")
        require(callbacks[0]["native_thread"] != main_thread, "callback executed on main Python thread")
        require(len(starts) == len(ends) and all(e["active"] == 1 for e in starts), "callback concurrency/cleanup failed")
        require(all(e["lifetime_guard_alive"] for e in starts), "active callback lacked non-daemon Python lifetime guard")
        returned = [e for e in observed if e["event"] == "start_returned"]
        require(len(returned) == 1 and returned[0]["closed"] and returned[0]["active"] == 0
                and returned[0]["callback_thread_absent"] and returned[0]["lifetime_guard_absent"],
                "start returned before callback cleanup")
        close = next(e for e in observed if e["event"] == "close")
        require(close["active"] == 0 and all(e["monotonic"] < close["monotonic"] for e in ends),
                "facade closed before actual synchronous callback completed")
        heartbeats = [e for e in observed if e["event"] == "heartbeat"
                      and callbacks[0]["monotonic"] < e["monotonic"] < close["monotonic"]]
        require(len(heartbeats) >= 3, "Python heartbeat did not progress during blocking native start")
        require(any(e["event"] == "heartbeat_stopped" and not e["alive"] for e in observed),
                "heartbeat leaked after native start returned")
        forwarded = [e for e in observed if e["event"] == "worker_request"]
        if name == "json_sse":
            if kv_exact_history and not kv_completion_token_input:
                require(len({e["worker"] for e in forwarded}) == 1,
                        "equivalent exact Completion/Chat did not share advisory history")
                router_log = (case_dir / "router.log").read_text()
                require('"stage":"exact_history"' in router_log
                        and '"prefix_blocks":0' in router_log
                        and "kv_history_commit" in router_log,
                        "missing actual native zero-ownership history/commit decision")
            require(len(starts) == len(payloads) and {e["kind"] for e in starts} == {"completion", "chat"},
                    "did not exercise both raw ingress types")
            require(len(forwarded) == len(payloads), "unexpected dispatch/replay count")
            for index, (_, raw, request_headers, transformed) in enumerate(payloads):
                require(starts[index]["raw_hex"] == raw.hex(), "facade ingress bytes changed")
                expected = (raw.replace(b'"A\\u4e2d"', json.dumps(list(range(1, 33)), separators=(",", ":")).encode(), 1)
                            if transformed else raw)
                require(forwarded[index]["raw_hex"] == expected.hex(), "backend body did not match selected forwarding mode")
                worker_headers = {key.lower(): value for key, value in forwarded[index]["headers"].items()}
                require(int(worker_headers["content-length"]) == len(expected), "stale body length forwarded")
                for key, value in request_headers.items():
                    if transformed and key.lower() == "content-encoding":
                        require(key.lower() not in worker_headers, "derived identity encoding not removed")
                    else:
                        require(worker_headers.get(key.lower()) == value, "auth/request/fallback header changed")
        else:
            signalled = next(e for e in observed if e["event"] == "signal_owned_child")
            require(signalled["monotonic"] < ends[0]["monotonic"] < close["monotonic"],
                    "shutdown did not exercise an active callback drain")
        return dict(status="PASS", case=name, child_pid=child_process.pid,
                    callbacks=len(starts), python_heartbeats=len(heartbeats),
                    worker_requests=len(forwarded), events=str(event_path),
                    prepared_backend_requests=(sum(item[3] for item in payloads) if name == "json_sse" else 0),
                    completion_metrics_before=metrics_before, completion_metrics_after=metrics_after,
                    events_sha256=sha256(event_path), child_exit=child_process.returncode)
    finally:
        if child_process is not None and child_process.poll() is None:
            # Failure cleanup targets only the process created by this probe.
            child_process.kill()
            child_process.wait(timeout=5)
        workers.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--kv-load-guard", action="store_true")
    parser.add_argument("--kv-exact-history", action="store_true")
    parser.add_argument("--kv-completion-token-input", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    if not args.extension or not args.output:
        parser.error("--extension and --output are required")
    extension = args.extension.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="IN_PROGRESS", evidence_kind="synthetic facade, actual PyO3 extension",
                    claims_excluded=["actual vLLM rendering", "KV events/hits", "GPU", "TTFT"],
                    extension=str(extension), extension_sha256=sha256(extension),
                    kv_load_guard=args.kv_load_guard, kv_completion_token_input=args.kv_completion_token_input,
                    kv_exact_history=args.kv_exact_history,
                    capability_evidence=("synthetic descriptor via real Rust control plane"
                        if args.kv_completion_token_input else "legacy synthetic transport only"),
                    harness_sha256=sha256(__file__), python=sys.version, cases=[])
    try:
        for name in ("json_sse", "deadline", "disconnect"):
            manifest["cases"].append(run_case(name, extension, output,
                kv_load_guard=args.kv_load_guard, kv_completion_token_input=args.kv_completion_token_input,
                kv_exact_history=args.kv_exact_history))
        require(sha256(extension) == manifest["extension_sha256"], "native artifact changed during probe")
        manifest["status"] = "PASS"
    except Exception as error:
        manifest["status"] = "FAIL"
        manifest["error"] = f"{type(error).__name__}: {error}"
    finally:
        (output / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=manifest["status"], completed_cases=len(manifest["cases"]),
                          summary=str(output / "summary.json")), sort_keys=True))
    if manifest["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

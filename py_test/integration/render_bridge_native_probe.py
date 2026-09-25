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
            delay = json.loads(raw).get("probe_delay", 0)
            if delay:
                time.sleep(delay)
            self.active -= 1
            record(event_path, "render_end", active=self.active)
            return dict(status="exact", token_ids=list(range(1, 33)),
                        contract_id="synthetic-native-probe", epoch=1, cache_eligible=True)

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
        worker_startup_check_interval=1, log_level="warn", disable_retries=True,
        health_check_interval_secs=60, prometheus_host="127.0.0.1",
        prometheus_port=config["metrics_port"],
    )
    heartbeat_thread.start()
    try:
        router.start(
            render_facade=facade, render_contract_id="synthetic-native-probe",
            render_contract_epoch=1,
            render_limits=dict(max_pending_jobs=1, max_input_bytes=65536,
                               max_tokens_per_request=64, max_reserved_tokens=64,
                               queue_timeout_ms=100, execution_timeout_ms=config["deadline_ms"]),
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
    def __init__(self, event_path):
        self.servers = []
        self.threads = []
        self.urls = []
        for index in range(2):
            self._add(index, event_path)

    def _add(self, index, event_path):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                value = json.loads(raw)
                record(event_path, "worker_request", worker=index, path=self.path, raw_hex=raw.hex())
                if value.get("stream"):
                    payload = b'data: {"id":"synthetic","choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\ndata: [DONE]\n\n'
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
        self.urls.append(f"http://127.0.0.1:{server.server_port}")

    def close(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(2)
            require(not thread.is_alive(), "owned mock worker thread did not close")


def request(port, raw, path="/v1/completions"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
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


def run_case(name, extension, output):
    case_dir = output / name
    case_dir.mkdir()
    event_path = case_dir / "events.jsonl"
    workers = Workers(event_path)
    child_process = None
    try:
        config = dict(extension=str(extension), events=str(event_path), workers=workers.urls,
                      port=free_port(), deadline_ms=100 if name == "deadline" else 2000,
                      metrics_port=free_port(),
                      endpoints=[f"{url}=tcp://127.0.0.1:{free_port()}" for url in workers.urls])
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
            started = time.monotonic()
            if name == "json_sse":
                payloads = [
                    ("/v1/completions", b'{ "prompt" : "A\\u4e2d", "model":"synthetic-probe", "unknown_extra":{"a":1} }'),
                    ("/v1/chat/completions", b'{ "messages":[{"role":"user","content":"hi"}], "model":"synthetic-probe", "stream":true, "chat_template_kwargs":{"x":7} }'),
                ]
                for path, raw in payloads:
                    status, content_type, body = request(config["port"], raw, path)
                    require(status == 200, f"synthetic worker response failed: {status}")
                    if path == "/v1/chat/completions":
                        require("text/event-stream" in content_type and b"data: [DONE]" in body,
                                "SSE response was not relayed completely")
                    else:
                        require(json.loads(body)["raw_hex"] == raw.hex(), "worker bytes changed")
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
            record(event_path, "signal_owned_child", child_pid=child_process.pid)
            child_process.send_signal(signal.SIGTERM)
            require(child_process.wait(timeout=10) == 0, "Router child failed graceful shutdown")
        observed = events(event_path)
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
            require(len(starts) == 2 and {e["kind"] for e in starts} == {"completion", "chat"},
                    "did not exercise both raw ingress types")
            require([e["raw_hex"] for e in starts] == [e["raw_hex"] for e in forwarded],
                    "facade and actual worker did not receive identical original bytes")
        else:
            signalled = next(e for e in observed if e["event"] == "signal_owned_child")
            require(signalled["monotonic"] < ends[0]["monotonic"] < close["monotonic"],
                    "shutdown did not exercise an active callback drain")
        return dict(status="PASS", case=name, child_pid=child_process.pid,
                    callbacks=len(starts), python_heartbeats=len(heartbeats),
                    worker_requests=len(forwarded), events=str(event_path),
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
                    harness_sha256=sha256(__file__), python=sys.version, cases=[])
    try:
        for name in ("json_sse", "deadline", "disconnect"):
            manifest["cases"].append(run_case(name, extension, output))
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

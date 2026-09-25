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
"""

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import urllib.parse
import uuid

import kv_aware_cuda_validate as prior


ROOT = Path(__file__).resolve().parents[1]
BASE = "13b04aa2c3e811b9937abb3dbb3bd60f49a118c0"


def require(condition, message):
    prior.require(condition, message)


def save(path, value):
    prior.save(path, value)


def source_identity(source, candidate):
    source = Path(source).resolve()
    require(source == ROOT, "runner must belong to the candidate source")
    require(prior.command(["git", "rev-parse", "HEAD"], source) == candidate,
            "candidate SHA does not match source HEAD")
    require(not prior.command(["git", "status", "--porcelain"], source), "candidate source is dirty")
    prior.command(["git", "merge-base", "--is-ancestor", BASE, candidate], source)
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


def child(manifest_path):
    config = json.loads(Path(manifest_path).read_text())
    native_path = Path(config["native"])
    require(prior.sha256(native_path) == config["native_sha256"], "native artifact changed")
    spec = importlib.util.spec_from_file_location("vllm_router_rs", native_path)
    native = importlib.util.module_from_spec(spec)
    sys.modules["vllm_router_rs"] = native
    spec.loader.exec_module(native)
    sys.path.insert(0, str(ROOT / "py_src"))
    from vllm_router.router import Router
    from vllm_router.router_args import RouterArgs

    args = RouterArgs(
        host="127.0.0.1", port=config["router_port"], worker_urls=config["workers"],
        policy="kv_aware", kv_input_backend="vllm", kv_render_config=config["render_config"],
        kv_hash_algo="sha256_cbor", kv_hash_seed=0, kv_events_topic_filter="kv",
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
                    "python_module": str(Path(sys.modules[facade.__class__.__module__].__file__).resolve()),
                })

        facade.render, facade.startup = observed_render, observed_startup
        router.start()


@contextmanager
def owned_router(config, out):
    manifest = out / "router-child.json"
    save(manifest, config)
    with (out / "router.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                    "--child", str(manifest)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ.copy())
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


class Validation(prior.Validation):
    def __init__(self, args, out):
        super().__init__(args, out)
        self.observations = out / "facade-observations.jsonl"
        self.model = args.model
        self.worker_logs = [args.worker0_log, args.worker1_log]

    def case(self, name, callback):
        require(time.monotonic() < self.args.deadline, "finite GPU matrix budget exhausted")
        return super().case(name, callback)

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

    def routed(self, name, payload, expected=None, unsupported=False):
        self.idle()
        payload = {**payload, "return_token_ids": True}
        raw, tokens, route = self.oracle(payload, name)
        (self.out / f"{name}.request.json").write_bytes(raw)
        before = self.counters()
        before_prefix = [prefix_metrics(prior.metrics(worker)) for worker in self.workers]
        before_observation = len(read_observations(self.observations))
        before_access = render_access_count(self.worker_logs)
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
        scores = {item["worker"].rstrip("/"): item["prefix_blocks"] for item in decision["scores"]}
        require(set(scores) == set(self.workers), "decision lacks both worker scores")
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
        after_access = render_access_count(self.worker_logs)
        require(after_access == before_access, "generation unexpectedly used request-level Worker /render HTTP")
        after_prefix = [prefix_metrics(prior.metrics(worker)) for worker in self.workers]
        prefix_delta = [{key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}
                        for before, after in zip(before_prefix, after_prefix)]
        if expected is not None:
            hits = prefix_delta[expected].get("vllm:prefix_cache_hits_total", 0)
            queries = prefix_delta[expected].get("vllm:prefix_cache_queries_total", 0)
            require(0 < hits <= queries, "positive routing lacks a valid backend prefix-cache token hit/query delta")
        result = {"name": name, "status": "PASS", "actual_backend": actual,
                  "request": payload, "http_status": status, "response_sha256": hashlib.sha256(body).hexdigest(),
                  "decision": decision, "completed_request_deltas": delta,
                  "worker_token_count": len(tokens), "worker_token_ids_sha256": prior.token_digest(tokens),
                  "actual_generation_token_ids": actual_generation_tokens,
                  "facade_observation": observed, "prefix_cache_metric_deltas": prefix_delta,
                  "prefix_cache_token_hit_ratios": [
                      value.get("vllm:prefix_cache_hits_total", 0) / value["vllm:prefix_cache_queries_total"]
                      if value.get("vllm:prefix_cache_queries_total", 0) > 0 else None
                      for value in prefix_delta],
                  "worker_render_access_counts_before": before_access,
                  "worker_render_access_counts_after": after_access}
        save(self.out / f"{name}.json", result)
        return result

    def positive(self, name, kind, target, stream=False):
        text = (f"{uuid.uuid4().hex} Public GPU render bridge case {name}. " * 12
                + "Explain the water cycle in ordinary words.")
        payload = {"model": self.model, "max_tokens": 8, "temperature": 0, "stream": stream,
                   "return_token_ids": True}
        if kind == "chat":
            payload.update(messages=[{"role": "user", "content": text}],
                           chat_template_kwargs={"enable_thinking": False})
        else:
            payload.update(prompt=text, add_special_tokens=True)
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
        require(flags.get("--block-size") == ["16"]
                and flags.get("--prefix-caching-hash-algo") == ["sha256_cbor"], "Worker hash contract mismatch")
        require(env.get("PYTHONHASHSEED") == "0" and env.get("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") == "0",
                "Worker hash seed/full-byte events mismatch")
        require(env.get("VLLM_SERVER_DEV_MODE") in (None, "0"), "development mode is not allowed")
        kv = json.loads(flags["--kv-events-config"][0])
        require(kv.get("enable_kv_cache_events") and kv.get("publisher") == "zmq"
                and kv.get("endpoint") == config["publisher_endpoints"][index] and kv.get("topic") == "kv",
                "Worker publisher does not match declared Router event endpoint")
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
    out = prior.output_directory(args.output, args.source)
    report = {"status": "RUNNING", "started_at_unix": time.time(), "command": sys.argv,
              "limitations": ["No tokens-in/out; generation repeats preprocessing.",
                              "Queued cancellation uses existing CPU synthetic lifecycle proof; no artificial GPU delay is injected.",
                              "Raw facade ingress is recorded; stock Worker does not export raw generation bytes."]}
    save(out / "summary.json", report)
    args.deadline = time.monotonic() + args.budget_seconds
    try:
        require(platform.system() == "Linux", "run in the authorized GPU process /proc namespace")
        identity = source_identity(args.source, args.candidate)
        native_hash = prior.sha256(args.native)
        build = json.loads(Path(args.build_manifest).read_text())
        require(build.get("status") == "PASS" and build.get("candidate_sha") == args.candidate
                and build.get("native_sha256") == native_hash,
                "native build manifest must bind exact candidate and actual .so SHA")
        deployment = json.loads(Path(args.render_config).read_text())
        workers = [loopback_url(args.worker0), loopback_url(args.worker1)]
        require(deployment["worker_urls"] == workers and len(set(workers)) == 2, "deployment Worker URLs differ")
        config = {"native": str(Path(args.native).resolve()), "native_sha256": native_hash,
                  "workers": workers, "event_endpoints": [args.event0, args.event1],
                  "publisher_endpoints": [args.publisher0 or args.event0, args.publisher1 or args.event1],
                  "router_port": args.router_port, "metrics_port": args.metrics_port,
                  "render_config": str(Path(args.render_config).resolve()),
                  "serving_args": deployment["serving_args"],
                  "observations": str(out / "facade-observations.jsonl"),
                  "facade_identity": str(out / "facade-identity.json")}
        worker_processes = verify_workers(args, config)
        report.update(identity, native_sha256=native_hash, native=str(Path(args.native).resolve()),
                      build_manifest_sha256=prior.sha256(args.build_manifest),
                      render_config_sha256=prior.sha256(args.render_config), workers=worker_processes,
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
            save(out / "summary.json", report)
            validation = Validation(args, out)
            # Reuse the public, independently reviewed shape corpus. This imports
            # definitions only; it never launches the CPU/mock render service.
            sys.path.insert(0, str(ROOT / "py_test"))
            from test_render_bridge_vllm import actual_cases
            for name, _, payload in actual_cases(args.model):
                validation.case("tokens-" + name, lambda name=name, payload=payload:
                                validation.routed("tokens-" + name, payload))
            for kind in ("completion", "chat"):
                for target in (0, 1):
                    for stream in (False, True):
                        name = f"positive-{kind}-w{target}-{'sse' if stream else 'json'}"
                        validation.case(name, lambda name=name, kind=kind, target=target, stream=stream:
                                        validation.positive(name, kind, target, stream))
            validation.case("salt_fairness", validation.salt_fairness)
            # Existing helper has full first-record/active-before-close checks,
            # log/PID/ID correlation and no natural completion masquerading as abort.
            prior.MODEL = args.model
            validation.case("stream_cancel_cleanup", validation.cancel_cleanup)
            validation.idle()
            require(mapped_native(process.pid, args.native) == report["mapped_native"],
                    "mapped extension changed during validation")
            for before in worker_processes:
                expected = {key: value for key, value in before.items()
                            if key != "verified_preprocessing_arguments"}
                require(prior.process(before["pid"]) == expected, "Worker process identity changed during matrix")
            report["cases"] = validation.results
        require(source_identity(args.source, args.candidate) == identity, "source changed during validation")
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
    print("PASS render bridge GPU evidence parsers (11 checks; no hardware)")
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
    require(60 <= args.budget_seconds <= 1800, "matrix budget must be 60..1800 seconds")
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("supervised matrix interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

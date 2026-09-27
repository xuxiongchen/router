"""Opt-in REAL vLLM 0.29 CPU render/public-HTTP equivalence and measurements.

Requires the optional dependency, public local assets, and explicit environment:
  CMB_RENDER_TEST_MODEL_DIRS=/assets/Qwen3-0.6B:/assets/Qwen3-1.7B
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS=''
  python -m unittest discover -s py_test -p 'test_render_bridge*.py' -v

No inference engine/model weights/GPU are used. Each temporary HTTP service runs
the official API router and ServingRender. It is a CPU oracle, NOT a generation
worker/hardware proof. CMB_RENDER_BENCHMARK=1 additionally measures serial Python
facade vs persistent loopback HTTP; Rust queue/GIL conversion is not measured here.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import socket
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("actual_render_bridge", ROOT / "py_src/vllm_router/render_bridge.py")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)
MODEL_DIRS = [Path(p) for p in os.environ.get("CMB_RENDER_TEST_MODEL_DIRS", "").split(os.pathsep) if p]
# Verification fixture identity, never a runtime model/template allowlist.
# Same canonical Qwen3 template fingerprint as PR1's independently checked corpus.
GOLDEN_CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


@contextmanager
def official_http_renderer(argv, *, generation_probe=False):
    """Bind only this test's own ephemeral loopback socket; bounded teardown."""
    import uvicorn
    from fastapi import FastAPI, Request
    from vllm import AsyncEngineArgs, envs
    from vllm.config import VllmConfig
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.launchers.render.app_state import init_render_app_state
    from vllm.entrypoints.scale_out.render.api_router import router
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    failures = []
    ready = threading.Event()
    @asynccontextmanager
    async def lifespan(app):
        # Independent baseline: use the OFFICIAL app-state initializer, not
        # this adapter's _load_runtime or its CaptureRenderer. Configuration
        # construction is the CPU launcher's run_launch_fastapi sequence.
        args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
        validate_parsed_serve_args(args)
        model_config = AsyncEngineArgs.from_cli_args(args).create_model_config()
        model_config.quantization = None
        envs.VLLM_CPU_KVCACHE_SPACE = 0
        config = VllmConfig(model_config=model_config)
        await init_render_app_state(config, app.state, args)
        try:
            ready.set()
            yield
        finally:
            # OnlineRenderer/OnlineDerenderer intentionally share one renderer.
            app.state.online_renderer.renderer.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    if generation_probe:
        # Explicit CPU benchmark sink, NOT an inference implementation. The
        # unmodified official /render routes above remain the startup oracle.
        @app.get("/health")
        async def probe_health():
            return {"status": "cpu-benchmark-mock"}

        @app.get("/v1/models")
        async def probe_models():
            options = bridge._serving_options(argv)
            return {"object": "list", "data": [{"id": options["--served-model-name"], "object": "model"}]}

        @app.post("/v1/chat/completions")
        @app.post("/v1/completions")
        async def probe_generation(request: Request):
            raw = await request.body()
            body = json.loads(raw)
            chat = request.url.path == "/v1/chat/completions"
            choice = ({"message": {"role": "assistant", "content": "cpu-probe"}}
                      if chat else {"text": "cpu-probe"})
            return {"id": "cpu-only-generation-probe", "created": 0,
                    "object": "chat.completion" if chat else "text_completion",
                    "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop", **choice}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "cmb_mock_generation": True,
                    "cmb_request_sha256": hashlib.sha256(raw).hexdigest()}
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False,
                                        lifespan="on", timeout_graceful_shutdown=5))
    def serve():
        try:
            asyncio.run(server.serve(sockets=[sock]))
        except BaseException as exc:
            failures.append(type(exc).__name__)
        finally:
            ready.set()
    thread = threading.Thread(target=serve, name="official-render-test", daemon=True)
    thread.start()
    try:
        if not ready.wait(120) or failures:
            raise RuntimeError("official_render_test_startup_failed")
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("official_render_test_not_listening")
        yield "http://127.0.0.1:" + str(port)
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("official_render_test_shutdown_timeout")


def actual_cases(model):
    base = {"model": model, "max_tokens": 1, "temperature": 0}
    normal = [{"role": "user", "content": "What is café 中文 🙂?"}]
    yield "normal", "chat", {**base, "messages": normal}
    yield "stream", "chat", {**base, "messages": normal, "stream": True,
                              "stream_options": {"include_usage": True}}
    yield "text_parts", "chat", {**base, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "中文"}, {"type": "text", "text": "\n🙂 café"}]}]}
    history = [*normal, {"role": "assistant", "content": "Answer.", "reasoning": "Historical reasoning."},
               {"role": "user", "content": "Continue."}]
    for thinking in (False, True):
        yield "thinking_" + str(thinking), "chat", {
            **base, "messages": history, "chat_template_kwargs": {"enable_thinking": thinking}}
    yield "reasoning_none", "chat", {**base, "messages": normal, "reasoning_effort": "none"}
    yield "drop_thinking", "chat", {**base, "messages": history,
                                    "chat_template_kwargs": {"drop_thinking": True}}
    yield "null_kwargs", "chat", {**base, "messages": normal, "chat_template_kwargs": None}
    yield "json_object", "chat", {**base, "messages": normal, "response_format": {"type": "json_object"}}
    yield "json_schema", "chat", {**base, "messages": normal, "response_format": {
        "type": "json_schema", "json_schema": {"name": "answer", "schema": {
            "type": "object", "properties": {"z": {"type": "integer"}, "a": {"type": "string"}},
            "required": ["z", "a"], "additionalProperties": False}}}}
    yield "structured_choice", "chat", {**base, "messages": normal,
                                          "structured_outputs": {"choice": ["red", "blue"]}}
    for reverse in (False, True):
        properties = {"z": {"type": "integer"}, "a": {"type": "string"}}
        if reverse:
            properties = dict(reversed(list(properties.items())))
        arguments = '{"a":"中文","z":1}' if reverse else '{"z":1,"a":"中文"}'
        yield "tool_order_" + str(reverse), "chat", {
            **base, "messages": [*normal, {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "lookup", "arguments": arguments}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "found"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {
                "type": "object", "properties": properties, "required": list(properties)}}}],
            "tool_choice": "auto",
        }
    yield "completion_text", "completion", {**base, "prompt": "中英文 café 🙂\n"}
    yield "completion_ids", "completion", {**base, "prompt": [151644, 872, 198, 40, 151645]}
    yield "completion_truncate", "completion", {**base, "prompt": "one two three four five six", "truncate_prompt_tokens": 4}


def _percentiles(samples):
    values = sorted(samples)
    return {"p" + str(p) + "_ms": values[max(0, math.ceil(len(values) * p / 100) - 1)]
            for p in (50, 95, 99)}


def measure_serial(facade, url):
    """Bounded serial measurement; no invented Rust conversion/queue timings."""
    port = int(url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    results = []
    try:
        for length in (32, 4096, 32768, 65536):
            if length + 128 >= facade.effective_config["max_model_len"]:
                results.append({"target_tokens": length, "status": "NOT_RUN", "reason": "model_length_limit"})
                continue
            raw = json.dumps({"model": facade.model, "messages": [{"role": "user", "content": " x" * length}],
                              "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}}).encode()
            expected = facade.render("chat", raw)
            if expected["status"] != "exact":
                results.append({"target_tokens": length, "status": "NOT_RUN", "reason": expected["reason"]})
                continue
            for mode in ("python_facade", "official_http_keepalive"):
                samples = []
                before_cpu = time.process_time()
                before_wall = time.perf_counter()
                for _ in range(10):
                    start = time.perf_counter_ns()
                    if mode == "python_facade":
                        actual = facade.render("chat", raw)["token_ids"]
                    else:
                        connection.request("POST", "/v1/chat/completions/render", raw,
                                           {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        payload = response.read()
                        if response.status != 200:
                            raise RuntimeError("benchmark_official_http_failed")
                        actual = json.loads(payload)["token_ids"]
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                    if actual != expected["token_ids"]:
                        raise RuntimeError("benchmark_token_mismatch")
                elapsed = time.perf_counter() - before_wall
                results.append({"mode": mode, "concurrency": 1, "iterations": 10,
                                "actual_tokens": len(actual), "target_tokens": length,
                                "requests_per_second": 10 / elapsed,
                                "cpu_seconds": time.process_time() - before_cpu,
                                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                "cancelled": 0, "timeouts": 0, **_percentiles(samples)})
    finally:
        connection.close()
    for result in results:
        result["model_directory"] = facade.effective_config["model"]
    print("CMB_RENDER_SERIAL_MEASUREMENTS=" + json.dumps(results, sort_keys=True))


@unittest.skipUnless(MODEL_DIRS, "set CMB_RENDER_TEST_MODEL_DIRS for real vLLM CPU validation")
class RealVllmRenderTests(unittest.TestCase):
    def test_real_cpu_facade_public_http_and_public_fixtures(self):
        reports = []
        for model_dir in MODEL_DIRS:
            with self.subTest(model_directory=str(model_dir)):
                model = "render-test-model"
                argv = ["--model", str(model_dir), "--tokenizer", str(model_dir),
                        "--served-model-name", model, "--enable-auto-tool-choice",
                        "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"]
                with official_http_renderer(argv) as url, tempfile.TemporaryDirectory(prefix="real-render-test-") as directory:
                    config_path = Path(directory) / "render.json"
                    config_path.write_text(json.dumps({"serving_args": argv, "worker_urls": [url],
                        "cache_layout": {"kind": "qwen3_dense_full_attention", "block_size": 16,
                                         "hash_algorithm": "sha256_cbor", "hash_seed": 0},
                        # Include template overhead above a 64k content body.
                        "bridge_limits": {"max_tokens_per_request": 131072,
                                          "max_reserved_tokens": 524288}}))
                    facade = bridge.create_facade(config_path)
                    try:
                        facade.startup()
                        self.assertTrue(facade.conformance)
                        passed = []
                        for name, kind, request in actual_cases(model):
                            raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
                            result = facade.render(kind, raw)
                            self.assertEqual(result["status"], "exact", name + ": " + str(result))
                            if os.environ.get("VLLM_ROUTER_KV_STAGE_TIMING") == "1":
                                stages = result["stage_durations_ns"]
                                self.assertTrue({"python_total", "asset_check", "schema_json", "raw_json",
                                                 "cache_eligibility", "serving", "online_renderer",
                                                 "python_result"} <= stages.keys(), name)
                                self.assertTrue(all(type(value) is int and value >= 0
                                                    for value in stages.values()), name)
                                self.assertEqual(result["stage_counters"]["asset_read_bytes"], 0)
                                self.assertEqual(result["stage_counters"]["asset_hash_bytes"], 0)
                            expected = bridge._remote_render(url, kind, raw, 10, None)
                            self.assertEqual(result["token_ids"], expected, name)
                            passed.append(name)
                        # Last-key-wins at actual FastAPI JSON ingress and direct
                        # schema parsing must agree; raw ordering is untouched.
                        duplicate = ('{"model":"' + model + '","messages":[{"role":"user",'
                                     '"content":"first","content":"second"}],"max_tokens":1}').encode()
                        self.assertEqual(facade.render("chat", duplicate)["token_ids"],
                                         bridge._remote_render(url, "chat", duplicate, 10, None))
                        passed.append("duplicate_last_key_wins")
                        oracle = json.loads((ROOT / "tests/fixtures/kv_qwen3/python_oracle.json").read_text())
                        digest = hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest()
                        effective_template = facade._runtime.renderer.tokenizer.get_chat_template()
                        template_digest = hashlib.sha256(effective_template.encode()).hexdigest()
                        golden_eligible = {
                            "completion": digest == oracle["tokenizer_sha256"],
                            "chat": (digest == oracle["tokenizer_sha256"]
                                     and template_digest == GOLDEN_CHAT_TEMPLATE_SHA256),
                        }
                        for kind in ("chat", "completion"):
                            if golden_eligible[kind]:
                                for case in oracle[kind]:
                                    request = {**case["request"], "model": model, "max_tokens": 1}
                                    result = facade.render(kind, json.dumps(request, ensure_ascii=False).encode())
                                    self.assertEqual(result["status"], "exact", case["name"])
                                    self.assertEqual(result["token_ids"], case["token_ids"], case["name"])
                                    passed.append("public_golden_" + case["name"])
                        unsupported = [
                            ("completion", {"prompt": ["a", "b"]}),
                            ("completion", {"prompt": [[1], [2]]}),
                            ("chat", {"messages": [{"role": "user", "content": "x"}], "cache_salt": "salt"}),
                            ("chat", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://must-not-fetch.invalid/image"}}]}]}),
                            ("chat", {"messages": [{"role": "user", "content": "x"}], "kv_transfer_params": {"prompt_token_ids": [1]}}),
                        ]
                        for kind, body in unsupported:
                            self.assertEqual(facade.render(kind, json.dumps({**body, "model": model}).encode())["status"], "unsupported")
                        self.assertEqual(facade.render("chat", b'{"messages":"malformed"}')["status"], "invalid")
                        wrong_model = {"model": "missing-model", "messages": [{"role": "user", "content": "x"}]}
                        missing = facade.render("chat", json.dumps(wrong_model).encode())
                        self.assertEqual(missing["status"], "invalid")
                        self.assertEqual(missing["http_status"], 404)
                        # This negative fixture depends on the original template's
                        # enable_thinking semantics. Instruct-2507 legitimately
                        # ignores it; do not require a mismatch where none exists.
                        if template_digest == GOLDEN_CHAT_TEMPLATE_SHA256:
                            mismatch_path = Path(directory) / "mismatch.json"
                            mismatch_config = json.loads(config_path.read_text())
                            mismatch_config["serving_args"] += [
                                "--default-chat-template-kwargs", '{"enable_thinking":false}']
                            mismatch_path.write_text(json.dumps(mismatch_config))
                            mismatch = bridge.create_facade(mismatch_path)
                            try:
                                with self.assertRaises(bridge.RenderConfigurationError):
                                    mismatch.startup()
                                self.assertEqual(mismatch.startup_failure["code"], "worker_conformance_mismatch")
                                self.assertEqual(mismatch.render("chat", b"{}")["status"], "unavailable")
                            finally:
                                mismatch.close()
                            passed.append("actual_default_kwargs_mismatch_refuses_startup")
                        reports.append({"model_directory": str(model_dir), "contract_id": facade.contract_id,
                                        "effective_config": facade.effective_config, "cases": passed,
                                        "effective_template_sha256": template_digest,
                                        "public_golden_eligible": golden_eligible,
                                        "unsupported_cases": len(unsupported), "invalid_cases": 2,
                                        "stage_timing": os.environ.get("VLLM_ROUTER_KV_STAGE_TIMING") == "1",
                                        "startup_conformance": facade.conformance})
                        if os.environ.get("CMB_RENDER_BENCHMARK") == "1":
                            measure_serial(facade, url)
                    finally:
                        facade.close()
        print("CMB_REAL_RENDER_RESULTS=" + json.dumps(reports, sort_keys=True))


@unittest.skipUnless(MODEL_DIRS, "set CMB_RENDER_TEST_MODEL_DIRS for real vLLM CPU validation")
class RealVllmCompletionInputTests(unittest.TestCase):
    def test_prepared_completion_tokens_sampling_and_skipped_tokenizer(self):
        """Real serving/input equivalence, NOT GPU output or Worker evidence.

        No manual profile is needed: this test loads the same automatic Dense
        renderer adapter for both Qwen and non-Qwen public tokenizer assets.
        Only vLLM's real preprocessing is used, with no inference engine.
        """
        import msgspec
        from vllm import envs

        reports = []
        for model_dir in MODEL_DIRS:
            async def run_model():
                envs.VLLM_CPU_KVCACHE_SPACE = 0
                model = "completion-token-input-test"
                runtime = bridge._load_runtime([
                    "--model", str(model_dir), "--tokenizer", str(model_dir),
                    "--served-model-name", model,
                ], worker_capabilities=True)
                # Isolate the real per-request facade method. Startup/cohort
                # conformance is covered by separate integration tests; this
                # fixture does not claim a remote Worker exists on the CPU.
                facade = object.__new__(bridge.RenderFacade)
                facade._runtime = runtime
                facade.model = model
                facade.limits = {"max_tokens_per_request": 65536}
                facade.contract_id = "cpu-input-equivalence-only"
                facade.epoch = 1
                tokenize_calls = 0
                original_tokenize = runtime.renderer._tokenize_prompt_async

                async def counted_tokenize(*args, **kwargs):
                    nonlocal tokenize_calls
                    tokenize_calls += 1
                    return await original_tokenize(*args, **kwargs)

                runtime.renderer._tokenize_prompt_async = counted_tokenize
                base = {"model": model, "max_tokens": 3, "temperature": 0,
                        "seed": 17, "add_special_tokens": False}
                # Include literal special strings, whitespace, Unicode and
                # longer BPE boundaries; never construct tokens by fragments.
                cases = [(name, {**base, "prompt": prompt}) for name, prompt in (
                    ("ascii", "The next number after two is"),
                    ("unicode", "café e\u0301 中文 🙂\n\tA  B"),
                    ("special_literals", "<|im_start|>user\nHello<|im_end|>"),
                    ("whitespace", " \n\t\r\n "),
                    ("boundary", " word" * 127 + " café"),
                )]
                for name, update in (
                    ("return_ids", {"return_token_ids": True}),
                    ("stream_usage", {"stream": True, "stream_options": {"include_usage": True}}),
                    ("stop_sampling", {"temperature": 0.7, "top_p": 0.8,
                        "top_k": 8, "min_p": 0.1, "stop": ["END"],
                        "stop_token_ids": [3], "include_stop_str_in_output": True,
                        "frequency_penalty": 0.2, "presence_penalty": 0.1,
                        "repetition_penalty": 1.1, "ignore_eos": True, "min_tokens": 1}),
                    ("nulls_missing", {"temperature": None, "top_p": None,
                        "stop": None, "stream": None, "return_token_ids": None}),
                    ("explicit_count", {"n": 1, "use_beam_search": False,
                        "skip_special_tokens": False, "spaces_between_special_tokens": False}),
                ):
                    cases.append((name, {**base, "prompt": "hello café", **update}))
                passed = []
                try:
                    for name, body in cases:
                        raw = json.dumps(body, ensure_ascii=False).encode()
                        exact = await facade._render("completion", raw)
                        self.assertEqual(exact["status"], "exact", (name, exact))
                        self.assertTrue(exact["completion_token_input_eligible"], name)
                        original_schema = runtime.schemas["completion"].model_validate_json(raw)
                        text_result = await runtime.serving.render_completion_request(original_schema)
                        text_inputs = runtime.capture.engine_inputs
                        calls_before = tokenize_calls
                        derived = {**body, "prompt": exact["token_ids"]}
                        token_schema = runtime.schemas["completion"].model_validate_json(json.dumps(derived).encode())
                        token_result = await runtime.serving.render_completion_request(token_schema)
                        token_inputs = runtime.capture.engine_inputs
                        self.assertEqual(tokenize_calls, calls_before, name)
                        self.assertEqual(text_inputs[0]["prompt_token_ids"], token_inputs[0]["prompt_token_ids"], name)
                        self.assertEqual(exact["token_ids"], token_result[0].token_ids, name)
                        self.assertEqual(msgspec.to_builtins(text_result[0].sampling_params),
                                         msgspec.to_builtins(token_result[0].sampling_params), name)
                        self.assertEqual(text_result[0].model_dump(exclude={"request_id", "sampling_params"}),
                                         token_result[0].model_dump(exclude={"request_id", "sampling_params"}), name)
                        passed.append(name)
                    for name, update in (
                        ("negative_max_tokens", {"max_tokens": -1}),
                        ("negative_temperature", {"temperature": -1}),
                        ("null_n", {"n": None}),
                        ("null_special", {"add_special_tokens": None}),
                        ("invalid_prompt", {"prompt": {"not": "text"}}),
                    ):
                        invalid = await facade._render("completion", json.dumps({
                            **base, "prompt": "hello", **update}).encode())
                        self.assertEqual(invalid["status"], "invalid", (name, invalid))
                        self.assertNotIn("completion_token_input_eligible", invalid)
                    after_invalid = await facade._render("completion", json.dumps({
                        **base, "prompt": "valid after client errors"}).encode())
                    self.assertEqual(after_invalid["status"], "exact")
                    self.assertTrue(after_invalid["completion_token_input_eligible"])
                    reports.append({"model_directory": str(model_dir),
                        "cases": passed, "invalid_cases": 5,
                        "token_array_text_tokenization_calls": 0,
                        "gpu_generation": "NOT_RUN"})
                finally:
                    runtime.renderer.shutdown()
            with self.subTest(model_directory=str(model_dir)):
                asyncio.run(run_model())
        print("CMB_REAL_COMPLETION_INPUT_RESULTS=" + json.dumps(reports, sort_keys=True))


if __name__ == "__main__":
    unittest.main()

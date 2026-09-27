"""Public CPU acceptance. Optional real vLLM tests never load weights or a GPU.

CMB_CHAT_MODEL_DIR=/oracle-assets python -m unittest discover -s py_test -p test_chat_serving.py -v
"""

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from chat_serving_semantics import (  # noqa: E402
    TOOL,
    ChatAccumulator,
    check_semantics,
    decode_response,
    execute_tools,
    followup,
)


def message(arguments='{"a":17,"b":25}', ident="call_public"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": ident,
                "type": "function",
                "function": {"name": "synthetic_add", "arguments": arguments},
            }
        ],
    }


class ChatContractTests(unittest.TestCase):
    def test_native_provenance_rejects_product_changes(self):
        from render_bridge_gpu_validate import verify_native_source

        sha = "a" * 40
        for changed in (
            "src/lib.rs",
            "Cargo.lock",
            "build.rs",
            "py_src/vllm_router/router.py",
            "setup.py",
            "scripts/build_wheel.sh",
        ):
            with self.subTest(changed=changed), patch(
                "kv_aware_cuda_validate.command", side_effect=["", changed]
            ):
                with self.assertRaises((AssertionError, RuntimeError)):
                    verify_native_source(".", "b" * 40, sha)
        with patch(
            "kv_aware_cuda_validate.command",
            side_effect=["", "docs/chat.md\npy_test/test_chat.py"],
        ):
            self.assertEqual(
                verify_native_source(".", "b" * 40, sha)["native_source_candidate"], sha
            )

    def test_actual_local_result_and_immutable_history(self):
        payload = {"messages": [{"role": "user", "content": "add"}], "tools": [TOOL]}
        original = copy.deepcopy(payload)
        result = followup(payload, message())
        self.assertEqual(payload, original)
        self.assertEqual(json.loads(result["messages"][-1]["content"]), {"result": 42})
        self.assertEqual(result["messages"][-1]["tool_call_id"], "call_public")

    def test_multiple_calls_and_reject_unsafe_arguments(self):
        value = message()
        value["tool_calls"] += message('{"a":-1,"b":2}', "second")["tool_calls"]
        self.assertEqual(len(execute_tools(value)), 2)
        for raw in (
            '{"a":true,"b":2}',
            '{"a":1,"a":2,"b":3}',
            '{"a":1,"b":2,"cmd":"ls"}',
            '{"a":1000001,"b":2}',
            "[]",
            "{broken",
        ):
            with self.subTest(raw=raw), self.assertRaises((AssertionError, ValueError)):
                execute_tools(message(raw))
        value["tool_calls"][1]["id"] = "call_public"
        with self.assertRaises(AssertionError):
            execute_tools(value)
        bad = message()
        bad["tool_calls"][0]["function"]["name"] = "shell"
        with self.assertRaises(AssertionError):
            execute_tools(bad)

    def test_stream_identity_finish_usage_and_error_guards(self):
        acc = ChatAccumulator()
        acc.feed(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            0.1,
        )
        self.assertEqual(acc.first, {})
        for arguments, ident in (('{"a":', "first"), ('17,"b":25}', "changed")):
            event = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": ident,
                                    "function": {"arguments": arguments},
                                }
                            ]
                        },
                    }
                ]
            }
            if ident == "changed":
                with self.assertRaises(AssertionError):
                    acc.feed(event)
            else:
                acc.feed(event, 0.2)
        self.assertEqual(acc.first, {"tool": 0.2})
        for body in (
            b'data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n',
            b'data: {"choices":[]}\n\n',
            b"data: [DONE]\n\ndata: {}\n\n",
        ):
            with self.assertRaises(AssertionError):
                decode_response(body, True)

    def test_schema_rejects_wrong_actual_output(self):
        for text in (
            '{"answer":true}',
            '{"answer":42,"extra":1}',
            '"42"',
            '{"answer":41}',
        ):
            with self.assertRaises(AssertionError):
                check_semantics({"message": {"content": text}}, "schema")


@unittest.skipUnless(
    os.environ.get("CMB_CHAT_MODEL_DIR"),
    "set CMB_CHAT_MODEL_DIR for real vLLM CPU parser tests",
)
class OfficialServingReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "chat_bridge_fixture", ROOT / "py_src/vllm_router/render_bridge.py"
        )
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        cls.runtime = bridge._load_runtime(
            [
                "--model",
                os.environ["CMB_CHAT_MODEL_DIR"],
                "--served-model-name",
                "chat-fixture",
                "--max-model-len",
                "8192",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--reasoning-parser",
                "qwen3",
                "--generation-config",
                "vllm",
            ]
        )
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

        online = cls.runtime.capture.inner
        cls.online = online
        engine = SimpleNamespace(
            model_config=online.model_config,
            renderer=cls.runtime.renderer,
            input_processor=None,
            vllm_config=None,
        )
        cls.serving = OpenAIServingChat(
            engine,
            cls.runtime.serving.models,
            "assistant",
            online_renderer=online,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            reasoning_parser="qwen3",
            tool_parser="hermes",
            enable_auto_tools=True,
            enable_prompt_tokens_details=True,
        )
        cls.tokenizer = cls.runtime.renderer.tokenizer
        cls.measurements = []

    @classmethod
    def tearDownClass(cls):
        cls.runtime.renderer.shutdown()
        output = os.environ.get("CMB_CHAT_REPLAY_REPORT")
        if output:
            import inspect
            import vllm

            files = [
                inspect.getfile(cls.serving.__class__),
                inspect.getfile(cls.online.__class__),
            ]
            parser = cls.serving.parser_cls
            files += [
                inspect.getfile(parser.tool_parser_cls),
                inspect.getfile(parser.reasoning_parser_cls),
            ]
            Path(output).write_text(
                json.dumps(
                    {
                        "vllm_version": vllm.__version__,
                        "source_files": {
                            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in files
                        },
                        "scope": "fixed-output official serving replay, not model inference or GPU performance",
                        "measurements": cls.measurements,
                    },
                    indent=2,
                )
                + "\n"
            )

    async def replay(
        self,
        text,
        *,
        stream,
        step=1,
        extra=None,
        finish="stop",
        failure=False,
        cancelled=False,
    ):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
        from vllm.outputs import CompletionOutput, RequestOutput

        request = ChatCompletionRequest(
            **{
                "model": "chat-fixture",
                "messages": [{"role": "user", "content": "public fixture"}],
                "stream": stream,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
                **(extra or {}),
                **({"stream_options": {"include_usage": True}} if stream else {}),
            }
        )
        conversation, inputs = await self.online.render_chat(request)
        prompt = self.serving._extract_prompt_components(inputs[0]).token_ids
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        kwargs = self.serving._effective_chat_template_kwargs(request)

        async def outputs():
            if cancelled:
                raise asyncio.CancelledError()
            if failure:
                raise ValueError("public injected engine failure")
            if not stream:
                yield RequestOutput(
                    "replay",
                    None,
                    prompt,
                    None,
                    [CompletionOutput(0, text, ids, None, None, finish_reason=finish)],
                    True,
                    num_cached_tokens=0,
                )
                return
            # Official incremental detokenization; never tokenizer.decode(delta).
            from vllm.v1.engine.detokenizer import IncrementalDetokenizer
            from vllm import SamplingParams
            from vllm.v1.engine import EngineCoreRequest

            engine_request = EngineCoreRequest(
                request_id="replay",
                prompt_token_ids=prompt,
                mm_features=None,
                sampling_params=SamplingParams(
                    max_tokens=8192, skip_special_tokens=False
                ),
                pooling_params=None,
                arrival_time=0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )
            detok = IncrementalDetokenizer.from_new_request(
                self.tokenizer, engine_request
            )
            for start in range(0, len(ids), step):
                chunk = ids[start : start + step]
                detok.update(chunk, False)
                final = start + step >= len(ids)
                delta = detok.get_next_output_text(final, delta=True)
                yield RequestOutput(
                    "replay",
                    None,
                    prompt,
                    None,
                    [
                        CompletionOutput(
                            0,
                            delta,
                            chunk,
                            None,
                            None,
                            finish_reason=finish if final else None,
                        )
                    ],
                    final,
                    num_cached_tokens=0,
                )

        metadata = RequestResponseMetadata(request_id="replay")
        started = time.process_time()
        if stream:
            chunks = [
                x
                async for x in self.serving.chat_completion_stream_generator(
                    request,
                    outputs(),
                    "replay",
                    "chat-fixture",
                    conversation,
                    self.tokenizer,
                    metadata,
                    chat_template_kwargs=kwargs,
                )
            ]
            body = "".join(chunks).encode()
        else:
            parser = self.serving.parser_cls(
                self.tokenizer,
                request.tools,
                chat_template_kwargs=kwargs,
                model_config=self.online.model_config,
            )
            result = await self.serving.chat_completion_full_generator(
                request,
                outputs(),
                "replay",
                "chat-fixture",
                conversation,
                self.tokenizer,
                metadata,
                parser=parser,
            )
            body = result.model_dump_json().encode()
        self.measurements.append(
            {
                "output_tokens": len(ids),
                "step": step,
                "stream": stream,
                "cpu_seconds": time.process_time() - started,
            }
        )
        if failure or cancelled or finish == "error":
            return body
        decoded = decode_response(body, stream)
        self.last_wire = body
        self.assertEqual(decoded["usage"]["prompt_tokens"], len(prompt))
        self.assertEqual(decoded["usage"]["completion_tokens"], len(ids))
        return decoded

    def test_unicode_json_sse_chunk_boundaries(self):
        for step in (1, 3, 11):
            for stream in (False, True):
                with self.subTest(step=step, stream=stream):
                    result = asyncio.run(
                        self.replay("café 中文 🙂 hello", stream=stream, step=step)
                    )
                    self.assertEqual(result["message"]["content"], "café 中文 🙂 hello")

    def test_tools_and_real_synthetic_result_history(self):
        text = '<tool_call>\n{"name":"synthetic_add","arguments":{"a":17,"b":25}}\n</tool_call>'
        for stream in (False, True):
            for step in (1, 7):
                for multiple in (False, True):
                    with self.subTest(stream=stream, step=step, multiple=multiple):
                        result = asyncio.run(
                            self.replay(
                                text + ("\n" + text if multiple else ""),
                                stream=stream,
                                step=step,
                                extra={"tools": [TOOL], "tool_choice": "auto"},
                            )
                        )
                        self.assertEqual(result["finish_reason"], "tool_calls")
                        check_semantics(result, "tool")
                        calls = result["message"]["tool_calls"]
                        self.assertEqual(len(calls), 2 if multiple else 1)
                        payload = followup(
                            {"messages": [], "tools": [TOOL]}, result["message"]
                        )
                        answer = asyncio.run(
                            self.replay(
                                "The result is 42.", stream=stream, extra=payload
                            )
                        )
                        self.assertIn("42", answer["message"]["content"])

    def test_thinking_and_suppression(self):
        for stream in (False, True):
            for include in (False, True):
                result = asyncio.run(
                    self.replay(
                        "A short thought.</think>\nThe answer is 42.",
                        stream=stream,
                        step=1,
                        extra={
                            "chat_template_kwargs": {"enable_thinking": True},
                            "include_reasoning": include,
                        },
                    )
                )
                self.assertIn("42", result["message"]["content"])
                self.assertEqual(bool(result["message"]["reasoning"]), include)
                self.assertNotIn("short thought", result["message"]["content"])

    def test_named_required_and_structured_outputs(self):
        # Pinned Hermes declares a structural-tag grammar; required/named
        # therefore use its tagged output, NOT the generic JSON-only grammar.
        tool_text = '<tool_call>{"name":"synthetic_add","arguments":{"a":17,"b":25}}</tool_call>'
        fixtures = [
            (
                tool_text,
                {
                    "tools": [TOOL],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "synthetic_add"},
                    },
                },
                "tool",
            ),
            (tool_text, {"tools": [TOOL], "tool_choice": "required"}, "tool"),
            ('{"answer":42}', {"response_format": {"type": "json_object"}}, "schema"),
            (
                '{"answer":42}',
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "integer", "enum": [42]}
                                },
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                        },
                    }
                },
                "schema",
            ),
            ("red", {"structured_outputs": {"choice": ["red", "blue"]}}, "choice"),
        ]
        for text, extra, expected in fixtures:
            for stream in (False, True):
                with self.subTest(extra=extra, stream=stream):
                    result = asyncio.run(
                        self.replay(text, stream=stream, step=2, extra=extra)
                    )
                    check_semantics(result, expected)

    def test_engine_errors_not_success(self):
        for kwargs in ({"failure": True}, {"finish": "error"}):
            body = asyncio.run(self.replay("partial", stream=True, **kwargs))
            self.assertIn(b'"error"', body)
            with self.assertRaises(AssertionError):
                decode_response(body, True)

    def test_fixed_dependency_rejects_parsed_stream_derender(self):
        from vllm.renderers.online_derenderer import OnlineDerenderer

        derenderer = OnlineDerenderer(
            self.online.model_config,
            self.runtime.renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
            tool_parser="hermes",
            reasoning_parser="qwen3",
        )
        with self.assertRaisesRegex(NotImplementedError, "reasoning or tool parser"):
            asyncio.run(
                derenderer.derender_chat_stream(
                    "chat-fixture", SimpleNamespace(choices=[])
                )
            )

    def test_cancelled_official_generator_and_next_request(self):
        body = asyncio.run(self.replay("", stream=False, cancelled=True))
        self.assertIn("Client disconnected", json.loads(body)["error"]["message"])
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(self.replay("", stream=True, cancelled=True))
        next_request = asyncio.run(self.replay("healthy", stream=True))
        self.assertEqual(next_request["message"]["content"], "healthy")

    def test_bounded_long_output(self):
        # Three finite output lengths, one measurement each. Not a throughput
        # claim; includes the actual official streaming parser/serialization.
        for repetitions in (64, 256, 1024):
            text = " public" * repetitions
            result = asyncio.run(self.replay(text, stream=True, step=8))
            self.assertEqual(result["message"]["content"], text)
            result = asyncio.run(
                self.replay(
                    text + "</think>Answer.",
                    stream=True,
                    step=8,
                    extra={"chat_template_kwargs": {"enable_thinking": True}},
                )
            )
            self.assertEqual(result["message"]["reasoning"], text)
            self.assertEqual(result["message"]["content"], "Answer.")

    @unittest.skipUnless(
        os.environ.get("CMB_CHAT_NATIVE"),
        "set CMB_CHAT_NATIVE for actual extension/SDK wire replay",
    )
    def test_official_output_through_native_sdk_and_tool_loop(self):
        import httpx
        from openai import OpenAI

        spec = importlib.util.spec_from_file_location(
            "chat_native_probe",
            ROOT / "py_test/integration/render_bridge_native_probe.py",
        )
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        tool_text = '<tool_call>{"name":"synthetic_add","arguments":{"a":17,"b":25}}</tool_call>'
        wires = {}
        for stream in (False, True):
            for turn, text in ((0, tool_text), (1, "The result is 42. café 中文 🙂")):
                asyncio.run(
                    self.replay(
                        text,
                        stream=stream,
                        step=1,
                        extra={
                            "tools": [TOOL],
                            "tool_choice": "auto" if turn == 0 else "none",
                        },
                    )
                )
                wires[stream, turn] = self.last_wire

        def provider(payload):
            turn = int(any(x["role"] == "tool" for x in payload["messages"]))
            stream = payload.get("stream", False)
            return (
                200,
                "text/event-stream" if stream else "application/json",
                wires[stream, turn],
            )

        with tempfile.TemporaryDirectory(prefix="chat-native-") as directory:
            out = Path(directory)
            event_path = out / "events.jsonl"
            workers = probe.Workers(
                event_path, capabilities=True, response_provider=provider
            )
            child = None
            sent = []
            try:
                config = dict(
                    extension=os.environ["CMB_CHAT_NATIVE"],
                    events=str(event_path),
                    workers=workers.urls,
                    port=probe.free_port(),
                    metrics_port=probe.free_port(),
                    deadline_ms=2000,
                    kv_load_guard=True,
                    kv_completion_token_input=True,
                    cohort={"workers": workers.descriptors, "api_key_env": None},
                    endpoints=workers.endpoints,
                )
                config_path = out / "config.json"
                config_path.write_text(json.dumps(config))
                with (out / "router.log").open("wb") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            str(
                                ROOT
                                / "py_test/integration/render_bridge_native_probe.py"
                            ),
                            "--child",
                            str(config_path),
                        ],
                        stdout=log,
                        stderr=log,
                    )
                    probe.wait_until(
                        lambda: child.poll() is not None or probe.ready(config["port"])
                    )
                    self.assertIsNone(
                        child.poll(), (out / "router.log").read_text()[-2000:]
                    )
                    before = probe.wait_until(
                        lambda: probe.require_complete_metrics(config["metrics_port"])
                    )
                    with OpenAI(
                        base_url=f"http://127.0.0.1:{config['port']}/v1",
                        api_key="public-synthetic-only",
                        max_retries=0,
                        timeout=15,
                        http_client=httpx.Client(
                            event_hooks={
                                "request": [
                                    lambda request: sent.append(
                                        (request.content, dict(request.headers))
                                    )
                                ]
                            }
                        ),
                    ) as client:
                        for stream in (False, True):
                            payload = {
                                "model": "synthetic-probe",
                                "messages": [
                                    {"role": "user", "content": "Add 17 and 25"}
                                ],
                                "tools": [TOOL],
                                "tool_choice": "auto",
                                "stream": stream,
                            }
                            if stream:
                                payload["stream_options"] = {"include_usage": True}

                            def sdk_call(value):
                                response = client.chat.completions.create(**value)
                                accumulator = ChatAccumulator()
                                if stream:
                                    with response:
                                        for chunk in response:
                                            accumulator.feed(
                                                chunk.model_dump(exclude_none=True)
                                            )
                                else:
                                    accumulator.feed(
                                        response.model_dump(exclude_none=True)
                                    )
                                return accumulator.result()

                            first = sdk_call(payload)
                            check_semantics(first, "tool")
                            second = sdk_call(followup(payload, first["message"]))
                            self.assertIn("42", second["message"]["content"])
                    forwarded = [
                        e
                        for e in probe.events(event_path)
                        if e["event"] == "worker_request"
                    ]
                    self.assertEqual(len(forwarded), 4)
                    self.assertEqual(len(sent), 4)
                    for event, (raw, headers) in zip(forwarded, sent):
                        self.assertEqual(bytes.fromhex(event["raw_hex"]), raw)
                        received = {k.lower(): v for k, v in event["headers"].items()}
                        for key in (
                            "authorization",
                            "user-agent",
                            "x-stainless-lang",
                            "x-stainless-package-version",
                        ):
                            self.assertEqual(received.get(key), headers[key])
                    after = probe.completion_metrics(config["metrics_port"])
                    self.assertEqual(
                        after, before
                    )  # Real Completion counters; not invented Chat metrics.
                    self.measurements.append(
                        {
                            "scope": "actual native + official replay + OpenAI SDK; synthetic render facade",
                            "native_sha256": probe.sha256(config["extension"]),
                            "sdk_requests": 4,
                            "observed_raw_chat_dispatches": len(forwarded),
                            "observed_prepared_chat_dispatches": 0,
                            "completion_metrics_before": before,
                            "completion_metrics_after": after,
                        }
                    )
                    child.send_signal(signal.SIGTERM)
                    child.wait(timeout=15)
                    self.assertEqual(child.returncode, 0)
            finally:
                if child is not None and child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                workers.close()


if __name__ == "__main__":
    unittest.main()

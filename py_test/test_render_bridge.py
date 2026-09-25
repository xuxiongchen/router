"""Dependency-free boundary tests; these are NOT evidence of vLLM equivalence.

The separate test_render_bridge_vllm module runs the real optional dependency.
Run in the isolated Linux environment with unittest discovery, without importing
the package's native extension (which these adapter boundary tests do not need).
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "py_src/vllm_router/render_bridge.py"
SPEC = importlib.util.spec_from_file_location("render_bridge_under_test", MODULE)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class SchemaError(ValueError):
    pass


class BoundarySchema:
    """Minimal test double to isolate bridge ownership, never a vLLM substitute."""

    @staticmethod
    def model_validate_json(raw):
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise SchemaError()
        return SimpleNamespace(model_extra=obj.get("unknown"), raw=obj)


class BoundaryRuntime:
    def __init__(self):
        self.capture = SimpleNamespace(engine_inputs=None)
        self.schemas = {"chat": BoundarySchema, "completion": BoundarySchema}
        self.validation_type = SchemaError
        self.error_type = type("ErrorResponse", (), {})
        self.renderer = self
        self.serving = self
        self.effective = {"synthetic_test_double": True}
        self.inputs = [{"type": "token", "prompt_token_ids": [4, 2, 19]}]
        self.seen = []
        self.closed = False

    def shutdown(self):
        self.closed = True

    async def render_chat_request(self, request):
        self.seen.append(request.raw)
        self.capture.engine_inputs = self.inputs
        return SimpleNamespace(token_ids=[4, 2, 19], features=None, cache_salt=None)

    async def render_completion_request(self, request):
        return [await self.render_chat_request(request)]


class RenderBridgeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="render-bridge-boundary-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
            "use_sliding_window": False, "sliding_window": None,
        }))
        (self.root / "tokenizer.json").write_text('{"model":{"type":"BPE","dropout":null}}')
        (self.root / "tokenizer_config.json").write_text('{"chat_template":"test only"}')
        self.config = {
            "serving_args": ["--model", str(self.root), "--served-model-name", "public-test"],
            "worker_urls": ["http://127.0.0.1:12345"],
            "cache_layout": {"kind": "qwen3_dense_full_attention", "block_size": 16,
                             "hash_algorithm": "sha256_cbor", "hash_seed": 0},
        }
        self.path = self.root / "bridge.json"
        self.runtime = BoundaryRuntime()

    def facade(self):
        self.path.write_text(json.dumps(self.config))
        facade = bridge.create_facade(self.path)
        self.addCleanup(facade.close)
        return facade

    def ready(self):
        facade = self.facade()
        with patch.object(bridge, "_load_runtime", return_value=self.runtime), \
             patch.object(bridge, "_remote_render", return_value=[4, 2, 19]), \
             patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": ""}):
            facade.startup()
        self.runtime.seen.clear()
        return facade

    def test_construct_without_importing_vllm_or_creating_eventloop(self):
        filters = {name: list(logging.getLogger(name).filters) for name in bridge._CONTENT_FREE_LOGGERS}
        facade = self.facade()
        self.assertIsNone(facade._loop)
        self.assertEqual(facade.model, "public-test")
        self.assertEqual(facade.epoch, 1)
        self.assertEqual(facade.render("chat", b"{}")['status'], "unavailable")
        for name, original in filters.items():
            self.assertEqual(logging.getLogger(name).filters, original)

    def test_sensitive_logs_are_content_free_in_renderer_threads(self):
        filters = {name: list(logging.getLogger(name).filters) for name in bridge._CONTENT_FREE_LOGGERS}
        facade = self.ready()
        sentinel = "synthetic-request-content-must-not-appear-7429"
        for name in bridge._CONTENT_FREE_LOGGERS:
            logger = logging.getLogger(name)
            with self.assertLogs(logger, level="DEBUG") as captured:
                def emit():
                    try:
                        raise ValueError(sentinel)
                    except ValueError:
                        logger.exception("request=%s", sentinel, stack_info=True)
                thread = threading.Thread(target=emit)
                thread.start()
                thread.join(2)
                self.assertFalse(thread.is_alive())
            self.assertNotIn(sentinel, "\n".join(captured.output))
            self.assertEqual(len(captured.records), 1)
            record = captured.records[0]
            self.assertEqual(record.name, name)
            self.assertEqual(record.levelno, logging.ERROR)
            self.assertEqual(record.getMessage(), "cmb_render_diagnostic")
            self.assertEqual(record.args, ())
            self.assertIsNone(record.exc_info)
            self.assertIsNone(record.exc_text)
            self.assertIsNone(record.stack_info)
        with self.assertLogs("vllm.unrelated_boundary_logger", level="WARNING") as unrelated:
            logging.getLogger("vllm.unrelated_boundary_logger").warning("ordinary diagnostic")
        self.assertIn("ordinary diagnostic", unrelated.output[0])
        facade.close()
        for name, original in filters.items():
            self.assertEqual(logging.getLogger(name).filters, original)

    def test_log_filters_are_owned_per_facade_and_removed_after_failure(self):
        name = bridge._CONTENT_FREE_LOGGERS[0]
        logger = logging.getLogger(name)
        original = list(logger.filters)
        first = self.ready()
        second = self.ready()
        self.assertEqual(len(logger.filters), len(original) + 2)
        first.close()
        self.assertEqual(len(logger.filters), len(original) + 1)
        with self.assertLogs(logger, level="WARNING") as remaining:
            logger.warning("synthetic-secret-must-be-filtered")
        self.assertEqual(remaining.records[0].getMessage(), "cmb_render_diagnostic")
        second.close()
        self.assertEqual(logger.filters, original)
        failed = self.facade()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(bridge.RenderConfigurationError):
                failed.startup()
        self.assertEqual(logger.filters, original)

    def test_log_filters_are_removed_even_when_renderer_close_raises(self):
        name = bridge._CONTENT_FREE_LOGGERS[0]
        logger = logging.getLogger(name)
        original = list(logger.filters)
        facade = self.ready()
        with patch.object(self.runtime, "shutdown", side_effect=RuntimeError("synthetic close failure")):
            with self.assertRaises(RuntimeError):
                facade.close()
        self.assertEqual(logger.filters, original)

    def test_offline_setting_is_required_not_mutated(self):
        facade = self.facade()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(bridge.RenderConfigurationError, "render_startup_failed"):
                facade.startup()
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_mismatch_prevents_ready_and_disposes_runtime(self):
        facade = self.facade()
        with patch.object(bridge, "_load_runtime", return_value=self.runtime), \
             patch.object(bridge, "_remote_render", return_value=[99]), \
             patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": ""}):
            with self.assertRaisesRegex(bridge.RenderConfigurationError, "render_startup_failed"):
                facade.startup()
        self.assertTrue(self.runtime.closed)
        self.assertEqual(facade.render("chat", b"{}")['status'], "unavailable")

    def test_raw_order_null_and_last_duplicate_survive(self):
        facade = self.ready()
        raw = b'{"messages":[{"role":"assistant","content":null}],"z":1,"a":2,"z":3}'
        result = facade.render("chat", raw)
        self.assertEqual(result['status'], "exact")
        self.assertEqual(list(self.runtime.seen[-1]), ["messages", "z", "a"])
        self.assertIsNone(self.runtime.seen[-1]["messages"][0]["content"])
        self.assertEqual(self.runtime.seen[-1]["z"], 3)

    def test_valid_fields_are_not_removed(self):
        facade = self.ready()
        request = {"messages": [], "tools": [], "response_format": {"type": "json_object"},
                   "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}}
        self.assertEqual(facade.render("chat", json.dumps(request).encode())["status"], "exact")
        self.assertEqual(self.runtime.seen[-1], request)

    def test_unknown_model_is_rejected_before_serving_logs(self):
        facade = self.ready()
        result = facade.render("chat", b'{"model":"sensitive-test-name","messages":[]}')
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["http_status"], 404)
        self.assertNotIn("sensitive-test-name", str(result))
        self.assertEqual(self.runtime.seen, [])

    def test_unsupported_identity_and_media_do_not_call_renderer(self):
        facade = self.ready()
        requests = [
            ("chat", {"messages": [], "cache_salt": "private"}),
            ("chat", {"messages": [], "kv_transfer_params": {"prompt_token_ids": [2]}}),
            ("chat", {"messages": [], "lora_path": "private"}),
            ("chat", {"messages": [], "unknown": {"new_cache_key": 1}}),
            ("chat", {"messages": [], "use_beam_search": True}),
            ("chat", {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "http://never-fetch"}}]}]}),
            ("completion", {"prompt": ["one", "two"]}),
            ("completion", {"prompt": [[1], [2]]}),
            ("completion", {"prompt_embeds": "never-decode"}),
        ]
        for kind, request in requests:
            with self.subTest(request=request):
                self.assertEqual(facade.render(kind, json.dumps(request).encode())["status"], "unsupported")
        self.assertEqual(self.runtime.seen, [])

    def test_null_salt_is_not_nonempty_identity(self):
        facade = self.ready()
        self.assertEqual(facade.render("completion", b'{"prompt":[3],"cache_salt":null}')["status"], "exact")

    def test_engine_shape_is_checked_after_serving(self):
        facade = self.ready()
        for inputs in ([{"type": "multimodal", "prompt_token_ids": [4]}],
                       [{"type": "token", "prompt_token_ids": [4], "cache_salt": "x"}],
                       [{"type": "token", "prompt_token_ids": [4]}] * 2):
            self.runtime.inputs = inputs
            self.assertEqual(facade.render("chat", b'{"messages":[]}')["status"], "unsupported")

    def test_invalid_schema_is_not_unavailable(self):
        facade = self.ready()
        result = facade.render("chat", b"{secret-invalid")
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["http_status"], 400)
        self.assertNotIn("secret", str(result))

    def test_exception_is_sanitized_and_fences_provider(self):
        facade = self.ready()
        async def fail(_):
            raise RuntimeError("secret prompt")
        self.runtime.render_chat_request = fail
        result = facade.render("chat", b'{"messages":[]}')
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("secret", str(result))
        self.assertTrue(facade._invalidated)

    def test_changed_assets_return_identity_fence_not_fallback(self):
        facade = self.ready()
        (self.root / "tokenizer.json").write_text('{"changed":true}')
        result = facade.render("chat", b'{"messages":[]}')
        self.assertEqual(result["status"], "invalidated")
        self.assertNotEqual(result["epoch"], facade.epoch)

    def test_new_template_file_also_fences_identity(self):
        facade = self.ready()
        (self.root / "chat_template.jinja").write_text("new effective template")
        self.assertEqual(facade.render("chat", b'{"messages":[]}')["status"], "invalidated")
        self.assertEqual(facade.render("chat", b'{"messages":[]}')["status"], "invalidated")

    def test_wrong_thread_cannot_render_or_close(self):
        facade = self.ready()
        received = []
        def other_thread():
            received.append(facade.render("chat", b'{"messages":[]}'))
            try:
                facade.close()
            except bridge.RenderConfigurationError:
                received.append("close_refused")
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(received[0]["status"], "unavailable")
        self.assertEqual(received[1], "close_refused")

    def test_close_disposes_single_loop_and_is_idempotent(self):
        facade = self.ready()
        loop = facade._loop
        facade.close()
        facade.close()
        self.assertTrue(loop.is_closed())
        self.assertTrue(self.runtime.closed)
        self.assertEqual(facade.render("chat", b"{}")["status"], "unavailable")

    def test_input_and_output_budgets(self):
        self.config["bridge_limits"] = {"max_input_bytes": 64}
        facade = self.ready()
        self.assertEqual(facade.render("chat", b"x" * 65)["reason"], "input_byte_budget")
        facade.limits["max_tokens_per_request"] = 2
        self.assertEqual(facade.render("chat", b'{"messages":[]}')["reason"], "token_budget")

    def test_invalid_configs_fail_closed(self):
        invalid = [
            ("serving_args", ["--model", "https://remote", "--served-model-name", "x"]),
            ("serving_args", self.config["serving_args"] + ["--trust-remote-code"]),
            ("serving_args", self.config["serving_args"] + ["--chat-template", "/tmp/custom"]),
            ("worker_urls", []),
            ("worker_urls", ["https://user:secret@localhost"]),
            ("bridge_limits", {"max_pending_jobs": 0}),
            ("cache_layout", {"kind": "hybrid"}),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                old = self.config[key] if key in self.config else None
                self.config[key] = value
                with self.assertRaises(bridge.RenderConfigurationError):
                    self.facade()
                if old is None:
                    del self.config[key]
                else:
                    self.config[key] = old

    def test_hybrid_layout_and_nondeterministic_tokenizer_rejected(self):
        (self.root / "config.json").write_text('{"model_type":"qwen3_5"}')
        with self.assertRaises(bridge.RenderConfigurationError):
            self.facade()
        (self.root / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
        (self.root / "tokenizer.json").write_text('{"model":{"dropout":0.1}}')
        with self.assertRaises(bridge.RenderConfigurationError):
            self.facade()

    def test_contract_changes_with_configuration(self):
        first = self.facade().contract_id
        self.config["serving_args"] += ["--default-chat-template-kwargs", '{"enable_thinking":false}']
        self.assertNotEqual(first, self.facade().contract_id)

    def test_hash_seed_uses_router_u32_domain(self):
        self.config["cache_layout"]["hash_seed"] = 2**32 - 1
        self.assertEqual(self.facade().hash_seed, 2**32 - 1)
        self.config["cache_layout"]["hash_seed"] = 2**32
        with self.assertRaises(bridge.RenderConfigurationError):
            self.facade()


if __name__ == "__main__":
    unittest.main()

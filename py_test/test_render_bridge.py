"""Dependency-free boundary tests; these are NOT evidence of vLLM equivalence.

The separate test_render_bridge_vllm module runs the real optional dependency.
Run in the isolated Linux environment with unittest discovery, without importing
the package's native extension (which these adapter boundary tests do not need).
"""

import asyncio
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

    def test_completion_input_proof_is_narrow_and_after_exact_render(self):
        facade = self.ready()
        base = {"prompt": "café 中文 🙂", "add_special_tokens": False}
        for update in ({}, {"n": 1}, {"use_beam_search": False},
                       {"return_token_ids": True}, {"return_token_ids": False},
                       {"stream": True, "stream_options": {"include_usage": True}},
                       {"temperature": None, "stop": None}):
            request = {**base, **update}
            result = facade.render("completion", json.dumps(request).encode())
            self.assertEqual(result["status"], "exact")
            self.assertTrue(result["completion_token_input_eligible"], update)
            self.assertEqual(self.runtime.seen[-1], request)
        for key, value in (
            ("echo", False), ("echo", None), ("suffix", None),
            ("logprobs", None), ("prompt_logprobs", None),
            ("return_token_offsets", False), ("truncate_prompt_tokens", None),
            ("truncation_side", None), ("cache_salt", None),
            ("add_special_tokens", True), ("add_special_tokens", None),
            ("n", "1"), ("n", True), ("n", 2),
            ("response_format", {"type": "text"}),
            ("user_unreviewed", "unchanged"), ("prompt", [4, 2, 19]),
        ):
            request = {**base, key: value}
            result = facade.render("completion", json.dumps(request).encode())
            self.assertFalse(result.get("completion_token_input_eligible", False), (key, value))
        missing_special = facade.render("completion", b'{"prompt":"text"}')
        self.assertFalse(missing_special["completion_token_input_eligible"])
        chat = facade.render("chat", b'{"messages":[]}')
        self.assertFalse(chat["completion_token_input_eligible"])
        self.runtime.inputs = [{"type": "other", "prompt_token_ids": [4, 2, 19]}]
        unsupported = facade.render("completion", json.dumps(base).encode())
        self.assertNotEqual(unsupported["status"], "exact")
        self.assertNotIn("completion_token_input_eligible", unsupported)

    def test_runtime_validation_error_is_client_error_not_provider_invalidation(self):
        # vLLMValidationError inherits Exception, not ValueError in v0.29.
        class RuntimeValidationError(Exception):
            pass

        self.runtime.render_validation_type = RuntimeValidationError
        facade = self.ready()
        raw = b'{"prompt":"valid later","add_special_tokens":false}'
        with patch.object(self.runtime, "render_completion_request",
                          side_effect=RuntimeValidationError("synthetic private detail")):
            failed = facade.render("completion", raw)
        self.assertEqual(failed["status"], "invalid")
        self.assertEqual(failed["http_status"], 400)
        self.assertNotIn("completion_token_input_eligible", failed)
        self.assertNotIn("synthetic private detail", str(failed))
        self.assertFalse(facade._invalidated)
        recovered = facade.render("completion", raw)
        self.assertEqual(recovered["status"], "exact")
        self.assertTrue(recovered["completion_token_input_eligible"])

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

    def test_stat_signature_stats_each_entry_once_on_every_call(self):
        paths = [self.root / "z.json", self.root / "a.json", self.root / "z.json"]
        metadata = [SimpleNamespace(st_size=11, st_mtime_ns=101),
                    SimpleNamespace(st_size=12, st_mtime_ns=102),
                    SimpleNamespace(st_size=11, st_mtime_ns=101)]
        expected = ((str(paths[0]), 11, 101), (str(paths[1]), 12, 102),
                    (str(paths[2]), 11, 101))
        observer = bridge._StageObserver()
        with patch.object(Path, "stat", autospec=True, side_effect=metadata * 2) as stat:
            self.assertEqual(bridge._stat_signature(paths, observer), expected)
            self.assertEqual(bridge._stat_signature(paths, observer), expected)
        self.assertEqual([entry.args[0] for entry in stat.call_args_list], paths * 2)
        self.assertEqual(observer.counters["asset_stat_calls"], 6)

    def test_asset_size_change_with_same_mtime_fences_identity(self):
        facade = self.ready()
        path = self.root / "tokenizer.json"
        initial = path.stat()
        path.write_bytes(path.read_bytes() + b" ")
        os.utime(path, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        current = path.stat()
        self.assertNotEqual(current.st_size, initial.st_size)
        self.assertEqual(current.st_mtime_ns, initial.st_mtime_ns)
        result = facade.render("chat", b'{"messages":[]}')
        self.assertEqual((result["status"], result["reason"], result["epoch"]),
                         ("invalidated", "contract_changed", facade.epoch + 1))
        self.assertEqual(self.runtime.seen, [])

    def test_asset_mtime_change_with_same_size_fences_identity(self):
        facade = self.ready()
        path = self.root / "tokenizer.json"
        initial = path.stat()
        os.utime(path, ns=(initial.st_atime_ns, initial.st_mtime_ns + 1_000_000_000))
        current = path.stat()
        self.assertEqual(current.st_size, initial.st_size)
        self.assertNotEqual(current.st_mtime_ns, initial.st_mtime_ns)
        result = facade.render("chat", b'{"messages":[]}')
        self.assertEqual((result["status"], result["reason"], result["epoch"]),
                         ("invalidated", "contract_changed", facade.epoch + 1))
        self.assertEqual(self.runtime.seen, [])

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


class RenderBridgeTimingTests(unittest.TestCase):
    """Instrumentation contracts only; these doubles do not prove vLLM speed."""

    setUp = RenderBridgeBoundaryTests.setUp
    facade = RenderBridgeBoundaryTests.facade
    ready = RenderBridgeBoundaryTests.ready

    def timed_ready(self):
        with patch.dict(os.environ, {"VLLM_ROUTER_KV_STAGE_TIMING": "1"}):
            return self.ready()

    def test_disabled_by_default_and_only_literal_one_enables(self):
        for setting in ("", "0", "true"):
            with self.subTest(setting=setting), patch.dict(
                os.environ, {"VLLM_ROUTER_KV_STAGE_TIMING": setting}
            ):
                facade = self.ready()
                with patch.object(bridge.time, "perf_counter_ns", side_effect=AssertionError("unexpected clock")):
                    result = facade.render("chat", b'{"messages":[]}')
                self.assertEqual(result["status"], "exact")
                self.assertNotIn("stage_durations_ns", result)
                self.assertNotIn("stage_counters", result)
                facade.close()

    def test_observation_preserves_full_result_and_contract(self):
        facade = self.timed_ready()
        result = facade.render("chat", b'{"messages":[]}')
        stages = result.pop("stage_durations_ns")
        counters = result.pop("stage_counters")
        facade._stage_timing_enabled = False
        self.assertEqual(result, facade.render("chat", b'{"messages":[]}'))
        self.assertEqual(set(stages), {"python_total", "asset_check", "schema_json", "raw_json",
                                       "cache_eligibility", "serving", "python_result"})
        self.assertTrue(all(type(value) is int and value >= 0 for value in stages.values()))
        self.assertGreaterEqual(stages["python_total"], stages["serving"])
        self.assertEqual(counters, {
            "asset_scan_calls": 5, "asset_is_file_calls": 4, "asset_exists_calls": 1,
            "asset_stat_calls": 9, "asset_read_calls": 0, "asset_read_bytes": 0,
            "asset_hash_calls": 0, "asset_hash_bytes": 0,
        })

    def test_setting_is_sampled_at_construction_not_each_request(self):
        facade = self.timed_ready()
        with patch.dict(os.environ, {"VLLM_ROUTER_KV_STAGE_TIMING": "0"}):
            self.assertIn("stage_durations_ns", facade.render("chat", b'{"messages":[]}'))

    def test_clock_failure_never_changes_or_repeats_render(self):
        facade = self.timed_ready()
        with patch.object(bridge.time, "perf_counter_ns", side_effect=RuntimeError("clock failure")):
            result = facade.render("chat", b'{"messages":[]}')
        self.assertEqual(result["status"], "exact")
        self.assertEqual(result["token_ids"], [4, 2, 19])
        self.assertEqual(len(self.runtime.seen), 1)
        self.assertEqual(result["stage_durations_ns"], {})
        self.assertIsNone(bridge._ACTIVE_STAGE_OBSERVER.get())

    def test_clock_exit_and_counter_failure_do_not_escape(self):
        class BrokenDict(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("observation storage failure")
        observer = bridge._StageObserver()
        observer.durations = BrokenDict()
        observer.counters = BrokenDict(observer.counters)
        with observer.measure("python_total"):
            observer.count("asset_stat_calls")
        with patch.object(bridge.time, "perf_counter_ns", side_effect=[1, RuntimeError("exit failure")]):
            with observer.measure("python_total"):
                pass
        self.assertEqual(observer.durations, {})

    def test_schema_failure_has_only_entered_stages_and_no_stale_values(self):
        facade = self.timed_ready()
        failed = facade.render("chat", b"not-json")
        self.assertEqual(failed["reason"], "request_schema")
        self.assertEqual(set(failed["stage_durations_ns"]), {"python_total", "asset_check", "schema_json"})
        result = facade.render("chat", b'{"messages":[]}')
        self.assertIn("serving", result["stage_durations_ns"])
        self.assertIsNot(failed["stage_durations_ns"], result["stage_durations_ns"])
        self.assertIsNone(bridge._ACTIVE_STAGE_OBSERVER.get())

    def test_unsupported_does_not_claim_renderer_time(self):
        facade = self.timed_ready()
        result = facade.render("chat", b'{"cache_salt":"private","messages":[]}')
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("cache_eligibility", result["stage_durations_ns"])
        self.assertNotIn("serving", result["stage_durations_ns"])
        self.assertNotIn("private", json.dumps(result))

    def test_asset_mutation_still_invalidates_and_observation_is_fresh(self):
        facade = self.timed_ready()
        (self.root / "new.jinja").write_text("test-only added asset")
        result = facade.render("chat", b'{"messages":[]}')
        self.assertEqual((result["status"], result["epoch"]), ("invalidated", 2))
        self.assertIn("asset_check", result["stage_durations_ns"])
        self.assertNotIn("schema_json", result["stage_durations_ns"])
        later = facade.render("chat", b'{"messages":[]}')
        self.assertEqual(set(later["stage_durations_ns"]), {"python_total"})
        self.assertEqual(sum(later["stage_counters"].values()), 0)

    def test_provider_exception_records_serving_and_restores_context(self):
        facade = self.timed_ready()
        async def fail(_):
            raise RuntimeError("private failure")
        self.runtime.render_chat_request = fail
        result = facade.render("chat", b'{"messages":[]}')
        self.assertEqual(result["reason"], "provider_exception")
        self.assertIn("serving", result["stage_durations_ns"])
        self.assertNotIn("python_result", result["stage_durations_ns"])
        self.assertIsNone(bridge._ACTIVE_STAGE_OBSERVER.get())
        self.assertNotIn("private failure", json.dumps(result))

    def test_wrong_thread_does_not_use_owner_runtime_or_leak_context(self):
        facade = self.timed_ready()
        replies = []
        def invoke():
            replies.append(facade.render("chat", b'{"messages":[]}'))
            self.assertIsNone(bridge._ACTIVE_STAGE_OBSERVER.get())
        thread = threading.Thread(target=invoke)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(replies[0]["reason"], "provider_not_ready")
        self.assertEqual(set(replies[0]["stage_durations_ns"]), {"python_total"})
        self.assertEqual(facade.render("chat", b'{"messages":[]}')["status"], "exact")

    def test_asset_read_and_hash_bytes_count_actual_helper_work(self):
        observer = bridge._StageObserver()
        path = self.root / "tokenizer.json"
        digest = bridge._asset_digest(path, observer)
        self.assertEqual(digest, bridge.hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(observer.counters["asset_read_calls"], 1)
        self.assertEqual(observer.counters["asset_read_bytes"], path.stat().st_size)
        self.assertEqual(observer.counters["asset_hash_calls"], 1)
        self.assertEqual(observer.counters["asset_hash_bytes"], path.stat().st_size)

    def test_nested_instance_async_observers_preserve_results_and_are_local(self):
        class Renderer:
            async def _tokenize_prompt_async(self, value):
                await asyncio.sleep(0)
                return value

            async def _apply_chat_template_async(self, value):
                return value

            async def render_chat(self, value):
                value = await self._apply_chat_template_async(value)
                ids = await self._tokenize_prompt_async(value)
                return "prompt", [{"prompt_token_ids": ids}]

            async def render_completion(self, value):
                return [{"prompt_token_ids": await self._tokenize_prompt_async(value)}]

        first, second = Renderer(), Renderer()
        bridge._observe_renderer_async(first)
        self.assertNotIn("_tokenize_prompt_async", second.__dict__)
        capture = bridge._CaptureRenderer(first)
        observer = bridge._StageObserver()
        token = bridge._ACTIVE_STAGE_OBSERVER.set(observer)
        try:
            result = asyncio.run(capture.render_chat([4, 2, 19]))
        finally:
            bridge._ACTIVE_STAGE_OBSERVER.reset(token)
        self.assertEqual(result[1], capture.engine_inputs)
        self.assertEqual(set(observer.durations), {"online_renderer", "tokenize_async", "template_async"})
        self.assertGreaterEqual(observer.durations["online_renderer"], observer.durations["tokenize_async"])
        previous = dict(observer.durations)
        with patch.object(bridge.time, "perf_counter_ns", side_effect=AssertionError("unexpected clock")):
            self.assertEqual(asyncio.run(capture.render_completion([7])), [{"prompt_token_ids": [7]}])
        self.assertEqual(observer.durations, previous)


if __name__ == "__main__":
    unittest.main()

"""Dependency-free entrypoint wiring checks, not vLLM rendering evidence.

The Rust extension is deliberately replaced only while loading the wrapper
under a private module name. Actual PyO3/GIL execution has separate Rust/native
integration coverage and must not be inferred from these boundary tests.
"""

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from vllm_router.router_args import RouterArgs


class _Policy:
    Random = "Random"
    RoundRobin = "RoundRobin"
    CacheAware = "CacheAware"
    PowerOfTwo = "PowerOfTwo"
    ConsistentHash = "ConsistentHash"
    KvAware = "KvAware"


class _NativeRouter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.start_calls = []

    def start(self, **kwargs):
        self.start_calls.append(kwargs)


def _wrapper():
    extension = types.ModuleType("vllm_router_rs")
    extension.PolicyType = _Policy
    extension.Router = _NativeRouter
    source = Path(__file__).resolve().parents[2] / "py_src/vllm_router/router.py"
    spec = importlib.util.spec_from_file_location("_kv_entrypoint_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"vllm_router_rs": extension}):
        spec.loader.exec_module(module)
    return module


def _facade():
    return types.SimpleNamespace(
        model="reviewed-model",
        tokenizer_path="/reviewed/assets/tokenizer.json",
        block_size=16,
        hash_algorithm="sha256_cbor",
        hash_seed=0,
        worker_urls=["http://127.0.0.1:8000", "http://127.0.0.1:8001"],
        contract_id="reviewed-contract",
        epoch=1,
        limits={"max_pending_jobs": 4, "execution_timeout_ms": 1000},
        worker_capabilities=False,
        capability_cohort=None,
    )


class TestKvRenderEntrypoint(unittest.TestCase):
    def test_exact_history_args_and_native_kwargs_are_opt_in(self):
        self.assertEqual(list(RouterArgs.__dataclass_fields__)[-3:], [
            "disable_circuit_breaker", "kv_fallback_policy", "kv_fallback_history_ttl_secs"])
        self.assertEqual(RouterArgs().kv_fallback_policy, "least_load")
        args = self.args(kv_fallback_policy="cache_aware", kv_fallback_history_ttl_secs=45,
                         max_tree_size=1024, cache_threshold=0.7, kv_load_guard=True)
        router = self.module.Router.from_args(args)
        for key in ("kv_fallback_policy", "kv_fallback_history_ttl_secs", "max_tree_size",
                    "cache_threshold", "kv_load_guard"):
            self.assertEqual(router._router.kwargs[key], getattr(args, key))
        for overrides in ({"kv_fallback_policy": "unknown"},
                          {"kv_fallback_policy": "cache_aware", "kv_fallback_history_ttl_secs": 0},
                          {"policy": "round_robin", "kv_fallback_policy": "cache_aware"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.args(**overrides)._validate_router_args()

    def test_exact_history_cli_round_trip(self):
        for prefix in ("", "router-"):
            parser = argparse.ArgumentParser()
            RouterArgs.add_cli_args(parser, use_router_prefix=bool(prefix))
            argv = [
                f"--{prefix}policy", "kv_aware",
                f"--{prefix}kv-tokenizer-path", "/public/tokenizer.json",
                f"--{prefix}kv-hash-algo", "sha256_cbor",
                f"--{prefix}kv-fallback-policy", "cache_aware",
                f"--{prefix}kv-fallback-history-ttl-secs", "45",
            ]
            if not prefix:
                argv.extend(["--worker-urls", "http://127.0.0.1:8000"])
            namespace = parser.parse_args(argv)
            args = RouterArgs.from_cli_args(namespace, use_router_prefix=bool(prefix))
            self.assertEqual(args.kv_fallback_policy, "cache_aware")
            self.assertEqual(args.kv_fallback_history_ttl_secs, 45)

    def setUp(self):
        self.module = _wrapper()
        self.facade = _facade()
        self.render_module = types.ModuleType("vllm_router.render_bridge")
        self.render_module.create_facade = Mock(return_value=self.facade)
        self.modules = patch.dict(sys.modules, {"vllm_router.render_bridge": self.render_module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def args(self, **kwargs):
        values = dict(
            worker_urls=list(self.facade.worker_urls), policy="kv_aware",
            kv_hash_algo="sha256_cbor", kv_input_backend="vllm",
            kv_render_config="/reviewed/deployment.json",
        )
        values.update(kwargs)
        return RouterArgs(**values)

    def test_native_defaults_do_not_load_optional_backend(self):
        router = self.module.Router.from_args(RouterArgs())
        router.start()
        self.render_module.create_facade.assert_not_called()
        self.assertNotIn("kv_model", router._router.kwargs)
        self.assertNotIn("kv_block_size", router._router.kwargs)
        self.assertEqual(router._router.start_calls, [{}])

    def test_existing_native_kv_options_are_forwarded(self):
        args = self.args(
            kv_input_backend="native", kv_render_config=None,
            kv_tokenizer_path="/local/tokenizer.json", kv_model="local-model",
            kv_block_size=32, kv_hash_seed=42, kv_events_topic_filter="kv",
            kv_events_port=6557, kv_events_endpoints=["http://127.0.0.1:8000=tcp://127.0.0.1:6557"],
            kv_index_max_entries=123,
        )
        router = self.module.Router.from_args(args)
        for name in ("kv_tokenizer_path", "kv_model", "kv_block_size", "kv_hash_seed",
                     "kv_hash_algo", "kv_events_topic_filter", "kv_events_port",
                     "kv_events_endpoints", "kv_index_max_entries"):
            self.assertEqual(router._router.kwargs[name], getattr(args, name))
        self.assertEqual(router._router.kwargs["policy"], _Policy.KvAware)
        self.render_module.create_facade.assert_not_called()

    def test_render_single_source_is_resolved_and_injected(self):
        args = self.args()
        router = self.module.Router.from_args(args)
        self.render_module.create_facade.assert_called_once_with(args.kv_render_config)
        self.assertEqual(router._router.kwargs["kv_tokenizer_path"], self.facade.tokenizer_path)
        self.assertEqual(router._router.kwargs["kv_model"], self.facade.model)
        self.assertEqual(router._router.kwargs["kv_block_size"], self.facade.block_size)
        self.assertNotIn("kv_render_config", router._router.kwargs)
        self.assertNotIn("kv_input_backend", router._router.kwargs)
        router.start()
        self.assertEqual(router._router.start_calls, [dict(
            render_facade=self.facade, render_contract_id=self.facade.contract_id,
            render_contract_epoch=1, render_limits=self.facade.limits,
        )])

    def test_from_args_does_not_mutate_caller_configuration(self):
        args = self.args()
        before = copy.deepcopy(vars(args))
        self.module.Router.from_args(args)
        self.assertEqual(vars(args), before)

    def test_load_guard_is_opt_in_and_forwarded(self):
        self.assertFalse(self.module.Router.from_args(self.args())._router.kwargs["kv_load_guard"])
        router = self.module.Router.from_args(self.args(kv_load_guard=True))
        self.assertTrue(router._router.kwargs["kv_load_guard"])
        with self.assertRaisesRegex(ValueError, "require policy=kv_aware"):
            RouterArgs(policy="round_robin", kv_load_guard=True)._validate_router_args()

    def test_completion_token_input_is_opt_in_and_requires_vllm(self):
        self.assertFalse(self.module.Router.from_args(self.args())._router.kwargs["kv_completion_token_input"])
        with self.assertRaisesRegex(ValueError, "requires automatic Worker capabilities"):
            self.module.Router.from_args(self.args(kv_completion_token_input=True))
        self.facade.capability_cohort = {
            "workers": {url: {} for url in self.facade.worker_urls}, "api_key_env": None}
        router = self.module.Router.from_args(self.args(kv_completion_token_input=True))
        self.assertTrue(router._router.kwargs["kv_completion_token_input"])
        with self.assertRaisesRegex(ValueError, "requires the vllm input backend"):
            self.module.Router(kv_completion_token_input=True)
        with self.assertRaisesRegex(ValueError, "requires the vllm input backend"):
            RouterArgs(kv_completion_token_input=True)._validate_router_args()

    def test_explicit_matching_overrides_are_permitted(self):
        self.module.Router.from_args(self.args(
            kv_model=self.facade.model, kv_tokenizer_path=self.facade.tokenizer_path,
            kv_block_size=self.facade.block_size,
        ))

    def test_conflicting_deployment_overrides_fail(self):
        for overrides in (
            {"kv_model": "another-model"},
            {"kv_tokenizer_path": "/another/tokenizer.json"},
            {"kv_block_size": 32},
            {"kv_hash_seed": 42},
            {"worker_urls": ["http://127.0.0.1:9000"]},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.module.Router.from_args(self.args(**overrides))

    def test_backend_mode_errors_fail_before_facade_creation(self):
        for overrides in (
            {"kv_input_backend": "unknown"}, {"kv_render_config": None},
            {"policy": "round_robin"}, {"mini_lb": True},
            {"vllm_pd_disaggregation": True}, {"service_discovery": True},
            {"intra_node_data_parallel_size": 2}, {"enable_igw": True},
            {"enable_program_scheduling": True},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.module.Router.from_args(self.args(**overrides))
        self.render_module.create_facade.assert_not_called()

    def test_legacy_hash_algorithm_remains_explicit(self):
        with self.assertRaisesRegex(ValueError, "kv_hash_algo"):
            self.module.Router.from_args(self.args(kv_hash_algo=None))

    def test_automatic_defaults_use_observed_values_and_pass_cohort_to_native(self):
        import json
        self.facade.worker_capabilities = True
        self.facade.hash_seed = 42
        self.facade.block_size = 32
        self.facade.capability_cohort = {
            "workers": {url: {} for url in self.facade.worker_urls}, "api_key_env": None}
        router = self.module.Router.from_args(self.args(kv_hash_algo=None))
        self.assertEqual(router._router.kwargs["kv_hash_seed"], 42)
        self.assertEqual(router._router.kwargs["kv_block_size"], 32)
        self.assertEqual(router._router.kwargs["kv_hash_algo"], "sha256_cbor")
        router.start()
        self.assertEqual(json.loads(router._router.start_calls[0]["kv_capabilities_json"]),
                         self.facade.capability_cohort)
        for override in ({"kv_hash_seed": 0}, {"kv_block_size": 16}, {"kv_model": "other"}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "requested=.*effective="):
                self.module.Router.from_args(self.args(**override))

    def test_automatic_cohort_preserves_registry_trailing_slash_keys(self):
        import json
        self.facade.worker_capabilities = True
        self.facade.capability_cohort = {
            "workers": {url: {} for url in self.facade.worker_urls}, "api_key_env": None}
        configured = [url + "/" for url in self.facade.worker_urls]
        router = self.module.Router.from_args(self.args(worker_urls=configured))
        router.start()
        cohort = json.loads(router._router.start_calls[0]["kv_capabilities_json"])
        self.assertEqual(set(cohort["workers"]), set(configured))
        self.assertEqual(set(self.facade.capability_cohort["workers"]), set(self.facade.worker_urls))

    def test_native_rejects_render_config_or_missing_tokenizer(self):
        for overrides in (
            {"kv_input_backend": "native"},
            {"kv_input_backend": "native", "kv_render_config": None},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.module.Router.from_args(self.args(**overrides))

    def test_stopped_render_facade_cannot_be_restarted(self):
        router = self.module.Router.from_args(self.args())
        router.start()
        with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
            router.start()
        self.assertEqual(len(router._router.start_calls), 1)

    def test_cli_and_prefixed_endpoint_mapping(self):
        for prefix in ("", "router-"):
            with self.subTest(prefix=prefix):
                parser = argparse.ArgumentParser()
                RouterArgs.add_cli_args(parser, use_router_prefix=bool(prefix))
                namespace = parser.parse_args([
                    f"--{prefix}policy", "kv_aware",
                    f"--{prefix}kv-input-backend", "vllm",
                    f"--{prefix}kv-render-config", "/reviewed/deployment.json",
                    f"--{prefix}kv-hash-algo", "sha256_cbor",
                    f"--{prefix}kv-events-endpoint", "http://w0:8000=tcp://w0:5557",
                    f"--{prefix}kv-events-endpoint", "http://w1:8000=tcp://w1:5558",
                ])
                args = RouterArgs.from_cli_args(namespace, use_router_prefix=bool(prefix))
                self.assertEqual(args.kv_input_backend, "vllm")
                self.assertEqual(args.kv_render_config, "/reviewed/deployment.json")
                self.assertEqual(len(args.kv_events_endpoints), 2)
                self.assertIsNone(args.kv_model)
                self.assertIsNone(args.kv_block_size)


if __name__ == "__main__":
    unittest.main()

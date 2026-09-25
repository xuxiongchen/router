"""Public startup/transport boundaries; synthetic metadata is NOT GPU proof."""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from test_render_bridge import RenderBridgeBoundaryTests, bridge


FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/kv_capabilities/descriptor.json"


def descriptor():
    return json.loads(FIXTURE.read_text())


class DescriptorTests(unittest.TestCase):
    def test_family_independent_and_optional_fields_ignored(self):
        original = descriptor()
        expected = bridge._capability_contract(original, "public-test")
        original["model_family_optional"] = "not-qwen"
        original["namespace"]["diagnostic_optional"] = "not a capability"
        original["hash"]["future_optional"] = True
        self.assertEqual(bridge._capability_contract(original, "public-test"), expected)

    def test_required_fields_units_and_semantics_fail_closed(self):
        changes = [
            ("schema_version", 2), ("mechanism", "custom_full_attention"),
            ("mechanism_version", 2), ("vllm_version", "0.30.0"),
            ("groups", []), ("groups", [descriptor()["groups"][0]] * 2),
            ("groups.0.kind", "sliding_window"), ("groups.0.layer_count", 0),
            ("groups.0.effective_block_tokens", 32), ("hash.block_tokens", 0),
            ("hash.width_bytes", 8), ("hash.root_hex", "0" * 64),
            ("hash.seed", 2**32), ("hash.seed", True), ("hash.extra_keys", "salt"),
            ("reuse.terminal_recompute_tokens", 0), ("execution.speculation", True),
            ("execution.offload", True), ("execution.connector", True),
            ("execution.dp", 2), ("execution.tp", 2), ("execution.dcp", 2),
            ("execution.prefix_caching", False), ("events.epoch", "old"),
            ("events.topic", "kv.wrong-epoch"), ("events.dp_rank", 1),
            ("events.next_sequence", -1), ("namespace.served_model_names", ["other"]),
        ]
        for path, replacement in changes:
            with self.subTest(path=path):
                value = descriptor()
                keys = path.split(".")
                target = value
                for key in keys[:-1]:
                    target = target[int(key)] if isinstance(target, list) else target[key]
                target[keys[-1]] = replacement
                with self.assertRaises(bridge.RenderConfigurationError):
                    bridge._capability_contract(value, "public-test")
        for key in descriptor():
            value = descriptor()
            del value[key]
            with self.subTest(missing=key), self.assertRaises(bridge.RenderConfigurationError):
                bridge._capability_contract(value, "public-test")


class AutomaticFacadeTests(RenderBridgeBoundaryTests):
    """Inherit legacy boundary regressions, adding the automatic opt-in path."""

    def automatic(self):
        self.config.pop("cache_layout", None)
        self.config["kv_capabilities"] = "worker"
        # This deliberately does not name any real model implementation.
        (self.root / "config.json").write_text(json.dumps({"model_type": "not-qwen"}))

    def test_family_gate_changes_only_with_verified_worker_metadata(self):
        (self.root / "config.json").write_text('{"model_type":"not-qwen"}')
        with self.assertRaisesRegex(bridge.RenderConfigurationError, "unsupported_cache_layout_model"):
            self.facade()
        self.automatic()
        with patch.object(bridge, "_remote_capabilities", return_value=descriptor()):
            facade = self.facade()
        self.assertTrue(facade.worker_capabilities)
        self.assertEqual((facade.block_size, facade.hash_seed), (16, 0))
        self.assertEqual(len(facade.capability_cohort["workers"]), 1)

    def test_manual_layout_and_automatic_source_are_mutually_exclusive(self):
        self.config["kv_capabilities"] = "worker"
        with self.assertRaisesRegex(bridge.RenderConfigurationError, "conflict"):
            self.facade()

    def test_replacement_during_conformance_is_not_admitted(self):
        self.automatic()
        after = descriptor()
        after["events"]["epoch"] = "f" * 32
        after["events"]["topic"] = "kv." + "f" * 32
        with patch.object(bridge, "_remote_capabilities", side_effect=[descriptor(), after]), \
             patch.object(bridge, "_remote_render", return_value=[4, 2, 19]), \
             patch.object(bridge, "_load_runtime", return_value=self.runtime), \
             patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": ""}):
            facade = self.facade()
            with self.assertRaises(bridge.RenderConfigurationError):
                facade.startup()
        self.assertEqual(facade.startup_failure["code"], "worker_changed_during_conformance")

    def test_http_control_plane_counters_and_request_exclusions(self):
        self.automatic()
        counts = {"metadata": 0, "render": 0}
        observed_auth = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                counts["metadata"] += 1
                observed_auth.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(descriptor()).encode())

            def do_POST(self):
                counts["render"] += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                value = {"token_ids": [4, 2, 19], "features": None, "cache_salt": None}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps([value] if "/completions/render" in self.path
                                           and "/chat/" not in self.path else value).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.config["worker_urls"] = [f"http://127.0.0.1:{server.server_port}"]
        self.config["worker_api_key_env"] = "CMB_SYNTHETIC_TEST_KEY"
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                    "VLLM_PLUGINS": "", "CMB_SYNTHETIC_TEST_KEY": "public-test-key"}), \
             patch.object(bridge, "_load_runtime", return_value=self.runtime) as loader:
            facade = self.facade()
            facade.startup()
            loader.assert_called_once_with(facade._argv, worker_capabilities=True)
            before = copy.deepcopy(counts)
            for _ in range(10):
                self.assertEqual(facade.render("completion", b'{"prompt":"hello"}')["status"], "exact")
            for key, extra in [("skip_reading_prefix_cache", True), ("cache_salt", "salt"),
                               ("lora_request", {}), ("multi_modal_data", {})]:
                raw = json.dumps({"prompt": "hello", key: extra}).encode()
                self.assertFalse(facade.render("completion", raw)["cache_eligible"])
        self.assertEqual(counts, before)
        self.assertEqual(counts["metadata"], 2)
        self.assertGreater(counts["render"], 0)
        self.assertEqual(observed_auth, ["Bearer public-test-key"] * 2)


class CapabilityTransportTests(unittest.TestCase):
    def test_absent_unsupported_invalid_and_redirect_are_distinct(self):
        from urllib.error import HTTPError
        for status, expected in [(404, "capabilities_endpoint_unavailable"),
                                 (409, "worker_mechanism_unsupported"),
                                 (401, "capabilities_authentication_failed")]:
            with self.subTest(status=status), patch.object(bridge, "build_opener") as opener:
                opener.return_value.open.side_effect = HTTPError("http://local", status, "", {}, None)
                with self.assertRaisesRegex(bridge.RenderConfigurationError, expected):
                    bridge._remote_capabilities("http://local", 1, None)
        with patch.object(bridge, "build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read1.side_effect = [b"not json", b""]
            with self.assertRaisesRegex(bridge.RenderConfigurationError, "invalid_worker_capabilities"):
                bridge._remote_capabilities("http://local", 1, None)
        with self.assertRaises(bridge.RenderConfigurationError):
            bridge._NoRedirect().redirect_request(None, None, 302, None, {}, "http://other")


if __name__ == "__main__":
    unittest.main()

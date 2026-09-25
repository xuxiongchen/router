"""Opt-in real non-Qwen vLLM CPU rendering through automatic capability mode.

Set CMB_CAPABILITIES_TEST_MODEL_DIR to a local config/tokenizer-only model asset
directory. The pinned SmolLM2-135M-Instruct assets are a test candidate, never a
production family/revision allowlist. No weights, GPU or inference engine run.

The official HTTP renderer is real vLLM 0.29; the capability descriptor is an
explicit controlled fixture. Thus this verifies automatic-path preprocessing
and family-gate removal, NOT the model's initialized Worker KV layout, generated
tokens, live KV events, cache hits or a hardware acceptance result.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from py_test.test_render_bridge_vllm import bridge, official_http_renderer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = os.environ.get("CMB_CAPABILITIES_TEST_MODEL_DIR")


def request_cases(model):
    base = {"model": model, "max_tokens": 1, "temperature": 0}
    messages = [{"role": "user", "content": "Explain café, 中文 and 🙂 briefly."}]
    for stream in (False, True):
        yield f"completion_stream_{stream}", "completion", {
            **base, "prompt": "Hello café 中文 🙂\n", "stream": stream,
        }
        yield f"chat_stream_{stream}", "chat", {
            **base, "messages": messages, "stream": stream,
        }
    yield "completion_token_ids", "completion", {
        **base, "prompt": [1, 223, 198, 500, 2],
    }
    yield "completion_truncate", "completion", {
        **base, "prompt": "one two three four five six seven", "truncate_prompt_tokens": 4,
    }
    yield "chat_system_history", "chat", {
        **base, "messages": [
            {"role": "system", "content": "Answer briefly."},
            *messages,
            {"role": "assistant", "content": "These are text examples."},
            {"role": "user", "content": "Continue."},
        ],
    }
    yield "chat_text_parts", "chat", {
        **base, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "中文"}, {"type": "text", "text": "\n🙂 café"},
        ]}],
    }
    yield "chat_null_template_kwargs", "chat", {
        **base, "messages": messages, "chat_template_kwargs": None,
    }
    yield "chat_json_schema", "chat", {
        **base, "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "answer", "schema": {"type": "object", "properties": {
                "text": {"type": "string"}}, "required": ["text"],
                "additionalProperties": False},
        }},
    }


@unittest.skipUnless(MODEL_DIR, "set CMB_CAPABILITIES_TEST_MODEL_DIR for real non-Qwen CPU render")
class NonQwenRealVllmCapabilitiesTests(unittest.TestCase):
    def test_actual_non_qwen_cpu_render_matches_official_http(self):
        model_dir = Path(MODEL_DIR).resolve()
        model_config = json.loads((model_dir / "config.json").read_text())
        # This assertion qualifies the test evidence; it is not admission logic.
        self.assertFalse(model_config["model_type"].startswith("qwen"))
        descriptor = json.loads((
            ROOT / "tests/fixtures/kv_capabilities/descriptor.json"
        ).read_text())
        model = "public-test"
        descriptor["namespace"]["served_model_names"] = [model]
        argv = ["--model", str(model_dir), "--tokenizer", str(model_dir),
                "--served-model-name", model, "--max-model-len", "2048",
                "--generation-config", "vllm"]
        reports = []
        real_remote_render = bridge._remote_render
        with official_http_renderer(argv) as url, tempfile.TemporaryDirectory(
            prefix="kv-capabilities-real-render-"
        ) as directory:
            config_path = Path(directory) / "render.json"
            configuration = {"serving_args": argv, "worker_urls": [url],
                             "kv_capabilities": "worker"}
            config_path.write_text(json.dumps(configuration))
            with patch.object(bridge, "_remote_capabilities", side_effect=(
                lambda *_args: copy.deepcopy(descriptor)
            )) as metadata_reads, patch.object(
                bridge, "_remote_render", wraps=real_remote_render
            ) as conformance_reads:
                facade = bridge.create_facade(config_path)
                try:
                    facade.startup()
                    self.assertTrue(facade.worker_capabilities)
                    self.assertIsNotNone(facade.capability_cohort)
                    self.assertTrue(facade.conformance)
                    self.assertEqual(metadata_reads.call_count, 2)
                    startup_render_reads = conformance_reads.call_count
                    self.assertGreater(startup_render_reads, 0)
                    for name, kind, request in request_cases(model):
                        with self.subTest(case=name):
                            raw = json.dumps(request, ensure_ascii=False,
                                             separators=(",", ":")).encode()
                            result = facade.render(kind, raw)
                            self.assertEqual(result["status"], "exact", result)
                            self.assertTrue(result["cache_eligible"])
                            # Explicit independent oracle HTTP calls are not
                            # Router request-time RPC. Bypass the counted facade
                            # hook to prove that hook is startup-only.
                            expected = real_remote_render(url, kind, raw, 10, None)
                            self.assertEqual(result["token_ids"], expected)
                            reports.append({"name": name, "kind": kind,
                                            "status": "PASS", "token_ids": expected,
                                            "request_sha256": hashlib.sha256(raw).hexdigest()})
                    negatives = [
                        ("salt", {"cache_salt": "not-supported"}),
                        ("adapter", {"lora_request": {"lora_int_id": 1}}),
                        ("read_disabled", {"skip_reading_prefix_cache": True}),
                        ("multimodal", {"messages": [{"role": "user", "content": [{
                            "type": "image_url", "image_url": {
                                "url": "http://must-not-fetch.invalid/image"}}]}]}),
                    ]
                    for name, extra in negatives:
                        raw = json.dumps({"model": model, "max_tokens": 1,
                                          "messages": [{"role": "user", "content": "x"}],
                                          **extra}).encode()
                        result = facade.render("chat", raw)
                        self.assertEqual(result["status"], "unsupported", (name, result))
                        self.assertFalse(result["cache_eligible"])
                        reports.append({"name": name, "status": "PASS_UNSUPPORTED",
                                        "reason": result["reason"]})
                    malformed = facade.render("chat", b'{"messages":"malformed"}')
                    self.assertEqual(malformed["status"], "invalid")
                    missing_model = facade.render("chat", json.dumps({
                        "model": "wrong-alias", "messages": [{"role": "user", "content": "x"}],
                    }).encode())
                    self.assertEqual(missing_model["http_status"], 404)
                    self.assertEqual(metadata_reads.call_count, 2)
                    self.assertEqual(conformance_reads.call_count, startup_render_reads)
                    template = facade._runtime.renderer.tokenizer.get_chat_template()
                    evidence = {
                        "classification": "CPU_RENDER_EQUIVALENCE_WITH_SYNTHETIC_DESCRIPTOR",
                        "worker_runtime_layout_verified": False,
                        "gpu_generation_verified": False,
                        "model_type": model_config["model_type"],
                        "model_directory": str(model_dir),
                        "effective_config": facade.effective_config,
                        "contract_id": facade.contract_id,
                        "conformance": facade.conformance,
                        "metadata_reads_startup": metadata_reads.call_count,
                        "remote_render_reads_startup": startup_render_reads,
                        "metadata_reads_per_request": 0,
                        "remote_render_reads_per_request": 0,
                        "explicit_http_oracle_requests": len(list(request_cases(model))),
                        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                        "asset_sha256": {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                                         for name in ("config.json", "tokenizer.json", "tokenizer_config.json")},
                        "cases": reports,
                        "schema_and_served_alias_validation": "PASS",
                    }
                    output = os.environ.get("CMB_CAPABILITIES_CPU_EVIDENCE")
                    if output:
                        Path(output).write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
                    print("CMB_CAPABILITIES_CPU_RENDER=" + json.dumps({
                        key: value for key, value in evidence.items()
                        if key not in ("cases", "conformance")
                    }, sort_keys=True))
                finally:
                    facade.close()
            # Same assets must still be rejected by the original legacy family
            # gate: only the verified-capability input path is generalized.
            legacy = dict(configuration)
            legacy.pop("kv_capabilities")
            legacy["cache_layout"] = {
                "kind": "qwen3_dense_full_attention", "block_size": 16,
                "hash_algorithm": "sha256_cbor", "hash_seed": 0,
            }
            legacy_path = Path(directory) / "legacy.json"
            legacy_path.write_text(json.dumps(legacy))
            with self.assertRaises(bridge.RenderConfigurationError):
                bridge.create_facade(legacy_path)


if __name__ == "__main__":
    unittest.main()

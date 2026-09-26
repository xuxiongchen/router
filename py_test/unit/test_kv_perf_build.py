"""Packaging switches, without compiling or invoking setuptools side effects."""

import os
from pathlib import Path
import runpy
import types
import unittest
from unittest.mock import patch

SETUP = Path(__file__).resolve().parents[2] / "setup.py"


class PerfBuildTests(unittest.TestCase):
    def configure(self, *, benchmark=False, no_rust=False):
        calls = []
        setuptools = types.SimpleNamespace(setup=lambda **kwargs: calls.append(kwargs))
        rust = types.SimpleNamespace(
            Binding=types.SimpleNamespace(PyO3="pyo3"),
            RustExtension=lambda **kwargs: kwargs,
        )
        with patch.dict(
            os.environ,
            {
                "VLLM_ROUTER_BUILD_KV_PERF": "1" if benchmark else "0",
                "VLLM_ROUTER_BUILD_NO_RUST": "1" if no_rust else "0",
            },
        ), patch.dict(
            "sys.modules", {"setuptools": setuptools, "setuptools_rust": rust}
        ):
            runpy.run_path(str(SETUP))
        return calls[0]["rust_extensions"]

    def test_production_default_has_no_benchmark_feature(self):
        (extension,) = self.configure()
        self.assertEqual(extension["features"], [])
        self.assertEqual(extension["args"], ["--locked"])

    def test_benchmark_requires_explicit_build_opt_in(self):
        (extension,) = self.configure(benchmark=True)
        self.assertEqual(extension["features"], ["kv-perf"])
        self.assertEqual(extension["args"], ["--locked"])

    def test_python_only_build_remains_available(self):
        self.assertEqual(self.configure(no_rust=True), [])

    def test_benchmark_cannot_silently_build_without_native(self):
        with self.assertRaisesRegex(RuntimeError, "requires the native extension"):
            self.configure(benchmark=True, no_rust=True)


if __name__ == "__main__":
    unittest.main()

import os

from setuptools import setup

no_rust = os.environ.get("VLLM_ROUTER_BUILD_NO_RUST") == "1"
# A separate diagnostic artifact, not an additional production policy. The
# native handshake and startup guard also require explicit runtime opt-in.
kv_perf = os.environ.get("VLLM_ROUTER_BUILD_KV_PERF") == "1"
if kv_perf and no_rust:
    raise RuntimeError("kv-perf requires the native extension build")

rust_extensions = []
if not no_rust:
    from setuptools_rust import Binding, RustExtension

    rust_extensions.append(
        RustExtension(
            target="vllm_router_rs",
            path="Cargo.toml",
            binding=Binding.PyO3,
            features=["kv-perf"] if kv_perf else [],
            args=["--locked"],
        )
    )

setup(
    rust_extensions=rust_extensions,
    zip_safe=False,
)

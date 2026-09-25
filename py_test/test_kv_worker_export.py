"""Source-executed linked vLLM proposal tests; no installed Worker is patched.

Set VLLM_KV_PROPOSAL_SOURCE to the proposal's modified tree. These CPU tests
construct real initialized vLLM KV managers/specs without loading model weights.
They do not claim GPU EngineCore initialization or actual hardware acceptance.
"""

import ast
import importlib.util
import inspect
import os
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _method(path, class_name, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    source = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[name]


def _capture_inventory(proposal, engine, configs):
    """Execute the exact opt-in initialization block without constructing GPU engines."""
    path = proposal.root / "v1/engine/core.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_initialize_kv_caches"
    )
    capture = next(
        n
        for n in method.body
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.Attribute)
            and child.attr == "_kv_capability_source_groups"
            for child in ast.walk(n)
        )
    )
    module = ast.fix_missing_locations(ast.Module(body=[capture], type_ignores=[]))
    exec(
        compile(module, str(path), "exec"),
        {
            "self": engine,
            "vllm_config": engine.vllm_config,
            "kv_cache_configs": configs,
        },
    )


@pytest.fixture
def proposal(monkeypatch):
    path = os.environ.get("VLLM_KV_PROPOSAL_SOURCE")
    if not path:
        pytest.skip("set VLLM_KV_PROPOSAL_SOURCE for linked Worker proposal tests")
    root = Path(path) / "vllm"
    # Import installed dependencies before replacing only the selected modules
    # in this test interpreter. No on-disk installed source changes occur.
    import vllm.config.kv_events
    import vllm.distributed.kv_events
    import vllm.v1.core.sched.scheduler
    import vllm.v1.core.sched.async_scheduler

    config = _load("vllm.config.kv_events", root / "config/kv_events.py", monkeypatch)
    publisher = _load(
        "vllm.distributed.kv_events", root / "distributed/kv_events.py", monkeypatch
    )
    monkeypatch.setattr(vllm.config, "kv_events", config)
    monkeypatch.setattr(vllm.distributed, "kv_events", publisher)
    exporter = _load(
        "vllm.v1.engine.kv_capabilities",
        root / "v1/engine/kv_capabilities.py",
        monkeypatch,
    )
    api = _load(
        "cmb_proposal_api",
        root / "entrypoints/serve/kv_capabilities/api_router.py",
        monkeypatch,
    )
    return SimpleNamespace(
        root=root, config=config, publisher=publisher, exporter=exporter, api=api
    )


@pytest.fixture
def runtime(proposal, monkeypatch):
    import torch
    from vllm.utils.hashing import sha256_cbor
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        get_kv_cache_spec_kind,
    )

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", sha256_cbor("0"), raising=False)
    monkeypatch.setattr(kv_cache_utils, "_NONE_HASH_SEED", "0")
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    group = KVCacheGroupSpec(layer_names=["layer.0", "layer.1"], kv_cache_spec=spec)
    kv_config = KVCacheConfig(
        num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[group]
    )
    cache = KVCacheManager(
        kv_cache_config=kv_config,
        max_model_len=1024,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_kv_cache_events=True,
    )
    publisher = proposal.publisher.ZmqEventPublisher(
        0, endpoint="tcp://127.0.0.1:*", topic="kv", enable_capabilities=True
    )
    config = SimpleNamespace(
        kv_events_config=SimpleNamespace(enable_capabilities=True),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            kv_offloading_size=None,
            prefix_caching_hash_algo="sha256_cbor",
            cache_dtype="auto",
        ),
        speculative_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        lora_config=None,
        weight_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            data_parallel_index=0,
        ),
        model_config=SimpleNamespace(
            model="family-independent",
            served_model_name=["model"],
            revision=None,
            dtype=torch.bfloat16,
            quantization=None,
            runner_type="generate",
            is_encoder_decoder=False,
            is_diffusion=False,
            multimodal_config=None,
        ),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = cache
    scheduler.kv_cache_config = kv_config
    scheduler.block_size = scheduler.hash_block_size = 16
    scheduler.enable_kv_cache_events = True
    scheduler.use_eagle = False
    scheduler.num_prefill_lookahead = 0
    scheduler.dcp_world_size = scheduler.pcp_world_size = 1
    scheduler.connector = scheduler.ec_connector = None
    scheduler.kv_event_publisher = publisher
    engine = SimpleNamespace(
        vllm_config=config,
        scheduler=scheduler,
        use_spec_decode=False,
        _weight_version="default",
        request_block_hasher=kv_cache_utils.get_request_block_hasher(16, sha256_cbor),
    )
    _capture_inventory(proposal, engine, [kv_config])
    for name in ("get_kv_cache_group_metadata", "get_kv_cache_capabilities"):
        function = _method(
            proposal.root / "v1/engine/core.py",
            "EngineCore",
            name,
            {"Any": Any, "get_kv_cache_spec_kind": get_kv_cache_spec_kind},
        )
        setattr(engine, name, MethodType(function, engine))
    try:
        yield SimpleNamespace(
            engine=engine,
            config=config,
            scheduler=scheduler,
            cache=cache,
            spec=spec,
            group=group,
            publisher=publisher,
        )
    finally:
        publisher.shutdown()


def test_real_initialized_manager_export_is_family_independent(runtime):
    first = runtime.engine.get_kv_cache_capabilities()
    assert "unsupported" not in first
    runtime.config.model_config.model = "non-qwen-other-family"
    second = runtime.engine.get_kv_cache_capabilities()
    assert first["groups"] == second["groups"]
    assert first["mechanism"] == second["mechanism"] == "normal_full_attention"
    assert (
        first["hash"]["root_hex"]
        == "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e"
    )
    assert first["events"]["topic"].endswith("." + first["events"]["epoch"])
    assert first["events"]["resolved_endpoint"].startswith("tcp://127.0.0.1:")
    assert "*" not in first["events"]["resolved_endpoint"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.scheduler.kv_cache_config.kv_cache_groups.clear(),
        lambda x: x.group.layer_names.clear(),
        lambda x: setattr(x.group, "is_eagle_group", True),
        lambda x: object.__setattr__(x.spec, "sliding_window", 128),
        lambda x: object.__setattr__(x.spec, "attention_chunk_size", 128),
        lambda x: object.__setattr__(x.spec, "non_causal", True),
        lambda x: setattr(x.config, "speculative_config", object()),
        lambda x: setattr(x.scheduler, "num_prefill_lookahead", 1),
        lambda x: setattr(x.config.cache_config, "kv_offloading_size", 1),
        lambda x: setattr(x.scheduler, "connector", object()),
        lambda x: setattr(x.config.parallel_config, "decode_context_parallel_size", 2),
        lambda x: setattr(x.config.parallel_config, "data_parallel_size", 2),
        lambda x: setattr(x.scheduler, "hash_block_size", 8),
        lambda x: setattr(x.cache, "enable_caching", False),
        lambda x: setattr(x.engine, "_weight_version", "updated"),
        lambda x: setattr(x.engine, "request_block_hasher", lambda _: []),
        lambda x: setattr(x.config, "lora_config", object()),
        lambda x: setattr(x.config.model_config, "multimodal_config", object()),
    ],
)
def test_unsupported_runtime_facts(runtime, mutation):
    mutation(runtime)
    assert "unsupported" in runtime.engine.get_kv_cache_capabilities()


@pytest.mark.parametrize("target", ["manager", "scheduler", "spec"])
def test_custom_subclasses_do_not_inherit_capability(runtime, target):
    if target == "manager":
        obj = runtime.cache.coordinator.single_type_managers[0]
    elif target == "scheduler":
        obj = runtime.scheduler
    else:
        obj = runtime.spec
    cls = type(
        type(obj).__name__, (type(obj),), {}
    )  # even an identical name is rejected
    object.__setattr__(obj, "__class__", cls)
    assert "unsupported" in runtime.engine.get_kv_cache_capabilities()


@pytest.mark.parametrize("seed", ["vllm-none-hash", "4294967296", "-1", "00"])
def test_unsupported_seed_not_reinterpreted(runtime, monkeypatch, seed):
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setattr(kv_cache_utils, "_NONE_HASH_SEED", seed)
    assert "unsupported" in runtime.engine.get_kv_cache_capabilities()


def test_full_width_identity_required(runtime, monkeypatch):
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    assert "unsupported" in runtime.engine.get_kv_cache_capabilities()
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", kv_cache_utils.NONE_HASH[:8])
    assert "unsupported" in runtime.engine.get_kv_cache_capabilities()


def test_export_does_not_touch_allocator_or_increment_sequence(runtime):
    before = runtime.cache.block_pool.get_num_free_blocks()
    first = runtime.engine.get_kv_cache_capabilities()
    for _ in range(4):
        assert runtime.engine.get_kv_cache_capabilities() == first
    assert runtime.cache.block_pool.get_num_free_blocks() == before
    assert first["events"]["next_sequence"] == 0


def test_stopped_publisher_cannot_grant_capability(runtime):
    runtime.publisher._running = False
    try:
        assert "unsupported" in runtime.engine.get_kv_cache_capabilities()
    finally:
        runtime.publisher._running = True


def test_incomplete_inventory_and_publisher_rank(runtime):
    original = runtime.engine.get_kv_cache_group_metadata
    runtime.engine.get_kv_cache_group_metadata = lambda: []
    assert "unsupported" in runtime.engine.get_kv_cache_capabilities()
    runtime.engine.get_kv_cache_group_metadata = original
    runtime.publisher._dp_rank = 1
    assert "unsupported" in runtime.engine.get_kv_cache_capabilities()


@pytest.mark.parametrize(
    "hidden", ["ordinary", "window", "custom", "wrong-unit", "missing"]
)
def test_original_uniform_inventory_survives_scheduler_collapse(
    proposal, runtime, hidden
):
    from dataclasses import replace
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
    from vllm.v1.kv_cache_interface import (
        UniformTypeKVCacheSpecs,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    first = runtime.spec
    second = replace(first, head_size=128)
    if hidden == "window":
        second = replace(second, sliding_window=128)
    elif hidden == "wrong-unit":
        second = replace(second, block_size=32)
    elif hidden == "custom":
        second = type("FullAttentionSpec", (type(first),), {})(
            block_size=16, num_kv_heads=2, head_size=128, dtype=first.dtype
        )
    leaf_map = {"layer.0": first, "layer.1": second}
    if hidden == "missing":
        leaf_map.pop("layer.1")
    wrapped = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=leaf_map)
    original = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=runtime.group.layer_names, kv_cache_spec=wrapped
            )
        ],
    )
    normalized = generate_scheduler_kv_cache_config([original])
    # Actual pinned normalization retains only the first leaf: the scheduler's
    # reported full_attention kind is insufficient to classify the entire group.
    assert type(normalized.kv_cache_groups[0].kv_cache_spec) is type(first)
    _capture_inventory(proposal, runtime.engine, [original])
    result = runtime.engine.get_kv_cache_capabilities()
    assert ("unsupported" not in result) == (hidden == "ordinary")


def test_fixed_authenticated_http_to_utility_chain(proposal, runtime):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

    calls = []

    async def utility(name, *args):
        calls.append((name, args))
        return runtime.engine.get_kv_cache_capabilities()

    method = _method(
        proposal.root / "v1/engine/async_llm.py",
        "AsyncLLM",
        "get_kv_cache_capabilities",
        {"Any": Any},
    )
    client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
    client.get_kv_cache_capabilities = MethodType(method, client)
    app = FastAPI()
    app.state.vllm_config = runtime.config
    app.state.engine_client = client
    app.add_middleware(AuthenticationMiddleware, tokens=["test-only-key"])
    proposal.api.attach_router(app)
    with TestClient(app) as http:
        path = "/v1/kv-cache/capabilities"
        assert http.get(path).status_code == 401
        assert calls == []
        headers = {"Authorization": "Bearer test-only-key"}
        runtime.config.kv_events_config.enable_capabilities = False
        assert http.get(path, headers=headers).status_code == 404
        assert calls == []
        runtime.config.kv_events_config.enable_capabilities = True
        response = http.get(path, headers=headers)
        assert response.status_code == 200
        assert response.json()["mechanism"] == "normal_full_attention"
        assert calls == [("get_kv_cache_capabilities", ())]
        runtime.scheduler.connector = object()
        assert http.get(path, headers=headers).status_code == 409


def test_publisher_default_and_custom_constructor_compatibility(proposal):
    module = proposal.publisher
    publisher = module.ZmqEventPublisher(0, endpoint="tcp://127.0.0.1:*", topic="old")
    try:
        assert publisher._topic_bytes == b"old"
        with pytest.raises(ValueError):
            publisher.get_capability_binding()
    finally:
        publisher.shutdown()
    args_seen = []

    def custom(**kwargs):
        args_seen.append(kwargs)
        return object()

    module.EventPublisherFactory._registry["test-custom"] = custom
    config = proposal.config.KVEventsConfig(enable_kv_cache_events=True)
    config.publisher = "test-custom"
    module.EventPublisherFactory.create(config)
    assert "enable_capabilities" not in args_seen[0]
    config.enable_capabilities = True
    with pytest.raises(ValueError):
        module.EventPublisherFactory.create(config)


def test_actual_loopback_wire_epoch_and_watermark(proposal, runtime):
    import zmq

    publisher = runtime.publisher
    binding = publisher.get_capability_binding()
    subscriber = zmq.Context.instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, binding["topic"].encode())
    subscriber.connect(binding["resolved_endpoint"])
    try:
        # Bounded slow-joiner handshake: no engine/allocator mutation. Each
        # synthetic wire batch has a real sequence from the actual publisher.
        received = None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and received is None:
            publisher.publish(
                proposal.publisher.KVEventBatch(ts=time.time(), events=[])
            )
            if subscriber.poll(100):
                received = subscriber.recv_multipart()
        assert received is not None and len(received) == 3
        assert received[0] == binding["topic"].encode()
        sequence = int.from_bytes(received[1], "big")
        assert publisher.get_capability_binding()["next_sequence"] > sequence
        replacement = proposal.publisher.ZmqEventPublisher(
            0, endpoint="tcp://127.0.0.1:*", topic="kv", enable_capabilities=True
        )
        try:
            assert replacement.get_capability_binding()["epoch"] != binding["epoch"]
            assert replacement.get_capability_binding()["next_sequence"] == 0
        finally:
            replacement.shutdown()
    finally:
        subscriber.close(linger=0)


def test_public_route_has_no_rpc_selector(proposal):
    assert list(
        inspect.signature(proposal.api.get_kv_cache_capabilities).parameters
    ) == ["request"]
    assert (
        "get_kv_cache_capabilities"
        in (proposal.root / "engine/protocol.py").read_text()
    )
    assert (
        "attach_capabilities_router(app)"
        in (proposal.root / "entrypoints/serve/__init__.py").read_text()
    )

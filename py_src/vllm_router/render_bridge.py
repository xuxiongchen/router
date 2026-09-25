"""Optional, engine-free adapter to the installed vLLM 0.29 render service.

This module is deliberately importable without vLLM. ``startup``, ``render``
and ``close`` belong to ONE dedicated Rust executor thread, not a Tokio worker
or the Python CLI thread. The executor owns admission, deadlines and cancellation;
in particular, timing out a caller must not interrupt/release a running call.

No templates or tokenization algorithms are implemented here. The version-
specific integration is ``_load_runtime``; it follows vLLM's CPU render launcher
and calls its public ServingRender methods, including model/sampling validation.
Engine health and inference-engine checks remain the generation worker's job.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


VLLM_VERSION = "0.29.0"
_LAYOUT = "qwen3_dense_full_attention"
_LIMIT_DEFAULTS = {
    "max_pending_jobs": 32,
    "max_input_bytes": 1_048_576,
    "max_tokens_per_request": 65_536,
    "max_reserved_tokens": 262_144,
    "queue_timeout_ms": 1000,
    "execution_timeout_ms": 10_000,
}
# Only preprocessing arguments belong in this reviewed source. No arbitrary
# plugin loading, inference-engine flags, remote-code trust or media fetching.
_VALUE_ARGS = {
    "--model", "--tokenizer", "--served-model-name", "--revision",
    "--tokenizer-revision", "--tokenizer-mode", "--max-model-len", "--dtype",
    "--chat-template-content-format", "--default-chat-template-kwargs",
    "--tool-call-parser", "--reasoning-parser", "--generation-config",
    "--override-generation-config",
}
_BOOL_ARGS = {"--enable-auto-tool-choice", "--exclude-tools-when-tool-choice-none"}
_UNSAFE_REQUEST_KEYS = {
    "cache_salt", "kv_transfer_params", "lora_request", "lora_path",
    "session_params", "prompt_embeds", "mm_processor_kwargs",
    "multi_modal_data", "multi_modal_uuids",
}
_ENGINE_KEYS = {
    "type", "prompt_token_ids", "prompt", "prompt_token_offsets",
    "assistant_tokens_mask", "arrival_time",
}
_CONTENT_FREE_LOGGERS = (
    "vllm.renderers.hf",
    "vllm.entrypoints.chat_utils",
    "vllm.entrypoints.scale_out.render.serving",
)


class _ContentFreeRenderLog(logging.Filter):
    """Bound known v0.29 diagnostics, including renderer-owned worker threads.

    These module loggers are shared process-wide: while a facade is active,
    other consumers of precisely these loggers also receive content-free
    diagnostics. Do not disable global logging or alter another logger's level.
    """

    def filter(self, record):
        record.msg = "cmb_render_diagnostic"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        record.message = "cmb_render_diagnostic"
        return True


class RenderConfigurationError(ValueError):
    """A fixed, content-free code suitable for a startup error."""


def _require(condition, code):
    if not condition:
        raise RenderConfigurationError(code)


def _config_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_configuration_key")
        result[key] = value
    return result


def _read_json(path):
    try:
        return json.loads(path.read_bytes(), object_pairs_hook=_config_object)
    except (OSError, UnicodeError, ValueError):
        raise RenderConfigurationError("invalid_local_configuration") from None


def _serving_options(argv):
    _require(isinstance(argv, list) and all(isinstance(v, str) for v in argv),
             "serving_args_must_be_argv")
    result = {}
    offset = 0
    while offset < len(argv):
        key = argv[offset]
        _require(key not in result, "duplicate_serving_option")
        if key in _BOOL_ARGS:
            result[key] = True
            offset += 1
        else:
            _require(key in _VALUE_ARGS and offset + 1 < len(argv),
                     "unsupported_serving_option")
            result[key] = argv[offset + 1]
            offset += 2
    for key in ("--model", "--served-model-name"):
        _require(bool(result.get(key)), "missing_serving_identity")
    _require(result.get("--generation-config", "auto") in ("auto", "vllm"),
             "external_generation_config_unsupported")
    return result


def _local_directory(value):
    path = Path(value)
    _require(path.is_absolute() and path.is_dir(), "assets_must_be_local_directory")
    return path.resolve()


def _asset_files(model_dir, tokenizer_dir):
    """Hash local configuration/tokenizer assets, never weights or a cache clone."""
    paths = set()
    for directory in {model_dir, tokenizer_dir}:
        for pattern in ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"):
            paths.update(p for p in directory.glob(pattern) if p.is_file())
        templates = directory / "chat_templates"
        if templates.exists():
            paths.update(p for p in templates.rglob("*") if p.is_file())
    for path in paths:
        _require(path.stat().st_size <= 128 * 1024 * 1024,
                 "oversized_preprocessing_asset")
    return sorted(paths)


def _validate_layout(model):
    # Input rendering support is broader than the PR1 event matcher. Do not
    # admit hybrid/MoE/MM merely because vLLM can return their token IDs.
    _require(isinstance(model, dict) and model.get("model_type") == "qwen3"
             and model.get("architectures") == ["Qwen3ForCausalLM"],
             "unsupported_cache_layout_model")
    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size",
                "vision_config", "text_config", "mamba_d_state",
                "hybrid_layer_pattern", "attention_chunk_size", "sliding_window"):
        _require(model.get(key) is None, "unsupported_cache_layout_model")
    _require(model.get("use_sliding_window") in (None, False),
             "unsupported_cache_layout_model")
    layers = model.get("layer_types")
    _require(layers is None or (isinstance(layers, list) and layers
                               and all(v == "full_attention" for v in layers)),
             "unsupported_cache_layout_model")


def _stat_signature(paths):
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def _validate_template_determinism(tokenizer_dir):
    # Static inspection only; vLLM/Transformers still own compilation/execution.
    # Unknown syntax is outside this v1 contract, not silently considered safe.
    from jinja2 import Environment, nodes

    config = _read_json(tokenizer_dir / "tokenizer_config.json")
    configured = config.get("chat_template")
    templates = []
    if isinstance(configured, str):
        templates.append(configured)
    elif isinstance(configured, dict):
        templates.extend(configured.values())
    elif isinstance(configured, list):
        templates.extend(item.get("template") for item in configured if isinstance(item, dict))
    templates.extend(p.read_text() for p in tokenizer_dir.glob("*.jinja"))
    if (tokenizer_dir / "chat_templates").is_dir():
        templates.extend(p.read_text() for p in (tokenizer_dir / "chat_templates").rglob("*.jinja"))
    environment = Environment(extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"])
    for template in templates:
        _require(isinstance(template, str), "invalid_template_asset")
        tree = environment.parse(template)
        _require(not any(node.name == "random" for node in tree.find_all(nodes.Filter))
                 and not any(node.name == "strftime_now" for node in tree.find_all(nodes.Name)),
                 "nondeterministic_template")


class _CaptureRenderer:
    """Observe EngineInput without replacing any vLLM preprocessing operation."""

    def __init__(self, renderer):
        self.inner = renderer
        self.engine_inputs = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def render_chat(self, request, **kwargs):
        result = await self.inner.render_chat(request, **kwargs)
        if isinstance(result, tuple):
            self.engine_inputs = result[1]
        return result

    async def render_completion(self, request, **kwargs):
        result = await self.inner.render_completion(request, **kwargs)
        if isinstance(result, list):
            self.engine_inputs = result
        return result


def _load_runtime(argv):
    """vLLM 0.29's launchers/render/entry.py and app_state.py, without HTTP.

    The same official CLI parser resolves defaults; the same EngineArgs builds
    ModelConfig. As in the official CPU-only launcher, clear quantization before
    VllmConfig validation: no quantized kernels, weights or KV cache are needed.
    Do not instantiate EngineClient or call plugin/derender app initialization.
    """
    _require(importlib.metadata.version("vllm").split("+")[0] == VLLM_VERSION,
             "unsupported_vllm_version")
    from vllm import AsyncEngineArgs
    from vllm.config import VllmConfig
    from vllm.entrypoints.launchers.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
    from vllm.entrypoints.scale_out.render.serving import ServingRender
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
    from vllm.renderers import renderer_from_config
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from pydantic import ValidationError

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)
    _validate_template_determinism(Path(args.tokenizer or args.model))
    model_config = AsyncEngineArgs.from_cli_args(args).create_model_config()
    _require(not model_config.trust_remote_code, "remote_code_not_supported")
    _validate_layout(model_config.hf_config.to_dict())
    model_config.quantization = None
    config = VllmConfig(model_config=model_config)
    renderer = renderer_from_config(config)
    try:
        online = OnlineRenderer(
            model_config=model_config, renderer=renderer, request_logger=None,
            chat_template=None,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=False,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser, reasoning_parser=args.reasoning_parser,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
            log_error_stack=False,
        )
        online.warmup()
        capture = _CaptureRenderer(online)
        models = OpenAIModelRegistry(
            model_config=model_config,
            base_model_paths=[BaseModelPath(name=name, model_path=args.model)
                              for name in (args.served_model_name or [args.model])],
        )
        serving = ServingRender(models, capture, request_logger=None)
        return SimpleNamespace(
            serving=serving, capture=capture, renderer=renderer,
            schemas={"chat": ChatCompletionRequest, "completion": CompletionRequest},
            error_type=ErrorResponse, validation_type=ValidationError,
            effective={
                "model": model_config.model, "tokenizer": model_config.tokenizer,
                "tokenizer_mode": model_config.tokenizer_mode,
                "revision": model_config.revision,
                "tokenizer_revision": model_config.tokenizer_revision,
                "max_model_len": model_config.max_model_len,
                "renderer": type(renderer).__module__ + "." + type(renderer).__name__,
                "content_format": args.chat_template_content_format,
                "default_chat_template_kwargs": args.default_chat_template_kwargs,
                "tool_parser": args.tool_call_parser,
                "reasoning_parser": args.reasoning_parser,
            },
        )
    except BaseException:
        renderer.shutdown()
        raise


def _request_cache_reason(request, raw_object, kind):
    # Keep every valid schema field for ServingRender. Unknown top-level fields
    # could introduce future cache identity; they are not silently neutralized.
    if getattr(request, "model_extra", None):
        return "unknown_request_fields"
    if any(raw_object.get(k) is not None for k in _UNSAFE_REQUEST_KEYS):
        return "unsupported_cache_identity"
    if raw_object.get("use_beam_search"):
        # Generation supports beams, the official Chat Render API does not.
        return "beam_render_not_supported"
    if raw_object.get("chat_template") is not None:
        return "untrusted_request_template"
    kwargs = raw_object.get("chat_template_kwargs") or {}
    if isinstance(kwargs, dict) and kwargs.get("chat_template") is not None:
        return "untrusted_request_template"
    if kind == "completion":
        prompt = raw_object.get("prompt")
        if isinstance(prompt, list) and (not prompt or not all(type(v) is int for v in prompt)):
            return "batched_prompt"
    else:
        for message in raw_object.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                if any(not isinstance(part, dict) or part.get("type") != "text"
                       for part in content):
                    return "non_text_input"
    return None


class RenderFacade:
    """Single reviewed cohort, one loop, one renderer, one contract epoch."""

    def __init__(self, config_path):
        self._config_path = Path(config_path).resolve()
        config = _read_json(self._config_path)
        _require(isinstance(config, dict) and not set(config) - {
            "serving_args", "worker_urls", "cache_layout", "bridge_limits",
            "worker_api_key_env", "conformance_timeout_seconds",
        }, "unsupported_render_configuration")
        self._argv = config.get("serving_args")
        options = _serving_options(self._argv)
        model_dir = _local_directory(options["--model"])
        tokenizer_dir = _local_directory(options.get("--tokenizer", str(model_dir)))
        _require((tokenizer_dir / "tokenizer.json").is_file(), "missing_local_tokenizer")
        tokenizer_json = _read_json(tokenizer_dir / "tokenizer.json")
        _require(isinstance(tokenizer_json, dict)
                 and isinstance(tokenizer_json.get("model"), dict)
                 and tokenizer_json["model"].get("dropout") in (None, 0),
                 "nondeterministic_tokenizer")
        self.model = options["--served-model-name"]
        self.tokenizer_path = str(tokenizer_dir / "tokenizer.json")
        _validate_layout(_read_json(model_dir / "config.json"))
        layout = config.get("cache_layout", {})
        _require(isinstance(layout, dict) and set(layout) == {
            "kind", "block_size", "hash_algorithm", "hash_seed"
        }, "invalid_cache_layout_contract")
        _require(layout["kind"] == _LAYOUT
                 and layout["hash_algorithm"] == "sha256_cbor"
                 and type(layout["hash_seed"]) is int and 0 <= layout["hash_seed"] < 2**32
                 and type(layout["block_size"]) is int and layout["block_size"] > 0,
                 "unsupported_cache_layout_contract")
        self.block_size = layout["block_size"]
        self.hash_algorithm = layout["hash_algorithm"]
        self.hash_seed = layout["hash_seed"]
        urls = config.get("worker_urls")
        _require(isinstance(urls, list) and urls and all(isinstance(u, str) for u in urls),
                 "missing_conformance_workers")
        for url in urls:
            parsed = urlsplit(url)
            _require(parsed.scheme in ("http", "https") and parsed.hostname
                     and not parsed.username and not parsed.password
                     and parsed.path in ("", "/") and not parsed.query and not parsed.fragment,
                     "invalid_conformance_worker_url")
        self.worker_urls = tuple(u.rstrip("/") for u in urls)
        _require(len(set(self.worker_urls)) == len(self.worker_urls), "duplicate_worker_url")
        limits = config.get("bridge_limits", {})
        _require(isinstance(limits, dict) and not set(limits) - set(_LIMIT_DEFAULTS),
                 "invalid_bridge_limits")
        self.limits = {**_LIMIT_DEFAULTS, **limits}
        _require(all(type(v) is int and 0 < v <= 2**32 - 1 for v in self.limits.values())
                 and self.limits["max_reserved_tokens"] >= self.limits["max_tokens_per_request"],
                 "invalid_bridge_limits")
        self._timeout = config.get("conformance_timeout_seconds", 10)
        _require(type(self._timeout) in (int, float) and 0 < self._timeout <= 60,
                 "invalid_conformance_timeout")
        self._api_key_env = config.get("worker_api_key_env")
        _require(self._api_key_env is None or isinstance(self._api_key_env, str),
                 "invalid_worker_api_key_env")
        self._asset_directories = (model_dir, tokenizer_dir)
        self._paths = [self._config_path, *_asset_files(model_dir, tokenizer_dir)]
        assets = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self._paths}
        # Templates using clock/random globals cannot establish deterministic
        # prefix identity. No arbitrary user template/plugin override is enabled.
        for path in self._paths:
            if path.name == "tokenizer_config.json" or path.suffix == ".jinja":
                content = path.read_text()
                _require("strftime_now" not in content and not re.search(r"\|\s*random\b", content),
                         "nondeterministic_template")
        identity = {"adapter": 1, "vllm": VLLM_VERSION, "configuration": config,
                    "assets": assets}
        self.contract_id = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
        self.epoch = 1
        self._signature = _stat_signature(self._paths)
        self._thread = None
        self._loop = None
        self._runtime = None
        self._ready = False
        self._closed = False
        self._invalidated = False
        self._identity_changed = False
        self.effective_config = None
        self.conformance = []
        self.startup_failure = None
        self._log_filters = []

    def _result(self, status, reason, **extra):
        return {"status": status, "reason": reason, "cache_eligible": False,
                "contract_id": self.contract_id, "epoch": self.epoch, **extra}

    def _assets_intact(self):
        # Also detect newly added higher-precedence templates/configuration,
        # not only mutations/deletions in the startup list of assets.
        try:
            current = [self._config_path, *_asset_files(*self._asset_directories)]
            return _stat_signature(current) == self._signature
        except (OSError, RenderConfigurationError):
            return False

    def startup(self):
        _require(self._thread is None and not self._closed, "invalid_startup_state")
        self._thread = threading.get_ident()
        self._loop = asyncio.new_event_loop()
        try:
            # Install before vLLM imports/renderer construction and retain until
            # actual cleanup. Its async renderer offloads templates to its own
            # thread pool, so filtering only our calling thread is insufficient.
            for name in _CONTENT_FREE_LOGGERS:
                logger = logging.getLogger(name)
                owned_filter = _ContentFreeRenderLog()
                logger.addFilter(owned_filter)
                self._log_filters.append((logger, owned_filter))
            _require(self._assets_intact(), "assets_changed")
            # Offline, deterministic environment is explicit, never silently
            # enabled by this library or relaxed to download missing assets.
            _require(os.environ.get("HF_HUB_OFFLINE") == "1"
                     and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                     "offline_environment_required")
            _require(os.environ.get("VLLM_PLUGINS") == "",
                     "explicit_disabled_plugins_required")
            self._loop.run_until_complete(self._initialize())
            self._ready = True
        except BaseException as exc:
            # Preserve only our own fixed diagnostic codes or an exception's
            # class, never its message (which may include prompt/asset content).
            self.startup_failure = {
                "kind": type(exc).__name__,
                "code": str(exc) if isinstance(exc, RenderConfigurationError) else "runtime_initialization",
            }
            try:
                self.close()
            finally:
                raise RenderConfigurationError("render_startup_failed") from None

    async def _initialize(self):
        # Official app initialization runs within a live event loop, too.
        self._runtime = _load_runtime(self._argv)
        self.effective_config = self._runtime.effective
        await self._verify_workers()

    async def _render(self, kind, raw):
        runtime = self._runtime
        try:
            # vLLM's FastAPI ingress uses JSON last-key-wins semantics. Parse the
            # ORIGINAL bytes directly with its schema, never a Rust JSON tree.
            request = runtime.schemas[kind].model_validate_json(raw)
            raw_object = json.loads(raw)
        except (runtime.validation_type, ValueError, UnicodeError, TypeError):
            return self._result("invalid", "request_schema", http_status=400)
        # This cohort has exactly one reviewed served alias. Match vLLM's model
        # gate before its error logger can include an arbitrary request.model.
        # Matching requests still execute the complete ServingRender validation.
        if raw_object.get("model") not in (None, self.model):
            return self._result("invalid", "model_not_served", http_status=404)
        reason = _request_cache_reason(request, raw_object, kind)
        if reason:
            return self._result("unsupported", reason)
        runtime.capture.engine_inputs = None
        try:
            if kind == "chat":
                result = await runtime.serving.render_chat_request(request)
            else:
                result = await runtime.serving.render_completion_request(request)
        except (ValueError, TypeError):
            # The same renderer errors become client errors in vLLM's serving
            # frontend. Never include exception messages containing prompt text.
            return self._result("invalid", "render_validation", http_status=400)
        if isinstance(result, runtime.error_type):
            code = result.error.code
            if isinstance(code, int) and 400 <= code < 500:
                return self._result("invalid", "serving_validation", http_status=code)
            return self._result("unavailable", "serving_failure")
        inputs = runtime.capture.engine_inputs
        if not isinstance(inputs, list) or len(inputs) != 1:
            return self._result("unsupported", "multiple_engine_inputs")
        engine_input = inputs[0]
        if (engine_input.get("type") != "token"
                or set(engine_input) - _ENGINE_KEYS):
            return self._result("unsupported", "engine_cache_identity")
        ids = engine_input.get("prompt_token_ids")
        if not isinstance(ids, list) or not ids or any(type(v) is not int or not 0 <= v < 2**32 for v in ids):
            return self._result("unavailable", "invalid_engine_tokens")
        if len(ids) > self.limits["max_tokens_per_request"]:
            return self._result("unsupported", "token_budget")
        outputs = [result] if kind == "chat" else result
        if (not isinstance(outputs, list) or len(outputs) != 1
                or outputs[0].token_ids != ids or outputs[0].features is not None
                or outputs[0].cache_salt is not None):
            return self._result("unavailable", "render_output_mismatch")
        return self._result("exact", "verified_text", token_ids=list(ids), cache_eligible=True)

    def render(self, kind, raw):
        if self._identity_changed:
            return self._result("invalidated", "contract_changed", epoch=self.epoch + 1)
        if (threading.get_ident() != self._thread or not self._ready
                or self._closed or self._invalidated):
            return self._result("unavailable", "provider_not_ready")
        if kind not in ("chat", "completion") or type(raw) is not bytes:
            return self._result("invalid", "invalid_bridge_input", http_status=400)
        if len(raw) > self.limits["max_input_bytes"]:
            return self._result("unsupported", "input_byte_budget")
        if not self._assets_intact():
            self._identity_changed = True
            self._invalidated = True
            return self._result("invalidated", "contract_changed", epoch=self.epoch + 1)
        try:
            return self._loop.run_until_complete(self._render(kind, raw))
        except BaseException:
            self._invalidated = True
            return self._result("unavailable", "provider_exception")

    async def _verify_workers(self):
        for name, kind, request in _conformance_requests(self.model, self._argv):
            raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
            local = await self._render(kind, raw)
            _require(local["status"] == "exact", "local_conformance_failed")
            for worker in self.worker_urls:
                remote = _remote_render(worker, kind, raw, self._timeout, self._api_key_env)
                _require(remote == local["token_ids"], "worker_conformance_mismatch")
                self.conformance.append({"worker": worker, "case": name,
                                         "tokens": len(remote),
                                         "token_sha256": hashlib.sha256(json.dumps(remote).encode()).hexdigest()})

    def close(self):
        if self._closed:
            return
        _require(self._thread is None or threading.get_ident() == self._thread,
                 "close_on_wrong_thread")
        self._ready = False
        self._closed = True
        try:
            try:
                if self._runtime is not None:
                    self._runtime.renderer.shutdown()
            finally:
                self._runtime = None
                if self._loop is not None:
                    self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                    self._loop.run_until_complete(self._loop.shutdown_default_executor())
                    self._loop.close()
                    self._loop = None
        finally:
            for logger, owned_filter in self._log_filters:
                logger.removeFilter(owned_filter)
            self._log_filters.clear()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RenderConfigurationError("conformance_redirect_refused")


def _remote_render(worker, kind, raw, timeout, api_key_env):
    route = "/v1/chat/completions/render" if kind == "chat" else "/v1/completions/render"
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        _require(bool(key), "missing_worker_api_key")
        headers["Authorization"] = "Bearer " + key
    request = Request(worker + route, data=raw, headers=headers, method="POST")
    # Startup only. There is no call from the per-request render path.
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        payload = response.read(8 * 1024 * 1024 + 1)
    _require(len(payload) <= 8 * 1024 * 1024, "conformance_response_too_large")
    result = json.loads(payload)
    if kind == "completion":
        _require(isinstance(result, list) and len(result) == 1, "conformance_batch")
        result = result[0]
    _require(isinstance(result, dict) and result.get("features") is None
             and result.get("cache_salt") is None, "conformance_cache_identity")
    ids = result.get("token_ids")
    _require(isinstance(ids, list) and ids and all(type(v) is int for v in ids),
             "conformance_token_shape")
    return ids


def _conformance_requests(model, argv):
    base = {"model": model, "max_tokens": 1, "temperature": 0}
    user = {"role": "user", "content": "Render contract: café 中文 🙂\n2 + 2?"}
    yield "completion_default", "completion", {**base, "prompt": user["content"]}
    yield "completion_no_special", "completion", {**base, "prompt": user["content"], "add_special_tokens": False}
    yield "chat_default", "chat", {**base, "messages": [user]}
    for thinking in (False, True):
        yield "chat_thinking_" + str(thinking), "chat", {
            **base, "messages": [{"role": "system", "content": "Be concise."}, user,
                                    {"role": "assistant", "content": "4", "reasoning": "Compute."},
                                    {"role": "user", "content": [{"type": "text", "text": "Again?"}]}],
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
    yield "json_output", "chat", {**base, "messages": [user], "response_format": {"type": "json_object"}}
    if "--enable-auto-tool-choice" in argv:
        yield "ordered_tool_history", "chat", {
            **base, "messages": [user,
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_probe", "type": "function", "function": {
                        "name": "lookup", "arguments": '{"z":1,"a":"中文"}'}}]},
                {"role": "tool", "tool_call_id": "call_probe", "content": "found"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {
                "type": "object", "properties": {"z": {"type": "integer"}, "a": {"type": "string"}},
                "required": ["z", "a"]}}}], "tool_choice": "auto",
        }


def create_facade(config_path):
    """Construct only; Rust must call startup on its dedicated render thread."""
    return RenderFacade(config_path)

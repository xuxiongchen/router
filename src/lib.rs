use pyo3::prelude::*;
pub mod backend;
pub mod config;
pub mod logging;
use std::collections::HashMap;

pub mod core;
pub mod data_connector;
pub mod kv_capabilities;
pub mod kv_events;
pub mod kv_index;
pub mod metrics;
pub mod middleware;
pub mod otel_http;
pub mod otel_trace;
pub mod policies;
pub mod program_scheduling;
pub mod prompt_tokens;
pub mod protocols;
pub mod routers;
pub mod server;
pub mod service_discovery;
mod token_estimator;
pub mod tokenizer;
pub mod tree;
pub mod wasm_middleware;
use crate::metrics::PrometheusConfig;

#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
    PowerOfTwo,
    ConsistentHash,
    KvAware,
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
struct Router {
    host: String,
    port: u16,
    worker_urls: Vec<String>,
    policy: PolicyType,
    kv_tokenizer_path: Option<String>,
    kv_model: String,
    kv_hash_algo: Option<String>,
    kv_block_size: usize,
    kv_hash_seed: u32,
    kv_events_topic_filter: String,
    kv_events_port: u16,
    kv_events_endpoints: Vec<String>,
    kv_index_max_entries: usize,
    kv_load_guard: bool,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval: u64,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    eviction_interval_secs: u64,
    max_tree_size: usize,
    max_payload_size: usize,
    wasm_middleware: Option<String>,
    wasm_middleware_sha256: Option<String>,
    wasm_middleware_routes: Vec<String>,
    intra_node_data_parallel_size: usize,
    api_key: Option<String>,
    api_key_validation_urls: Vec<String>,
    log_dir: Option<String>,

    log_level: Option<String>,
    service_discovery: bool,
    selector: HashMap<String, String>,
    service_discovery_port: u16,
    service_discovery_namespace: Option<String>,
    prefill_selector: HashMap<String, String>,
    decode_selector: HashMap<String, String>,
    bootstrap_port_annotation: String,
    prometheus_port: Option<u16>,
    prometheus_host: Option<String>,
    request_timeout_secs: u64,
    request_id_headers: Option<Vec<String>>,
    vllm_pd_disaggregation: bool,
    vllm_discovery_address: Option<String>,
    prefill_urls: Option<Vec<(String, Option<u16>)>>,
    decode_urls: Option<Vec<String>>,
    prefill_policy: Option<PolicyType>,
    decode_policy: Option<PolicyType>,
    max_concurrent_requests: usize,
    cors_allowed_origins: Vec<String>,
    // Retry configuration
    retry_max_retries: u32,
    retry_initial_backoff_ms: u64,
    retry_max_backoff_ms: u64,
    retry_backoff_multiplier: f32,
    retry_jitter_factor: f32,
    disable_retries: bool,
    // Circuit breaker configuration
    cb_failure_threshold: u32,
    cb_success_threshold: u32,
    cb_timeout_duration_secs: u64,
    cb_window_duration_secs: u64,
    disable_circuit_breaker: bool,
    // Health check configuration
    health_failure_threshold: u32,
    health_success_threshold: u32,
    health_check_timeout_secs: u64,
    health_check_interval_secs: u64,
    health_check_endpoint: String,
    // IGW (Inference Gateway) configuration
    enable_igw: bool,
    queue_size: usize,
    queue_timeout_secs: u64,
    rate_limit_tokens_per_second: Option<usize>,
    // OpenTelemetry tracing
    enable_trace: bool,
    otlp_traces_endpoint: Option<String>,
    // KV connector for PD disaggregation ("nixl" or "mooncake")
    kv_connector: String,
    // Explicit Program-level scheduling feature switch and optional overrides.
    enable_program_scheduling: bool,
    program_scheduling_config_json: Option<String>,
}

impl Router {
    /// Convert PyO3 Router to RouterConfig
    pub fn to_router_config(&self) -> config::ConfigResult<config::RouterConfig> {
        use config::{
            DiscoveryConfig, MetricsConfig, PolicyConfig as ConfigPolicyConfig, RoutingMode,
        };

        let kv_config = if self.policy == PolicyType::KvAware
            || self.prefill_policy == Some(PolicyType::KvAware)
            || self.decode_policy == Some(PolicyType::KvAware)
        {
            if self.kv_hash_algo.as_deref() != Some("sha256_cbor") {
                return Err(config::ConfigError::ValidationFailed {
                    reason: "kv_aware requires kv_hash_algo=sha256_cbor and matching workers"
                        .into(),
                });
            }
            let mut worker_endpoints = HashMap::new();
            for entry in &self.kv_events_endpoints {
                let (worker, endpoint) = kv_events::parse_endpoint_mapping(entry)
                    .map_err(|reason| config::ConfigError::ValidationFailed { reason })?;
                if worker_endpoints.insert(worker, endpoint).is_some() {
                    return Err(config::ConfigError::ValidationFailed {
                        reason: "duplicate KV worker endpoint mapping".into(),
                    });
                }
            }
            config::KvAwareConfig {
                block_size: self.kv_block_size,
                hash_seed: self.kv_hash_seed,
                tokenizer_path: self.kv_tokenizer_path.clone().unwrap_or_default(),
                model: self.kv_model.clone(),
                topic: self.kv_events_topic_filter.clone(),
                default_port: self.kv_events_port,
                worker_endpoints,
                index_max_entries: self.kv_index_max_entries,
                load_guard: self.kv_load_guard,
            }
        } else {
            if self.kv_tokenizer_path.is_some()
                || self.kv_hash_algo.is_some()
                || !self.kv_events_endpoints.is_empty()
                || self.kv_load_guard
            {
                return Err(config::ConfigError::ValidationFailed {
                    reason: "KV options require policy=kv_aware".into(),
                });
            }
            config::KvAwareConfig::default()
        };

        // Convert policy helper function
        let convert_policy = |policy: &PolicyType| -> ConfigPolicyConfig {
            match policy {
                PolicyType::Random => ConfigPolicyConfig::Random,
                PolicyType::RoundRobin => ConfigPolicyConfig::RoundRobin,
                PolicyType::CacheAware => ConfigPolicyConfig::CacheAware {
                    cache_threshold: self.cache_threshold,
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_tree_size: self.max_tree_size,
                },
                PolicyType::PowerOfTwo => ConfigPolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5, // Default value
                },
                PolicyType::ConsistentHash => ConfigPolicyConfig::ConsistentHash {
                    virtual_nodes: 160, // Default value
                },
                PolicyType::KvAware => ConfigPolicyConfig::KvAware {
                    config: Box::new(kv_config.clone()),
                },
            }
        };

        // Determine routing mode
        let mode = if self.enable_igw {
            // IGW mode - routing mode is not used in IGW, but we need to provide a placeholder
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if self.vllm_pd_disaggregation {
            RoutingMode::VllmPrefillDecode {
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                prefill_policy: self.prefill_policy.as_ref().map(convert_policy),
                decode_policy: self.decode_policy.as_ref().map(convert_policy),
                discovery_address: self.vllm_discovery_address.clone(),
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        // Convert main policy
        let policy = convert_policy(&self.policy);

        // Service discovery configuration
        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: self.selector.clone(),
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
            })
        } else {
            None
        };

        // Metrics configuration
        let metrics = match (self.prometheus_port, self.prometheus_host.as_ref()) {
            (Some(port), Some(host)) => Some(MetricsConfig {
                port,
                host: host.clone(),
            }),
            _ => None,
        };

        Ok(config::RouterConfig {
            mode,
            policy,
            host: self.host.clone(),
            port: self.port,
            connection_mode: config::ConnectionMode::Http,
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            intra_node_data_parallel_size: self.intra_node_data_parallel_size,
            api_key: self.api_key.clone(),
            api_key_validation_urls: self.api_key_validation_urls.clone(),
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: self.log_level.clone(),
            request_id_headers: self.request_id_headers.clone(),
            max_concurrent_requests: self.max_concurrent_requests,
            queue_size: self.queue_size,
            queue_timeout_secs: self.queue_timeout_secs,
            rate_limit_tokens_per_second: self.rate_limit_tokens_per_second,
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: config::RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: config::CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: self.disable_retries,
            disable_circuit_breaker: self.disable_circuit_breaker,
            health_check: config::HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            },
            enable_igw: self.enable_igw,
            history_backend: config::HistoryBackend::Memory,
            enable_profiling: false, // Profiling disabled in Python binding by default
            profile_timeout_secs: 10, // Default profiling timeout
            kv_connector: match self.kv_connector.to_ascii_lowercase().as_str() {
                "nixl" => config::KvConnector::Nixl,
                "mooncake" => config::KvConnector::Mooncake,
                "moriio" => config::KvConnector::MoriIO,
                other => {
                    return Err(config::ConfigError::ValidationFailed {
                        reason: format!(
                            "Invalid kv_connector '{}': expected 'nixl', 'mooncake', or 'moriio'",
                            other
                        ),
                    });
                }
            },
            program_scheduling: config::ProgramSchedulingConfig::resolve(
                self.enable_program_scheduling,
                self.program_scheduling_config_json.as_deref(),
            )?,
        })
    }
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("127.0.0.1"),
        port = 3001,
        worker_startup_timeout_secs = 600,
        worker_startup_check_interval = 30,
        cache_threshold = 0.3,
        balance_abs_threshold = 64,
        balance_rel_threshold = 1.5,
        eviction_interval_secs = 120,
        max_tree_size = 2usize.pow(26),
        max_payload_size = 512 * 1024 * 1024,  // 512MB default for large batches
        intra_node_data_parallel_size = 1,
        api_key = None,
        api_key_validation_urls = vec![],
        log_dir = None,
        log_level = None,
        service_discovery = false,
        selector = HashMap::new(),
        service_discovery_port = 80,
        service_discovery_namespace = None,
        prefill_selector = HashMap::new(),
        decode_selector = HashMap::new(),
        bootstrap_port_annotation = String::from("vllm.ai/bootstrap-port"),
        prometheus_port = None,
        prometheus_host = None,
        request_timeout_secs = 1800,  // Add configurable request timeout
        request_id_headers = None,  // Custom request ID headers
        vllm_pd_disaggregation = false,  // New flag for PD mode
        vllm_discovery_address = None,
        prefill_urls = None,
        decode_urls = None,
        prefill_policy = None,
        decode_policy = None,
        max_concurrent_requests = 32768,
        cors_allowed_origins = vec![],
        // Retry defaults
        retry_max_retries = 5,
        retry_initial_backoff_ms = 50,
        retry_max_backoff_ms = 30_000,
        retry_backoff_multiplier = 1.5,
        retry_jitter_factor = 0.2,
        disable_retries = false,
        // Circuit breaker defaults
        cb_failure_threshold = 10,
        cb_success_threshold = 3,
        cb_timeout_duration_secs = 60,
        cb_window_duration_secs = 120,
        disable_circuit_breaker = false,
        // Health check defaults
        health_failure_threshold = 3,
        health_success_threshold = 2,
        health_check_timeout_secs = 5,
        health_check_interval_secs = 60,
        health_check_endpoint = String::from("/health"),
        // IGW defaults
        enable_igw = false,
        queue_size = 100,
        queue_timeout_secs = 60,
        rate_limit_tokens_per_second = None,
        // Tracing defaults
        enable_trace = false,
        otlp_traces_endpoint = None,
        // KV connector default (PD disaggregation)
        kv_connector = String::from("nixl"),
        wasm_middleware = None,
        wasm_middleware_sha256 = None,
        wasm_middleware_routes = vec![],
        enable_program_scheduling = false,
        program_scheduling_config_json = None,
        kv_tokenizer_path = None,
        kv_model = String::from("Qwen/Qwen3-0.6B"),
        kv_hash_algo = None,
        kv_block_size = 16,
        kv_hash_seed = 0,
        kv_events_topic_filter = String::new(),
        kv_events_port = 5557,
        kv_events_endpoints = vec![],
        kv_index_max_entries = 100_000,
        kv_load_guard = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval: u64,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        max_payload_size: usize,
        intra_node_data_parallel_size: usize,
        api_key: Option<String>,
        api_key_validation_urls: Vec<String>,
        log_dir: Option<String>,
        log_level: Option<String>,
        service_discovery: bool,
        selector: HashMap<String, String>,
        service_discovery_port: u16,
        service_discovery_namespace: Option<String>,
        prefill_selector: HashMap<String, String>,
        decode_selector: HashMap<String, String>,
        bootstrap_port_annotation: String,
        prometheus_port: Option<u16>,
        prometheus_host: Option<String>,
        request_timeout_secs: u64,
        request_id_headers: Option<Vec<String>>,
        vllm_pd_disaggregation: bool,
        vllm_discovery_address: Option<String>,
        prefill_urls: Option<Vec<(String, Option<u16>)>>,
        decode_urls: Option<Vec<String>>,
        prefill_policy: Option<PolicyType>,
        decode_policy: Option<PolicyType>,
        max_concurrent_requests: usize,
        cors_allowed_origins: Vec<String>,
        retry_max_retries: u32,
        retry_initial_backoff_ms: u64,
        retry_max_backoff_ms: u64,
        retry_backoff_multiplier: f32,
        retry_jitter_factor: f32,
        disable_retries: bool,
        cb_failure_threshold: u32,
        cb_success_threshold: u32,
        cb_timeout_duration_secs: u64,
        cb_window_duration_secs: u64,
        disable_circuit_breaker: bool,
        health_failure_threshold: u32,
        health_success_threshold: u32,
        health_check_timeout_secs: u64,
        health_check_interval_secs: u64,
        health_check_endpoint: String,
        enable_igw: bool,
        queue_size: usize,
        queue_timeout_secs: u64,
        rate_limit_tokens_per_second: Option<usize>,
        enable_trace: bool,
        otlp_traces_endpoint: Option<String>,
        kv_connector: String,
        wasm_middleware: Option<String>,
        wasm_middleware_sha256: Option<String>,
        wasm_middleware_routes: Vec<String>,
        enable_program_scheduling: bool,
        program_scheduling_config_json: Option<String>,
        kv_tokenizer_path: Option<String>,
        kv_model: String,
        kv_hash_algo: Option<String>,
        kv_block_size: usize,
        kv_hash_seed: u32,
        kv_events_topic_filter: String,
        kv_events_port: u16,
        kv_events_endpoints: Vec<String>,
        kv_index_max_entries: usize,
        kv_load_guard: bool,
    ) -> PyResult<Self> {
        if wasm_middleware_sha256
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .is_some()
            && wasm_middleware
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .is_none()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "wasm_middleware_sha256 requires wasm_middleware",
            ));
        }
        Ok(Router {
            host,
            port,
            worker_urls,
            policy,
            kv_tokenizer_path,
            kv_model,
            kv_hash_algo,
            kv_block_size,
            kv_hash_seed,
            kv_events_topic_filter,
            kv_events_port,
            kv_events_endpoints,
            kv_index_max_entries,
            kv_load_guard,
            worker_startup_timeout_secs,
            worker_startup_check_interval,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
            max_payload_size,
            wasm_middleware,
            wasm_middleware_sha256,
            wasm_middleware_routes,
            enable_program_scheduling,
            intra_node_data_parallel_size,
            api_key,
            api_key_validation_urls,
            log_dir,
            log_level,
            service_discovery,
            selector,
            service_discovery_port,
            service_discovery_namespace,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation,
            prometheus_port,
            prometheus_host,
            request_timeout_secs,
            request_id_headers,
            vllm_pd_disaggregation,
            vllm_discovery_address,
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
            max_concurrent_requests,
            cors_allowed_origins,
            retry_max_retries,
            retry_initial_backoff_ms,
            retry_max_backoff_ms,
            retry_backoff_multiplier,
            retry_jitter_factor,
            disable_retries,
            cb_failure_threshold,
            cb_success_threshold,
            cb_timeout_duration_secs,
            cb_window_duration_secs,
            disable_circuit_breaker,
            health_failure_threshold,
            health_success_threshold,
            health_check_timeout_secs,
            health_check_interval_secs,
            health_check_endpoint,
            enable_igw,
            queue_size,
            queue_timeout_secs,
            rate_limit_tokens_per_second,
            enable_trace,
            otlp_traces_endpoint,
            kv_connector,
            program_scheduling_config_json,
        })
    }

    #[pyo3(signature = (*, render_facade=None, render_contract_id=None, render_contract_epoch=1, render_limits=None, kv_capabilities_json=None))]
    fn start(
        &self,
        py: Python<'_>,
        render_facade: Option<Py<PyAny>>,
        render_contract_id: Option<String>,
        render_contract_epoch: u64,
        render_limits: Option<HashMap<String, u64>>,
        kv_capabilities_json: Option<String>,
    ) -> PyResult<()> {
        let render_input =
            match (render_facade, render_contract_id) {
                (Some(facade), Some(id)) if self.policy == PolicyType::KvAware => Some((
                    facade,
                    prompt_tokens::bridge::RenderContract {
                        id,
                        epoch: render_contract_epoch,
                    },
                )),
                (None, None) if render_limits.is_none() => None,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "render injection requires kv_aware, a facade and a render contract identity",
                )),
            };
        let bridge_limits = render_bridge_limits(render_limits)?;
        if kv_capabilities_json.is_some() && render_input.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "worker capabilities require the vllm render backend",
            ));
        }
        let capability_cohort = kv_capabilities_json
            .map(|value| {
                if value.len() > 16 * 1024 * 1024 {
                    return Err("capability cohort exceeds size limit".to_string());
                }
                serde_json::from_str::<kv_capabilities::CapabilityCohort>(&value)
                    .map_err(|_| "invalid capability cohort".to_string())
            })
            .transpose()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        // Convert to RouterConfig and validate
        let router_config = self.to_router_config().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration error: {}", e))
        })?;

        // Validate the configuration
        router_config.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Configuration validation failed: {}",
                e
            ))
        })?;

        // Create service discovery config if enabled
        let service_discovery_config = if self.service_discovery {
            Some(service_discovery::ServiceDiscoveryConfig {
                enabled: true,
                selector: self.selector.clone(),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                // HTTP service discovery only supports the vLLM PD router.
                pd_mode: self.vllm_pd_disaggregation,
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
            })
        } else {
            None
        };

        // Create Prometheus config if enabled
        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self
                .prometheus_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
        });

        let startup_timeout = std::time::Duration::from_secs(self.worker_startup_timeout_secs);
        let server_config = server::ServerConfig {
            host: self.host.clone(),
            port: self.port,
            router_config,
            max_payload_size: self.max_payload_size,
            wasm_middleware: self.wasm_middleware.clone(),
            wasm_middleware_sha256: self.wasm_middleware_sha256.clone(),
            wasm_middleware_routes: self.wasm_middleware_routes.clone(),
            log_dir: self.log_dir.clone(),
            log_level: self.log_level.clone(),
            service_discovery_config,
            prometheus_config,
            request_timeout_secs: self.request_timeout_secs,
            request_id_headers: self.request_id_headers.clone(),
            trace_config: if self.enable_trace {
                Some(config::TraceConfig {
                    otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
                    ..Default::default()
                })
            } else {
                None
            },
        };

        // PyO3 0.26 provides detach(). In particular, the long-lived server
        // must not retain the GIL while its dedicated render thread needs it.
        // Only owned Python handles cross this boundary; no Bound/Python token.
        py.detach(move || -> Result<(), String> {
            let runtime = tokio::runtime::Runtime::new().map_err(|error| error.to_string())?;
            let bridge = render_input
                .map(|(facade, contract)| {
                    prompt_tokens::bridge::RenderBridge::new(facade, contract, bridge_limits)
                        .map(|bridge| bridge.with_capabilities(capability_cohort))
                        .map(std::sync::Arc::new)
                })
                .transpose()?;
            let result = runtime.block_on(async {
                if let Some(bridge) = &bridge {
                    bridge.wait_ready(startup_timeout).await?;
                }
                server::startup_with_render_bridge(server_config, bridge.clone())
                    .await
                    .map_err(|error| error.to_string())
            });
            if let Some(bridge) = bridge {
                bridge.shutdown();
                // A started synchronous Python computation cannot be killed.
                // The bridge's non-daemon Python lifetime guard prevents normal
                // interpreter finalization until its actual callback/ref drain.
                if !runtime.block_on(bridge.wait_closed(std::time::Duration::from_secs(30))) {
                    return Err("render shutdown deadline exceeded; Python callback may still be active and the interpreter lifetime guard remains; irrecoverable work requires terminating the whole process externally".into());
                }
            }
            result
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

fn render_bridge_limits(
    overrides: Option<HashMap<String, u64>>,
) -> PyResult<prompt_tokens::bridge::BridgeLimits> {
    let mut limits = prompt_tokens::bridge::BridgeLimits::default();
    for (name, value) in overrides.unwrap_or_default() {
        let size = || {
            usize::try_from(value).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "render limit exceeds the platform size bound",
                )
            })
        };
        match name.as_str() {
            "max_pending_jobs" => limits.max_pending_jobs = size()?,
            "max_input_bytes" => limits.max_input_bytes = size()?,
            "max_tokens_per_request" => limits.max_tokens_per_request = size()?,
            "max_reserved_tokens" => limits.max_reserved_tokens = size()?,
            "queue_timeout_ms" => limits.queue_timeout = std::time::Duration::from_millis(value),
            "execution_timeout_ms" => {
                limits.execution_timeout = std::time::Duration::from_millis(value)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown render bridge limit",
                ))
            }
        }
    }
    Ok(limits)
}

/// Explicit native capability handshake for the controlled performance harness.
/// No environment value means ordinary production behavior, including in a
/// feature-enabled build. Invalid/unsupported requests fail before measurement.
#[pyfunction]
fn kv_perf_capabilities(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let selected_mode = routers::http::router::kv_perf_requested_mode()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let result = pyo3::types::PyDict::new(py);
    result.set_item("enabled", cfg!(feature = "kv-perf"))?;
    let modes: &[&str] = if cfg!(feature = "kv-perf") {
        &routers::http::router::KV_PERF_MODES
    } else {
        &[]
    };
    result.set_item("modes", modes)?;
    result.set_item("environment_variable", routers::http::router::KV_PERF_ENV)?;
    result.set_item("loopback_only", true)?;
    result.set_item("selected_mode", selected_mode)?;
    Ok(result)
}

#[pymodule]
fn vllm_router_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<Router>()?;
    m.add_function(wrap_pyfunction!(kv_perf_capabilities, m)?)?;
    Ok(())
}

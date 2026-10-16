pub mod types;
pub mod validation;

pub use types::*;
pub use validation::*;

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Invalid value for field '{field}': {value} - {reason}")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    #[error("Incompatible configuration: {reason}")]
    IncompatibleConfig { reason: String },

    #[error("Missing required field: {field}")]
    MissingRequired { field: String },
}

/// Result type for configuration operations
pub type ConfigResult<T> = Result<T, ConfigError>;

/// Parse repeatable `WORKER_URL=ENDPOINT[,ENDPOINT...]` KV event sources.
pub fn parse_static_kv_event_sources(
    specs: &[String],
    replay_specs: &[String],
    block_size: u32,
    model: Option<&str>,
) -> ConfigResult<std::collections::HashMap<String, StaticKvEventSources>> {
    if specs.is_empty() {
        if !replay_specs.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "--kv-event-replay-source requires a matching --kv-event-source"
                    .to_string(),
            });
        }
        return Ok(std::collections::HashMap::new());
    }
    if block_size == 0 {
        return Err(ConfigError::InvalidValue {
            field: "kv_event_block_size".to_string(),
            value: block_size.to_string(),
            reason: "Must be greater than zero".to_string(),
        });
    }
    let model = model
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| ConfigError::ValidationFailed {
            reason: "--kv-event-model is required when --kv-event-source is used".to_string(),
        })?;

    let mut sources = std::collections::HashMap::new();
    for spec in specs {
        let Some((worker_url, endpoints)) = spec.split_once('=') else {
            return Err(ConfigError::ValidationFailed {
                reason: format!(
                    "--kv-event-source expects WORKER_URL=ENDPOINT[,ENDPOINT...], got '{spec}'"
                ),
            });
        };
        let worker_url = worker_url.trim();
        if worker_url.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("--kv-event-source needs a worker URL, got '{spec}'"),
            });
        }
        let endpoints: Vec<String> = endpoints
            .split(',')
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(str::to_owned)
            .collect();
        if endpoints.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("--kv-event-source needs at least one endpoint, got '{spec}'"),
            });
        }
        for endpoint in &endpoints {
            let parsed =
                url::Url::parse(endpoint).map_err(|error| ConfigError::ValidationFailed {
                    reason: format!("invalid KV event endpoint '{endpoint}': {error}"),
                })?;
            if parsed.scheme() != "tcp"
                || parsed
                    .host_str()
                    .is_none_or(|host| host == "*" || host == "0.0.0.0" || host == "::")
                || parsed.port().is_none_or(|port| port == 0)
            {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "KV event endpoint must be reachable tcp://HOST:PORT: '{endpoint}'"
                    ),
                });
            }
        }
        let source = StaticKvEventSources {
            endpoints,
            replay_endpoints: None,
            block_size,
            model: model.to_owned(),
        };
        if sources.insert(worker_url.to_owned(), source).is_some() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("duplicate --kv-event-source worker URL '{worker_url}'"),
            });
        }
    }
    for spec in replay_specs {
        let Some((worker_url, endpoints)) = spec.split_once('=') else {
            return Err(ConfigError::ValidationFailed {
                reason: format!(
                    "--kv-event-replay-source expects WORKER_URL=ENDPOINT[,ENDPOINT...], got '{spec}'"
                ),
            });
        };
        let worker_url = worker_url.trim();
        let endpoints: Vec<String> = endpoints
            .split(',')
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(str::to_owned)
            .collect();
        if endpoints.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("--kv-event-replay-source needs an endpoint, got '{spec}'"),
            });
        }
        let source = sources
            .get_mut(worker_url)
            .ok_or_else(|| ConfigError::ValidationFailed {
                reason: format!(
                    "--kv-event-replay-source worker '{worker_url}' has no --kv-event-source"
                ),
            })?;
        if source.replay_endpoints.replace(endpoints).is_some() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("duplicate --kv-event-replay-source worker URL '{worker_url}'"),
            });
        }
    }
    Ok(sources)
}

/// Parse repeatable static PD KV-source specifications for both launchers.
pub fn parse_pd_kv_events(
    specs: &[String],
) -> ConfigResult<std::collections::HashMap<String, PdKvEventSources>> {
    let mut sources = std::collections::HashMap::new();
    for spec in specs {
        let Some((url, endpoints)) = spec.split_once('=') else {
            return Err(ConfigError::ValidationFailed {
                reason: format!("--pd-kv-events expects URL=ENDPOINT[,ENDPOINT...], got '{spec}'"),
            });
        };
        let url = url.trim();
        let endpoints: Vec<String> = endpoints
            .split(',')
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(str::to_owned)
            .collect();
        if url.is_empty() || endpoints.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: format!(
                    "--pd-kv-events needs a URL and at least one endpoint, got '{spec}'"
                ),
            });
        }
        sources.insert(
            url.to_owned(),
            PdKvEventSources {
                endpoints,
                replay_endpoints: None,
                block_size: 16,
                model: None,
            },
        );
    }
    Ok(sources)
}

#[cfg(test)]
mod static_kv_event_source_tests {
    use super::*;

    #[test]
    fn parses_static_event_sources_with_shared_model_and_block_size() {
        let specs = vec![
            "http://worker-a:8000=tcp://10.0.0.1:5557,tcp://10.0.0.1:5559".to_string(),
            "grpc://worker-b:50051=tcp://10.0.0.2:5557".to_string(),
        ];
        let replay_specs =
            vec!["http://worker-a:8000=tcp://10.0.0.1:5558,tcp://10.0.0.1:5560".to_string()];
        let sources =
            parse_static_kv_event_sources(&specs, &replay_specs, 32, Some("model-a")).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources["http://worker-a:8000"].block_size, 32);
        assert_eq!(sources["http://worker-a:8000"].model, "model-a");
        assert_eq!(sources["http://worker-a:8000"].endpoints.len(), 2);
        assert_eq!(
            sources["http://worker-a:8000"]
                .replay_endpoints
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(sources["grpc://worker-b:50051"].endpoints.len(), 1);
    }

    #[test]
    fn static_event_sources_require_model_and_reachable_endpoints() {
        let missing_model = vec!["http://worker:8000=tcp://10.0.0.1:5557".to_string()];
        assert!(parse_static_kv_event_sources(&missing_model, &[], 16, None).is_err());
        for endpoint in ["tcp://*:5557", "tcp://0.0.0.0:5557", "tcp://10.0.0.1:0"] {
            let specs = vec![format!("http://worker:8000={endpoint}")];
            assert!(parse_static_kv_event_sources(&specs, &[], 16, Some("model")).is_err());
        }
    }
}

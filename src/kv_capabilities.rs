//! Audited vLLM 0.29 capability proposal. This is NOT a stock HTTP API.
//! The fixed read-only endpoint is queried only by control-plane subscribers.

use std::{
    collections::{HashMap, HashSet},
    io::Read,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CAPABILITIES_PATH: &str = "/v1/kv-cache/capabilities";
const MAX_DESCRIPTOR_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Namespace {
    pub model: String,
    pub served_model_names: Vec<String>,
    #[serde(deserialize_with = "required_value")]
    pub revision: serde_json::Value,
    pub dtype: String,
    #[serde(deserialize_with = "required_value")]
    pub quantization: serde_json::Value,
    pub cache_dtype: String,
    pub weight_version: String,
}

fn required_value<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<serde_json::Value, D::Error> {
    serde_json::Value::deserialize(deserializer)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheGroup {
    pub group_id: u32,
    pub kind: String,
    pub layer_count: u32,
    pub allocation_block_tokens: usize,
    pub effective_block_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HashContract {
    pub algorithm: String,
    pub width_bytes: usize,
    pub representation: String,
    pub seed: u32,
    pub root_hex: String,
    pub extra_keys: String,
    pub block_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReuseContract {
    pub alignment_tokens: usize,
    pub terminal_recompute_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Execution {
    pub mode: String,
    pub dp: usize,
    pub tp: usize,
    pub pp: usize,
    pub dcp: usize,
    pub pcp: usize,
    pub prefix_caching: bool,
    pub speculation: bool,
    pub connector: bool,
    pub offload: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventSource {
    pub publisher: String,
    pub epoch: String,
    pub topic: String,
    pub configured_endpoint: String,
    pub resolved_endpoint: String,
    pub dp_rank: usize,
    pub sequence: String,
    pub next_sequence: u64,
    pub payload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Descriptor {
    pub schema_version: u32,
    pub vllm_version: String,
    pub mechanism: String,
    pub mechanism_version: u32,
    pub namespace: Namespace,
    pub groups: Vec<CacheGroup>,
    pub hash: HashContract,
    pub reuse: ReuseContract,
    pub execution: Execution,
    pub events: EventSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityCohort {
    pub workers: HashMap<String, Descriptor>,
    pub api_key_env: Option<String>,
}

fn require(valid: bool, message: &str) -> Result<(), String> {
    valid.then_some(()).ok_or_else(|| message.to_string())
}

fn nonempty(value: &str) -> bool {
    !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
}

fn endpoint_port(endpoint: &str, allow_wildcard_port: bool) -> Result<Option<u16>, String> {
    // The descriptor never chooses the destination host. Its bound port is
    // compared with the independently configured worker -> publisher mapping.
    let endpoint = endpoint
        .strip_prefix("tcp://")
        .ok_or("capability publisher is not TCP")?;
    let (host, port) = endpoint
        .rsplit_once(':')
        .ok_or("capability endpoint has no port")?;
    require(
        nonempty(host)
            && !host.contains(['/', '@', '?', '#'])
            && !host.chars().any(char::is_whitespace),
        "invalid capability endpoint host",
    )?;
    if port == "*" && allow_wildcard_port {
        return Ok(None);
    }
    require(
        !port.is_empty() && port.bytes().all(|c| c.is_ascii_digit()),
        "invalid capability endpoint port",
    )?;
    let port: u16 = port
        .parse()
        .map_err(|_| "invalid capability endpoint port")?;
    require(port != 0, "zero capability endpoint port")?;
    Ok(Some(port))
}

pub fn seed_root_hex(seed: u32) -> String {
    // Canonical CBOR of a u32 decimal seed string (at most ten UTF-8 bytes).
    let text = seed.to_string();
    let mut cbor = vec![0x60 + text.len() as u8];
    cbor.extend_from_slice(text.as_bytes());
    format!("{:x}", Sha256::digest(cbor))
}

impl Descriptor {
    pub fn validate(
        &self,
        endpoint: &str,
        block_size: usize,
        seed: u32,
        model: &str,
    ) -> Result<(), String> {
        require(
            self.schema_version == 1 && self.vllm_version.split('+').next() == Some("0.29.0"),
            "unsupported capability schema/package version",
        )?;
        require(
            self.mechanism == "normal_full_attention" && self.mechanism_version == 1,
            "unsupported required cache mechanism semantics",
        )?;
        require(
            block_size > 0 && block_size <= 65_536,
            "invalid logical block token bound",
        )?;
        require(
            self.groups.len() == 1,
            "capability requires exactly one complete cache group",
        )?;
        let group = &self.groups[0];
        require(
            group.group_id == 0
                && group.kind == "full_attention"
                && group.layer_count > 0
                && group.layer_count <= 1_000_000
                && group.allocation_block_tokens == block_size
                && group.effective_block_tokens == block_size,
            "unsupported cache group, inventory or block token units",
        )?;
        require(
            self.hash.algorithm == "sha256_cbor"
                && self.hash.width_bytes == 32
                && self.hash.representation == "bytes"
                && self.hash.extra_keys == "none"
                && self.hash.seed == seed
                && self.hash.root_hex == seed_root_hex(seed)
                && self.hash.block_tokens == block_size,
            "unsupported hash domain or requested/effective hash conflict",
        )?;
        require(
            self.reuse.alignment_tokens == block_size && self.reuse.terminal_recompute_tokens == 1,
            "unsupported Dense reuse or terminal recompute semantics",
        )?;
        let execution = &self.execution;
        require(
            execution.mode == "normal"
                && execution.dp == 1
                && execution.tp == 1
                && execution.pp == 1
                && execution.dcp == 1
                && execution.pcp == 1
                && execution.prefix_caching
                && !execution.speculation
                && !execution.connector
                && !execution.offload,
            "unsupported effective cache execution qualifiers",
        )?;
        let namespace = &self.namespace;
        require(
            nonempty(&namespace.model)
                && nonempty(&namespace.dtype)
                && nonempty(&namespace.cache_dtype)
                && nonempty(&namespace.weight_version)
                && namespace.quantization.is_null()
                && (namespace.revision.is_null()
                    || namespace.revision.as_str().is_some_and(nonempty))
                && namespace.served_model_names.len() == 1
                && namespace
                    .served_model_names
                    .iter()
                    .all(|name| nonempty(name))
                && namespace
                    .served_model_names
                    .iter()
                    .collect::<HashSet<_>>()
                    .len()
                    == namespace.served_model_names.len()
                && namespace
                    .served_model_names
                    .iter()
                    .any(|name| name == model),
            "invalid or incompatible model/serving/cache namespace",
        )?;
        let events = &self.events;
        require(
            events.publisher == "zmq"
                && events.dp_rank == 0
                && events.sequence == "u64_be_monotonic_per_epoch"
                && events.payload == "vllm_kv_events_v1"
                && events.epoch.len() == 32
                && events
                    .epoch
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                && events.topic.ends_with(&format!(".{}", events.epoch))
                && nonempty(&events.topic)
                && events.topic.len() <= 512
                && events.next_sequence < u64::MAX,
            "invalid or unsupported publisher identity/epoch/sequence",
        )?;
        let port = endpoint_port(endpoint, false)?;
        require(
            endpoint_port(&events.resolved_endpoint, false)? == port,
            "configured Router endpoint and resolved publisher port conflict",
        )?;
        let configured = endpoint_port(&events.configured_endpoint, true)?;
        require(
            configured.is_none() || configured == port,
            "configured and resolved publisher port conflict",
        )
    }

    /// A boot epoch or observation watermark may change without changing the
    /// input/cache contract. Metadata is never a cache inventory.
    pub fn compatible_with(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.vllm_version.split('+').next() == other.vllm_version.split('+').next()
            && self.mechanism == other.mechanism
            && self.mechanism_version == other.mechanism_version
            && self.namespace == other.namespace
            && self.groups == other.groups
            && self.hash == other.hash
            && self.reuse == other.reuse
            && self.execution == other.execution
            && self.events.publisher == other.events.publisher
            && self.events.dp_rank == other.events.dp_rank
            && self.events.sequence == other.events.sequence
            && self.events.payload == other.events.payload
    }

    pub fn same_publisher(&self, other: &Self) -> bool {
        self.events.epoch == other.events.epoch
            && self.events.topic == other.events.topic
            && self.events.configured_endpoint == other.events.configured_endpoint
            && self.events.resolved_endpoint == other.events.resolved_endpoint
    }
}

impl CapabilityCohort {
    pub fn validate(
        &self,
        endpoints: &[(String, String)],
        block_size: usize,
        seed: u32,
        model: &str,
    ) -> Result<(), String> {
        require(
            !endpoints.is_empty() && self.workers.len() == endpoints.len(),
            "capability cohort worker inventory differs",
        )?;
        require(
            self.api_key_env.as_ref().is_none_or(|key| {
                !key.is_empty()
                    && key.len() <= 256
                    && key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            }),
            "invalid Worker API key environment name",
        )?;
        let mut initial = None;
        let mut epochs = HashSet::new();
        for (worker, endpoint) in endpoints {
            let descriptor = self
                .workers
                .get(worker)
                .ok_or("missing capability Worker")?;
            descriptor.validate(endpoint, block_size, seed, model)?;
            require(
                epochs.insert(&descriptor.events.epoch),
                "distinct Workers share a publisher epoch",
            )?;
            if let Some(previous) = initial {
                require(
                    descriptor.compatible_with(previous),
                    "incompatible capability cohort",
                )?;
            }
            initial = Some(descriptor);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchError {
    Unavailable,
    Unsupported,
    Invalid,
    Transport,
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Construct and drop on its subscriber thread, never inside a Tokio worker.
pub(crate) struct CapabilityClient {
    client: reqwest::blocking::Client,
    api_key: Option<String>,
}

impl CapabilityClient {
    pub(crate) fn new(api_key_env: Option<&str>) -> Result<Self, String> {
        let api_key = api_key_env
            .map(|name| {
                std::env::var(name).map_err(|_| "missing Worker API key environment".to_string())
            })
            .transpose()?;
        require(
            api_key.as_ref().is_none_or(|key| !key.is_empty()),
            "empty Worker API key",
        )?;
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(1))
            .timeout(Duration::from_secs(2))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| "cannot create capability HTTP client")?;
        Ok(Self { client, api_key })
    }

    pub(crate) fn fetch(&self, worker: &str) -> Result<Descriptor, FetchError> {
        let mut request = self.client.get(format!(
            "{}{CAPABILITIES_PATH}",
            worker.trim_end_matches('/')
        ));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().map_err(|_| FetchError::Transport)?;
        match response.status().as_u16() {
            404 => return Err(FetchError::Unavailable),
            409 => return Err(FetchError::Unsupported),
            200 => {}
            _ => return Err(FetchError::Transport),
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_DESCRIPTOR_BYTES)
        {
            return Err(FetchError::Invalid);
        }
        let mut bytes = Vec::new();
        response
            .take(MAX_DESCRIPTOR_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| FetchError::Transport)?;
        if bytes.len() as u64 > MAX_DESCRIPTOR_BYTES {
            return Err(FetchError::Invalid);
        }
        serde_json::from_slice(&bytes).map_err(|_| FetchError::Invalid)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn descriptor() -> Descriptor {
        serde_json::from_value(json!({
            "schema_version":1,"vllm_version":"0.29.0","mechanism":"normal_full_attention","mechanism_version":1,
            "namespace":{"model":"/model","served_model_names":["model"],"revision":null,"dtype":"torch.bfloat16",
                "quantization":null,"cache_dtype":"auto","weight_version":"default"},
            "groups":[{"group_id":0,"kind":"full_attention","layer_count":24,"allocation_block_tokens":16,"effective_block_tokens":16}],
            "hash":{"algorithm":"sha256_cbor","width_bytes":32,"representation":"bytes","seed":0,"root_hex":seed_root_hex(0),"extra_keys":"none","block_tokens":16},
            "reuse":{"alignment_tokens":16,"terminal_recompute_tokens":1},
            "execution":{"mode":"normal","dp":1,"tp":1,"pp":1,"dcp":1,"pcp":1,"prefix_caching":true,"speculation":false,"connector":false,"offload":false},
            "events":{"publisher":"zmq","epoch":"0123456789abcdef0123456789abcdef","topic":"kv.0123456789abcdef0123456789abcdef","configured_endpoint":"tcp://*:5557","resolved_endpoint":"tcp://0.0.0.0:5557","dp_rank":0,"sequence":"u64_be_monotonic_per_epoch","next_sequence":0,"payload":"vllm_kv_events_v1"}
        })).unwrap()
    }

    #[test]
    fn kv_capability_version_inventory_units_and_hash_are_required() {
        let value = serde_json::to_value(descriptor()).unwrap();
        for (pointer, replacement) in [
            ("/schema_version", json!(2)),
            ("/mechanism", json!("unknown")),
            ("/mechanism_version", json!(2)),
            ("/groups", json!([])),
            ("/groups/0/kind", json!("sliding_window")),
            ("/groups/0/layer_count", json!(0)),
            ("/groups/0/effective_block_tokens", json!(32)),
            ("/hash/width_bytes", json!(8)),
            ("/hash/root_hex", json!("00")),
            ("/hash/extra_keys", json!("salt")),
            ("/reuse/terminal_recompute_tokens", json!(0)),
            ("/execution/dcp", json!(2)),
            ("/execution/speculation", json!(true)),
            ("/execution/offload", json!(true)),
            ("/execution/connector", json!(true)),
            ("/execution/prefix_caching", json!(false)),
            ("/events/epoch", json!("short")),
            ("/events/topic", json!("kv")),
            ("/events/dp_rank", json!(1)),
            ("/events/resolved_endpoint", json!("tcp://*:5558")),
        ] {
            let mut changed = value.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            let parsed: Descriptor = serde_json::from_value(changed).unwrap();
            assert!(
                parsed
                    .validate("tcp://worker:5557", 16, 0, "model")
                    .is_err(),
                "{pointer}"
            );
        }
        for field in [
            "schema_version",
            "namespace",
            "groups",
            "hash",
            "reuse",
            "execution",
            "events",
        ] {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<Descriptor>(missing).is_err());
        }
        for field in ["revision", "quantization"] {
            let mut missing = value.clone();
            missing["namespace"].as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<Descriptor>(missing).is_err());
        }
        let mut extended = value;
        extended["optional_extension"] = json!({"future":true});
        let parsed: Descriptor = serde_json::from_value(extended).unwrap();
        assert!(parsed.validate("tcp://worker:5557", 16, 0, "model").is_ok());
    }

    #[test]
    fn kv_capability_family_independent_and_explicit_overrides_conflict() {
        let mut parsed = descriptor();
        parsed.namespace.model = "any-non-qwen-model".into();
        assert!(parsed.validate("tcp://worker:5557", 16, 0, "model").is_ok());
        assert!(parsed
            .validate("tcp://worker:5557", 32, 0, "model")
            .is_err());
        assert!(parsed
            .validate("tcp://worker:5557", 16, 1, "model")
            .is_err());
        assert!(parsed
            .validate("tcp://worker:5557", 16, 0, "other")
            .is_err());
        assert_eq!(
            seed_root_hex(0),
            "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e"
        );
    }

    #[test]
    fn kv_capability_epoch_is_not_namespace_and_metadata_is_not_inventory() {
        let original = descriptor();
        let mut next = original.clone();
        next.events.epoch = "fedcba9876543210fedcba9876543210".into();
        next.events.topic = format!("kv.{}", next.events.epoch);
        next.events.next_sequence = 42;
        assert!(next.compatible_with(&original));
        assert!(!next.same_publisher(&original));
        next.namespace.weight_version = "changed".into();
        assert!(!next.compatible_with(&original));
    }

    #[test]
    fn kv_capability_http_absent_unsupported_invalid_and_bounded() {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            thread,
        };
        fn fetch(status: &str, body: String) -> Result<Descriptor, FetchError> {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let status = status.to_owned();
            let server = thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                let mut request = [0; 4096];
                let n = socket.read(&mut request).unwrap();
                assert!(std::str::from_utf8(&request[..n])
                    .unwrap()
                    .starts_with("GET /v1/kv-cache/capabilities "));
                let _ = write!(
                    socket,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            });
            let result = CapabilityClient::new(None)
                .unwrap()
                .fetch(&format!("http://{address}"));
            server.join().unwrap();
            result
        }
        assert_eq!(
            fetch("404 Not Found", "{}".into()),
            Err(FetchError::Unavailable)
        );
        assert_eq!(
            fetch("409 Conflict", "{}".into()),
            Err(FetchError::Unsupported)
        );
        assert_eq!(fetch("200 OK", "{}".into()), Err(FetchError::Invalid));
        assert_eq!(
            fetch("200 OK", "x".repeat(MAX_DESCRIPTOR_BYTES as usize + 1)),
            Err(FetchError::Invalid)
        );
        assert_eq!(
            fetch("200 OK", serde_json::to_string(&descriptor()).unwrap()),
            Ok(descriptor())
        );
    }
}

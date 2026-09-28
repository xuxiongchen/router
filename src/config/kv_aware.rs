use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Opt-in contract for the first static, single-model KV-aware deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvAwareConfig {
    /// Must match every worker's block size and sha256_cbor configuration.
    pub block_size: usize,
    pub hash_seed: u32,
    /// Local tokenizer.json (or directory), with neighboring Qwen3 Dense
    /// config.json and tokenizer_config.json matching the workers.
    pub tokenizer_path: String,
    /// Accepted request model alias; omitted request.model is also accepted.
    pub model: String,
    pub topic: String,
    pub default_port: u16,
    pub worker_endpoints: HashMap<String, String>,
    /// Hard bound on (worker, block) ownership records.
    pub index_max_entries: usize,
    /// Opt-in cache-first guard: retain affinity within one excess in-flight
    /// request of the least-loaded eligible worker. Not a universal cost model.
    pub load_guard: bool,
    /// Opt-in derived single-text Completion body, only with the vLLM bridge.
    /// Unsupported shapes/headers remain on the original byte-forward path.
    pub completion_token_input: bool,
    pub history: KvHistoryConfig,
}

impl Default for KvAwareConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            hash_seed: 0,
            tokenizer_path: String::new(),
            model: "Qwen/Qwen3-0.6B".into(),
            topic: String::new(),
            default_port: 5557,
            worker_endpoints: HashMap::new(),
            index_max_entries: 100_000,
            load_guard: false,
            completion_token_input: false,
            history: KvHistoryConfig::default(),
        }
    }
}

/// Bounded advisory exact-token history, not the standalone cache_aware policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvHistoryConfig {
    pub enabled: bool,
    pub history_ttl_secs: u64,
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    /// Total prefix-token + session associations, including pending attempts.
    pub max_tree_size: usize,
}

impl Default for KvHistoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            history_ttl_secs: 300,
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 120,
            max_tree_size: 1 << 26,
        }
    }
}

impl KvHistoryConfig {
    pub(crate) fn store_config(&self) -> crate::policies::exact_history::ExactHistoryConfig {
        use std::time::Duration;
        crate::policies::exact_history::ExactHistoryConfig {
            cache_threshold: self.cache_threshold,
            balance_abs_threshold: self.balance_abs_threshold,
            balance_rel_threshold: self.balance_rel_threshold,
            history_ttl: Duration::from_secs(self.history_ttl_secs),
            eviction_interval: Duration::from_secs(self.eviction_interval_secs),
            max_tree_size: self.max_tree_size,
        }
    }
}

//! Controlled, default-off static Completion activation. Cold HTTP probes are
//! finite conformance evidence, not asset authentication or engine admission.
//! Workers must remain immutable until the router is drained and restarted.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use reqwest::{Client, Method};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

use crate::core::{Worker, WorkerRegistry};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractFile {
    assets_path: PathBuf,
    model: String,
    #[serde(default)]
    aliases: Vec<String>,
    supervised_immutable_workers: bool,
    #[serde(default)]
    local_effective_encoding: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalEffectiveEncoding {
    schema: String,
    asset_sha256: HashMap<String, String>,
    versions: EffectiveEncodingVersions,
    vllm_source_commit: String,
    backend_tokenizer: Value,
    split_special_tokens: bool,
    backend_encode_special_tokens: bool,
    route: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectiveEncodingVersions {
    vllm: String,
    transformers: String,
    tokenizers: String,
}

/// The exact owned asset snapshot used by the production Completion loader.
/// This does not establish a remote Worker's input contract.
pub struct CompletionInputAssets {
    pub tokenizer: Arc<dyn Tokenizer>,
    pub model_vocab_size: u32,
    pub model: String,
    aliases: Vec<String>,
    tokenizer_config: Value,
    // A local official-loader proof under supervised immutability, never a
    // remote tokenizer object or a full-input-domain conformance certificate.
    has_local_effective_encoding: bool,
}

impl CompletionInputAssets {
    pub fn accepts_model(&self, model: Option<&str>) -> bool {
        model.is_none_or(|model| model == self.model || self.aliases.iter().any(|a| a == model))
    }

    /// Establish model scope before using this model's tokenizer/vocabulary.
    pub fn prepare(
        &self,
        request: &crate::protocols::spec::CompletionRequest,
    ) -> Result<crate::backend::PreparedCompletion, crate::backend::CompletionPreparationError>
    {
        use crate::backend::CompletionPreparationError::{InvalidRequest, Unsupported};
        if !self.accepts_model(request.model.as_deref()) || request.lora_path.is_some() {
            return Err(Unsupported(
                "Completion input contract does not cover this model/adapter".into(),
            ));
        }
        if request
            .other
            .get("prompt_embeds")
            .is_some_and(|value| !value.is_null())
        {
            return Err(Unsupported(
                "Completion input contract does not cover prompt embeddings".into(),
            ));
        }
        if matches!(&request.prompt, crate::protocols::spec::PromptInput::String(text) if text.is_empty())
        {
            return Err(InvalidRequest("Completion prompt must be nonempty".into()));
        }
        crate::backend::prepare_completion_classified(
            request,
            self.tokenizer.as_ref(),
            self.model_vocab_size,
        )
    }

    pub async fn validate_workers(
        &self,
        workers: &[String],
        client: &Client,
        api_key: Option<&str>,
    ) -> Result<(), String> {
        let mut cohort_root = None;
        for worker in workers {
            let version = worker_json(client, api_key, worker, "/version", None).await?;
            if !matches!(version["version"].as_str(), Some("0.29.0" | "0.29.0+cpu")) {
                return Err(format!(
                    "Completion contract requires vLLM 0.29.0 at {worker}"
                ));
            }
            let server = worker_json(
                client,
                api_key,
                worker,
                "/server_info?config_format=json",
                None,
            )
            .await?;
            self.validate_effective_environment(&server)
                .map_err(|error| format!("{error} at {worker}"))?;
            let config = server["vllm_config"].as_object().ok_or_else(|| {
                format!("Completion contract requires JSON /server_info at {worker}; isolated Worker must enable VLLM_SERVER_DEV_MODE")
            })?;
            let model_config = config.get("model_config").ok_or("missing model_config")?;
            for (field, expected) in [
                ("tokenizer_mode", json!("hf")),
                ("skip_tokenizer_init", json!(false)),
                ("trust_remote_code", json!(false)),
                ("io_processor_plugin", Value::Null),
            ] {
                if model_config.get(field) != Some(&expected) {
                    return Err(format!(
                        "Completion contract incompatible model_config.{field} at {worker}"
                    ));
                }
            }
            if model_config
                .get("hf_overrides")
                .is_none_or(|value| value != &json!({}))
            {
                return Err(format!(
                    "Completion contract requires empty hf_overrides at {worker}"
                ));
            }
            if config
                .get("parallel_config")
                .and_then(|p| p.get("data_parallel_size"))
                != Some(&json!(1))
                || config.get("lora_config") != Some(&Value::Null)
                || config.get("speculative_config") != Some(&Value::Null)
            {
                return Err(format!(
                    "Completion contract requires DP=1 without LoRA/speculation at {worker}"
                ));
            }
            let models = worker_json(client, api_key, worker, "/v1/models", None).await?;
            let cards = models["data"].as_array().ok_or("missing model cards")?;
            let mut worker_root = None;
            for name in std::iter::once(&self.model).chain(&self.aliases) {
                let card = cards
                    .iter()
                    .find(|card| card["id"].as_str() == Some(name))
                    .ok_or_else(|| {
                        format!("Completion contract model/alias {name} missing at {worker}")
                    })?;
                let root = card["root"]
                    .as_str()
                    .filter(|root| !root.is_empty())
                    .ok_or("missing base model root")?;
                if card.get("parent") != Some(&Value::Null)
                    || worker_root
                        .as_ref()
                        .is_some_and(|previous| previous != root)
                {
                    return Err(format!(
                        "Completion contract requires aliases of one base model at {worker}"
                    ));
                }
                worker_root = Some(root.to_string());
            }
            if cards.iter().any(|card| {
                card.get("parent") != Some(&Value::Null)
                    || card["root"].as_str() != worker_root.as_deref()
            }) || cohort_root
                .as_ref()
                .is_some_and(|previous| Some(previous) != worker_root.as_ref())
                || model_config["model"].as_str() != worker_root.as_deref()
            {
                return Err(format!(
                    "Completion contract worker cohort has different model roots at {worker}"
                ));
            }
            cohort_root = worker_root;
            let info = worker_json(client, api_key, worker, "/tokenizer_info", None).await?;
            self.validate_tokenizer_info(&info).map_err(|error| format!("{error} at {worker}; isolated Worker must enable --enable-tokenizer-info-endpoint"))?;
            for text in [
                "Hello world",
                "中英文 café e\u{301} 🙂",
                " \t\n\r\n a\u{a0}b\u{2003}c\u{3000}d",
                "<|endoftext|><|im_start|>user\nhello<|im_end|><think>literal</think>",
            ] {
                for special in [None, Some(true), Some(false)] {
                    let mut request = json!({"model": self.model, "prompt": text});
                    if let Some(special) = special {
                        request["add_special_tokens"] = json!(special);
                    }
                    let actual =
                        worker_json(client, api_key, worker, "/tokenize", Some(&request)).await?;
                    let expected = self
                        .tokenizer
                        .encode(text, special.unwrap_or(true))
                        .map_err(|error| error.to_string())?;
                    if actual["tokens"] != json!(expected)
                        || actual["count"] != json!(expected.len())
                    {
                        return Err(format!("Completion contract /tokenize full-array conformance failed at {worker}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_tokenizer_info(&self, info: &Value) -> Result<(), String> {
        let class = info["tokenizer_class"]
            .as_str()
            .ok_or("missing tokenizer_class")?;
        let configured_class = self.tokenizer_config["tokenizer_class"]
            .as_str()
            .ok_or("missing local tokenizer_class")?;
        let fast_class = if configured_class.ends_with("Fast") {
            configured_class.to_string()
        } else {
            format!("{configured_class}Fast")
        };
        // Stock 0.29 pools only HF fast backends. Its cached concrete class may
        // omit "Fast" under Transformers 5. /tokenizer_info does not expose
        // is_fast: these names plus full-array probes are only a controlled-
        // launch compatibility check, not proof of tokenizer identity.
        if class != "TokenizersBackend"
            && class != "TokenizerPool"
            && class != fast_class
            && class != format!("Cached{fast_class}")
            && class != format!("TokenizerPoolCached{configured_class}")
            && class != format!("TokenizerPoolCached{fast_class}")
        {
            return Err(format!(
                "Completion contract requires an HF fast tokenizer, got {class}"
            ));
        }
        for field in [
            "add_bos_token",
            "add_eos_token",
            "padding_side",
            "truncation_side",
            "bos_token",
            "eos_token",
            "unk_token",
        ] {
            if let Some(local) = self.tokenizer_config.get(field) {
                let local = local.get("content").unwrap_or(local);
                if info.get(field).is_some_and(|remote| remote != local)
                    || (!self.has_local_effective_encoding && info.get(field).is_none())
                {
                    return Err(format!(
                        "Completion contract tokenizer setting {field} differs"
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_effective_environment(&self, server: &Value) -> Result<(), String> {
        if self.has_local_effective_encoding
            && (server["vllm_env"]["VLLM_USE_FASTOKENS"] != json!(false)
                || !server["system_env"]["pip_packages"]
                    .as_str()
                    .is_some_and(|packages| {
                        packages
                            .lines()
                            .any(|line| line.trim() == "transformers==5.17.0")
                    }))
        {
            return Err("Completion local effective encoding requires Worker Transformers 5.17.0 without fastokens provenance".into());
        }
        Ok(())
    }
}

/// Read the three input assets once; validate and load that same private copy.
/// No weights, network loader, family/revision allowlist, or hot-path files.
pub fn load_completion_input_assets(contract_path: &Path) -> Result<CompletionInputAssets, String> {
    let file: ContractFile =
        serde_json::from_slice(&std::fs::read(contract_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("invalid Completion input contract: {error}"))?;
    let mut names = HashSet::new();
    if !file.supervised_immutable_workers
        || std::iter::once(&file.model)
            .chain(&file.aliases)
            .any(|name| name.trim().is_empty() || !names.insert(name))
    {
        return Err("Completion contract requires supervised_immutable_workers=true and unique nonempty model/aliases".into());
    }
    let assets_path = if file.assets_path.is_absolute() {
        file.assets_path
    } else {
        contract_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(file.assets_path)
    };
    let snapshot = tempfile::tempdir().map_err(|error| error.to_string())?;
    let mut assets = HashMap::new();
    let mut asset_sha256 = HashMap::new();
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        let bytes = std::fs::read(assets_path.join(name))
            .map_err(|error| format!("Completion asset {name}: {error}"))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Completion asset {name}: {error}"))?;
        if !value.is_object() {
            return Err(format!("Completion asset {name} must be a JSON object"));
        }
        asset_sha256.insert(name.to_string(), format!("{:x}", Sha256::digest(&bytes)));
        std::fs::write(snapshot.path().join(name), bytes).map_err(|error| error.to_string())?;
        assets.insert(name, value);
    }
    let config = &assets["config.json"];
    let tokenizer_config = &assets["tokenizer_config.json"];
    if config["is_encoder_decoder"] == true
        || [config, tokenizer_config].iter().any(|value| {
            value
                .get("auto_map")
                .is_some_and(|map| !map.is_null() && map != &json!({}))
        })
        || tokenizer_config
            .get("processor_class")
            .is_some_and(|value| !value.is_null())
    {
        return Err(
            "Completion contract does not support encoder-decoder/custom remote processor assets"
                .into(),
        );
    }
    let model_vocab_size = config["vocab_size"]
        .as_u64()
        .and_then(|size| u32::try_from(size).ok())
        .filter(|&size| size != 0)
        .ok_or("Completion config requires a nonzero u32 vocab_size")?;
    crate::backend::validate_completion_tokenizer_definition(&assets["tokenizer.json"])?;
    let definition: tokenizers::Tokenizer =
        serde_json::from_value(assets["tokenizer.json"].clone())
            .map_err(|error| error.to_string())?;
    if definition
        .get_vocab(true)
        .values()
        .any(|&id| id >= model_vocab_size)
    {
        return Err("Completion tokenizer IDs exceed local model vocabulary".into());
    }
    validate_added_tokens(
        tokenizer_config,
        &assets["tokenizer.json"],
        model_vocab_size,
    )?;
    validate_special_tokens(tokenizer_config, &assets["tokenizer.json"])?;
    if let Some(path) = &file.local_effective_encoding {
        let path = if path.is_absolute() {
            path.clone()
        } else {
            contract_path.parent().unwrap_or(Path::new(".")).join(path)
        };
        let evidence: LocalEffectiveEncoding =
            serde_json::from_slice(&std::fs::read(path).map_err(|error| error.to_string())?)
                .map_err(|error| format!("invalid local effective encoding: {error}"))?;
        validate_local_effective_encoding(evidence, &asset_sha256, tokenizer_config, &definition)?;
    }
    let tokenizer = HuggingFaceTokenizer::new_hf(&snapshot.path().join("tokenizer.json"))
        .map_err(|error| error.to_string())?;
    if let Some(decoder) = tokenizer_config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
    {
        for (id, token) in decoder {
            if tokenizer.token_to_id(
                token["content"]
                    .as_str()
                    .ok_or("missing added-token content")?,
            ) != id.parse().ok()
            {
                return Err(format!(
                    "Completion loaded added-token ID differs from configuration: {id}"
                ));
            }
        }
    }
    for (content, id) in tokenizer.added_vocab() {
        if *id >= model_vocab_size || tokenizer.token_to_id(content) != Some(*id) {
            return Err(
                "Completion loaded added-token IDs are inconsistent with model vocabulary".into(),
            );
        }
    }
    Ok(CompletionInputAssets {
        tokenizer: Arc::new(tokenizer),
        model_vocab_size,
        model: file.model,
        aliases: file.aliases,
        tokenizer_config: tokenizer_config.clone(),
        has_local_effective_encoding: file.local_effective_encoding.is_some(),
    })
}

fn validate_local_effective_encoding(
    evidence: LocalEffectiveEncoding,
    asset_sha256: &HashMap<String, String>,
    config: &Value,
    definition: &tokenizers::Tokenizer,
) -> Result<(), String> {
    if evidence.schema != "vllm029-hf517-local-effective-v1"
        || evidence.asset_sha256 != *asset_sha256
        || !matches!(evidence.versions.vllm.as_str(), "0.29.0" | "0.29.0+cpu")
        || evidence.versions.transformers != "5.17.0"
        || evidence.versions.tokenizers != "0.23.2"
        || evidence.vllm_source_commit != "g98dff2a81"
        || evidence.route != "stock_hf_tokenizers_completion_v1"
        || evidence.split_special_tokens
        || evidence.backend_encode_special_tokens
    {
        return Err("Completion local effective encoding provenance/assets/flags differ".into());
    }
    let local = serde_json::to_value(definition).map_err(|error| error.to_string())?;
    // new_hf merges absent decoder entries. Do not compare the pre-merge
    // definition unless every configured token is already materialized.
    if config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .is_some_and(|decoder| {
            decoder.keys().any(|id| {
                !local["added_tokens"].as_array().is_some_and(|tokens| {
                    tokens
                        .iter()
                        .any(|token| id.parse::<u64>().ok() == token["id"].as_u64())
                })
            })
        })
    {
        return Err(
            "Completion local effective encoding requires materialized added tokens".into(),
        );
    }
    if input_encoding_projection(local) != input_encoding_projection(evidence.backend_tokenizer) {
        return Err("Completion effective backend input encoding differs from Rust".into());
    }
    Ok(())
}

fn input_encoding_projection(mut definition: Value) -> Value {
    // Responses are still forwarded unchanged: decoder behavior is not used.
    if let Some(fields) = definition.as_object_mut() {
        fields.remove("decoder");
        for stage in ["pre_tokenizer", "post_processor"] {
            if let Some(value) = fields.get_mut(stage) {
                remove_byte_level_offsets(value);
            }
        }
    }
    definition
}

fn remove_byte_level_offsets(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            // trim_offsets changes offsets, not Completion input token IDs.
            // Preserve add_prefix_space/use_regex and every other input field.
            if fields.get("type").and_then(Value::as_str) == Some("ByteLevel") {
                fields.remove("trim_offsets");
            }
            for value in fields.values_mut() {
                remove_byte_level_offsets(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                remove_byte_level_offsets(value);
            }
        }
        _ => {}
    }
}

fn validate_added_tokens(
    config: &Value,
    definition: &Value,
    vocab_size: u32,
) -> Result<(), String> {
    let Some(decoder) = config.get("added_tokens_decoder") else {
        return Ok(());
    };
    let decoder = decoder
        .as_object()
        .ok_or("Completion added_tokens_decoder must be an object")?;
    let mut ids = HashSet::new();
    for (key, token) in decoder {
        let id = key
            .parse::<u32>()
            .map_err(|_| format!("invalid Completion added-token ID {key}"))?;
        if id >= vocab_size || !ids.insert(id) || token["content"].as_str().is_none() {
            return Err(format!("invalid Completion added-token entry {key}"));
        }
        if token
            .get("id")
            .is_some_and(|value| value.as_u64() != Some(u64::from(id)))
        {
            return Err(format!("inconsistent Completion added-token ID {key}"));
        }
        let existing = definition["added_tokens"].as_array().and_then(|tokens| {
            tokens
                .iter()
                .find(|token| token["id"].as_u64() == Some(u64::from(id)))
        });
        if existing.is_some_and(|existing| existing["content"] != token["content"]) {
            return Err(format!("inconsistent Completion added-token content {key}"));
        }
        for field in ["single_word", "lstrip", "rstrip", "normalized", "special"] {
            let flag = token.get(field).and_then(Value::as_bool).ok_or_else(|| {
                format!("Completion added-token {field} must be explicit for {key}")
            })?;
            if existing.is_some_and(|existing| {
                existing
                    .get(field)
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    != flag
            }) {
                return Err(format!(
                    "inconsistent Completion added-token {field} for {key}"
                ));
            }
        }
    }
    Ok(())
}

/// Transformers may add/reconfigure special tokens beyond decoder entries.
/// Accept only declarations already materialized with identical properties in
/// the shared assets; do not duplicate the Python constructor in the router.
fn validate_special_tokens(config: &Value, definition: &Value) -> Result<(), String> {
    const NAMED: [&str; 7] = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ];
    if config
        .get("model_specific_special_tokens")
        .is_some_and(|value| !value.is_null() && value != &json!({}))
        || config.as_object().is_some_and(|fields| {
            fields.iter().any(|(key, value)| {
                key.ends_with("_token")
                    && !NAMED.contains(&key.as_str())
                    && (value.is_string() || value.is_object())
            })
        })
    {
        return Err(
            "Completion custom/model-specific special-token declarations are unsupported".into(),
        );
    }
    if config
        .get("split_special_tokens")
        .is_some_and(|value| value != &json!(false))
    {
        return Err("Completion split_special_tokens must be false".into());
    }
    let decoder = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object);
    let check = |declared: &Value, named: bool| -> Result<(), String> {
        if declared.is_null() {
            return Ok(());
        }
        let content = declared
            .as_str()
            .or_else(|| declared.get("content").and_then(Value::as_str))
            .ok_or("invalid Completion special-token declaration")?;
        let configured = decoder.and_then(|tokens| {
            tokens
                .values()
                .find(|token| token["content"].as_str() == Some(content))
        });
        let materialized = definition["added_tokens"]
            .as_array()
            .and_then(|tokens| {
                tokens
                    .iter()
                    .find(|token| token["content"].as_str() == Some(content))
            })
            .or(configured)
            .ok_or_else(|| format!("Completion special-token {content} is not materialized"))?;
        let effective = if named {
            configured.unwrap_or(declared)
        } else {
            declared
        };
        if materialized["special"] != true {
            return Err(format!(
                "Completion special-token {content} is not marked special"
            ));
        }
        for field in ["single_word", "lstrip", "rstrip", "normalized"] {
            let expected = if effective.is_string() {
                false
            } else {
                effective
                    .get(field)
                    .and_then(Value::as_bool)
                    .ok_or_else(|| {
                        format!("Completion special-token {content} requires explicit {field}")
                    })?
            };
            if materialized.get(field).and_then(Value::as_bool) != Some(expected) {
                return Err(format!(
                    "Completion special-token {content} has different {field}"
                ));
            }
        }
        Ok(())
    };
    for field in NAMED {
        if let Some(token) = config.get(field) {
            check(token, true)?;
        }
    }
    for field in ["extra_special_tokens", "additional_special_tokens"] {
        if let Some(tokens) = config.get(field).filter(|value| !value.is_null()) {
            for token in tokens
                .as_array()
                .ok_or_else(|| format!("Completion {field} requires an array"))?
            {
                check(token, false)?;
            }
        }
    }
    Ok(())
}

async fn worker_json(
    client: &Client,
    api_key: Option<&str>,
    worker: &str,
    path: &str,
    body: Option<&Value>,
) -> Result<Value, String> {
    let mut request = client
        .request(
            if body.is_some() {
                Method::POST
            } else {
                Method::GET
            },
            format!("{}{path}", worker.trim_end_matches('/')),
        )
        .timeout(Duration::from_secs(10));
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    if let Some(body) = body {
        request = request.json(body);
    }
    request
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|error| format!("Completion contract {worker}{path}: {error}"))?
        .json()
        .await
        .map_err(|error| format!("Completion contract {worker}{path}: {error}"))
}

/// Immutable registry identities and revision. URL equality cannot renew this
/// contract. Monotonic revision changes close admission until router restart;
/// already-dispatched response guards retain their existing drain ownership.
pub(crate) struct CompletionWorkerBinding {
    revision: u64,
    workers: Vec<Arc<dyn Worker>>,
}

impl std::fmt::Debug for CompletionWorkerBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionWorkerBinding")
            .field("revision", &self.revision)
            .finish_non_exhaustive()
    }
}

impl CompletionWorkerBinding {
    pub(crate) fn capture(registry: &WorkerRegistry) -> Self {
        Self {
            revision: registry.revision(),
            workers: registry.get_all(),
        }
    }

    pub(crate) fn valid(&self, registry: &WorkerRegistry) -> bool {
        registry.revision() == self.revision
            && self.workers.iter().all(|bound| {
                registry
                    .get_by_url(bound.url())
                    .is_some_and(|current| Arc::ptr_eq(bound, &current))
            })
    }

    pub(crate) fn contains(&self, worker: &Arc<dyn Worker>) -> bool {
        self.workers.iter().any(|bound| Arc::ptr_eq(bound, worker))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::CompletionPreparationError;
    use crate::core::{BasicWorker, WorkerType};

    pub(crate) fn assets_fixture() -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            include_bytes!("../../tests/fixtures/tokenizer/completion_word_level.json"),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            r#"{"vocab_size":16,"is_encoder_decoder":false}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"tokenizer_class":"PreTrainedTokenizerFast","added_tokens_decoder":{}}"#,
        )
        .unwrap();
        let contract = directory.path().join("contract.json");
        std::fs::write(&contract, json!({"assets_path": directory.path(), "model": "base", "aliases": ["alias"], "supervised_immutable_workers": true}).to_string()).unwrap();
        (directory, contract)
    }

    pub(crate) fn effective_fixture(directory: &Path, contract: &Path) -> Value {
        let config_path = directory.join("tokenizer_config.json");
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["add_bos_token"] = json!(false);
        std::fs::write(config_path, config.to_string()).unwrap();
        let definition: tokenizers::Tokenizer =
            serde_json::from_slice(&std::fs::read(directory.join("tokenizer.json")).unwrap())
                .unwrap();
        let hashes: HashMap<_, _> = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            .into_iter()
            .map(|name| {
                (
                    name,
                    format!(
                        "{:x}",
                        Sha256::digest(std::fs::read(directory.join(name)).unwrap())
                    ),
                )
            })
            .collect();
        // Unit fixture only. Real startup uses the public official-loader
        // exporter, not this hand-written shape as a Python behavior oracle.
        let evidence = json!({
            "schema": "vllm029-hf517-local-effective-v1",
            "asset_sha256": hashes,
            "versions": {"vllm":"0.29.0+cpu", "transformers":"5.17.0", "tokenizers":"0.23.2"},
            "vllm_source_commit": "g98dff2a81",
            "backend_tokenizer": serde_json::to_value(definition).unwrap(),
            "split_special_tokens": false,
            "backend_encode_special_tokens": false,
            "route": "stock_hf_tokenizers_completion_v1",
        });
        std::fs::write(directory.join("effective.json"), evidence.to_string()).unwrap();
        let mut body: Value = serde_json::from_slice(&std::fs::read(contract).unwrap()).unwrap();
        body["local_effective_encoding"] = json!("effective.json");
        std::fs::write(contract, body.to_string()).unwrap();
        evidence
    }

    #[test]
    fn completion_missing_setting_requires_local_effective_evidence() {
        let (directory, contract) = assets_fixture();
        effective_fixture(directory.path(), &contract);
        let mut body: Value = serde_json::from_slice(&std::fs::read(&contract).unwrap()).unwrap();
        body.as_object_mut()
            .unwrap()
            .remove("local_effective_encoding");
        std::fs::write(&contract, body.to_string()).unwrap();
        let assets = load_completion_input_assets(&contract).unwrap();
        assert!(assets
            .validate_tokenizer_info(&json!({
                "tokenizer_class":"TokenizerPoolCachedPreTrainedTokenizerFast"
            }))
            .unwrap_err()
            .contains("add_bos_token"));
    }

    #[test]
    fn completion_captured_stock_info_reproduces_missing_declaration_failure() {
        let info: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/tokenizer/vllm029_hf517_tokenizer_info.json"
        ))
        .unwrap();
        assert!(info.get("add_bos_token").is_none());
        assert!(info.get("add_eos_token").is_none());
        let (_directory, contract) = assets_fixture();
        let mut assets = load_completion_input_assets(&contract).unwrap();
        // This tests the captured metadata absence, not a claim that the toy
        // fixture is the remote Qwen tokenizer or an official encoding oracle.
        assets.tokenizer_config["tokenizer_class"] = json!("Qwen2Tokenizer");
        assets.tokenizer_config["add_bos_token"] = json!(false);
        assert!(assets
            .validate_tokenizer_info(&info)
            .unwrap_err()
            .contains("add_bos_token"));
    }

    #[test]
    fn completion_effective_evidence_accepts_missing_not_conflicting_settings() {
        let (directory, contract) = assets_fixture();
        effective_fixture(directory.path(), &contract);
        let assets = load_completion_input_assets(&contract).unwrap();
        let info = json!({"tokenizer_class":"TokenizerPoolCachedPreTrainedTokenizerFast"});
        assert!(assets.validate_tokenizer_info(&info).is_ok());
        let mut conflicting = info;
        conflicting["add_bos_token"] = json!(true);
        assert!(assets.validate_tokenizer_info(&conflicting).is_err());
        assert!(assets.validate_effective_environment(&json!({})).is_err());
        let valid = json!({
            "system_env":{"pip_packages":"torch==2.13.0\ntransformers==5.17.0\ntokenizers==0.23.2\n"},
            "vllm_env":{"VLLM_USE_FASTOKENS":false}
        });
        assert!(assets.validate_effective_environment(&valid).is_ok());
        for fastokens in [json!(true), Value::Null] {
            let mut incompatible = valid.clone();
            incompatible["vllm_env"]["VLLM_USE_FASTOKENS"] = fastokens;
            assert!(assets
                .validate_effective_environment(&incompatible)
                .is_err());
        }
        // Stock collect_env does not export tokenizers. Its version is a
        // supervised-launch prerequisite, not an invented HTTP attestation.
        let mut stock = valid.clone();
        stock["system_env"]["pip_packages"] = json!("transformers==5.17.0");
        assert!(assets.validate_effective_environment(&stock).is_ok());
        for packages in ["transformers==5.16.0", "torch==2.13.0"] {
            let mut incompatible = valid.clone();
            incompatible["system_env"]["pip_packages"] = json!(packages);
            assert!(assets
                .validate_effective_environment(&incompatible)
                .is_err());
        }
    }

    #[test]
    fn completion_same_class_does_not_authorize_different_effective_encoding() {
        let (directory, contract) = assets_fixture();
        let evidence = effective_fixture(directory.path(), &contract);
        for field in ["post_processor", "added_tokens"] {
            let mut different = evidence.clone();
            if field == "post_processor" {
                different["backend_tokenizer"][field]["single"] =
                    json!([{"Sequence":{"id":"A", "type_id":0}}]);
            } else {
                different["backend_tokenizer"][field][0]["lstrip"] = json!(true);
            }
            std::fs::write(
                directory.path().join("effective.json"),
                different.to_string(),
            )
            .unwrap();
            assert!(load_completion_input_assets(&contract)
                .err()
                .unwrap()
                .contains("input encoding differs"));
        }
    }

    #[test]
    fn completion_effective_evidence_binds_assets_versions_and_encoding_flags() {
        let (directory, contract) = assets_fixture();
        let evidence = effective_fixture(directory.path(), &contract);
        for (pointer, value) in [
            ("/schema", json!("unverified")),
            ("/asset_sha256/tokenizer.json", json!("unbound")),
            ("/versions/transformers", json!("5.16.0")),
            ("/versions/tokenizers", json!("0.22.2")),
            ("/vllm_source_commit", json!("different")),
            ("/route", json!("custom")),
            ("/split_special_tokens", json!(true)),
            ("/backend_encode_special_tokens", json!(true)),
        ] {
            let mut different = evidence.clone();
            *different.pointer_mut(pointer).unwrap() = value;
            std::fs::write(
                directory.path().join("effective.json"),
                different.to_string(),
            )
            .unwrap();
            assert!(load_completion_input_assets(&contract)
                .err()
                .unwrap()
                .contains("provenance/assets/flags"));
        }
    }

    #[test]
    fn completion_effective_evidence_refuses_implicit_added_token_merge() {
        let (directory, contract) = assets_fixture();
        let config_path = directory.path().join("tokenizer_config.json");
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["added_tokens_decoder"]["8"] = json!({"content":"<extra>",
            "single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true});
        std::fs::write(config_path, config.to_string()).unwrap();
        effective_fixture(directory.path(), &contract);
        assert!(load_completion_input_assets(&contract)
            .err()
            .unwrap()
            .contains("materialized added tokens"));
    }

    #[test]
    fn completion_input_projection_ignores_only_decoder_and_byte_level_offsets() {
        let local = json!({"pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"ByteLevel","trim_offsets":true,"add_prefix_space":false,"use_regex":true}
        ]},"post_processor":{"type":"ByteLevel","trim_offsets":true,"add_prefix_space":false},
        "decoder":{"type":"ByteLevel"},"normalizer":null});
        let mut offsets = local.clone();
        offsets["decoder"] = Value::Null;
        offsets["pre_tokenizer"]["pretokenizers"][0]["trim_offsets"] = json!(false);
        offsets["post_processor"]["trim_offsets"] = json!(false);
        assert_eq!(
            input_encoding_projection(local.clone()),
            input_encoding_projection(offsets.clone())
        );
        offsets["pre_tokenizer"]["pretokenizers"][0]["add_prefix_space"] = json!(true);
        assert_ne!(
            input_encoding_projection(local),
            input_encoding_projection(offsets)
        );
    }

    #[test]
    fn completion_tokenizer_info_accepts_stock_fast_pool_names() {
        let (_directory, contract) = assets_fixture();
        let mut assets = load_completion_input_assets(&contract).unwrap();
        for configured in ["Qwen2Tokenizer", "ExampleTokenizer"] {
            assets.tokenizer_config["tokenizer_class"] = json!(configured);
            for class in [
                format!("TokenizerPoolCached{configured}"),
                format!("TokenizerPoolCached{configured}Fast"),
            ] {
                assert!(assets
                    .validate_tokenizer_info(&json!({"tokenizer_class": class}))
                    .is_ok());
            }
        }
    }

    #[test]
    fn completion_tokenizer_info_rejects_nonmatching_pool_names_and_settings() {
        let (_directory, contract) = assets_fixture();
        let mut assets = load_completion_input_assets(&contract).unwrap();
        assets.tokenizer_config["tokenizer_class"] = json!("ExampleTokenizer");
        for class in [
            "ExampleTokenizer",
            "CachedExampleTokenizer",
            "TokenizerPoolCachedOtherTokenizer",
            "TokenizerPoolCachedExampleTokenizerExtra",
            "OtherTokenizerPoolCachedExampleTokenizer",
        ] {
            assert!(assets
                .validate_tokenizer_info(&json!({"tokenizer_class": class}))
                .is_err());
        }
        assets.tokenizer_config["bos_token"] = json!("<bos>");
        assert!(assets
            .validate_tokenizer_info(&json!({"tokenizer_class": "TokenizerPoolCachedExampleTokenizer", "bos_token": "<other>"}))
            .is_err());
    }

    #[test]
    fn completion_assets_use_owned_snapshot_and_scope_before_vocabulary() {
        let (directory, contract) = assets_fixture();
        let assets = load_completion_input_assets(&contract).unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            "invalid original after load",
        )
        .unwrap();
        let request = |value| serde_json::from_value(value).unwrap();
        for model in [Value::Null, json!("base"), json!("alias")] {
            let body = request(json!({"model": model, "prompt": "hello world"}));
            assert_eq!(assets.prepare(&body).unwrap().token_ids(), [1, 3, 4]);
        }
        for fields in [
            json!({"model":"other", "prompt":[999]}),
            json!({"model":"alias", "prompt":[999], "lora_path":"adapter"}),
            json!({"prompt":[999], "prompt_embeds":"opaque"}),
        ] {
            assert!(matches!(
                assets.prepare(&request(fields)),
                Err(CompletionPreparationError::Unsupported(_))
            ));
        }
        assert!(matches!(
            assets.prepare(&request(json!({"prompt":[16]}))),
            Err(CompletionPreparationError::InvalidRequest(_))
        ));
        assert!(matches!(
            assets.prepare(&request(json!({"prompt":""}))),
            Err(CompletionPreparationError::InvalidRequest(_))
        ));
        assert!(matches!(
            assets.prepare(&request(json!({"prompt":"", "prompt_embeds":"opaque"}))),
            Err(CompletionPreparationError::Unsupported(_))
        ));
        assert_eq!(
            assets
                .prepare(&request(
                    json!({"prompt":"hello", "add_special_tokens":false})
                ))
                .unwrap()
                .token_ids(),
            [3]
        );
    }

    #[test]
    fn completion_assets_reject_incompatible_definitions_and_decoder_entries() {
        for (file, fields) in [
            ("config.json", json!({"vocab_size":7})),
            ("config.json", json!({"vocab_size":0})),
            (
                "config.json",
                json!({"vocab_size":16,"is_encoder_decoder":true}),
            ),
            (
                "tokenizer_config.json",
                json!({"auto_map":{"AutoTokenizer":"remote"}}),
            ),
            (
                "tokenizer_config.json",
                json!({"added_tokens_decoder":{"invalid":{"content":"x"}}}),
            ),
            (
                "tokenizer_config.json",
                json!({"added_tokens_decoder":{"16":{"content":"x"}}}),
            ),
            (
                "tokenizer_config.json",
                json!({"added_tokens_decoder":{"2":{"content":"wrong"}}}),
            ),
            (
                "tokenizer_config.json",
                json!({"added_tokens_decoder":{"2":{"content":"<added>","normalized":true}}}),
            ),
            (
                "tokenizer_config.json",
                json!({"added_tokens_decoder":{"3":{"content":"x","special":"yes"}}}),
            ),
        ] {
            let (directory, contract) = assets_fixture();
            std::fs::write(directory.path().join(file), fields.to_string()).unwrap();
            assert!(
                load_completion_input_assets(&contract).is_err(),
                "accepted {file}: {fields}"
            );
        }
        let (directory, contract) = assets_fixture();
        let mut definition: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/tokenizer/completion_word_level.json"
        ))
        .unwrap();
        definition["truncation"] =
            json!({"direction":"Right","max_length":2,"strategy":"LongestFirst","stride":0});
        std::fs::write(
            directory.path().join("tokenizer.json"),
            definition.to_string(),
        )
        .unwrap();
        assert!(load_completion_input_assets(&contract).is_err());
    }

    #[test]
    fn completion_worker_binding_closes_on_health_revision_or_same_url_replacement() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://worker:8000".into(),
            WorkerType::Regular,
        ));
        registry.register(worker.clone());
        let binding = CompletionWorkerBinding::capture(&registry);
        assert!(binding.valid(&registry));
        assert!(binding.contains(&worker));
        registry.notify_worker_state_change();
        assert!(!binding.valid(&registry));
        let fresh = CompletionWorkerBinding::capture(&registry);
        let replacement: Arc<dyn Worker> =
            Arc::new(BasicWorker::new(worker.url().into(), WorkerType::Regular));
        registry.register(replacement.clone());
        assert!(!fresh.valid(&registry));
        assert!(!fresh.contains(&replacement));
    }

    #[test]
    fn completion_assets_reject_unmaterialized_special_token_declarations() {
        for fields in [
            json!({"model_specific_special_tokens":{"custom_input_token":"<|custom_input|>"}}),
            json!({"custom_input_token":"<|custom_input|>"}),
            json!({"custom_input_token":{"__type":"AddedToken","content":"<added>"}}),
        ] {
            let (directory, contract) = assets_fixture();
            std::fs::write(
                directory.path().join("tokenizer_config.json"),
                fields.to_string(),
            )
            .unwrap();
            assert!(load_completion_input_assets(&contract).is_err());
        }
        for field in [
            "eos_token",
            "extra_special_tokens",
            "additional_special_tokens",
        ] {
            let (directory, contract) = assets_fixture();
            let value = if field == "eos_token" {
                json!("<|custom_input|>")
            } else {
                json!(["<|custom_input|>"])
            };
            let config = json!({"tokenizer_class":"PreTrainedTokenizerFast",field:value});
            std::fs::write(
                directory.path().join("tokenizer_config.json"),
                config.to_string(),
            )
            .unwrap();
            assert!(
                load_completion_input_assets(&contract).is_err(),
                "accepted {config}"
            );
        }
        for invalid in [json!(true), json!("false"), Value::Null] {
            let (directory, contract) = assets_fixture();
            std::fs::write(
                directory.path().join("tokenizer_config.json"),
                json!({"split_special_tokens":invalid}).to_string(),
            )
            .unwrap();
            assert!(load_completion_input_assets(&contract).is_err());
        }
        let (directory, contract) = assets_fixture();
        let config = json!({"tokenizer_class":"PreTrainedTokenizerFast","eos_token":"<added>",
            "extra_special_tokens":["<added>"],"additional_special_tokens":["<s>"],"split_special_tokens":false});
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            config.to_string(),
        )
        .unwrap();
        assert!(load_completion_input_assets(&contract).is_ok());
        let mut definition: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/tokenizer/completion_word_level.json"
        ))
        .unwrap();
        definition["added_tokens"][1]["lstrip"] = json!(true);
        std::fs::write(
            directory.path().join("tokenizer.json"),
            definition.to_string(),
        )
        .unwrap();
        assert!(load_completion_input_assets(&contract).is_err());
    }
}

//! Startup snapshot for the opt-in, static HTTP Completion input contract.
//!
//! The operator binds an alias to immutable local frontend assets and an
//! identically configured Worker cohort. This is not remote attestation.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use vllm_tokenizer::DynTokenizer;

use super::{validate_completion_tokenizer_definition, EngineFrontend};

#[derive(Clone)]
pub(crate) struct CompletionInput {
    pub model: String,
    pub vocab_size: u32,
    pub tokenizer: DynTokenizer,
}

impl std::fmt::Debug for CompletionInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionInput")
            .field("model", &self.model)
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl CompletionInput {
    pub async fn from_env(frontend: &EngineFrontend) -> Result<Option<Self>> {
        let Some(model) = std::env::var("VLLM_ROUTER_COMPLETION_MODEL_ALIAS").ok() else {
            return Ok(None);
        };
        if model.trim().is_empty() || model != model.trim() {
            bail!("VLLM_ROUTER_COMPLETION_MODEL_ALIAS must be a nonempty exact served-model alias");
        }
        let path = PathBuf::from(std::env::var("VLLM_ROUTER_MODEL").context(
            "Completion requires VLLM_ROUTER_MODEL as an immutable local model directory",
        )?);
        let path = path.canonicalize().context("Completion model directory")?;
        if !path.is_dir() {
            bail!("Completion model assets must be a local directory");
        }
        if let Ok(tokenizer) = std::env::var("VLLM_ROUTER_TOKENIZER") {
            let tokenizer = PathBuf::from(tokenizer)
                .canonicalize()
                .context("Completion tokenizer override")?;
            let directory = if tokenizer.is_file() {
                tokenizer.parent().unwrap()
            } else {
                tokenizer.as_path()
            };
            if directory != path {
                bail!("Completion tokenizer override must use the validated model directory");
            }
        }
        let config: Value =
            serde_json::from_slice(&tokio::fs::read(path.join("config.json")).await?)
                .context("Completion model config")?;
        let vocab_size = config
            .get("vocab_size")
            .and_then(Value::as_u64)
            .and_then(|size| u32::try_from(size).ok())
            .filter(|size| *size != 0)
            .context("Completion config.json requires a nonzero u32 vocab_size")?;
        let definition: Value =
            serde_json::from_slice(&tokio::fs::read(path.join("tokenizer.json")).await?)
                .context("Completion tokenizer definition")?;
        validate_completion_tokenizer_definition(&definition).map_err(anyhow::Error::msg)?;
        let tokenizer = frontend.completion_tokenizer(&model).await?;
        Ok(Some(Self {
            model,
            vocab_size,
            tokenizer,
        }))
    }
}

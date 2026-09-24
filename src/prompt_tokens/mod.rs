//! Exact prompt extraction for the explicitly pinned Qwen3-0.6B text profile.
//!
//! These tokens are routing hints only. Callers must forward the original request
//! and use a non-affinity fallback on every error, never an approximate prompt.
//!
//! The restricted renderer is derived from the Qwen team's Apache-2.0 template
//! at the revision below. See tests/fixtures/kv_qwen3/README.md for provenance.

use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::protocols::spec::{CompletionRequest, PromptInput};

pub const QWEN3_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";
pub const QWEN3_TOKENIZER_SHA256: &str =
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";

/// A local, hash-verified tokenizer. This never downloads model assets.
#[derive(Debug)]
pub struct PromptTokenizer {
    tokenizer: Tokenizer,
}

impl PromptTokenizer {
    /// `local_path` is the pinned tokenizer.json file or its containing directory.
    /// The configured workers must serve this same model/tokenizer revision and
    /// its unmodified chat template; the file hash alone cannot verify workers.
    pub fn load(local_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = local_path.as_ref();
        let path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let bytes = std::fs::read(&path).map_err(|error| format!("read tokenizer: {error}"))?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if actual != QWEN3_TOKENIZER_SHA256 {
            return Err(format!(
                "tokenizer SHA256 mismatch for Qwen3-0.6B revision {QWEN3_REVISION}: {actual}"
            ));
        }
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| format!("load pinned tokenizer: {error}"))?;
        Ok(Self { tokenizer })
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| format!("tokenize prompt: {error}"))
    }

    /// Only a single string or a single token-ID sequence has one exact prefix.
    pub fn completion(&self, request: &CompletionRequest) -> Result<Vec<u32>, String> {
        if request.lora_path.is_some()
            || request.session_params.is_some()
            || request.suffix.is_some()
        {
            return Err("completion adapter, session, or suffix is outside the profile".into());
        }
        if let Some(key) = request
            .other
            .keys()
            .find(|key| key.as_str() != "add_special_tokens")
        {
            return Err(format!("unsupported completion field: {key}"));
        }
        let add_special_tokens = match request.other.get("add_special_tokens") {
            None => true,
            Some(Value::Bool(value)) => *value,
            Some(_) => return Err("add_special_tokens must be boolean".into()),
        };
        match &request.prompt {
            PromptInput::String(text) => self.encode(text, add_special_tokens),
            PromptInput::IntArray(ids) if !ids.is_empty() => ids
                .iter()
                .map(|&id| u32::try_from(id).map_err(|_| "negative prompt token ID".into()))
                .collect(),
            PromptInput::IntArray(_) => Err("empty prompt token sequence".into()),
            PromptInput::StringArray(_) | PromptInput::IntBatch(_) => {
                Err("batched completion has no single exact prefix".into())
            }
        }
    }

    /// Use the lossless JSON body, before protocol deserialization can discard
    /// unknown message fields (especially reasoning, tools and multimodal data).
    pub fn chat_raw(&self, request: &Value) -> Result<Vec<u32>, String> {
        self.encode(&render_qwen3_chat(request)?, false)
    }
}

// These fields affect sampling, output, or routing metadata, not the rendered
// input in this pinned text-only profile. Unknown fields fail closed because
// vLLM extensions can change prompt tokens or cache namespaces.
const CHAT_FIELDS: &[&str] = &[
    "model",
    "messages",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "logprobs",
    "top_logprobs",
    "response_format",
    "top_k",
    "min_p",
    "min_tokens",
    "repetition_penalty",
    "stop_token_ids",
    "ignore_eos",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "include_stop_str_in_output",
    "length_penalty",
    "priority",
    "request_id",
    "return_tokens_as_token_ids",
    "add_generation_prompt",
    "continue_final_message",
    "add_special_tokens",
    "chat_template_kwargs",
];

/// The restricted text branch of Qwen3-0.6B's pinned Apache-2.0 chat template.
/// Supports an optional initial system message followed by alternating user and
/// assistant text, ending in a user message. Tools/reasoning/continuations fall
/// back instead of approximating the upstream template's other branches.
pub fn render_qwen3_chat(request: &Value) -> Result<String, String> {
    let object = request.as_object().ok_or("chat body must be an object")?;
    for key in object.keys() {
        if !CHAT_FIELDS.contains(&key.as_str()) {
            return Err(format!("unsupported chat field: {key}"));
        }
    }
    for (key, expected) in [
        ("add_generation_prompt", true),
        ("continue_final_message", false),
        ("add_special_tokens", false),
    ] {
        if let Some(value) = object.get(key) {
            if value.as_bool() != Some(expected) {
                return Err(format!("unsupported {key}"));
            }
        }
    }
    let mut enable_thinking = true;
    if let Some(kwargs) = object.get("chat_template_kwargs") {
        let kwargs = kwargs
            .as_object()
            .ok_or("chat_template_kwargs must be an object")?;
        for (key, value) in kwargs {
            if key != "enable_thinking" {
                return Err(format!("unsupported chat template argument: {key}"));
            }
            enable_thinking = value.as_bool().ok_or("enable_thinking must be boolean")?;
        }
    }
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .ok_or("messages must be a nonempty array")?;
    let mut rendered = String::new();
    let mut expected_role = "user";
    for (index, message) in messages.iter().enumerate() {
        let message = message.as_object().ok_or("message must be an object")?;
        if message.keys().any(|key| key != "role" && key != "content") {
            return Err("only role and string content are supported in each message".into());
        }
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or("missing role")?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or("message content must be text")?;
        if role == "system" && index == 0 {
            // An initial system message does not consume a user turn.
        } else if role == expected_role {
            expected_role = if role == "user" { "assistant" } else { "user" };
        } else {
            return Err("expected optional system, then alternating user/assistant text".into());
        }
        if content.contains("<think>")
            || content.contains("</think>")
            || content.contains("<tool_response>")
            || content.contains("</tool_response>")
        {
            return Err("reasoning or tool markers are outside the text profile".into());
        }
        rendered.push_str("<|im_start|>");
        rendered.push_str(role);
        rendered.push('\n');
        rendered.push_str(content);
        rendered.push_str("<|im_end|>\n");
    }
    if expected_role != "assistant" {
        return Err("the final message must be a user message".into());
    }
    rendered.push_str("<|im_start|>assistant\n");
    if !enable_thinking {
        rendered.push_str("<think>\n\n</think>\n\n");
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tokenizer() -> PromptTokenizer {
        // Small synthetic tokenizer tests extraction and rejection independent
        // of network/model assets. The pinned oracle tests token IDs separately.
        PromptTokenizer {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
        }
    }

    #[test]
    fn completion_token_ids_are_not_encoded_as_text() {
        let body = serde_json::from_value(json!({"prompt": [151644, 8948, 198, 42]})).unwrap();
        assert_eq!(
            tokenizer().completion(&body).unwrap(),
            [151644, 8948, 198, 42]
        );
    }

    #[test]
    fn completion_ambiguous_or_transformed_inputs_fall_back() {
        for body in [
            json!({"prompt": [[1, 2]]}),
            json!({"prompt": ["hello"]}),
            json!({"prompt": [-1]}),
            json!({"prompt": [1], "cache_salt": "salt"}),
            json!({"prompt": [1], "truncate_prompt_tokens": 1}),
            json!({"prompt": [1], "prompt_embeds": []}),
            json!({"prompt": [1], "add_special_tokens": "false"}),
            json!({"prompt": [1], "suffix": "tail"}),
        ] {
            let body = serde_json::from_value(body).unwrap();
            assert!(tokenizer().completion(&body).is_err());
        }
    }

    #[test]
    fn public_rendered_fixtures_match() {
        let fixtures: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/kv_qwen3/prompts.json"))
                .unwrap();
        for case in fixtures["chat"].as_array().unwrap() {
            assert_eq!(
                render_qwen3_chat(&case["request"]).unwrap(),
                case["rendered"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn lossy_or_unverified_chat_forms_fall_back() {
        let base = json!({"messages": [{"role": "user", "content": "Hello"}]});
        for (key, value) in [
            ("tools", json!([])),
            ("cache_salt", json!("salt")),
            ("truncate_prompt_tokens", json!(8)),
            ("chat_template", json!("custom")),
            ("lora_path", json!("adapter")),
            ("continue_final_message", json!(true)),
            ("add_generation_prompt", json!(false)),
            ("add_special_tokens", json!(true)),
            ("chat_template_kwargs", json!({"documents": []})),
            ("chat_template_kwargs", json!({"enable_thinking": "false"})),
        ] {
            let mut request = base.clone();
            request[key] = value;
            assert!(render_qwen3_chat(&request).is_err(), "{key}");
        }
        for extra in [
            "name",
            "reasoning",
            "reasoning_content",
            "tool_calls",
            "unknown",
        ] {
            let mut request = base.clone();
            request["messages"][0][extra] = Value::Null;
            assert!(render_qwen3_chat(&request).is_err(), "{extra}");
        }
        for messages in [
            json!([]),
            json!([{"role": "user", "content": [{"type":"text", "text":"Hello"}]}]),
            json!([{"role":"assistant", "content":"Hello"}]),
            json!([{"role":"user", "content":"<think>reason</think>"}]),
            json!([{"role":"system", "content":"alone"}]),
        ] {
            assert!(render_qwen3_chat(&json!({"messages": messages})).is_err());
        }
    }

    #[test]
    #[ignore = "requires the hash-pinned local tokenizer"]
    fn pinned_python_oracle_token_ids() {
        let path = std::env::var("QWEN3_TOKENIZER_PATH").expect("QWEN3_TOKENIZER_PATH");
        let tokenizer = PromptTokenizer::load(path).unwrap();
        let oracle: Value = match std::env::var("QWEN3_ORACLE_PATH") {
            Ok(path) => serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap(),
            Err(_) => serde_json::from_str(include_str!(
                "../../tests/fixtures/kv_qwen3/python_oracle.json"
            ))
            .unwrap(),
        };
        assert_eq!(oracle["mode"], "canonical-tokenizer");
        assert_eq!(oracle["tokenizer_sha256"], QWEN3_TOKENIZER_SHA256);
        assert_eq!(oracle["revision"], QWEN3_REVISION);
        let fixtures: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/kv_qwen3/prompts.json"))
                .unwrap();
        let expected_chat = fixtures["chat"].as_array().unwrap();
        let expected_completion = fixtures["completion"].as_array().unwrap();
        assert_eq!(
            oracle["chat"].as_array().unwrap().len(),
            expected_chat.len()
        );
        assert_eq!(
            oracle["completion"].as_array().unwrap().len(),
            expected_completion.len()
        );
        for (case, fixture) in oracle["chat"].as_array().unwrap().iter().zip(expected_chat) {
            assert_eq!(case["request"], fixture["request"]);
            assert_eq!(case["rendered"], fixture["rendered"]);
            let actual = tokenizer.chat_raw(&case["request"]).unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(actual, expected, "{}", case["name"]);
        }
        for (case, fixture) in oracle["completion"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected_completion)
        {
            assert_eq!(case["request"], fixture["request"]);
            let request = serde_json::from_value(case["request"].clone()).unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(
                tokenizer.completion(&request).unwrap(),
                expected,
                "{}",
                case["name"]
            );
        }
    }
}

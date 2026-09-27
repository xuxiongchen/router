//! Narrow, opt-in Completion token-input forwarding, not a new rendering path.
//!
//! Ingress is never edited. Only the top-level prompt value is replaced; all
//! other bytes (including missing/null fields and numeric spelling) survive.
//! The successful facade render supplies the eligibility proof and exact IDs.
//! The caller must fence the same contract again at every dispatch/retry.

use std::{collections::HashSet, fmt};

use bytes::Bytes;
use http::HeaderMap;
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;

use super::{
    bridge::{PreparedTokens, RenderContract},
    timing::{self, StageTimer},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionInputFallback {
    NotEligible,
    StaleContract,
    BodyHeaders,
    InvalidEnvelope,
    PayloadLimit,
}

impl CompletionInputFallback {
    pub fn reason(self) -> &'static str {
        match self {
            Self::NotEligible => "not_eligible",
            Self::StaleContract => "stale_contract",
            Self::BodyHeaders => "body_headers",
            Self::InvalidEnvelope => "invalid_envelope",
            Self::PayloadLimit => "payload_limit",
        }
    }
}

/// Headers are a separate owned copy. Content-Length is recalculated by the
/// transport; identity encoding and transfer framing cannot describe this body.
#[derive(Clone, Debug)]
pub struct DerivedCompletionInput {
    pub body: Bytes,
    pub headers: HeaderMap,
}

/// A conservative header allowlist makes unknown body-integrity/authentication
/// extensions fall back, instead of forwarding an old signature over new bytes.
/// Bearer/Basic authentication and ordinary tracing/request IDs are unchanged.
fn derived_headers(headers: Option<&HeaderMap>) -> Option<HeaderMap> {
    let mut derived = headers.cloned().unwrap_or_default();
    for (name, value) in &derived {
        let allowed = matches!(
            name.as_str(),
            "accept"
                | "accept-encoding"
                | "accept-language"
                | "authorization"
                | "content-type"
                | "content-length"
                | "content-encoding"
                | "host"
                | "connection"
                | "transfer-encoding"
                | "user-agent"
                | "x-request-id"
                | "x-correlation-id"
                | "traceparent"
                | "tracestate"
                | "baggage"
                | "x-api-key"
                | "api-key"
        );
        if !allowed {
            return None;
        }
        if name == "content-encoding"
            && !value.to_str().ok()?.trim().eq_ignore_ascii_case("identity")
        {
            return None;
        }
        if name == "authorization" {
            let scheme = value.to_str().ok()?.split_ascii_whitespace().next()?;
            if !scheme.eq_ignore_ascii_case("bearer") && !scheme.eq_ignore_ascii_case("basic") {
                return None;
            }
        }
    }
    derived.remove("content-length");
    derived.remove("content-encoding");
    derived.remove("transfer-encoding");
    Some(derived)
}

struct PromptSlice<'a>(&'a RawValue);

impl<'de> Deserialize<'de> for PromptSlice<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PromptVisitor;

        impl<'de> Visitor<'de> for PromptVisitor {
            type Value = PromptSlice<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a unique-key Completion object")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut keys = HashSet::new();
                let mut prompt = None;
                while let Some((key, value)) = map.next_entry::<String, &'de RawValue>()? {
                    // The existing path still accepts last-key-wins requests.
                    // Duplicates simply do not qualify for this optimization.
                    if !keys.insert(key.clone()) {
                        return Err(de::Error::custom("duplicate Completion field"));
                    }
                    if key == "prompt" {
                        prompt = Some(value);
                    }
                }
                prompt
                    .map(PromptSlice)
                    .ok_or_else(|| de::Error::missing_field("prompt"))
            }
        }

        deserializer.deserialize_map(PromptVisitor)
    }
}

/// Call only for Completion, with `prepared` produced from this exact `raw`
/// ingress. No network calls, re-rendering, token decoding or request defaults.
/// A fallback is not a request error: forward the untouched ingress normally.
pub fn derive_completion_input(
    raw: &[u8],
    headers: Option<&HeaderMap>,
    prepared: &PreparedTokens,
    current_contract: &RenderContract,
    max_payload_bytes: usize,
) -> Result<DerivedCompletionInput, CompletionInputFallback> {
    let _timing = StageTimer::start("completion_backend_body");
    if !prepared.completion_token_input_eligible
        || !prepared.cache_eligible
        || prepared.token_ids.is_empty()
    {
        return Err(CompletionInputFallback::NotEligible);
    }
    if prepared.contract != *current_contract {
        return Err(CompletionInputFallback::StaleContract);
    }
    let headers = derived_headers(headers).ok_or(CompletionInputFallback::BodyHeaders)?;
    let PromptSlice(prompt) = serde_json::from_slice::<PromptSlice<'_>>(raw)
        .map_err(|_| CompletionInputFallback::InvalidEnvelope)?;
    if !prompt.get().starts_with('"') {
        return Err(CompletionInputFallback::NotEligible);
    }
    // RawValue borrows exactly one JSON value inside `raw`. Retaining its
    // original byte range avoids reserializing arbitrary sampling parameters.
    let start = (prompt.get().as_ptr() as usize)
        .checked_sub(raw.as_ptr() as usize)
        .ok_or(CompletionInputFallback::InvalidEnvelope)?;
    let end = start
        .checked_add(prompt.get().len())
        .filter(|end| *end <= raw.len())
        .ok_or(CompletionInputFallback::InvalidEnvelope)?;
    let encoded = serde_json::to_vec(&*prepared.token_ids)
        .map_err(|_| CompletionInputFallback::InvalidEnvelope)?;
    let total = raw
        .len()
        .checked_sub(end - start)
        .and_then(|size| size.checked_add(encoded.len()))
        .filter(|size| *size <= max_payload_bytes)
        .ok_or(CompletionInputFallback::PayloadLimit)?;
    let mut body = Vec::with_capacity(total);
    body.extend_from_slice(&raw[..start]);
    body.extend_from_slice(&encoded);
    body.extend_from_slice(&raw[end..]);
    timing::count("completion_original_bytes", raw.len() as u64);
    timing::count("completion_backend_bytes", body.len() as u64);
    Ok(DerivedCompletionInput {
        body: body.into(),
        headers,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn prepared() -> PreparedTokens {
        PreparedTokens {
            token_ids: Arc::from([7, 11, 23]),
            contract: RenderContract {
                id: "same-cohort".into(),
                epoch: 3,
            },
            cache_eligible: true,
            completion_token_input_eligible: true,
        }
    }

    #[test]
    fn replaces_only_prompt_and_preserves_ingress_and_all_other_bytes() {
        let raw = br#" { "prompt" : "caf\u00e9 \\" , "temperature": 1.000e-1, "stop":null, "stream":true, "nested":{"prompt":"not replaced"}, "n":1 } "#;
        let before = raw.to_vec();
        let prepared = prepared();
        let derived =
            derive_completion_input(raw, None, &prepared, &prepared.contract, 1024).unwrap();
        assert_eq!(raw.as_slice(), before);
        assert_eq!(derived.body.as_ref(), br#" { "prompt" : [7,11,23] , "temperature": 1.000e-1, "stop":null, "stream":true, "nested":{"prompt":"not replaced"}, "n":1 } "#);
    }

    #[test]
    fn escaped_prompt_key_and_unicode_preserve_other_fields() {
        let raw = "{\"pro\\u006dpt\":\"中文🙂\",\"user\":\"é\"}";
        let prepared = prepared();
        let derived =
            derive_completion_input(raw.as_bytes(), None, &prepared, &prepared.contract, 1024)
                .unwrap();
        assert_eq!(
            derived.body.as_ref(),
            "{\"pro\\u006dpt\":[7,11,23],\"user\":\"é\"}".as_bytes()
        );
    }

    #[test]
    fn requires_proof_contract_text_and_unique_keys() {
        let mut prepared = prepared();
        let contract = prepared.contract.clone();
        prepared.completion_token_input_eligible = false;
        assert_eq!(
            derive_completion_input(b"{}", None, &prepared, &contract, 1024).unwrap_err(),
            CompletionInputFallback::NotEligible
        );
        prepared.completion_token_input_eligible = true;
        prepared.contract.epoch += 1;
        assert_eq!(
            derive_completion_input(b"{}", None, &prepared, &contract, 1024).unwrap_err(),
            CompletionInputFallback::StaleContract
        );
        for raw in [
            br#"{"prompt":[7]}"#.as_slice(),
            br#"{"prompt":"x","prompt":"y"}"#,
            br#"{"prompt":"x","n":1,"n":1}"#,
            b"null",
            b"{",
            b"{}",
        ] {
            assert!(
                derive_completion_input(raw, None, &prepared, &prepared.contract, 1024).is_err()
            );
        }
    }

    #[test]
    fn exact_payload_budget_and_body_headers() {
        let prepared = prepared();
        let raw = br#"{"prompt":"x"}"#;
        let total = br#"{"prompt":[7,11,23]}"#.len();
        assert!(derive_completion_input(raw, None, &prepared, &prepared.contract, total).is_ok());
        assert_eq!(
            derive_completion_input(raw, None, &prepared, &prepared.contract, total - 1)
                .unwrap_err(),
            CompletionInputFallback::PayloadLimit
        );
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer unchanged".parse().unwrap());
        headers.insert("x-request-id", "same-id".parse().unwrap());
        headers.insert("content-length", "14".parse().unwrap());
        headers.insert("content-encoding", "identity".parse().unwrap());
        headers.insert("transfer-encoding", "chunked".parse().unwrap());
        let derived =
            derive_completion_input(raw, Some(&headers), &prepared, &prepared.contract, total)
                .unwrap();
        assert_eq!(derived.headers["authorization"], headers["authorization"]);
        assert_eq!(derived.headers["x-request-id"], headers["x-request-id"]);
        for removed in ["content-length", "content-encoding", "transfer-encoding"] {
            assert!(!derived.headers.contains_key(removed));
            assert!(headers.contains_key(removed));
        }
        for (name, value) in [
            ("content-encoding", "gzip"),
            ("content-digest", "sha-256=:old:"),
            ("x-custom-body-signature", "old"),
            ("authorization", "AWS4-HMAC-SHA256 old"),
        ] {
            let mut unsafe_headers = HeaderMap::new();
            unsafe_headers.insert(
                http::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
            assert_eq!(
                derive_completion_input(
                    raw,
                    Some(&unsafe_headers),
                    &prepared,
                    &prepared.contract,
                    1024
                )
                .unwrap_err(),
                CompletionInputFallback::BodyHeaders
            );
        }
    }
}

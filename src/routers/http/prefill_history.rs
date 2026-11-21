//! Request and response evidence for advisory HTTP Prefill history.

use crate::policies::HistoryObservation;
use serde_json::Value;

pub(super) struct PrefillObservation {
    /// Full prepared input length, not the block-quantized history prefix.
    pub input_tokens: usize,
    pub ticket: HistoryObservation,
}

/// Correlate the single-prompt vLLM Completion/NIXL pull response with this
/// HTTP attempt: `cmpl-<header>`, then the engine's item index and optional
/// random suffix. This does not authenticate the remote engine or its KV.
pub(super) fn fresh_prefill_handoff(
    body: &Value,
    request_id: &str,
    input_tokens: usize,
) -> Result<(), &'static str> {
    let Some(params) = body.get("kv_transfer_params") else {
        return Err("missing_handoff");
    };
    let Some(id) = body.get("id").and_then(Value::as_str) else {
        return Err("missing_request_id");
    };
    let Some(remote) = params.get("remote_request_id").and_then(Value::as_str) else {
        return Err("missing_remote_request_id");
    };
    if id.strip_prefix("cmpl-") != Some(request_id)
        || body.get("object").and_then(Value::as_str) != Some("text_completion")
        || body.get("error").is_some_and(|error| !error.is_null())
        || !body
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                matches!(choices.as_slice(), [choice] if choice["index"].as_u64() == Some(0)
                && matches!(choice["finish_reason"].as_str(), Some("length" | "stop")))
            })
    {
        return Err("invalid_completion");
    }
    let Some(suffix) = remote
        .strip_prefix(id)
        .and_then(|rest| rest.strip_prefix("-0"))
    else {
        return Err("mismatched_attempt");
    };
    // vLLM may randomize the engine item ID, but never accept an arbitrary
    // suffix: a handoff from another item or attempt is not this observation.
    let item_matches = suffix.is_empty()
        || suffix
            .strip_prefix('-')
            .is_some_and(|s| s.len() == 8 && s.bytes().all(|b| b.is_ascii_hexdigit()));
    if !item_matches {
        return Err("mismatched_item");
    }
    let single_group = params["remote_block_ids"].as_array().is_some_and(|groups| {
        let [group] = groups.as_slice() else {
            return false;
        };
        group
            .as_array()
            .is_some_and(|ids| !ids.is_empty() && ids.iter().all(Value::is_u64))
    });
    let supported = single_group
        && params["transfer_mode"].as_str() == Some("pull")
        && params["do_remote_prefill"].as_bool() == Some(true)
        && params["do_remote_decode"].as_bool() == Some(false)
        && params["tp_size"].as_u64() == Some(1)
        && ["pp_size", "dcp_size"]
            .into_iter()
            .all(|k| params[k].as_u64() == Some(1))
        && ["remote_engine_id", "remote_host"]
            .into_iter()
            .all(|k| params[k].as_str().is_some_and(|s| !s.trim().is_empty()))
        && params["remote_port"]
            .as_u64()
            .is_some_and(|p| (1..=65535).contains(&p))
        // We record the prepared request prefix, not a shorter transferred
        // range. Partial handoffs cannot establish this observation.
        && params["remote_num_tokens"].as_u64() == Some(input_tokens as u64);
    if supported {
        Ok(())
    } else {
        Err("unsupported_or_partial_handoff")
    }
}

pub(super) fn prefill_history_model(request: &Value) -> Option<&str> {
    let fields = request.as_object()?;
    if !fields.iter().all(|(key, value)| match key.as_str() {
        // Known output controls do not change the prepared prompt. P is
        // always nonstreaming; the client's stream flag applies to D.
        "model" | "prompt" | "max_tokens" | "temperature" | "top_p" | "n" | "stop"
        | "stop_token_ids" | "seed" | "logprobs" | "top_k" | "min_p" | "min_tokens"
        | "presence_penalty" | "frequency_penalty" | "repetition_penalty" => true,
        "session_id" => value.is_string(),
        // Typed Completion serializes these output controls even at their defaults.
        "stream"
        | "echo"
        | "no_stop_trim"
        | "ignore_eos"
        | "skip_special_tokens"
        | "return_hidden_states" => value.is_boolean(),
        _ => false,
    }) || fields.get("n").is_some_and(|v| v.as_u64() != Some(1))
        // Preparation validates the full token vector; only screen its shape here.
        || !(request["prompt"].is_string()
            || request["prompt"]
                .as_array()
                .and_then(|v| v.first())
                .is_some_and(Value::is_number))
    {
        return None;
    }
    fields.get("model")?.as_str().filter(|m| !m.is_empty())
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(super) fn fixture_handoff(request_id: &str, index: usize) -> Value {
    use serde_json::json;
    json!({
        "id":format!("cmpl-{request_id}"),
        "object":"text_completion",
        "choices":[{"index":0,"text":"", "finish_reason":"length"}],
        "kv_transfer_params":{
            "do_remote_prefill":true, "do_remote_decode":false,
            "remote_engine_id":format!("prefill-{}",index),
            "remote_request_id":format!("cmpl-{request_id}-0-0123abcd"),
            "remote_block_ids":[[1,2,3]],
            "remote_host":"127.0.0.1", "remote_port":19000,
            "tp_size":1, "dcp_size":1, "pp_size":1, "transfer_mode":"pull",
            "remote_num_tokens":33,
        }
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prefill_history_input_controls_and_handoff_shape_are_not_producer_identity() {
        let request = json!({"model":"m", "prompt":[1,2], "stop":["end"],
            "seed":7, "logprobs":2, "top_k":4, "min_tokens":1, "stream":true});
        assert_eq!(prefill_history_model(&request), Some("m"));
        for key in [
            "lora_name",
            "cache_salt",
            "truncate_prompt_tokens",
            "unknown_input",
        ] {
            let mut altered = request.clone();
            altered[key] = json!("unmodeled");
            assert_eq!(prefill_history_model(&altered), None, "{key}");
        }
        // This only establishes the supported response shape and attempt
        // correlation, not that engine/host/port belong to the selected P.
        let mut body = fixture_handoff("attempt", 0);
        body["kv_transfer_params"]["remote_request_id"] = json!("cmpl-attempt-0");
        assert_eq!(fresh_prefill_handoff(&body, "attempt", 33), Ok(()));
        for field in ["pp_size", "dcp_size", "transfer_mode"] {
            let mut incomplete = body.clone();
            incomplete["kv_transfer_params"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            assert_eq!(
                fresh_prefill_handoff(&incomplete, "attempt", 33),
                Err("unsupported_or_partial_handoff"),
                "missing {field}"
            );
        }
        assert_eq!(
            fresh_prefill_handoff(&body, "other", 33),
            Err("invalid_completion")
        );
        assert_eq!(
            fresh_prefill_handoff(&body, "attempt", 32),
            Err("unsupported_or_partial_handoff")
        );
    }
}

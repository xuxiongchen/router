//! Print full preparation results using the production owned-asset loader.
//! Usage: completion_input_probe CONTRACT.json < completion-requests.jsonl
use std::io::{self, BufRead};
use std::path::PathBuf;

use serde_json::json;
use vllm_router_rs::backend::completion_activation::load_completion_input_assets;
use vllm_router_rs::backend::CompletionPreparationError;
use vllm_router_rs::protocols::spec::CompletionRequest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let contract = PathBuf::from(std::env::args_os().nth(1).ok_or("expected CONTRACT.json")?);
    let assets = load_completion_input_assets(&contract)?;
    for line in io::stdin().lock().lines() {
        let result = match serde_json::from_str::<CompletionRequest>(&line?) {
            Ok(request) => match assets.prepare(&request) {
                Ok(prepared) => json!({"status": "exact", "token_ids": prepared.token_ids()}),
                Err(error) => {
                    let status = match &error {
                        CompletionPreparationError::Unsupported(_) => "unsupported",
                        CompletionPreparationError::InvalidRequest(_) => "invalid_request",
                        CompletionPreparationError::ServiceFailure(_) => "service_failure",
                    };
                    json!({"status": status, "error": error.to_string()})
                }
            },
            Err(error) => json!({"status": "invalid_request", "error": error.to_string()}),
        };
        println!("{result}");
    }
    Ok(())
}

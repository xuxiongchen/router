//! Opt-in, low-cardinality duration measurements. Nested intervals are not additive.
//!
//! No request text or tokens are recorded. Request correlation is carried by the
//! caller's tracing span, never a metric label. Detailed tracing has a hard event
//! cap and is intended only for a separate diagnostic run, not headline timings.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
    time::{Duration, Instant},
};

static ENABLED: OnceLock<bool> = OnceLock::new();
static TRACE: OnceLock<bool> = OnceLock::new();
static TRACE_EVENTS: AtomicUsize = AtomicUsize::new(0);
const MAX_TRACE_EVENTS: usize = 65_536;

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("VLLM_ROUTER_KV_STAGE_TIMING").as_deref() == Ok("1"))
}

pub fn record(stage: &'static str, elapsed: Duration) {
    if !enabled() {
        return;
    }
    metrics::histogram!("vllm_router_kv_stage_duration_seconds", "stage" => stage)
        .record(elapsed.as_secs_f64());
    if *TRACE.get_or_init(|| std::env::var("VLLM_ROUTER_KV_STAGE_TRACE").as_deref() == Ok("1"))
        && TRACE_EVENTS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < MAX_TRACE_EVENTS).then_some(n + 1)
            })
            .is_ok()
    {
        tracing::info!(stage, duration_ns = elapsed.as_nanos() as u64, "kv_stage");
    }
}

/// Names originate in compiled code, never in arbitrary provider-returned keys.
pub fn count(name: &'static str, value: u64) {
    if enabled() {
        metrics::counter!("vllm_router_kv_stage_operations_total", "operation" => name)
            .increment(value);
    }
}

pub fn gauge(name: &'static str, value: usize) {
    if enabled() {
        metrics::gauge!("vllm_router_kv_bridge_usage", "resource" => name).set(value as f64);
    }
}

pub struct StageTimer {
    stage: &'static str,
    start: Option<Instant>,
}

impl StageTimer {
    pub fn start(stage: &'static str) -> Self {
        Self {
            stage,
            start: enabled().then(Instant::now),
        }
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            record(self.stage, start.elapsed());
        }
    }
}

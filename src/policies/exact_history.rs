// Adapted from downstream fdcc40d3b90c114a069c3c8a15a10ab93b555ce6 and
// 2fdca9bc178279443bf5efdfa2cfd524cd44cef3; Regular history only.
//! Exact-token historical affinity for the KV-aware fallback chain.
//!
//! This module deliberately does not inspect raw request text and does not
//! mutate the real [`KVBlockIndex`](crate::kv_index::KVBlockIndex). Historical
//! entries are bounded, generation-scoped routing hints only. A SHA-256 prefix
//! index keeps lookups inexpensive, while every digest hit is verified against
//! the complete token prefix before it can influence routing.

use crate::core::Worker;
use crate::kv_index::WorkerInvalidationObserver;
use parking_lot::{Condvar, Mutex};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_TOKENIZER_CONTRACT_ID: AtomicU64 = AtomicU64::new(1);

/// Process-local identity of one exact tokenizer contract.
///
/// Historical state is process-local and never persisted, so a monotonic
/// identity is both stronger and cheaper than incorporating filesystem paths
/// into a key. One extractor must reuse the same value for compatible Chat
/// and Completion token sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ProcessTokenizerContractId(NonZeroU64);

impl ProcessTokenizerContractId {
    pub(crate) fn mint() -> Self {
        let value = NEXT_TOKENIZER_CONTRACT_ID.fetch_add(1, Ordering::Relaxed);
        Self(NonZeroU64::new(value).expect("process tokenizer-contract identity exhausted"))
    }
}

/// Stable model/profile scope inside this Router process.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ModelHistoryScope(Arc<str>);

impl ModelHistoryScope {
    pub(crate) fn new(value: impl AsRef<str>) -> Option<Self> {
        let value = value.as_ref();
        (!value.trim().is_empty()).then(|| Self(Arc::from(value)))
    }

    #[cfg(test)]
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ModelHistoryScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelHistoryScope")
            .finish_non_exhaustive()
    }
}

/// Scope in which identical token sequences are compatible history keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ExactHistoryScope {
    model: ModelHistoryScope,
    tokenizer_contract: ProcessTokenizerContractId,
}

impl ExactHistoryScope {
    pub(crate) fn new(
        model: ModelHistoryScope,
        tokenizer_contract: ProcessTokenizerContractId,
    ) -> Self {
        Self {
            model,
            tokenizer_contract,
        }
    }

    pub(crate) fn model(&self) -> &ModelHistoryScope {
        &self.model
    }
}

/// Exact tokens and their compatibility scope.
#[derive(Clone)]
pub(crate) struct ExactHistoryRequest {
    scope: ExactHistoryScope,
    tokens: Arc<[u32]>,
}

impl ExactHistoryRequest {
    pub(crate) fn new(scope: ExactHistoryScope, tokens: Arc<[u32]>) -> Option<Self> {
        (!tokens.is_empty()).then_some(Self { scope, tokens })
    }
}

impl fmt::Debug for ExactHistoryRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExactHistoryRequest")
            .field("scope", &self.scope)
            .field("token_count", &self.tokens.len())
            .finish()
    }
}

/// A session identifier which is guaranteed to be nonblank.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SessionHistoryKey(Arc<str>);

impl SessionHistoryKey {
    pub(crate) fn new(value: impl AsRef<str>) -> Option<Self> {
        let value = value.as_ref();
        // Whitespace is only a blank-key check. It is not normalization:
        // distinct nonblank session identifiers must remain distinct.
        (!value.trim().is_empty()).then(|| Self(Arc::from(value)))
    }
}

impl fmt::Debug for SessionHistoryKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionHistoryKey")
            .finish_non_exhaustive()
    }
}

/// All historical inputs derived once from one inference request.
#[derive(Clone)]
pub(crate) struct HistoricalRoutingRequest {
    model: ModelHistoryScope,
    exact: Option<ExactHistoryRequest>,
    session: Option<SessionHistoryKey>,
}

impl HistoricalRoutingRequest {
    pub(crate) fn new(
        model: ModelHistoryScope,
        exact: Option<ExactHistoryRequest>,
        session: Option<SessionHistoryKey>,
    ) -> Result<Self, String> {
        if exact
            .as_ref()
            .is_some_and(|exact| exact.scope.model() != &model)
        {
            return Err("exact-token and session history model scopes differ".to_owned());
        }
        Ok(Self {
            model,
            exact,
            session,
        })
    }

    pub(crate) fn has_affinity_key(&self) -> bool {
        self.exact.is_some() || self.session.is_some()
    }
}

impl fmt::Debug for HistoricalRoutingRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HistoricalRoutingRequest")
            .field("has_exact_tokens", &self.exact.is_some())
            .field(
                "exact_token_count",
                &self.exact.as_ref().map(|exact| exact.tokens.len()),
            )
            .field("has_session", &self.session.is_some())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExactHistoryConfig {
    pub(crate) cache_threshold: f32,
    pub(crate) balance_abs_threshold: usize,
    pub(crate) balance_rel_threshold: f32,
    pub(crate) history_ttl: Duration,
    pub(crate) eviction_interval: Duration,
    /// Maximum number of exact-prefix plus session index associations.
    pub(crate) max_tree_size: usize,
}

impl ExactHistoryConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.cache_threshold.is_finite() || !(0.0..=1.0).contains(&self.cache_threshold) {
            return Err("exact-history cache_threshold must be finite and within [0, 1]".into());
        }
        if !self.balance_rel_threshold.is_finite() || self.balance_rel_threshold < 1.0 {
            return Err("exact-history balance_rel_threshold must be finite and >= 1.0".into());
        }
        if self.history_ttl.is_zero() {
            return Err("exact-history TTL must be positive".into());
        }
        if Instant::now().checked_add(self.history_ttl).is_none() {
            return Err("exact-history TTL is too large for the monotonic clock".into());
        }
        if !self.eviction_interval.is_zero()
            && Instant::now().checked_add(self.eviction_interval).is_none()
        {
            return Err(
                "exact-history eviction interval is too large for the monotonic clock".into(),
            );
        }
        if self.max_tree_size == 0 {
            return Err("exact-history capacity must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HistoricalSelectionStage {
    ExactHistory,
    Session,
    LeastLoad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HistoryLoadGateDecision {
    NotEvaluated,
    Accepted {
        candidate_load: usize,
        minimum_load: usize,
    },
    Bypassed {
        candidate_load: usize,
        minimum_load: usize,
    },
}

/// Atomic historical fallback decision plus its pending reservation.
pub(crate) struct HistoricalSelection {
    pub(crate) worker_index: usize,
    pub(crate) stage: HistoricalSelectionStage,
    pub(crate) matched_tokens: usize,
    pub(crate) input_tokens: usize,
    pub(crate) load_gate: HistoryLoadGateDecision,
    pub(crate) reservation: Option<HistoryReservation>,
}

impl fmt::Debug for HistoricalSelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HistoricalSelection")
            .field("worker_index", &self.worker_index)
            .field("stage", &self.stage)
            .field("matched_tokens", &self.matched_tokens)
            .field("input_tokens", &self.input_tokens)
            .field("load_gate", &self.load_gate)
            .field("has_reservation", &self.reservation.is_some())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ExactHistoryStatsSnapshot {
    pub(crate) entry_count: usize,
    pub(crate) reservation_count: usize,
    pub(crate) exact_lookup_hits: u64,
    pub(crate) exact_lookup_misses: u64,
    pub(crate) session_lookup_hits: u64,
    pub(crate) session_lookup_misses: u64,
    pub(crate) reservation_rollbacks: u64,
    pub(crate) ttl_evictions: u64,
    pub(crate) capacity_evictions: u64,
    pub(crate) generation_purges: u64,
    pub(crate) load_gate_bypasses: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct HistoryPurgeOutcome {
    pub(crate) generations: usize,
    pub(crate) entries: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct HistoryRollbackOutcome {
    pub(crate) pending_removed: bool,
    pub(crate) stale_source_invalidated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WorkerGenerationId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct HistoryGroupId(u64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrefixIndexKey {
    scope: ExactHistoryScope,
    token_count: usize,
    digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SessionIndexKey {
    model: ModelHistoryScope,
    session: SessionHistoryKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvidenceState {
    Pending,
    Committed,
}

struct WorkerGeneration {
    url: Arc<str>,
    worker: Weak<dyn Worker>,
    active: bool,
}

struct HistoryGroup {
    worker_generation: WorkerGenerationId,
    exact: Option<ExactHistoryRequest>,
    session: Option<SessionIndexKey>,
    state: EvidenceState,
    version: u64,
    expires_at: Instant,
    update_order: u64,
    association_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct FailureInvalidation {
    group_id: HistoryGroupId,
    version: u64,
}

#[derive(Debug, Clone, Copy)]
struct InternalHistoryMatch {
    worker_index: usize,
    matched_tokens: usize,
    group_id: HistoryGroupId,
    state: EvidenceState,
    version: u64,
}

#[derive(Default)]
struct StateCounters {
    exact_lookup_hits: u64,
    exact_lookup_misses: u64,
    session_lookup_hits: u64,
    session_lookup_misses: u64,
    reservation_rollbacks: u64,
    ttl_evictions: u64,
    capacity_evictions: u64,
    generation_purges: u64,
    load_gate_bypasses: u64,
}

struct HistoryState {
    next_generation: u64,
    next_group: u64,
    next_version: u64,
    next_update_order: u64,
    generations: HashMap<WorkerGenerationId, WorkerGeneration>,
    current_generation_by_url: HashMap<Arc<str>, WorkerGenerationId>,
    groups: HashMap<HistoryGroupId, HistoryGroup>,
    prefixes: HashMap<PrefixIndexKey, Vec<HistoryGroupId>>,
    sessions: HashMap<SessionIndexKey, Vec<HistoryGroupId>>,
    association_count: usize,
    pending_count: usize,
    counters: StateCounters,
}

impl Default for HistoryState {
    fn default() -> Self {
        Self {
            next_generation: 1,
            next_group: 1,
            next_version: 1,
            next_update_order: 1,
            generations: HashMap::new(),
            current_generation_by_url: HashMap::new(),
            groups: HashMap::new(),
            prefixes: HashMap::new(),
            sessions: HashMap::new(),
            association_count: 0,
            pending_count: 0,
            counters: StateCounters::default(),
        }
    }
}

struct ExactHistoryInner {
    config: ExactHistoryConfig,
    state: Mutex<HistoryState>,
}

struct CleanerControl {
    stopped: Mutex<bool>,
    wake: Condvar,
}

/// Bounded exact-token and session history owned by a KV-aware composite.
pub(crate) struct ExactHistoryStore {
    inner: Arc<ExactHistoryInner>,
    cleaner_control: Arc<CleanerControl>,
    cleaner: Option<JoinHandle<()>>,
}

impl fmt::Debug for ExactHistoryStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExactHistoryStore")
            .field("config", &self.inner.config)
            .field("stats", &self.stats())
            .finish()
    }
}

impl ExactHistoryStore {
    pub(crate) fn new(config: ExactHistoryConfig) -> Result<Self, String> {
        config.validate()?;
        let eviction_interval = config.eviction_interval;
        let inner = Arc::new(ExactHistoryInner {
            config,
            state: Mutex::new(HistoryState::default()),
        });
        let cleaner_control = Arc::new(CleanerControl {
            stopped: Mutex::new(false),
            wake: Condvar::new(),
        });

        let cleaner = if eviction_interval.is_zero() {
            None
        } else {
            let inner = Arc::clone(&inner);
            let control = Arc::clone(&cleaner_control);
            Some(
                thread::Builder::new()
                    .name("kv-exact-history-cleaner".to_owned())
                    .spawn(move || cleaner_loop(inner, control, eviction_interval))
                    .map_err(|error| {
                        format!("failed to start exact-history cleanup thread: {error}")
                    })?,
            )
        };

        Ok(Self {
            inner,
            cleaner_control,
            cleaner,
        })
    }

    /// Atomically query exact history, then session history, then reserve the
    /// final worker. The fair least-load selector is invoked only when no
    /// historical candidate can be honored with a rollback-capable pending
    /// reservation, so accepted history never consumes the policy's fair
    /// tie-break cursor.
    pub(crate) fn select_and_reserve<F>(
        &self,
        request: &HistoricalRoutingRequest,
        workers: &[Arc<dyn Worker>],
        fair_least_load: F,
    ) -> Option<HistoricalSelection>
    where
        F: FnOnce() -> Option<usize>,
    {
        // Hashing can be substantial for long prompts. Compute it once before
        // entering the atomic lookup-and-reserve critical section.
        let exact_digests = request
            .exact
            .as_ref()
            .map(|exact| prefix_digests(&exact.tokens));
        let now = Instant::now();
        let mut state = self.inner.state.lock();
        cleanup_expired_locked(&mut state, now);

        let input_tokens = request.exact.as_ref().map_or(0, |exact| exact.tokens.len());
        let mut selected_index = None;
        let mut stage = HistoricalSelectionStage::LeastLoad;
        let mut matched_tokens = 0;
        let mut load_gate = HistoryLoadGateDecision::NotEvaluated;
        let mut invalidation = None;

        if let Some(exact) = request.exact.as_ref() {
            let exact_match = lookup_exact_locked(
                &state,
                exact,
                exact_digests
                    .as_deref()
                    .expect("exact tokens always have precomputed prefix digests"),
                workers,
                self.inner.config.cache_threshold,
            );
            if exact_match.is_some() {
                state.counters.exact_lookup_hits =
                    state.counters.exact_lookup_hits.saturating_add(1);
                metrics::counter!("vllm_router_kv_history_lookup_total", "outcome" => "hit")
                    .increment(1);
            } else {
                state.counters.exact_lookup_misses =
                    state.counters.exact_lookup_misses.saturating_add(1);
                metrics::counter!("vllm_router_kv_history_lookup_total", "outcome" => "miss")
                    .increment(1);
            }

            if let Some(history_match) = exact_match {
                load_gate = evaluate_load_gate(
                    workers,
                    history_match.worker_index,
                    self.inner.config.balance_abs_threshold,
                    self.inner.config.balance_rel_threshold,
                );
                if matches!(load_gate, HistoryLoadGateDecision::Accepted { .. }) {
                    selected_index = Some(history_match.worker_index);
                    stage = HistoricalSelectionStage::ExactHistory;
                    matched_tokens = history_match.matched_tokens;
                    invalidation = failure_invalidation(history_match);
                } else if matches!(load_gate, HistoryLoadGateDecision::Bypassed { .. }) {
                    state.counters.load_gate_bypasses =
                        state.counters.load_gate_bypasses.saturating_add(1);
                    metrics::counter!("vllm_router_kv_history_load_gate_bypass_total").increment(1);
                }
            }
        }

        if stage == HistoricalSelectionStage::LeastLoad
            && matches!(load_gate, HistoryLoadGateDecision::NotEvaluated)
        {
            if let Some(session_match) = lookup_session_locked(&state, request, workers) {
                state.counters.session_lookup_hits =
                    state.counters.session_lookup_hits.saturating_add(1);
                load_gate = evaluate_load_gate(
                    workers,
                    session_match.worker_index,
                    self.inner.config.balance_abs_threshold,
                    self.inner.config.balance_rel_threshold,
                );
                if matches!(load_gate, HistoryLoadGateDecision::Accepted { .. }) {
                    selected_index = Some(session_match.worker_index);
                    stage = HistoricalSelectionStage::Session;
                    invalidation = failure_invalidation(session_match);
                } else if matches!(load_gate, HistoryLoadGateDecision::Bypassed { .. }) {
                    state.counters.load_gate_bypasses =
                        state.counters.load_gate_bypasses.saturating_add(1);
                    metrics::counter!("vllm_router_kv_history_load_gate_bypass_total").increment(1);
                }
            } else if request.session.is_some() {
                state.counters.session_lookup_misses =
                    state.counters.session_lookup_misses.saturating_add(1);
            }
        }

        if let Some(historical_index) = selected_index {
            let selected_worker = workers.get(historical_index)?;
            if let Some(reservation) = reserve_locked(
                &self.inner,
                &mut state,
                request,
                exact_digests.as_deref(),
                selected_worker,
                invalidation,
                now,
            ) {
                publish_gauges_locked(&state);
                return Some(HistoricalSelection {
                    worker_index: historical_index,
                    stage,
                    matched_tokens,
                    input_tokens,
                    load_gate,
                    reservation: Some(reservation),
                });
            }

            // A historical candidate without a pending reservation could not
            // be rolled back after dispatch failure. Do not honor it.
            stage = HistoricalSelectionStage::LeastLoad;
            matched_tokens = 0;
            load_gate = HistoryLoadGateDecision::NotEvaluated;
        }

        let selected_index = fair_least_load()?;
        let selected_worker = workers.get(selected_index)?;
        if !selected_worker.is_available() {
            return None;
        }
        let reservation = reserve_locked(
            &self.inner,
            &mut state,
            request,
            exact_digests.as_deref(),
            selected_worker,
            None,
            now,
        );
        publish_gauges_locked(&state);

        Some(HistoricalSelection {
            worker_index: selected_index,
            stage,
            matched_tokens,
            input_tokens,
            load_gate,
            reservation,
        })
    }

    /// Reserve history for a real HBM owner without querying or overriding
    /// that primary selection.
    pub(crate) fn reserve_selected(
        &self,
        request: &HistoricalRoutingRequest,
        worker: &Arc<dyn Worker>,
    ) -> Option<HistoryReservation> {
        if !request.has_affinity_key() {
            return None;
        }
        let exact_digests = request
            .exact
            .as_ref()
            .map(|exact| prefix_digests(&exact.tokens));
        let now = Instant::now();
        let mut state = self.inner.state.lock();
        cleanup_expired_locked(&mut state, now);
        let reservation = reserve_locked(
            &self.inner,
            &mut state,
            request,
            exact_digests.as_deref(),
            worker,
            None,
            now,
        );
        publish_gauges_locked(&state);
        reservation
    }

    #[cfg(test)]
    pub(crate) fn register_worker(&self, worker: &Arc<dyn Worker>) -> HistoryPurgeOutcome {
        let mut state = self.inner.state.lock();
        let (_, purge) = ensure_worker_generation_locked(&mut state, worker);
        publish_gauges_locked(&state);
        purge
    }

    /// Purge only the generation represented by this exact Worker Arc.
    #[cfg(test)]
    pub(crate) fn purge_worker_generation(&self, worker: &Arc<dyn Worker>) -> HistoryPurgeOutcome {
        let mut state = self.inner.state.lock();
        let generation_ids = state
            .generations
            .iter()
            .filter_map(|(&generation_id, generation)| {
                generation
                    .worker
                    .upgrade()
                    .is_some_and(|candidate| Arc::ptr_eq(&candidate, worker))
                    .then_some(generation_id)
            })
            .collect::<Vec<_>>();
        let outcome = purge_generations_locked(&mut state, &generation_ids);
        publish_gauges_locked(&state);
        outcome
    }

    /// Conservative URL-scoped purge used by an AllBlocksCleared event or a
    /// lifecycle fence whose exact Arc is no longer available.
    pub(crate) fn purge_worker_url(&self, worker_url: &str) -> HistoryPurgeOutcome {
        let mut state = self.inner.state.lock();
        let generation_ids = state
            .generations
            .iter()
            .filter_map(|(&generation_id, generation)| {
                (generation.url.as_ref() == worker_url).then_some(generation_id)
            })
            .collect::<Vec<_>>();
        let outcome = purge_generations_locked(&mut state, &generation_ids);
        publish_gauges_locked(&state);
        outcome
    }

    #[cfg(test)]
    fn cleanup_now(&self) -> usize {
        let mut state = self.inner.state.lock();
        let removed = cleanup_expired_locked(&mut state, Instant::now());
        publish_gauges_locked(&state);
        removed
    }

    #[cfg(test)]
    fn expire_all_for_test(&self) {
        let mut state = self.inner.state.lock();
        let expired = Instant::now()
            .checked_sub(Duration::from_nanos(1))
            .expect("test expiration instant must be representable");
        for group in state.groups.values_mut() {
            group.expires_at = expired;
        }
    }

    pub(crate) fn stats(&self) -> ExactHistoryStatsSnapshot {
        let state = self.inner.state.lock();
        ExactHistoryStatsSnapshot {
            entry_count: state.association_count,
            reservation_count: state.pending_count,
            exact_lookup_hits: state.counters.exact_lookup_hits,
            exact_lookup_misses: state.counters.exact_lookup_misses,
            session_lookup_hits: state.counters.session_lookup_hits,
            session_lookup_misses: state.counters.session_lookup_misses,
            reservation_rollbacks: state.counters.reservation_rollbacks,
            ttl_evictions: state.counters.ttl_evictions,
            capacity_evictions: state.counters.capacity_evictions,
            generation_purges: state.counters.generation_purges,
            load_gate_bypasses: state.counters.load_gate_bypasses,
        }
    }
}

impl Drop for ExactHistoryStore {
    fn drop(&mut self) {
        *self.cleaner_control.stopped.lock() = true;
        self.cleaner_control.wake.notify_all();
        if let Some(cleaner) = self.cleaner.take() {
            let _ = cleaner.join();
        }
    }
}

impl WorkerInvalidationObserver for ExactHistoryStore {
    fn invalidate_worker(&self, worker_id: &str) {
        // Legacy event batches identify a worker by canonical URL but do not
        // carry the Router's process-local history generation. Conservatively
        // purging every same-URL generation is a safe false negative and can
        // never transfer stale affinity to a replacement process.
        self.purge_worker_url(worker_id);
    }
}

/// A pending historical reservation. It contains no request text/token
/// diagnostics and keeps only a weak reference to the owning store.
pub(crate) struct HistoryReservation {
    inner: Weak<ExactHistoryInner>,
    group_id: Option<HistoryGroupId>,
    invalidation: Option<FailureInvalidation>,
    finished: AtomicBool,
}

impl fmt::Debug for HistoryReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HistoryReservation")
            .field("finished", &self.finished.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl HistoryReservation {
    /// Commit only while the exact selected Worker generation is still active.
    pub(crate) fn commit(&mut self) -> bool {
        if self.finished.swap(true, Ordering::AcqRel) {
            return false;
        }
        let Some(inner) = self.inner.upgrade() else {
            return false;
        };
        let Some(group_id) = self.group_id.take() else {
            return false;
        };
        let mut state = inner.state.lock();
        let now = Instant::now();
        cleanup_expired_locked(&mut state, now);
        // Cleanup can expire this reservation (and other groups) before the
        // commit check. Keep exported gauges coherent on every early return.
        publish_gauges_locked(&state);

        let generation_is_active = state
            .groups
            .get(&group_id)
            .and_then(|group| state.generations.get(&group.worker_generation))
            .is_some_and(|generation| generation.active && generation.worker.upgrade().is_some());
        if !generation_is_active {
            remove_group_locked(&mut state, group_id);
            publish_gauges_locked(&state);
            return false;
        }

        let version = take_counter(&mut state.next_version);
        let update_order = take_counter(&mut state.next_update_order);
        let Some(group) = state.groups.get_mut(&group_id) else {
            return false;
        };
        if group.state != EvidenceState::Pending {
            return false;
        }
        group.state = EvidenceState::Committed;
        group.version = version;
        group.update_order = update_order;
        group.expires_at = expiration_deadline(now, inner.config.history_ttl);
        state.pending_count = state.pending_count.saturating_sub(1);
        publish_gauges_locked(&state);
        self.invalidation = None;
        true
    }

    #[cfg(test)]
    pub(crate) fn rollback(&mut self) -> HistoryRollbackOutcome {
        if self.finished.swap(true, Ordering::AcqRel) {
            return HistoryRollbackOutcome::default();
        }
        self.rollback_inner()
    }

    fn rollback_inner(&mut self) -> HistoryRollbackOutcome {
        let Some(inner) = self.inner.upgrade() else {
            return HistoryRollbackOutcome::default();
        };
        let mut state = inner.state.lock();
        let pending_removed = self
            .group_id
            .take()
            .is_some_and(|group_id| remove_group_locked(&mut state, group_id));
        let stale_source_invalidated = self.invalidation.take().is_some_and(|invalidation| {
            state
                .groups
                .get(&invalidation.group_id)
                .is_some_and(|group| {
                    group.state == EvidenceState::Committed && group.version == invalidation.version
                })
                && remove_group_locked(&mut state, invalidation.group_id)
        });
        state.counters.reservation_rollbacks =
            state.counters.reservation_rollbacks.saturating_add(1);
        metrics::counter!("vllm_router_kv_history_reservation_rollback_total").increment(1);
        publish_gauges_locked(&state);
        HistoryRollbackOutcome {
            pending_removed,
            stale_source_invalidated,
        }
    }
}

impl Drop for HistoryReservation {
    fn drop(&mut self) {
        if !self.finished.swap(true, Ordering::AcqRel) {
            let _ = self.rollback_inner();
        }
    }
}

fn cleaner_loop(
    inner: Arc<ExactHistoryInner>,
    control: Arc<CleanerControl>,
    eviction_interval: Duration,
) {
    let mut stopped = control.stopped.lock();
    loop {
        if *stopped {
            return;
        }
        control.wake.wait_for(&mut stopped, eviction_interval);
        if *stopped {
            return;
        }
        drop(stopped);
        let mut state = inner.state.lock();
        cleanup_expired_locked(&mut state, Instant::now());
        publish_gauges_locked(&state);
        drop(state);
        stopped = control.stopped.lock();
    }
}

fn prefix_digests(tokens: &[u32]) -> Vec<[u8; 32]> {
    let mut hasher = Sha256::new();
    let mut result = Vec::with_capacity(tokens.len());
    for token in tokens {
        hasher.update(token.to_be_bytes());
        result.push(hasher.clone().finalize().into());
    }
    result
}

fn failure_invalidation(history_match: InternalHistoryMatch) -> Option<FailureInvalidation> {
    (history_match.state == EvidenceState::Committed).then_some(FailureInvalidation {
        group_id: history_match.group_id,
        version: history_match.version,
    })
}

fn lookup_exact_locked(
    state: &HistoryState,
    request: &ExactHistoryRequest,
    digests: &[[u8; 32]],
    workers: &[Arc<dyn Worker>],
    cache_threshold: f32,
) -> Option<InternalHistoryMatch> {
    let input_len = request.tokens.len();
    debug_assert_eq!(digests.len(), input_len);
    for token_count in (1..=input_len).rev() {
        let match_rate = token_count as f32 / input_len as f32;
        if match_rate <= cache_threshold {
            break;
        }
        let key = PrefixIndexKey {
            scope: request.scope.clone(),
            token_count,
            digest: digests[token_count - 1],
        };
        let Some(group_ids) = state.prefixes.get(&key) else {
            continue;
        };
        let candidate = group_ids
            .iter()
            .filter_map(|group_id| {
                let group = state.groups.get(group_id)?;
                let exact = group.exact.as_ref()?;
                // Digest is only an accelerator. Exact equality makes a
                // collision a harmless extra bucket candidate.
                (exact.tokens.len() >= token_count
                    && exact.tokens[..token_count] == request.tokens[..token_count])
                    .then_some(())?;
                let worker_index =
                    worker_index_for_generation(state, group.worker_generation, workers)?;
                Some(InternalHistoryMatch {
                    worker_index,
                    matched_tokens: token_count,
                    group_id: *group_id,
                    state: group.state,
                    version: group.version,
                })
            })
            .max_by_key(|history_match| {
                state
                    .groups
                    .get(&history_match.group_id)
                    .map_or(0, |group| group.update_order)
            });
        if candidate.is_some() {
            return candidate;
        }
    }
    None
}

fn lookup_session_locked(
    state: &HistoryState,
    request: &HistoricalRoutingRequest,
    workers: &[Arc<dyn Worker>],
) -> Option<InternalHistoryMatch> {
    let session = request.session.as_ref()?;
    let key = SessionIndexKey {
        model: request.model.clone(),
        session: session.clone(),
    };
    state
        .sessions
        .get(&key)?
        .iter()
        .filter_map(|group_id| {
            let group = state.groups.get(group_id)?;
            let worker_index =
                worker_index_for_generation(state, group.worker_generation, workers)?;
            Some(InternalHistoryMatch {
                worker_index,
                matched_tokens: 0,
                group_id: *group_id,
                state: group.state,
                version: group.version,
            })
        })
        .max_by_key(|history_match| {
            state
                .groups
                .get(&history_match.group_id)
                .map_or(0, |group| group.update_order)
        })
}

fn worker_index_for_generation(
    state: &HistoryState,
    generation_id: WorkerGenerationId,
    workers: &[Arc<dyn Worker>],
) -> Option<usize> {
    let generation = state.generations.get(&generation_id)?;
    if !generation.active {
        return None;
    }
    let historical_worker = generation.worker.upgrade()?;
    workers
        .iter()
        .position(|worker| worker.is_available() && Arc::ptr_eq(worker, &historical_worker))
}

fn evaluate_load_gate(
    workers: &[Arc<dyn Worker>],
    candidate_index: usize,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
) -> HistoryLoadGateDecision {
    let Some(candidate) = workers.get(candidate_index) else {
        return HistoryLoadGateDecision::NotEvaluated;
    };
    let candidate_load = candidate.load();
    let mut available_loads = workers
        .iter()
        .filter(|worker| worker.is_available())
        .map(|worker| worker.load());
    let Some(first_load) = available_loads.next() else {
        return HistoryLoadGateDecision::NotEvaluated;
    };
    let (minimum_load, maximum_load) = available_loads
        .fold((first_load, first_load), |(minimum, maximum), load| {
            (minimum.min(load), maximum.max(load))
        });
    // Ported advisory load gate, separate from physical-KV CL protection.
    let bypass = maximum_load.saturating_sub(minimum_load) > balance_abs_threshold
        && (maximum_load as f32) > (minimum_load as f32 * balance_rel_threshold);
    if bypass {
        HistoryLoadGateDecision::Bypassed {
            candidate_load,
            minimum_load,
        }
    } else {
        HistoryLoadGateDecision::Accepted {
            candidate_load,
            minimum_load,
        }
    }
}

fn reserve_locked(
    inner: &Arc<ExactHistoryInner>,
    state: &mut HistoryState,
    request: &HistoricalRoutingRequest,
    exact_digests: Option<&[[u8; 32]]>,
    worker: &Arc<dyn Worker>,
    invalidation: Option<FailureInvalidation>,
    now: Instant,
) -> Option<HistoryReservation> {
    let exact_digests = match request.exact.as_ref() {
        Some(exact) => {
            let digests = exact_digests?;
            if digests.len() != exact.tokens.len() {
                return None;
            }
            Some(digests)
        }
        None => None,
    };
    let association_count = request.exact.as_ref().map_or(0, |exact| exact.tokens.len())
        + usize::from(request.session.is_some());
    if association_count == 0 || association_count > inner.config.max_tree_size {
        return None;
    }

    let (worker_generation, _) = ensure_worker_generation_locked(state, worker);
    if !reserve_capacity_locked(state, association_count, inner.config.max_tree_size) {
        return None;
    }

    let group_id = HistoryGroupId(take_counter(&mut state.next_group));
    let version = take_counter(&mut state.next_version);
    let update_order = take_counter(&mut state.next_update_order);
    let session = request.session.as_ref().map(|session| SessionIndexKey {
        model: request.model.clone(),
        session: session.clone(),
    });
    let group = HistoryGroup {
        worker_generation,
        exact: request.exact.clone(),
        session: session.clone(),
        state: EvidenceState::Pending,
        version,
        expires_at: expiration_deadline(now, inner.config.history_ttl),
        update_order,
        association_count,
    };
    state.groups.insert(group_id, group);

    if let Some(exact) = request.exact.as_ref() {
        let digests =
            exact_digests.expect("exact requests validate prefix digests before mutation");
        for (index, digest) in digests.iter().copied().enumerate() {
            state
                .prefixes
                .entry(PrefixIndexKey {
                    scope: exact.scope.clone(),
                    token_count: index + 1,
                    digest,
                })
                .or_default()
                .push(group_id);
        }
    }
    if let Some(session) = session {
        state.sessions.entry(session).or_default().push(group_id);
    }
    state.association_count = state.association_count.saturating_add(association_count);
    state.pending_count = state.pending_count.saturating_add(1);

    Some(HistoryReservation {
        inner: Arc::downgrade(inner),
        group_id: Some(group_id),
        invalidation,
        finished: AtomicBool::new(false),
    })
}

fn ensure_worker_generation_locked(
    state: &mut HistoryState,
    worker: &Arc<dyn Worker>,
) -> (WorkerGenerationId, HistoryPurgeOutcome) {
    if let Some(&generation_id) = state.current_generation_by_url.get(worker.url()) {
        let same_generation = state
            .generations
            .get(&generation_id)
            .and_then(|generation| generation.worker.upgrade())
            .is_some_and(|registered| Arc::ptr_eq(&registered, worker));
        if same_generation {
            return (generation_id, HistoryPurgeOutcome::default());
        }
        let purge = purge_generations_locked(state, &[generation_id]);
        let generation_id = insert_worker_generation_locked(state, worker);
        return (generation_id, purge);
    }
    (
        insert_worker_generation_locked(state, worker),
        HistoryPurgeOutcome::default(),
    )
}

fn insert_worker_generation_locked(
    state: &mut HistoryState,
    worker: &Arc<dyn Worker>,
) -> WorkerGenerationId {
    let generation_id = WorkerGenerationId(take_counter(&mut state.next_generation));
    let url: Arc<str> = Arc::from(worker.url());
    state.generations.insert(
        generation_id,
        WorkerGeneration {
            url: Arc::clone(&url),
            worker: Arc::downgrade(worker),
            active: true,
        },
    );
    state.current_generation_by_url.insert(url, generation_id);
    generation_id
}

fn purge_generations_locked(
    state: &mut HistoryState,
    generation_ids: &[WorkerGenerationId],
) -> HistoryPurgeOutcome {
    if generation_ids.is_empty() {
        return HistoryPurgeOutcome::default();
    }
    let mut entries = 0;
    for generation_id in generation_ids {
        if let Some(generation) = state.generations.get_mut(generation_id) {
            generation.active = false;
            let url = Arc::clone(&generation.url);
            if state.current_generation_by_url.get(&url) == Some(generation_id) {
                state.current_generation_by_url.remove(&url);
            }
        }
        let group_ids = state
            .groups
            .iter()
            .filter_map(|(&group_id, group)| {
                (group.worker_generation == *generation_id).then_some(group_id)
            })
            .collect::<Vec<_>>();
        for group_id in group_ids {
            entries += usize::from(remove_group_locked(state, group_id));
        }
        state.generations.remove(generation_id);
    }
    state.counters.generation_purges = state
        .counters
        .generation_purges
        .saturating_add(generation_ids.len() as u64);
    for _ in generation_ids {
        metrics::counter!("vllm_router_kv_history_generation_purge_total").increment(1);
    }
    HistoryPurgeOutcome {
        generations: generation_ids.len(),
        entries,
    }
}

fn remove_group_locked(state: &mut HistoryState, group_id: HistoryGroupId) -> bool {
    let Some(group) = state.groups.remove(&group_id) else {
        return false;
    };
    if let Some(exact) = group.exact {
        for (index, digest) in prefix_digests(&exact.tokens).into_iter().enumerate() {
            let key = PrefixIndexKey {
                scope: exact.scope.clone(),
                token_count: index + 1,
                digest,
            };
            let remove_key = state.prefixes.get_mut(&key).is_some_and(|groups| {
                groups.retain(|candidate| *candidate != group_id);
                groups.is_empty()
            });
            if remove_key {
                state.prefixes.remove(&key);
            }
        }
    }
    if let Some(session) = group.session {
        let remove_key = state.sessions.get_mut(&session).is_some_and(|groups| {
            groups.retain(|candidate| *candidate != group_id);
            groups.is_empty()
        });
        if remove_key {
            state.sessions.remove(&session);
        }
    }
    state.association_count = state
        .association_count
        .saturating_sub(group.association_count);
    if group.state == EvidenceState::Pending {
        state.pending_count = state.pending_count.saturating_sub(1);
    }
    true
}

fn cleanup_expired_locked(state: &mut HistoryState, now: Instant) -> usize {
    let expired = state
        .groups
        .iter()
        .filter_map(|(&group_id, group)| (group.expires_at <= now).then_some(group_id))
        .collect::<Vec<_>>();
    for group_id in &expired {
        remove_group_locked(state, *group_id);
    }
    state.counters.ttl_evictions = state
        .counters
        .ttl_evictions
        .saturating_add(expired.len() as u64);
    for _ in &expired {
        metrics::counter!("vllm_router_kv_history_eviction_total", "reason" => "ttl").increment(1);
    }
    expired.len()
}

/// Make room without evicting active pending reservations. If committed
/// history cannot free enough capacity, reject the new reservation.
fn reserve_capacity_locked(state: &mut HistoryState, needed: usize, capacity: usize) -> bool {
    while state.association_count.saturating_add(needed) > capacity {
        let Some((&oldest, _)) = state
            .groups
            .iter()
            .filter(|(_, group)| group.state == EvidenceState::Committed)
            .min_by_key(|(_, group)| group.update_order)
        else {
            return false;
        };
        remove_group_locked(state, oldest);
        state.counters.capacity_evictions = state.counters.capacity_evictions.saturating_add(1);
        metrics::counter!("vllm_router_kv_history_eviction_total", "reason" => "capacity")
            .increment(1);
    }
    true
}

fn publish_gauges_locked(state: &HistoryState) {
    let committed_associations: usize = state
        .groups
        .values()
        .filter(|group| group.state == EvidenceState::Committed)
        .map(|group| group.association_count)
        .sum();
    metrics::gauge!("vllm_router_kv_history_associations").set(committed_associations as f64);
    metrics::gauge!("vllm_router_kv_history_pending").set(state.pending_count as f64);
}

fn expiration_deadline(now: Instant, ttl: Duration) -> Instant {
    // The configuration is checked during store construction. Keep this
    // operation fail-closed even on an exotic monotonic-clock boundary.
    now.checked_add(ttl).unwrap_or(now)
}

fn take_counter(counter: &mut u64) -> u64 {
    let value = *counter;
    *counter = counter
        .checked_add(1)
        .expect("exact-history monotonic identity exhausted");
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    fn config() -> ExactHistoryConfig {
        ExactHistoryConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 2,
            balance_rel_threshold: 1.5,
            history_ttl: Duration::from_secs(60),
            eviction_interval: Duration::ZERO,
            max_tree_size: 1_000,
        }
    }

    fn workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(BasicWorker::new(
                "http://worker-0:8000".to_owned(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://worker-1:8000".to_owned(),
                WorkerType::Regular,
            )),
        ]
    }

    fn model(value: &str) -> ModelHistoryScope {
        ModelHistoryScope::new(value).unwrap()
    }

    fn exact_request(
        model_value: &str,
        contract: ProcessTokenizerContractId,
        tokens: &[u32],
        session: Option<&str>,
    ) -> HistoricalRoutingRequest {
        let model = model(model_value);
        let exact = ExactHistoryRequest::new(
            ExactHistoryScope::new(model.clone(), contract),
            Arc::from(tokens),
        );
        HistoricalRoutingRequest::new(model, exact, session.and_then(SessionHistoryKey::new))
            .unwrap()
    }

    #[test]
    fn pending_reservation_colocates_an_immediate_cold_request() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let request = exact_request(
            "model",
            ProcessTokenizerContractId::mint(),
            &[1, 2, 3],
            None,
        );

        let first = store
            .select_and_reserve(&request, &workers, || Some(1))
            .unwrap();
        assert_eq!(first.stage, HistoricalSelectionStage::LeastLoad);
        assert_eq!(first.worker_index, 1);

        let second = store
            .select_and_reserve(&request, &workers, || Some(0))
            .unwrap();
        assert_eq!(second.stage, HistoricalSelectionStage::ExactHistory);
        assert_eq!(second.worker_index, 1);
        assert_eq!(second.matched_tokens, 3);
    }

    #[test]
    fn longest_exact_prefix_wins_and_threshold_is_strict() {
        let mut cfg = config();
        cfg.cache_threshold = 0.5;
        let store = ExactHistoryStore::new(cfg).unwrap();
        let workers = workers();
        let contract = ProcessTokenizerContractId::mint();
        let short = exact_request("model", contract, &[1, 2], None);
        let long = exact_request("model", contract, &[1, 2, 3], None);
        let query = exact_request("model", contract, &[1, 2, 3, 4], None);

        store
            .reserve_selected(&short, &workers[0])
            .unwrap()
            .commit();
        store.reserve_selected(&long, &workers[1]).unwrap().commit();
        let selected = store
            .select_and_reserve(&query, &workers, || Some(0))
            .unwrap();
        assert_eq!(selected.worker_index, 1);
        assert_eq!(selected.matched_tokens, 3);

        let exactly_half = exact_request("model", contract, &[1, 2, 9, 9], None);
        let selected = store
            .select_and_reserve(&exactly_half, &workers, || Some(1))
            .unwrap();
        assert_eq!(selected.stage, HistoricalSelectionStage::LeastLoad);
        assert_eq!(selected.worker_index, 1);
    }

    #[test]
    fn model_and_tokenizer_contracts_prevent_cross_hits() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let first_contract = ProcessTokenizerContractId::mint();
        let recorded = exact_request("model-a", first_contract, &[7, 8, 9], None);
        store
            .reserve_selected(&recorded, &workers[1])
            .unwrap()
            .commit();

        let other_model = exact_request("model-b", first_contract, &[7, 8, 9], None);
        assert_eq!(
            store
                .select_and_reserve(&other_model, &workers, || Some(0))
                .unwrap()
                .stage,
            HistoricalSelectionStage::LeastLoad
        );
        let other_contract = exact_request(
            "model-a",
            ProcessTokenizerContractId::mint(),
            &[7, 8, 9],
            None,
        );
        assert_eq!(
            store
                .select_and_reserve(&other_contract, &workers, || Some(0))
                .unwrap()
                .stage,
            HistoricalSelectionStage::LeastLoad
        );
    }

    #[test]
    fn digest_bucket_also_requires_full_token_equality() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let contract = ProcessTokenizerContractId::mint();
        let recorded = exact_request("model", contract, &[1, 2, 3], None);
        store
            .reserve_selected(&recorded, &workers[1])
            .unwrap()
            .commit();

        let query = exact_request("model", contract, &[9, 8, 7], None);
        let query_exact = query.exact.as_ref().unwrap();
        let fake_key = PrefixIndexKey {
            scope: query_exact.scope.clone(),
            token_count: 3,
            digest: prefix_digests(&query_exact.tokens)[2],
        };
        let group_id = *store.inner.state.lock().groups.keys().next().unwrap();
        store
            .inner
            .state
            .lock()
            .prefixes
            .entry(fake_key)
            .or_default()
            .push(group_id);

        assert_eq!(
            store
                .select_and_reserve(&query, &workers, || Some(0))
                .unwrap()
                .stage,
            HistoricalSelectionStage::LeastLoad
        );
    }

    #[test]
    fn blank_sessions_are_never_keys_and_nonblank_sessions_preserve_identity() {
        assert!(SessionHistoryKey::new(" \t\n ").is_none());
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let model = model("model");
        let request = HistoricalRoutingRequest::new(
            model.clone(),
            None,
            SessionHistoryKey::new(" session-1 "),
        )
        .unwrap();
        store
            .reserve_selected(&request, &workers[1])
            .unwrap()
            .commit();

        let query = HistoricalRoutingRequest::new(model, None, SessionHistoryKey::new("session-1"))
            .unwrap();
        let selected = store
            .select_and_reserve(&query, &workers, || Some(0))
            .unwrap();
        assert_eq!(selected.stage, HistoricalSelectionStage::LeastLoad);
        assert_eq!(selected.worker_index, 0);

        let exact_query = HistoricalRoutingRequest::new(
            ModelHistoryScope::new("model").unwrap(),
            None,
            SessionHistoryKey::new(" session-1 "),
        )
        .unwrap();
        let selected = store
            .select_and_reserve(&exact_query, &workers, || Some(0))
            .unwrap();
        assert_eq!(selected.stage, HistoricalSelectionStage::Session);
        assert_eq!(selected.worker_index, 1);
    }

    #[test]
    fn load_gate_bypasses_an_overloaded_history_owner() {
        let mut cfg = config();
        cfg.balance_abs_threshold = 0;
        cfg.balance_rel_threshold = 1.0;
        let store = ExactHistoryStore::new(cfg).unwrap();
        let workers = workers();
        let request = exact_request("model", ProcessTokenizerContractId::mint(), &[1, 2], None);
        store
            .reserve_selected(&request, &workers[1])
            .unwrap()
            .commit();
        workers[1].increment_load();

        let selected = store
            .select_and_reserve(&request, &workers, || Some(0))
            .unwrap();
        assert_eq!(selected.stage, HistoricalSelectionStage::LeastLoad);
        assert_eq!(selected.worker_index, 0);
        assert!(matches!(
            selected.load_gate,
            HistoryLoadGateDecision::Bypassed { .. }
        ));
    }

    #[test]
    fn drop_rolls_back_pending_and_versioned_failure_invalidates_old_hit_only() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let request = exact_request("model", ProcessTokenizerContractId::mint(), &[1, 2], None);
        store
            .reserve_selected(&request, &workers[1])
            .unwrap()
            .commit();

        let failed = store
            .select_and_reserve(&request, &workers, || Some(0))
            .unwrap();
        assert_eq!(failed.stage, HistoricalSelectionStage::ExactHistory);
        drop(failed);
        assert_eq!(store.stats().reservation_count, 0);
        let after_failure = store
            .select_and_reserve(&request, &workers, || Some(0))
            .unwrap();
        assert_eq!(after_failure.stage, HistoricalSelectionStage::LeastLoad);
    }

    #[test]
    fn successful_concurrent_refresh_survives_an_old_failure_invalidation() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let request = exact_request("model", ProcessTokenizerContractId::mint(), &[1, 2], None);
        store
            .reserve_selected(&request, &workers[1])
            .unwrap()
            .commit();

        let failed = store
            .select_and_reserve(&request, &workers, || Some(0))
            .unwrap();
        let mut refreshed = store
            .select_and_reserve(&request, &workers, || Some(0))
            .unwrap();
        refreshed.reservation.as_mut().unwrap().commit();
        drop(failed);

        let selected = store
            .select_and_reserve(&request, &workers, || Some(0))
            .unwrap();
        assert_eq!(selected.stage, HistoricalSelectionStage::ExactHistory);
        assert_eq!(selected.worker_index, 1);
    }

    #[test]
    fn shutdown_wakes_cleaner_and_pending_cannot_keep_store_alive() {
        let mut cfg = config();
        cfg.eviction_interval = Duration::from_secs(3600);
        let store = ExactHistoryStore::new(cfg).unwrap();
        let request = exact_request("model", ProcessTokenizerContractId::mint(), &[1, 2], None);
        let mut pending = store.reserve_selected(&request, &workers()[0]).unwrap();
        let weak = Arc::downgrade(&store.inner);
        let (sender, receiver) = std::sync::mpsc::channel();
        let handle = thread::spawn(move || {
            drop(store);
            sender.send(()).unwrap();
        });
        receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("cleaner must wake without waiting for its interval");
        handle.join().unwrap();
        assert!(weak.upgrade().is_none());
        assert!(!pending.commit());
    }

    #[test]
    fn ttl_and_capacity_remove_stale_entries() {
        let mut cfg = config();
        // Expiry is forced under the state lock below. A millisecond TTL can
        // expire a pending reservation before commit on slow/QEMU runners.
        cfg.max_tree_size = 3;
        let store = ExactHistoryStore::new(cfg).unwrap();
        let workers = workers();
        let contract = ProcessTokenizerContractId::mint();
        let first = exact_request("model", contract, &[1, 2], None);
        let mut first_reservation = store.reserve_selected(&first, &workers[0]).unwrap();
        assert!(first_reservation.commit());
        let second = exact_request("model", contract, &[3, 4], None);
        let mut second_reservation = store.reserve_selected(&second, &workers[1]).unwrap();
        assert!(second_reservation.commit());
        assert!(store.stats().entry_count <= 3);
        assert!(store.stats().capacity_evictions >= 1);

        store.expire_all_for_test();
        assert!(store.cleanup_now() >= 1);
        assert_eq!(store.stats().entry_count, 0);
    }

    #[test]
    fn capacity_never_evicts_pending_or_exceeds_the_bound() {
        let mut cfg = config();
        cfg.max_tree_size = 2;
        let store = ExactHistoryStore::new(cfg).unwrap();
        let workers = workers();
        let contract = ProcessTokenizerContractId::mint();
        let first = exact_request("model", contract, &[1, 2], None);
        let second = exact_request("model", contract, &[3], None);

        let pending = store.reserve_selected(&first, &workers[0]).unwrap();
        assert_eq!(store.stats().entry_count, 2);
        assert_eq!(store.stats().reservation_count, 1);
        assert!(store.reserve_selected(&second, &workers[1]).is_none());
        assert_eq!(store.stats().entry_count, 2);
        assert_eq!(store.stats().reservation_count, 1);

        let mut pending = pending;
        assert!(pending.rollback().pending_removed);
        assert!(!pending.rollback().pending_removed);
        drop(pending);
        assert_eq!(store.stats().entry_count, 0);
        assert_eq!(store.stats().reservation_count, 0);
    }

    #[test]
    fn unreservable_history_hit_is_not_honored_without_a_rollback_guard() {
        let mut cfg = config();
        cfg.max_tree_size = 1;
        let store = ExactHistoryStore::new(cfg).unwrap();
        let workers = workers();
        let contract = ProcessTokenizerContractId::mint();
        let recorded = exact_request("model", contract, &[1], None);
        store
            .reserve_selected(&recorded, &workers[1])
            .unwrap()
            .commit();

        // This query can match the old exact-token hint, but its exact plus
        // session associations cannot fit. The historical owner must not be
        // selected without a reservation that can invalidate it on failure.
        let query = exact_request("model", contract, &[1], Some("session"));
        let selected = store
            .select_and_reserve(&query, &workers, || Some(0))
            .unwrap();
        assert_eq!(selected.stage, HistoricalSelectionStage::LeastLoad);
        assert_eq!(selected.worker_index, 0);
        assert!(selected.reservation.is_none());
    }

    #[test]
    fn exact_generation_and_url_purges_prevent_same_url_inheritance() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let request = exact_request("model", ProcessTokenizerContractId::mint(), &[1, 2], None);
        store
            .reserve_selected(&request, &workers[0])
            .unwrap()
            .commit();
        assert_eq!(store.purge_worker_generation(&workers[0]).generations, 1);
        assert_eq!(store.stats().entry_count, 0);

        let replacement: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            workers[0].url().to_owned(),
            WorkerType::Regular,
        ));
        store.register_worker(&replacement);
        assert_eq!(
            store
                .select_and_reserve(
                    &request,
                    &[Arc::clone(&replacement), Arc::clone(&workers[1])],
                    || Some(1),
                )
                .unwrap()
                .stage,
            HistoricalSelectionStage::LeastLoad
        );
        store
            .reserve_selected(&request, &replacement)
            .unwrap()
            .commit();
        assert_eq!(store.purge_worker_url(replacement.url()).generations, 1);
        assert_eq!(store.stats().entry_count, 0);
    }

    #[test]
    fn all_blocks_cleared_observer_purges_the_exact_worker_generation() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let request = exact_request("model", ProcessTokenizerContractId::mint(), &[1, 2], None);
        store
            .reserve_selected(&request, &workers[0])
            .unwrap()
            .commit();

        store.invalidate_worker(workers[0].url());

        assert_eq!(store.stats().entry_count, 0);
        assert_eq!(store.stats().generation_purges, 1);
    }

    #[test]
    fn late_commit_cannot_resurrect_a_purged_generation() {
        let store = ExactHistoryStore::new(config()).unwrap();
        let workers = workers();
        let request = exact_request("model", ProcessTokenizerContractId::mint(), &[1, 2], None);
        let mut reservation = store.reserve_selected(&request, &workers[0]).unwrap();
        store.purge_worker_generation(&workers[0]);
        assert!(!reservation.commit());
        assert_eq!(store.stats().entry_count, 0);
    }

    #[test]
    fn scopes_reject_blank_preserve_identity_and_never_expose_debug_values() {
        let model = ModelHistoryScope::new(" model ").unwrap();
        assert_eq!(model.as_str(), " model ");
        assert!(!format!("{model:?}").contains("model"));
        let session = SessionHistoryKey::new(" secret-session ").unwrap();
        assert!(!format!("{session:?}").contains("secret-session"));
    }
}

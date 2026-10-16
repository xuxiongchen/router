//! Exact KV-placement and live-load aware worker selection.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use rand::Rng;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::kv_index::{CacheKey, MatchQuery, RoutableTarget, StorageTier, TierMatch, WorkerKvInfo};

/// Read-only placement port owned by its policy consumer.
pub trait TieredMatchProvider: Send + Sync {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch>;
}

/// Immutable P work estimate from one placement snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillWorkEstimate {
    pub new_prefill_tokens: u64,
    pub placement_verified: bool,
}

/// Optional operations used by adapters with tokenized requests and KV discovery.
pub trait KvAwareOps: Send + Sync {
    /// P/D uses the same placement/credit contract as colocated routing,
    /// but accounts token backlog in the shared P/D admission manager.
    fn estimate_prefill_work(
        &self,
        worker: &dyn Worker,
        tokens: &[u32],
        credit: bool,
    ) -> PrefillWorkEstimate;
    fn pick_prefill(&self, costs: &[f64], occupancy: &[usize]) -> Option<usize>;
    /// Decode-side selection for P/D: sessions stick to their bound decoder
    /// within the configured spill margin, otherwise the minimum-load
    /// decoder wins (ties rotate). `worker_urls` are rank-qualified worker
    /// URLs (DP ranks on one server must stay distinct), `loads` are
    /// projected stage occupancies aligned with them.
    fn pick_decode(
        &self,
        session_key: Option<&str>,
        worker_urls: &[&str],
        loads: &[usize],
    ) -> Option<DecodePick>;
    /// Per-candidate surcharge for abandoning the prefix-holding prefill
    /// worker: `hysteresis * (new_tokens - min_new)` beyond chunk noise.
    fn defection_tax(&self, new_tokens: &[u64]) -> Vec<f64>;
    fn register_worker(
        &self,
        worker: &dyn Worker,
        info: &WorkerKvInfo,
        matches: Option<Arc<dyn TieredMatchProvider>>,
    );
    fn unregister_worker(&self, worker_url: &str);
    fn select_worker_with_receipt(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<KvSelectionReceipt>;
    fn begin_request(&self, receipt: KvSelectionReceipt) -> Option<KvRequestSlot>;
}

/// Decode selection result: the picked candidate plus how session affinity
/// shaped the choice (carried on `StageDecision::fallback_reason` and into
/// `vllm_router_pd_decode_decisions_total`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodePick {
    pub index: usize,
    pub outcome: &'static str,
}

/// Session bindings kept before the affinity map is trimmed back to size.
const DECODE_AFFINITY_CAP: usize = 8192;
/// Extra slack that batches trims so the oldest-entry scan amortizes.
const DECODE_AFFINITY_SLACK: usize = 1024;

/// Resolve the optional capability at the policy boundary. Keep concrete policy
/// discovery here so adapters and LoadBalancingPolicy stay independent of it.
pub fn kv_aware_ops(policy: &dyn LoadBalancingPolicy) -> Option<&dyn KvAwareOps> {
    policy
        .as_any()
        .downcast_ref::<KvAwarePolicy>()
        .map(|policy| policy as &dyn KvAwareOps)
}

/// Shared request accounting handle. The adapter signals output and completion;
/// concurrent body/producer cleanup updates the counters only once.
#[derive(Clone)]
pub struct KvRequestSlot {
    manager: Arc<dyn SlotManager>,
    target: RoutableTarget,
    prefill_tokens: u64,
    // 0 = prefill, 1 = decode, 2 = freed.
    state: Arc<AtomicU8>,
}

impl KvRequestSlot {
    pub(crate) fn new(
        manager: Arc<dyn SlotManager>,
        target: RoutableTarget,
        prefill_tokens: u64,
    ) -> Self {
        Self {
            manager,
            target,
            prefill_tokens,
            state: Arc::new(AtomicU8::new(0)),
        }
    }

    pub fn begin_decode(&self) {
        if self
            .state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.manager
                .begin_decode(&self.target, &self.target, self.prefill_tokens);
        }
    }

    pub fn release(&self) {
        self.begin_decode();
        if self
            .state
            .compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.manager.free(&self.target);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadSample {
    pub active_prefill_tokens: u64,
    pub active_requests: usize,
}

pub trait LoadProvider: Send + Sync {
    fn sample(&self, target: &RoutableTarget) -> LoadSample;
}

pub trait SlotManager: LoadProvider {
    fn begin_prefill(&self, target: &RoutableTarget, prefill_tokens: u64);
    fn begin_decode(
        &self,
        prefill_target: &RoutableTarget,
        decode_target: &RoutableTarget,
        prefill_tokens: u64,
    );
    fn free(&self, target: &RoutableTarget);

    /// Drop idle state for a target that is no longer registered. Implementers
    /// may keep active state until its matching request cleanup arrives.
    fn unregister_target(&self, _target: &RoutableTarget) {}
}

#[derive(Debug, Default)]
struct SlotCounters {
    active_prefill_tokens: AtomicU64,
    active_requests: AtomicUsize,
}

/// Router-local, bracket-based load view keyed at the index's `(worker, rank)`
/// granularity. Saturating decrements make duplicate cleanup harmless.
#[derive(Debug, Default)]
pub struct InMemorySlotManager {
    slots: DashMap<RoutableTarget, Arc<SlotCounters>>,
}

impl InMemorySlotManager {
    pub fn new() -> Self {
        Self::default()
    }

    fn update_or_insert(&self, target: &RoutableTarget, update: impl FnOnce(&SlotCounters)) {
        let slot = self
            .slots
            .entry(target.clone())
            .or_insert_with(|| Arc::new(SlotCounters::default()));
        update(&slot);
    }

    fn prune_if_idle(&self, target: &RoutableTarget) {
        if let dashmap::mapref::entry::Entry::Occupied(entry) = self.slots.entry(target.clone()) {
            let slot = entry.get();
            if slot.active_requests.load(Ordering::Acquire) == 0
                && slot.active_prefill_tokens.load(Ordering::Acquire) == 0
            {
                // The occupied entry keeps the shard locked between checking
                // the counters and removal, so a concurrent begin cannot be
                // detached from the map.
                entry.remove();
            }
        }
    }
}

fn atomic_saturating_sub_u64(value: &AtomicU64, amount: u64) {
    let _ = value.try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(amount))
    });
}

fn atomic_saturating_sub_usize(value: &AtomicUsize, amount: usize) {
    let _ = value.try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(amount))
    });
}

impl LoadProvider for InMemorySlotManager {
    fn sample(&self, target: &RoutableTarget) -> LoadSample {
        let Some(slot) = self.slots.get(target) else {
            return LoadSample::default();
        };
        LoadSample {
            active_prefill_tokens: slot.active_prefill_tokens.load(Ordering::Acquire),
            active_requests: slot.active_requests.load(Ordering::Acquire),
        }
    }
}

impl SlotManager for InMemorySlotManager {
    fn begin_prefill(&self, target: &RoutableTarget, prefill_tokens: u64) {
        self.update_or_insert(target, |slot| {
            slot.active_prefill_tokens
                .fetch_add(prefill_tokens, Ordering::AcqRel);
            slot.active_requests.fetch_add(1, Ordering::AcqRel);
        });
    }

    fn begin_decode(
        &self,
        prefill_target: &RoutableTarget,
        decode_target: &RoutableTarget,
        prefill_tokens: u64,
    ) {
        if prefill_target == decode_target {
            self.update_or_insert(prefill_target, |prefill| {
                atomic_saturating_sub_u64(&prefill.active_prefill_tokens, prefill_tokens);
            });
        } else {
            self.update_or_insert(prefill_target, |prefill| {
                atomic_saturating_sub_u64(&prefill.active_prefill_tokens, prefill_tokens);
                atomic_saturating_sub_usize(&prefill.active_requests, 1);
            });
            self.update_or_insert(decode_target, |decode| {
                decode.active_requests.fetch_add(1, Ordering::AcqRel);
            });
            self.prune_if_idle(prefill_target);
        }
    }

    fn free(&self, target: &RoutableTarget) {
        if let Some(slot) = self.slots.get(target) {
            atomic_saturating_sub_usize(&slot.active_requests, 1);
        }
        self.prune_if_idle(target);
    }

    fn unregister_target(&self, target: &RoutableTarget) {
        self.prune_if_idle(target);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferCredits {
    pub device: f64,
    pub host_pinned: f64,
    pub external: f64,
    pub disk: f64,
}

impl Default for TransferCredits {
    fn default() -> Self {
        Self {
            device: 1.0,
            host_pinned: 0.8,
            external: 0.6,
            disk: 0.3,
        }
    }
}

impl TransferCredits {
    fn for_tier(self, tier: StorageTier) -> f64 {
        match tier {
            StorageTier::Device => self.device,
            StorageTier::HostPinned => self.host_pinned,
            StorageTier::External => self.external,
            StorageTier::Disk => self.disk,
        }
    }

    fn valid(self) -> bool {
        self.device == 1.0
            && [self.host_pinned, self.external, self.disk]
                .into_iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostParams {
    pub router_temperature: f64,
    pub transfer_credits: TransferCredits,
    pub block_size: u32,
    /// Decode stickiness margin: a bound session stays on its decoder until
    /// that decoder's occupancy exceeds the pool minimum by more than this.
    pub decode_spill_margin: usize,
    /// Prefill defection hysteresis: leaving the prefix-holding worker adds
    /// this multiple of the extra re-computed tokens to the defector's cost.
    pub defection_hysteresis: f64,
    /// Prefill defection dead zone, in blocks: match credit is block-aligned,
    /// so a gap of a few blocks between candidates is placement noise rather
    /// than a defection worth taxing. Scales with the discovered block size.
    pub defection_epsilon_blocks: u64,
}

impl Default for CostParams {
    fn default() -> Self {
        Self {
            router_temperature: 0.0,
            transfer_credits: TransferCredits::default(),
            block_size: 16,
            decode_spill_margin: 2,
            defection_hysteresis: 1.0,
            defection_epsilon_blocks: 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CandidateView<'a> {
    pub target: &'a RoutableTarget,
    pub matches: &'a [TierMatch],
    pub load: LoadSample,
    pub input_tokens: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CostComponents {
    pub ptoken: u64,
    /// Current active request count (max of the slot-manager sample and worker load).
    pub batchsize: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CostDecision {
    pub target: RoutableTarget,
    pub cost: f64,
    pub components: CostComponents,
}

/// Immutable result of one KV-aware selection. The router carries this receipt
/// into admission so scoring and slot accounting use the same index snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct KvSelectionReceipt {
    pub worker_index: usize,
    pub new_prefill_tokens: u64,
    pub decision: CostDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    InvalidParams,
    CostLengthMismatch,
    NonFiniteCost,
}

pub trait Scorer: Send + Sync + std::fmt::Debug {
    fn score(
        &self,
        candidates: &[CandidateView<'_>],
        params: &CostParams,
        costs: &mut [f64],
    ) -> Result<(), SelectionError>;
}

pub trait Picker: Send + Sync + std::fmt::Debug {
    fn pick(&self, costs: &[f64]) -> Option<usize>;
}

/// Device prefix plus one external continuation, matching vLLM's two-segment
/// reuse rule. The return value is in tokens and can be fractional because a
/// transfer penalty is a credit ratio.
fn credited_tokens(candidate: &CandidateView<'_>, params: &CostParams) -> f64 {
    let device_blocks = candidate
        .matches
        .iter()
        .filter(|hit| hit.tier == StorageTier::Device)
        .flat_map(|hit| hit.segments.iter().copied())
        .filter(|(start, _)| *start == 0)
        .map(|(_, end)| end)
        .max()
        .unwrap_or(0)
        // vLLM must recompute the last prompt token to obtain logits. Local
        // GPU prefix reuse is block-aligned, so never credit its final block
        // when that block ends at the prompt boundary.
        .min(
            candidate
                .input_tokens
                .saturating_sub(1)
                .checked_div(params.block_size as usize)
                .unwrap_or(0)
                .min(u32::MAX as usize) as u32,
        );

    let mut best_continuation = (0_u32, 0.0_f64);
    for hit in candidate
        .matches
        .iter()
        .filter(|hit| hit.tier != StorageTier::Device)
    {
        for &(start, end) in &hit.segments {
            if start <= device_blocks && device_blocks < end {
                let blocks = end - device_blocks;
                let credit = params.transfer_credits.for_tier(hit.tier);
                if blocks > best_continuation.0
                    || (blocks == best_continuation.0 && credit > best_continuation.1)
                {
                    best_continuation = (blocks, credit);
                }
            }
        }
    }

    let credited_blocks =
        f64::from(device_blocks) + f64::from(best_continuation.0) * best_continuation.1;
    // External reuse may have a different alignment, but it cannot eliminate
    // the last-token logits work either.
    (credited_blocks * f64::from(params.block_size))
        .min(candidate.input_tokens.saturating_sub(1) as f64)
}

fn request_prefill_tokens(candidate: &CandidateView<'_>, params: &CostParams) -> u64 {
    // Round remaining work up: a fractional transfer credit never erases a
    // whole token of compute that was not fully credited.
    ((candidate.input_tokens as f64 - credited_tokens(candidate, params)).max(0.0)).ceil() as u64
}

pub fn cost_components(candidate: &CandidateView<'_>, params: &CostParams) -> CostComponents {
    let new_prefill_tokens = request_prefill_tokens(candidate, params);
    CostComponents {
        ptoken: new_prefill_tokens.saturating_add(candidate.load.active_prefill_tokens),
        // Keep the observed count for tie-breaking. The scorer adds one to
        // account for the request being selected, which is not active yet.
        batchsize: candidate.load.active_requests,
    }
}

#[derive(Debug, Default)]
pub struct MultiplicativeScorer;

impl Scorer for MultiplicativeScorer {
    fn score(
        &self,
        candidates: &[CandidateView<'_>],
        params: &CostParams,
        costs: &mut [f64],
    ) -> Result<(), SelectionError> {
        if candidates.len() != costs.len() {
            return Err(SelectionError::CostLengthMismatch);
        }
        if params.block_size == 0
            || !params.router_temperature.is_finite()
            || params.router_temperature < 0.0
            || !params.transfer_credits.valid()
        {
            return Err(SelectionError::InvalidParams);
        }
        for (candidate, cost) in candidates.iter().zip(costs.iter_mut()) {
            let components = cost_components(candidate, params);
            let projected_batchsize = components.batchsize.saturating_add(1);
            let contribution = components.ptoken as f64 * projected_batchsize as f64;
            if !contribution.is_finite() {
                return Err(SelectionError::NonFiniteCost);
            }
            *cost += contribution;
            if !cost.is_finite() {
                return Err(SelectionError::NonFiniteCost);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TemperaturePicker {
    temperature: f64,
}

impl TemperaturePicker {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl Picker for TemperaturePicker {
    fn pick(&self, costs: &[f64]) -> Option<usize> {
        if costs.is_empty() || costs.iter().any(|cost| !cost.is_finite()) {
            return None;
        }
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return costs
                .iter()
                .enumerate()
                .min_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index);
        }

        let min = costs.iter().copied().reduce(f64::min)?;
        let weights: Vec<f64> = costs
            .iter()
            .map(|cost| (-(cost - min) / self.temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return costs
                .iter()
                .enumerate()
                .min_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index);
        }
        let mut draw = rand::rng().random::<f64>() * total;
        for (index, weight) in weights.into_iter().enumerate() {
            if draw < weight {
                return Some(index);
            }
            draw -= weight;
        }
        Some(costs.len() - 1)
    }
}

#[derive(Debug, Default)]
struct EmptyMatchProvider;

impl TieredMatchProvider for EmptyMatchProvider {
    fn find_tiered_matches(&self, _query: &MatchQuery) -> Vec<TierMatch> {
        Vec::new()
    }
}

#[derive(Clone)]
struct TargetBinding {
    target: RoutableTarget,
    cache_key: Option<CacheKey>,
    matches: Arc<dyn TieredMatchProvider>,
    block_size: u32,
}

/// Registered `kv_aware` policy. Discovery binds each worker target to the
/// placement provider for its cache identity.
pub struct KvAwarePolicy {
    matches: Arc<dyn TieredMatchProvider>,
    load: Arc<dyn LoadProvider>,
    slot_manager: Option<Arc<dyn SlotManager>>,
    scorers: Vec<Arc<dyn Scorer>>,
    picker: Arc<dyn Picker>,
    params: CostParams,
    targets: DashMap<String, TargetBinding>,
    /// Session -> (bound decoder base URL, last-use epoch). Decode
    /// stickiness keeps a session on the decoder that already pulled its
    /// prefix so later rounds transfer only the delta.
    decode_affinity: DashMap<String, (String, u64)>,
    /// Round-robin cursor across equal-load decoders, so a stable first-index
    /// tie-break cannot magnetize every unbound session onto one decoder.
    decode_tie_cursor: AtomicU64,
    /// Monotonic clock for the affinity LRU.
    affinity_epoch: AtomicU64,
    /// Smallest block size discovered from registered cache identities; 0
    /// until the first registration. The defection dead zone scales with it.
    discovered_block_size: AtomicU32,
}

impl std::fmt::Debug for KvAwarePolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KvAwarePolicy")
            .field("scorers", &self.scorers.len())
            .field("params", &self.params)
            .field("registered_targets", &self.targets.len())
            .finish()
    }
}

impl KvAwarePolicy {
    pub fn new(params: CostParams) -> Self {
        let slots = Arc::new(InMemorySlotManager::new());
        Self {
            matches: Arc::new(EmptyMatchProvider),
            load: slots.clone(),
            slot_manager: Some(slots),
            scorers: vec![Arc::new(MultiplicativeScorer)],
            picker: Arc::new(TemperaturePicker::new(params.router_temperature)),
            params,
            targets: DashMap::new(),
            decode_affinity: DashMap::new(),
            decode_tie_cursor: AtomicU64::new(0),
            affinity_epoch: AtomicU64::new(0),
            discovered_block_size: AtomicU32::new(0),
        }
    }

    pub fn with_components(
        matches: Arc<dyn TieredMatchProvider>,
        load: Arc<dyn LoadProvider>,
        scorers: Vec<Arc<dyn Scorer>>,
        picker: Arc<dyn Picker>,
        params: CostParams,
    ) -> Self {
        Self {
            matches,
            load,
            slot_manager: None,
            scorers,
            picker,
            params,
            targets: DashMap::new(),
            decode_affinity: DashMap::new(),
            decode_tie_cursor: AtomicU64::new(0),
            affinity_epoch: AtomicU64::new(0),
            discovered_block_size: AtomicU32::new(0),
        }
    }

    pub fn register_target(&self, worker_url: impl Into<String>, target: RoutableTarget) {
        self.targets.insert(
            worker_url.into(),
            TargetBinding {
                target,
                cache_key: None,
                matches: self.matches.clone(),
                block_size: self.params.block_size,
            },
        );
    }

    pub fn register_target_with_cache_identity(
        &self,
        worker_url: impl Into<String>,
        target: RoutableTarget,
        cache_key: CacheKey,
        matches: Arc<dyn TieredMatchProvider>,
    ) {
        let block_size = cache_key.block_size;
        // Track the smallest live block size so the defection dead zone
        // scales with the fleet instead of a config constant.
        if block_size != 0 {
            let mut current = self.discovered_block_size.load(Ordering::Relaxed);
            while current == 0 || block_size < current {
                match self.discovered_block_size.compare_exchange_weak(
                    current,
                    block_size,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
        }
        self.targets.insert(
            worker_url.into(),
            TargetBinding {
                target,
                cache_key: Some(cache_key),
                matches,
                block_size,
            },
        );
    }

    pub fn unregister_target(&self, worker_url: &str) {
        if let Some((_, binding)) = self.targets.remove(worker_url) {
            if let Some(slot_manager) = &self.slot_manager {
                slot_manager.unregister_target(&binding.target);
            }
        }
        // Sessions bound to the removed decoder rebind on their next request
        // anyway (a stale URL misses and rebinds); dropping the bindings here
        // keeps the affinity map honest. Affinity keys are base URLs, while
        // registration may use DP-suffixed URLs, so match both forms.
        let base_url = worker_url
            .rsplit_once('@')
            .map(|(head, _)| head)
            .unwrap_or(worker_url);
        let stale: Vec<String> = self
            .decode_affinity
            .iter()
            .filter(|entry| {
                let bound = entry.value().0.as_str();
                bound == worker_url || bound == base_url
            })
            .map(|entry| entry.key().clone())
            .collect();
        for key in stale {
            self.decode_affinity.remove(&key);
        }
    }

    fn next_affinity_epoch(&self) -> u64 {
        self.affinity_epoch.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn bind_decode(&self, session_key: &str, worker_url: &str) {
        self.decode_affinity.insert(
            session_key.to_owned(),
            (worker_url.to_owned(), self.next_affinity_epoch()),
        );
        self.trim_decode_affinity();
    }

    /// Trim in batches once the map overshoots the cap, so the oldest-entry
    /// scan amortizes across inserts instead of running per request.
    fn trim_decode_affinity(&self) {
        if self.decode_affinity.len() <= DECODE_AFFINITY_CAP + DECODE_AFFINITY_SLACK {
            return;
        }
        let mut epochs: Vec<(u64, String)> = self
            .decode_affinity
            .iter()
            .map(|entry| (entry.value().1, entry.key().clone()))
            .collect();
        epochs.sort_unstable();
        let excess = epochs.len().saturating_sub(DECODE_AFFINITY_CAP);
        for (_, key) in epochs.into_iter().take(excess) {
            self.decode_affinity.remove(&key);
        }
    }

    pub fn slot_manager(&self) -> Option<Arc<dyn SlotManager>> {
        self.slot_manager.clone()
    }

    pub fn target_for_worker(&self, worker: &dyn Worker) -> RoutableTarget {
        self.targets
            .get(worker.url())
            .map(|entry| entry.target.clone())
            .unwrap_or_else(|| RoutableTarget {
                instance_id: Arc::from(worker.base_url()),
                dp_rank: worker.dp_rank().unwrap_or(0) as u32,
            })
    }

    fn binding_for_worker(&self, worker: &dyn Worker) -> TargetBinding {
        self.targets
            .get(worker.url())
            .map(|entry| entry.clone())
            .unwrap_or_else(|| TargetBinding {
                target: RoutableTarget {
                    instance_id: Arc::from(worker.base_url()),
                    dp_rank: worker.dp_rank().unwrap_or(0) as u32,
                },
                cache_key: None,
                matches: self.matches.clone(),
                block_size: self.params.block_size,
            })
    }

    pub fn local_hashes(token_ids: &[u32], block_size: u32) -> Vec<Arc<str>> {
        Self::local_hashes_with_lora(token_ids, block_size, None)
    }

    fn tiers_of_interest() -> Vec<StorageTier> {
        vec![
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::External,
            StorageTier::Disk,
        ]
    }

    fn local_hashes_with_lora(
        token_ids: &[u32],
        block_size: u32,
        lora_name: Option<&str>,
    ) -> Vec<Arc<str>> {
        crate::kv_index::local_block_hash_strings(token_ids, block_size, lora_name)
    }

    fn placement_matches(
        binding: &TargetBinding,
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        enabled: bool,
    ) -> Vec<TierMatch> {
        let local_hashes = if enabled {
            token_ids
                .map(|ids| Self::local_hashes_with_lora(ids, binding.block_size, lora_name))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        if local_hashes.is_empty() {
            Vec::new()
        } else {
            binding.matches.find_tiered_matches(&MatchQuery {
                group_idx: 0,
                local_hashes,
                tiers_of_interest: Self::tiers_of_interest(),
            })
        }
    }

    pub fn select_worker_with_token_ids(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
    ) -> Option<usize> {
        self.select_worker_with_receipt(workers, token_ids, None, None)
            .map(|receipt| receipt.worker_index)
    }

    pub fn select_worker_with_receipt(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<KvSelectionReceipt> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let base_params = self.params;
        let bindings: Vec<TargetBinding> = healthy
            .iter()
            .map(|&index| self.binding_for_worker(workers[index].as_ref()))
            .collect();

        // KV events carry lora_name but not cache_salt. Hash LoRA requests the
        // same way as ingestion; for salted requests, skip placement credit
        // instead of claiming an incompatible unsalted cache entry.
        let placement_enabled = cache_salt.is_none_or(|salt| salt.is_empty());
        let mut matches_by_identity: HashMap<Option<CacheKey>, Vec<TierMatch>> = HashMap::new();
        for binding in &bindings {
            let identity = binding.cache_key.clone();
            if matches_by_identity.contains_key(&identity) {
                continue;
            }
            let matches = Self::placement_matches(binding, token_ids, lora_name, placement_enabled);
            matches_by_identity.insert(identity, matches);
        }

        let grouped: Vec<Vec<TierMatch>> = bindings
            .iter()
            .map(|binding| {
                matches_by_identity
                    .get(&binding.cache_key)
                    .into_iter()
                    .flatten()
                    .filter(|hit| hit.target == binding.target)
                    .cloned()
                    .collect()
            })
            .collect();
        let input_tokens = token_ids.map_or(0, <[u32]>::len);
        let candidates: Vec<CandidateView<'_>> = bindings
            .iter()
            .zip(grouped.iter())
            .zip(healthy.iter())
            .map(|((binding, matches), &worker_index)| {
                let mut load = self.load.sample(&binding.target);
                // Existing request paths already bracket Worker::load. Use it
                // as the live batch-size floor while the richer slot manager
                // additionally carries KV-credited prefill tokens.
                load.active_requests = load.active_requests.max(workers[worker_index].load());
                CandidateView {
                    target: &binding.target,
                    matches,
                    load,
                    input_tokens,
                }
            })
            .collect();

        let mut costs = vec![0.0; candidates.len()];
        let mut components = vec![CostComponents::default(); candidates.len()];
        let mut indices_by_identity: HashMap<Option<CacheKey>, Vec<usize>> = HashMap::new();
        for (index, binding) in bindings.iter().enumerate() {
            indices_by_identity
                .entry(binding.cache_key.clone())
                .or_default()
                .push(index);
        }
        for indices in indices_by_identity.values() {
            let mut params = base_params;
            params.block_size = bindings[*indices.first()?].block_size;
            let identity_candidates: Vec<CandidateView<'_>> =
                indices.iter().map(|&index| candidates[index]).collect();
            let mut identity_costs = vec![0.0; identity_candidates.len()];
            for scorer in &self.scorers {
                scorer
                    .score(&identity_candidates, &params, &mut identity_costs)
                    .ok()?;
            }
            for ((&candidate_index, cost), candidate) in indices
                .iter()
                .zip(identity_costs)
                .zip(identity_candidates.iter())
            {
                costs[candidate_index] = cost;
                components[candidate_index] = cost_components(candidate, &params);
            }
        }

        // Zero work remains possible for token-less requests.
        // Preserve KV preference, breaking equal-work ties by projected load.
        let picked = if costs.iter().all(|cost| *cost == 0.0)
            || (base_params.router_temperature <= 0.0 && costs.contains(&0.0))
        {
            if token_ids.is_none() {
                // HTTP pools do not expose token IDs. Degrade to live load
                // instead of deterministically concentrating on worker zero.
                components
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| costs[*index] == 0.0)
                    .min_by_key(|(_, component)| component.batchsize)
                    .map(|(index, _)| index)
            } else {
                components
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| costs[*index] == 0.0)
                    .min_by_key(|(_, component)| (component.ptoken, component.batchsize))
                    .map(|(index, _)| index)
            }
        } else {
            self.picker.pick(&costs)
        }?;
        let worker_index = *healthy.get(picked)?;
        let target = bindings.get(picked)?.target.clone();
        let mut selected_params = base_params;
        selected_params.block_size = bindings[picked].block_size;
        Some(KvSelectionReceipt {
            worker_index,
            new_prefill_tokens: request_prefill_tokens(&candidates[picked], &selected_params),
            decision: CostDecision {
                target,
                cost: costs[picked],
                components: components[picked],
            },
        })
    }

    pub fn decision_for(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
    ) -> Option<CostDecision> {
        self.select_worker_with_receipt(workers, token_ids, None, None)
            .map(|receipt| receipt.decision)
    }
}

impl Default for KvAwarePolicy {
    fn default() -> Self {
        Self::new(CostParams::default())
    }
}

impl LoadBalancingPolicy for KvAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_with_token_ids(workers, None)
    }

    fn name(&self) -> &'static str {
        "kv_aware"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl KvAwareOps for KvAwarePolicy {
    fn estimate_prefill_work(
        &self,
        worker: &dyn Worker,
        tokens: &[u32],
        credit: bool,
    ) -> PrefillWorkEstimate {
        let binding = self.binding_for_worker(worker);
        let verified = credit && binding.cache_key.is_some();
        let mut params = self.params;
        params.block_size = binding.block_size;
        let matches: Vec<_> = Self::placement_matches(&binding, Some(tokens), None, verified)
            .into_iter()
            .filter(|hit| hit.target == binding.target)
            .collect();
        let candidate = CandidateView {
            target: &binding.target,
            matches: &matches,
            load: LoadSample::default(),
            input_tokens: tokens.len(),
        };
        PrefillWorkEstimate {
            new_prefill_tokens: request_prefill_tokens(&candidate, &params),
            placement_verified: verified,
        }
    }

    fn pick_prefill(&self, costs: &[f64], occupancy: &[usize]) -> Option<usize> {
        if costs.len() != occupancy.len() || costs.iter().any(|cost| !cost.is_finite()) {
            return None;
        }
        if self.params.router_temperature <= 0.0 {
            costs
                .iter()
                .enumerate()
                .min_by(|(li, left), (ri, right)| {
                    left.total_cmp(right)
                        .then_with(|| occupancy[*li].cmp(&occupancy[*ri]))
                })
                .map(|(index, _)| index)
        } else {
            self.picker.pick(costs)
        }
    }

    fn pick_decode(
        &self,
        session_key: Option<&str>,
        worker_urls: &[&str],
        loads: &[usize],
    ) -> Option<DecodePick> {
        if worker_urls.is_empty() || worker_urls.len() != loads.len() {
            return None;
        }
        let min_load = *loads.iter().min()?;
        let tied: Vec<usize> = loads
            .iter()
            .enumerate()
            .filter(|&(_, load)| *load == min_load)
            .map(|(index, _)| index)
            .collect();
        let cursor = self.decode_tie_cursor.fetch_add(1, Ordering::Relaxed) as usize;
        let fresh = tied[cursor % tied.len()];

        let Some(session_key) = session_key else {
            return Some(DecodePick {
                index: fresh,
                outcome: "no_session",
            });
        };
        enum Affinity {
            Hit(usize),
            Spill(usize),
            Rebind,
        }
        let action = if let Some(mut entry) = self.decode_affinity.get_mut(session_key) {
            match worker_urls.iter().position(|url| *url == entry.0.as_str()) {
                Some(index) if loads[index] <= min_load + self.params.decode_spill_margin => {
                    entry.1 = self.next_affinity_epoch();
                    Affinity::Hit(index)
                }
                Some(_) => {
                    // The bound decoder is hot beyond the spill margin: hand
                    // the session to the lightest decoder and rebind there.
                    *entry = (worker_urls[fresh].to_owned(), self.next_affinity_epoch());
                    Affinity::Spill(fresh)
                }
                // The bound decoder left the pool; fall through to rebind.
                None => Affinity::Rebind,
            }
        } else {
            Affinity::Rebind
        };
        match action {
            Affinity::Hit(index) => Some(DecodePick {
                index,
                outcome: "sticky_hit",
            }),
            Affinity::Spill(index) => Some(DecodePick {
                index,
                outcome: "sticky_spill",
            }),
            Affinity::Rebind => {
                self.bind_decode(session_key, worker_urls[fresh]);
                Some(DecodePick {
                    index: fresh,
                    outcome: "bind",
                })
            }
        }
    }

    fn defection_tax(&self, new_tokens: &[u64]) -> Vec<f64> {
        let Some(&min_new) = new_tokens.iter().min() else {
            return Vec::new();
        };
        // Dead zone of a few blocks: match credit is block-aligned, so gaps
        // within it are placement noise, not a defection worth taxing. Scale
        // with the discovered block size; the config default covers the
        // window before the first registration.
        let block_size = match self.discovered_block_size.load(Ordering::Relaxed) {
            0 => self.params.block_size,
            discovered => discovered,
        } as u64;
        let epsilon = self
            .params
            .defection_epsilon_blocks
            .saturating_mul(block_size);
        new_tokens
            .iter()
            .map(|&tokens| {
                if tokens > min_new + epsilon {
                    self.params.defection_hysteresis * (tokens - min_new) as f64
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn register_worker(
        &self,
        worker: &dyn Worker,
        info: &WorkerKvInfo,
        matches: Option<Arc<dyn TieredMatchProvider>>,
    ) {
        let target = RoutableTarget {
            instance_id: Arc::from(info.instance_id.as_str()),
            dp_rank: worker.dp_rank().unwrap_or(0) as u32,
        };
        match (info.cache_key.clone(), matches) {
            (Some(cache_key), Some(matches)) => {
                self.register_target_with_cache_identity(worker.url(), target, cache_key, matches);
            }
            _ => self.register_target(worker.url(), target),
        }
    }

    fn unregister_worker(&self, worker_url: &str) {
        self.unregister_target(worker_url);
    }

    fn select_worker_with_receipt(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<KvSelectionReceipt> {
        KvAwarePolicy::select_worker_with_receipt(self, workers, token_ids, lora_name, cache_salt)
    }

    fn begin_request(&self, receipt: KvSelectionReceipt) -> Option<KvRequestSlot> {
        let manager = self.slot_manager()?;
        manager.begin_prefill(&receipt.decision.target, receipt.new_prefill_tokens);
        Some(KvRequestSlot::new(
            manager,
            receipt.decision.target,
            receipt.new_prefill_tokens,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, DPAwareWorker, WorkerType};
    use crate::kv_index::Locality;
    use crate::policies::RoundRobinPolicy;

    fn target(name: &str, rank: u32) -> RoutableTarget {
        RoutableTarget {
            instance_id: Arc::from(name),
            dp_rank: rank,
        }
    }

    fn tier_match(
        target: &RoutableTarget,
        tier: StorageTier,
        segments: &[(u32, u32)],
    ) -> TierMatch {
        TierMatch {
            target: target.clone(),
            tier,
            segments: segments.to_vec(),
            locality: Locality::Local,
        }
    }

    fn policy_with_index(
        index: Arc<crate::kv_index::KvBlockIndexer>,
        params: CostParams,
    ) -> KvAwarePolicy {
        KvAwarePolicy::with_components(
            index,
            Arc::new(InMemorySlotManager::new()),
            vec![Arc::new(MultiplicativeScorer)],
            Arc::new(TemperaturePicker::new(params.router_temperature)),
            params,
        )
    }

    #[test]
    fn slot_manager_moves_colocated_and_separated_load() {
        let slots = InMemorySlotManager::new();
        let prefill = target("p", 1);
        let decode = target("d", 2);
        slots.begin_prefill(&prefill, 90);
        assert_eq!(
            slots.sample(&prefill),
            LoadSample {
                active_prefill_tokens: 90,
                active_requests: 1
            }
        );
        slots.begin_decode(&prefill, &prefill, 90);
        assert_eq!(slots.sample(&prefill).active_prefill_tokens, 0);
        assert_eq!(slots.sample(&prefill).active_requests, 1);
        slots.free(&prefill);
        assert_eq!(slots.sample(&prefill).active_requests, 0);

        slots.begin_prefill(&prefill, 30);
        slots.begin_decode(&prefill, &decode, 30);
        assert_eq!(slots.sample(&prefill), LoadSample::default());
        assert_eq!(slots.sample(&decode).active_requests, 1);
        slots.free(&decode);
        slots.free(&decode); // duplicate cleanup saturates
        assert_eq!(slots.sample(&decode).active_requests, 0);
    }

    #[test]
    fn unregister_and_request_cleanup_prune_only_idle_slots() {
        let slots = InMemorySlotManager::new();
        let idle = target("idle", 0);
        slots
            .slots
            .insert(idle.clone(), Arc::new(SlotCounters::default()));
        slots.unregister_target(&idle);
        assert!(!slots.slots.contains_key(&idle));

        let active = target("active", 1);
        slots.begin_prefill(&active, 32);
        slots.unregister_target(&active);
        assert_eq!(slots.sample(&active).active_requests, 1);
        assert_eq!(slots.sample(&active).active_prefill_tokens, 32);

        slots.begin_decode(&active, &active, 32);
        slots.free(&active);
        assert!(!slots.slots.contains_key(&active));
    }

    #[test]
    fn two_segment_credit_uses_device_and_one_continuation() {
        let t = target("w", 0);
        let matches = vec![
            tier_match(&t, StorageTier::Device, &[(0, 5)]),
            tier_match(&t, StorageTier::HostPinned, &[(5, 9)]),
            tier_match(&t, StorageTier::Disk, &[(5, 8)]),
        ];
        let candidate = CandidateView {
            target: &t,
            matches: &matches,
            load: LoadSample {
                active_prefill_tokens: 7,
                active_requests: 3,
            },
            input_tokens: 160,
        };
        let params = CostParams {
            block_size: 16,
            transfer_credits: TransferCredits {
                host_pinned: 0.5,
                ..TransferCredits::default()
            },
            ..CostParams::default()
        };
        // 5 device blocks + 4 * .5 host blocks = 112 credited tokens.
        assert_eq!(
            cost_components(&candidate, &params),
            CostComponents {
                ptoken: 55,
                batchsize: 3
            }
        );
    }

    #[test]
    fn disconnected_external_segment_is_not_credited() {
        let t = target("w", 0);
        let matches = vec![
            tier_match(&t, StorageTier::Device, &[(0, 3)]),
            tier_match(&t, StorageTier::HostPinned, &[(4, 10)]),
        ];
        let candidate = CandidateView {
            target: &t,
            matches: &matches,
            load: LoadSample::default(),
            input_tokens: 160,
        };
        assert_eq!(
            cost_components(&candidate, &CostParams::default()).ptoken,
            112
        );
    }

    #[test]
    fn none_token_ids_reduce_ptoken_to_backlog() {
        let t = target("w", 0);
        let candidate = CandidateView {
            target: &t,
            matches: &[],
            load: LoadSample {
                active_prefill_tokens: 42,
                active_requests: 2,
            },
            input_tokens: 0,
        };
        assert_eq!(
            cost_components(&candidate, &CostParams::default()).ptoken,
            42
        );
    }

    #[test]
    fn gpu_credit_reserves_the_engine_last_token_block() {
        let t = target("w", 0);
        let params = CostParams::default();
        for (tokens, cached, new_prefill) in [
            (0, 0, 0),
            (16, 0, 16),
            (17, 16, 1),
            (32, 16, 16),
            (48, 32, 16),
            (49, 48, 1),
        ] {
            let hits = [tier_match(&t, StorageTier::Device, &[(0, 10)])];
            let candidate = CandidateView {
                target: &t,
                matches: &hits,
                load: LoadSample::default(),
                input_tokens: tokens,
            };
            assert_eq!(
                credited_tokens(&candidate, &params),
                cached as f64,
                "tokens={tokens}"
            );
            assert_eq!(
                request_prefill_tokens(&candidate, &params),
                new_prefill,
                "tokens={tokens}"
            );
        }
    }

    #[test]
    fn external_credit_cannot_remove_last_token_work() {
        let t = target("w", 0);
        let params = CostParams {
            transfer_credits: TransferCredits {
                external: 1.0,
                ..TransferCredits::default()
            },
            ..CostParams::default()
        };
        let hits = [tier_match(&t, StorageTier::External, &[(0, 3)])];
        let candidate = CandidateView {
            target: &t,
            matches: &hits,
            load: LoadSample::default(),
            input_tokens: 48,
        };
        assert_eq!(request_prefill_tokens(&candidate, &params), 1);
    }

    #[test]
    fn mixed_zero_cost_candidates_still_choose_the_idle_worker() {
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = ["http://decode", "http://idle", "http://prefill"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let slots = policy.slot_manager().unwrap();
        let decode = policy.target_for_worker(workers[0].as_ref());
        slots.begin_prefill(&decode, 8);
        slots.begin_decode(&decode, &decode, 8);
        assert_eq!(
            policy.select_worker_with_token_ids(&workers[..2], None),
            Some(1)
        );
        let prefill = policy.target_for_worker(workers[2].as_ref());
        slots.begin_prefill(&prefill, 8);
        let receipt = policy
            .select_worker_with_receipt(&workers, None, None, None)
            .unwrap();
        assert_eq!(receipt.worker_index, 1);
        assert_eq!(receipt.decision.cost, 0.0);
        assert_eq!(receipt.decision.components.batchsize, 0);
    }

    #[test]
    fn cached_and_empty_prompts_prefer_idle_worker_over_one_hundred_active_requests() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(index.clone(), CostParams::default());
        let workers: Vec<Arc<dyn Worker>> = ["http://busy", "http://idle"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let ids: Vec<u32> = (0..32).collect();
        let blocks: Vec<_> = KvAwarePolicy::local_hashes(&ids, 16)
            .into_iter()
            .enumerate()
            .map(|(i, local)| (Arc::from(format!("seq-{i}")), local))
            .collect();
        for worker in &workers {
            let t = policy.target_for_worker(worker.as_ref());
            index.store(
                0,
                crate::kv_index::ResidencyOwner::Worker {
                    source: t.instance_id,
                    dp_rank: t.dp_rank,
                    incarnation: 0,
                },
                StorageTier::Device,
                None,
                &blocks,
            );
        }
        for _ in 0..100 {
            workers[0].increment_load();
        }
        for (tokens, expected_prefill) in [(&ids[..0], 0), (&ids[..], 16)] {
            let receipt = policy
                .select_worker_with_receipt(&workers, Some(tokens), None, None)
                .unwrap();
            assert_eq!(receipt.worker_index, 1);
            assert_eq!(receipt.new_prefill_tokens, expected_prefill);
            assert_eq!(receipt.decision.components.batchsize, 0);
        }
    }

    #[test]
    fn scorer_counts_incoming_request_in_batchsize() {
        let t = target("w", 0);
        let candidate = CandidateView {
            target: &t,
            matches: &[],
            load: LoadSample {
                active_prefill_tokens: 5,
                active_requests: 2,
            },
            input_tokens: 10,
        };
        let mut costs = [3.0];
        MultiplicativeScorer
            .score(&[candidate], &CostParams::default(), &mut costs)
            .unwrap();
        assert_eq!(costs, [48.0]);
    }

    #[test]
    fn idle_cold_worker_is_charged_for_candidate_prefill() {
        let hot = target("hot", 0);
        let cold = target("cold", 0);
        let matches = [tier_match(&hot, StorageTier::Device, &[(0, 2468)])];
        let candidates = [
            CandidateView {
                target: &hot,
                matches: &matches,
                load: LoadSample {
                    active_requests: 1,
                    ..LoadSample::default()
                },
                input_tokens: 40_000,
            },
            CandidateView {
                target: &cold,
                matches: &[],
                load: LoadSample::default(),
                input_tokens: 40_000,
            },
        ];
        let mut costs = [0.0; 2];
        MultiplicativeScorer
            .score(&candidates, &CostParams::default(), &mut costs)
            .unwrap();
        assert_eq!(costs, [1024.0, 40_000.0]);
        assert_eq!(TemperaturePicker::new(0.0).pick(&costs), Some(0));
    }

    #[test]
    fn active_batch_size_preserves_saturation() {
        let t = target("w", 0);
        let candidate = CandidateView {
            target: &t,
            matches: &[],
            load: LoadSample {
                active_requests: usize::MAX,
                ..LoadSample::default()
            },
            input_tokens: 1,
        };
        assert_eq!(
            cost_components(&candidate, &CostParams::default()).batchsize,
            usize::MAX
        );
        let mut costs = [0.0];
        MultiplicativeScorer
            .score(&[candidate], &CostParams::default(), &mut costs)
            .unwrap();
        assert!(costs[0] > 0.0);
    }

    #[test]
    fn cache_hit_offsets_load_until_cold_prefill_becomes_cheaper() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let warm = target("http://warm", 0);
        policy.register_target("http://warm", warm.clone());
        policy.register_target("http://cold", target("http://cold", 0));
        let ids = [1_u32, 2, 3, 4, 5, 6];
        let hashes = KvAwarePolicy::local_hashes(&ids, 2);
        index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: warm.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), hashes[0].clone()),
                (Arc::from("seq-2"), hashes[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://warm".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://cold".into(), WorkerType::Regular)),
        ];
        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None)
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.new_prefill_tokens, 2);
        assert_eq!(receipt.decision.cost, 2.0);
        assert_eq!(receipt.decision.components.batchsize, 0);

        workers[0].increment_load();
        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None)
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.decision.cost, 4.0);
        assert_eq!(receipt.decision.components.batchsize, 1);

        workers[0].increment_load();
        workers[0].increment_load();
        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None)
            .unwrap();
        assert_eq!(receipt.worker_index, 1);
        assert_eq!(receipt.decision.cost, 6.0);
        assert_eq!(receipt.decision.components.batchsize, 0);
    }

    #[test]
    fn argmin_picker_is_deterministic() {
        assert_eq!(TemperaturePicker::new(0.0).pick(&[5.0, 2.0, 2.0]), Some(1));
    }

    #[test]
    fn missing_token_ids_fall_back_to_live_worker_load() {
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://busy".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://idle".into(), WorkerType::Regular)),
        ];
        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        assert_eq!(policy.select_worker_with_token_ids(&workers, None), Some(1));
    }

    #[test]
    fn lora_name_is_seeded_into_request_block_hashes() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let base = target("http://base", 0);
        let lora = target("http://lora", 0);
        policy.register_target("http://base", base.clone());
        policy.register_target("http://lora", lora.clone());
        let ids = [1_u32, 2, 3, 4];
        let base_hashes = KvAwarePolicy::local_hashes(&ids, 2);
        let lora_hashes = KvAwarePolicy::local_hashes_with_lora(&ids, 2, Some("adapter"));
        for (target, hashes) in [(&base, base_hashes), (&lora, lora_hashes)] {
            index.store(
                0,
                crate::kv_index::ResidencyOwner::Worker {
                    source: target.instance_id.clone(),
                    dp_rank: 0,
                    incarnation: 0,
                },
                StorageTier::Device,
                None,
                &[
                    (Arc::from("seq-1"), hashes[0].clone()),
                    (Arc::from("seq-2"), hashes[1].clone()),
                ],
            );
        }
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://base".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://lora".into(), WorkerType::Regular)),
        ];
        for worker in &workers {
            worker.increment_load();
        }

        assert_eq!(
            policy
                .select_worker_with_receipt(&workers, Some(&ids), Some("adapter"), None)
                .map(|receipt| receipt.worker_index),
            Some(1)
        );
    }

    #[test]
    fn cache_salt_disables_incompatible_placement_credit() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let cached = target("http://cached", 0);
        let uncached = target("http://uncached", 0);
        policy.register_target("http://cached", cached.clone());
        policy.register_target("http://uncached", uncached);
        let ids = [1_u32, 2, 3, 4];
        let hashes = KvAwarePolicy::local_hashes(&ids, 2);
        index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: cached.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), hashes[0].clone()),
                (Arc::from("seq-2"), hashes[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://cached".into(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://uncached".into(),
                WorkerType::Regular,
            )),
        ];
        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        assert_eq!(
            policy
                .select_worker_with_receipt(&workers, Some(&ids), None, Some("tenant-salt"),)
                .map(|receipt| receipt.worker_index),
            Some(1)
        );
        assert_eq!(
            policy
                .select_worker_with_receipt(
                    std::slice::from_ref(&workers[0]),
                    Some(&ids),
                    None,
                    Some("tenant-salt"),
                )
                .unwrap()
                .new_prefill_tokens,
            ids.len() as u64
        );
    }

    #[test]
    fn selection_queries_each_cache_identity_with_its_own_provider() {
        let first_index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let second_index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = KvAwarePolicy::default();
        let first = target("first-instance", 0);
        let second = target("second-instance", 0);
        let first_key = CacheKey {
            model: Arc::from("model-a"),
            hash_mode: crate::kv_index::HashMode::Sha256,
            block_size: 2,
        };
        let second_key = CacheKey {
            model: Arc::from("model-b"),
            hash_mode: crate::kv_index::HashMode::Sha256,
            block_size: 4,
        };
        policy.register_target_with_cache_identity(
            "http://first",
            first.clone(),
            first_key,
            first_index.clone(),
        );
        policy.register_target_with_cache_identity(
            "http://second",
            second,
            second_key,
            second_index,
        );

        let ids = [1_u32, 2, 3, 4];
        let hashes = KvAwarePolicy::local_hashes(&ids, 2);
        first_index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: first.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), hashes[0].clone()),
                (Arc::from("seq-2"), hashes[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://first".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new(
                "http://second".into(),
                WorkerType::Regular,
            )),
        ];
        for worker in &workers {
            worker.increment_load();
        }

        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None)
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.decision.target, first);
        assert_eq!(receipt.new_prefill_tokens, 2);
    }

    #[test]
    fn idle_pool_prefers_deep_hit() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let first = target("http://w1", 0);
        let second = target("http://w2", 0);
        policy.register_target("http://w1", first.clone());
        policy.register_target("http://w2", second);
        let ids = [1_u32, 2, 3, 4];
        let locals = KvAwarePolicy::local_hashes(&ids, 2);
        index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: first.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), locals[0].clone()),
                (Arc::from("seq-2"), locals[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://w2".into(), WorkerType::Regular)),
        ];
        assert_eq!(
            policy.select_worker_with_token_ids(&workers, Some(&ids)),
            Some(0)
        );
    }
    #[test]
    fn stateless_policy_has_no_kv_capability() {
        assert!(kv_aware_ops(&RoundRobinPolicy::new()).is_none());
    }

    #[test]
    fn registration_uses_the_discovered_cache_block_size_and_provider() {
        #[derive(Default)]
        struct QueryBlocks(AtomicUsize);
        impl TieredMatchProvider for QueryBlocks {
            fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
                self.0.store(query.local_hashes.len(), Ordering::Relaxed);
                Vec::new()
            }
        }

        let policy = KvAwarePolicy::default();
        let ops = kv_aware_ops(&policy).unwrap();
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            "grpc://worker:50051".into(),
            1,
            2,
            WorkerType::Regular,
        ));
        let matches = Arc::new(QueryBlocks::default());
        ops.register_worker(
            worker.as_ref(),
            &WorkerKvInfo {
                instance_id: "engine-instance".into(),
                cache_key: Some(CacheKey {
                    model: Arc::from("model"),
                    hash_mode: crate::kv_index::HashMode::Sha256,
                    block_size: 2,
                }),
                ..WorkerKvInfo::default()
            },
            Some(matches.clone()),
        );
        assert!(ops
            .select_worker_with_receipt(&[worker], Some(&[1, 2, 3, 4, 5]), None, None)
            .is_some());
        assert_eq!(matches.0.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn capability_preserves_rank_identity_and_request_lifecycle() {
        let policy = KvAwarePolicy::default();
        let ops = kv_aware_ops(&policy).unwrap();
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            "grpc://worker:50051".into(),
            2,
            4,
            WorkerType::Regular,
        ));
        ops.register_worker(
            worker.as_ref(),
            &WorkerKvInfo {
                instance_id: "engine-instance".into(),
                ..WorkerKvInfo::default()
            },
            None,
        );
        let target = policy.target_for_worker(worker.as_ref());
        assert_eq!(target.instance_id.as_ref(), "engine-instance");
        assert_eq!(target.dp_rank, 2);
        let manager = policy.slot_manager().unwrap();
        let receipt = ops
            .select_worker_with_receipt(std::slice::from_ref(&worker), Some(&[1, 2, 3]), None, None)
            .unwrap();
        assert_eq!(receipt.decision.target, target);
        let tokens = receipt.new_prefill_tokens;
        let slot = ops.begin_request(receipt).unwrap();
        assert_eq!(
            manager.sample(&target),
            LoadSample {
                active_prefill_tokens: tokens,
                active_requests: 1,
            }
        );
        let cloned_slot = slot.clone();
        slot.begin_decode();
        cloned_slot.begin_decode();
        assert_eq!(
            manager.sample(&target),
            LoadSample {
                active_prefill_tokens: 0,
                active_requests: 1,
            }
        );
        slot.release();
        cloned_slot.release();
        assert_eq!(manager.sample(&target), LoadSample::default());
        ops.unregister_worker(worker.url());
        assert_eq!(
            policy
                .target_for_worker(worker.as_ref())
                .instance_id
                .as_ref(),
            worker.base_url()
        );
    }

    #[test]
    fn pick_decode_binds_sticks_and_spills_within_margin() {
        let policy = KvAwarePolicy::default(); // spill margin 2
        let urls = ["http://d0", "http://d1"];

        let first = policy.pick_decode(Some("s1"), &urls, &[0, 0]).unwrap();
        assert_eq!((first.index, first.outcome), (0, "bind"));

        // The bound decoder stays sticky while within the margin of the
        // pool minimum (loads [2, 0]: 2 <= 0 + 2).
        let hit = policy.pick_decode(Some("s1"), &urls, &[2, 0]).unwrap();
        assert_eq!((hit.index, hit.outcome), (0, "sticky_hit"));

        // Beyond the margin the session spills to the lighter decoder and
        // rebinds there.
        let spill = policy.pick_decode(Some("s1"), &urls, &[3, 0]).unwrap();
        assert_eq!((spill.index, spill.outcome), (1, "sticky_spill"));
        let rehit = policy.pick_decode(Some("s1"), &urls, &[3, 3]).unwrap();
        assert_eq!((rehit.index, rehit.outcome), (1, "sticky_hit"));
    }

    #[test]
    fn pick_decode_rebinds_when_bound_worker_leaves_pool() {
        let policy = KvAwarePolicy::default();
        let _ = policy
            .pick_decode(Some("s1"), &["http://d0", "http://d1"], &[0, 0])
            .unwrap();
        // Pool shrank to d1 only: the stale binding must rebind, not fail.
        let pick = policy
            .pick_decode(Some("s1"), &["http://d1"], &[5])
            .unwrap();
        assert_eq!((pick.index, pick.outcome), (0, "bind"));
    }

    #[test]
    fn pick_decode_without_session_uses_pure_min_load() {
        let policy = KvAwarePolicy::default();
        let pick = policy
            .pick_decode(None, &["http://d0", "http://d1"], &[4, 1])
            .unwrap();
        assert_eq!((pick.index, pick.outcome), (1, "no_session"));
        assert!(policy.pick_decode(None, &[], &[]).is_none());
        assert!(policy.pick_decode(None, &["http://d0"], &[0, 0]).is_none());
    }

    #[test]
    fn pick_decode_rotates_equal_load_ties() {
        let policy = KvAwarePolicy::default();
        let first = policy
            .pick_decode(None, &["http://d0", "http://d1"], &[0, 0])
            .unwrap();
        let second = policy
            .pick_decode(None, &["http://d0", "http://d1"], &[0, 0])
            .unwrap();
        assert_ne!(first.index, second.index);
    }

    #[test]
    fn defection_tax_charges_only_recompute_gap() {
        let policy = KvAwarePolicy::default(); // hysteresis 1.0
                                               // 40 - 32 = 8 is inside the 4-block dead zone and stays untaxed.
        assert_eq!(policy.defection_tax(&[32, 1000, 40]), vec![0.0, 968.0, 0.0]);
        let untaxed = KvAwarePolicy::new(CostParams {
            defection_hysteresis: 0.0,
            ..CostParams::default()
        });
        assert!(untaxed
            .defection_tax(&[32, 1000])
            .iter()
            .all(|tax| *tax == 0.0));
        assert!(KvAwarePolicy::default().defection_tax(&[]).is_empty());
    }

    #[test]
    fn defection_dead_zone_scales_with_discovered_block_size() {
        let policy = KvAwarePolicy::default(); // dead zone = 4 blocks
                                               // Before any registration the config default (16-token blocks)
                                               // applies: a 64-token gap is exactly the dead zone, so untaxed.
        assert_eq!(policy.defection_tax(&[32, 96]), vec![0.0, 0.0]);
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            "http://w".into(),
            0,
            1,
            WorkerType::Regular,
        ));
        kv_aware_ops(&policy).unwrap().register_worker(
            worker.as_ref(),
            &WorkerKvInfo {
                instance_id: "engine-instance".into(),
                cache_key: Some(CacheKey {
                    model: Arc::from("model"),
                    hash_mode: crate::kv_index::HashMode::Sha256,
                    block_size: 32,
                }),
                ..WorkerKvInfo::default()
            },
            Some(Arc::new(EmptyMatchProvider)),
        );
        // With 32-token blocks the dead zone widens to 128 tokens: a
        // 64-token gap stays untaxed, a 129-token gap is taxed.
        assert_eq!(policy.defection_tax(&[32, 96]), vec![0.0, 0.0]);
        assert_eq!(policy.defection_tax(&[32, 161]), vec![0.0, 129.0]);
    }
}

//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tokio::sync::{broadcast, mpsc};
use tokio::task::{JoinError, JoinHandle};
use tracing::{debug, info, warn};

use crate::kv_events::decode_batch;
use crate::kv_index::indexer::KvBlockIndexer;
use crate::kv_index::types::{ClearScope, ResidencyOwner, SourceId, StorageTier};
use crate::kv_index::wire::{ExternalBlockHash, KVEvent, KVEventBatch};
use crate::kv_index::IngestionSignal;

/// sha256 of each block's token-id bytes, seeded by `lora_name`.
/// The seed is re-applied per block (no parent chaining here) so identical
/// tokens under different adapters land on different nodes.
// ponytail: cache_salt/MM isolation deferred — needs raw msgpack capture of
// per-block extra_keys tuples (custom newtype or rmpv); add when cost model
// scores cross-salt collisions. Until then lora_name alone distinguishes adapters.
pub fn local_hashes(token_ids: &[u32], block_size: u32, lora_name: Option<&str>) -> Vec<[u8; 32]> {
    let bs = block_size as usize;
    if bs == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(token_ids.len().div_ceil(bs));
    for chunk in token_ids.chunks(bs) {
        let mut hasher = Sha256::new();
        if let Some(l) = lora_name {
            hasher.update(l.as_bytes());
        }
        for &tid in chunk {
            hasher.update(tid.to_le_bytes());
        }
        out.push(hasher.finalize().into());
    }
    out
}

/// The signed sentinel ending a replay stream: `seq = -1`.
const END_SEQ: i64 = -1;

/// One event source for one (worker, rank).
#[derive(Clone, Debug)]
pub struct EventSource {
    pub source: SourceId,
    pub dp_rank: u32,
    pub pub_endpoint: String,
    pub replay_endpoint: Option<String>,
    pub topic: String,
    /// SUB receive high-water mark; None leaves the ZMQ default.
    pub hwm: Option<i32>,
}

/// A handle to a running subscriber task; drop it (or call `shutdown`) to stop.
pub struct SubscriberHandle {
    shutdown_tx: broadcast::Sender<()>,
    lease: SubscriberLease,
    task: Option<JoinHandle<()>>,
}

impl SubscriberHandle {
    /// Synchronously fence publication and clear this attachment's Worker claims.
    pub fn shutdown(&self) {
        self.lease.retire();
        let _ = self.shutdown_tx.send(());
        if let Some(task) = &self.task {
            task.abort();
        }
    }

    /// Retire and wait for task/socket destruction. Cancellation is expected;
    /// a task panic remains visible to the caller.
    pub async fn shutdown_and_wait(mut self) -> Result<(), JoinError> {
        self.shutdown();
        match self.task.take().expect("subscriber owns its task").await {
            Err(error) if error.is_cancelled() => Ok(()),
            result => result,
        }
    }
}

impl Drop for SubscriberHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct LeaseState {
    active: bool,
    incarnation: u64,
    last_seq: i64,
    last_live_seq: i64,
    applied_ranks: HashSet<u32>,
}

/// Shared only by registrations for one cache identity and publishing source.
pub(super) type RankClaims = Arc<Mutex<HashMap<u32, Arc<()>>>>;

#[derive(Clone)]
pub(super) struct RankOwnership {
    pub(super) claims: RankClaims,
    pub(super) token: Arc<()>,
}

impl RankOwnership {
    fn claim(&self, claims: &mut HashMap<u32, Arc<()>>, rank: u32) -> bool {
        match claims.entry(rank) {
            Entry::Occupied(entry) => Arc::ptr_eq(entry.get(), &self.token),
            Entry::Vacant(entry) => {
                entry.insert(self.token.clone());
                true
            }
        }
    }
}

enum ApplyResult {
    Applied,
    Skipped,
    RankConflict,
}

#[derive(Clone)]
struct SubscriberLease {
    state: Arc<Mutex<LeaseState>>,
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    dp_rank: u32,
    ownership: Option<RankOwnership>,
}

impl SubscriberLease {
    fn clear_worker(&self, state: &mut LeaseState, claims: Option<&HashMap<u32, Arc<()>>>) {
        for rank in state.applied_ranks.drain() {
            if let Some(ownership) = &self.ownership {
                if !claims
                    .and_then(|claims| claims.get(&rank))
                    .is_some_and(|token| Arc::ptr_eq(token, &ownership.token))
                {
                    continue;
                }
            }
            self.indexer.clear(
                &owner_for(&self.source, rank, state.incarnation),
                ClearScope::Worker,
            );
        }
    }

    fn retire(&self) {
        let mut state = self.state.lock();
        if state.active {
            let mut claims = self.ownership.as_ref().map(|owner| owner.claims.lock());
            state.active = false;
            self.clear_worker(&mut state, claims.as_deref());
            if let (Some(ownership), Some(claims)) = (&self.ownership, &mut claims) {
                // Clear before releasing a rank for another current writer.
                claims.retain(|_, token| !Arc::ptr_eq(token, &ownership.token));
            }
        }
    }

    fn apply(
        &self,
        seq: i64,
        batch: &KVEventBatch,
        signals: &Option<mpsc::Sender<IngestionSignal>>,
    ) -> ApplyResult {
        // No await inside this fence: retirement, writes, high-water and the
        // positive observation signal are one publication operation.
        let mut state = self.state.lock();
        if !state.active {
            return ApplyResult::Skipped;
        }
        let rank = batch.data_parallel_rank.unwrap_or(self.dp_rank);
        let mut claims = self.ownership.as_ref().map(|owner| owner.claims.lock());
        if let (Some(ownership), Some(claims)) = (&self.ownership, &mut claims) {
            if !ownership.claim(claims, rank) {
                warn!(
                    "kv_index {}: batch rank{} belongs to another subscriber",
                    self.source, rank
                );
                return ApplyResult::RankConflict;
            }
        }
        if seq <= state.last_seq {
            return ApplyResult::Skipped;
        }
        state.applied_ranks.insert(rank);
        apply_batch(
            &self.indexer,
            &self.source,
            self.dp_rank,
            state.incarnation,
            batch,
        );
        state.last_seq = seq;
        emit(
            signals,
            IngestionSignal::Advance {
                source: self.source.clone(),
                last_seq: seq,
            },
        );
        ApplyResult::Applied
    }
}

struct RetireOnDrop(SubscriberLease);

impl Drop for RetireOnDrop {
    fn drop(&mut self) {
        self.0.retire();
    }
}

/// Spawn one subscriber task for an event source. Applies events to the shared
/// indexer in `seq` order; on a gap, opens a DEALER replay to `replay_endpoint`.
/// `signal_tx` receives ingestion signals for the trust arbiter; None disables.
/// Standalone callers must exclusively own every actual source/rank they emit,
/// including batch-rank overrides, and retire before attaching a replacement.
/// Discovery coordinates supervised writers; mixing standalone and supervised
/// writers in the same ownership domain is not protected by this API.
pub fn spawn(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
) -> SubscriberHandle {
    spawn_with_ownership(source, indexer, signal_tx, None)
}

pub(super) fn spawn_with_ownership(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
    ownership: Option<RankOwnership>,
) -> SubscriberHandle {
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
    let pub_endpoint = source.pub_endpoint.clone();
    let replay_endpoint = source.replay_endpoint.clone();
    let topic = source.topic.clone();
    let hwm = source.hwm;
    let signals = signal_tx;
    let lease = SubscriberLease {
        state: Arc::new(Mutex::new(LeaseState {
            active: true,
            incarnation: 0,
            last_seq: -1,
            last_live_seq: -1,
            applied_ranks: HashSet::new(),
        })),
        indexer,
        source: source.source,
        dp_rank: source.dp_rank,
        ownership,
    };
    let task_lease = lease.clone();

    let task = tokio::spawn(async move {
        let lease = task_lease;
        let _retire = RetireOnDrop(lease.clone());
        let context = zmq::Context::new();
        let sub = match context.socket(zmq::SUB) {
            Ok(s) => s,
            Err(e) => {
                warn!("kv_index sub socket for {}: {}", lease.source, e);
                return;
            }
        };
        if let Err(e) = (|| {
            sub.set_linger(0)?;
            sub.set_subscribe(topic.as_bytes())?;
            if let Some(h) = hwm {
                sub.set_rcvhwm(h)?;
            }
            sub.connect(&pub_endpoint)
        })() {
            warn!(
                "kv_index configure/connect {} to {}: {}",
                lease.source, pub_endpoint, e
            );
            return;
        }
        info!(
            "kv_index subscriber for {} rank{} on {}",
            lease.source, lease.dp_rank, pub_endpoint
        );

        loop {
            if is_shutdown(&mut shutdown_rx) {
                info!("kv_index subscriber {} shutting down", lease.source);
                break;
            }
            match sub.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => {
                    // PUB frame: [topic, seq, msgpack_payload].
                    if frames.len() != 3 || frames[0] != topic.as_bytes() {
                        continue;
                    }
                    let Some(seq) = read_seq(&frames[1]).filter(|seq| *seq >= 0) else {
                        continue;
                    };
                    let Some(batch) = decode(&frames[2]) else {
                        continue;
                    };
                    let gap = {
                        let mut state = lease.state.lock();
                        if !state.active {
                            break;
                        }
                        let rank = batch.data_parallel_rank.unwrap_or(lease.dp_rank);
                        let mut claims = lease.ownership.as_ref().map(|owner| owner.claims.lock());
                        if let (Some(ownership), Some(claims)) = (&lease.ownership, &mut claims) {
                            if !ownership.claim(claims, rank) {
                                warn!(
                                    "kv_index {}: live batch rank{} belongs to another subscriber",
                                    lease.source, rank
                                );
                                continue;
                            }
                        }
                        // A validated backwards sequence preserves the existing
                        // restart heuristic; this is not remote-generation proof.
                        if state.last_live_seq != -1 && seq < state.last_live_seq {
                            lease.clear_worker(&mut state, claims.as_deref());
                            state.incarnation += 1;
                            state.last_seq = -1;
                            emit(
                                &signals,
                                IngestionSignal::IncarnationReset {
                                    source: lease.source.clone(),
                                    incarnation: state.incarnation,
                                },
                            );
                        }
                        state.last_live_seq = seq;
                        if seq <= state.last_seq {
                            continue;
                        }
                        if state.last_seq != -1 && seq > state.last_seq + 1 {
                            emit(
                                &signals,
                                IngestionSignal::Gap {
                                    source: lease.source.clone(),
                                    from_seq: state.last_seq + 1,
                                    to_seq: seq - 1,
                                },
                            );
                            Some(state.last_seq)
                        } else {
                            None
                        }
                    };
                    if let Some(last_seq) = gap {
                        let from = last_seq + 1;
                        let to = seq - 1;
                        debug!("kv_index {}: gap {}..{}", lease.source, from, to);
                        if let Some(ep) = replay_endpoint.as_deref() {
                            let applied =
                                replay(ep, &topic, last_seq, to, &context, &lease, &signals).await;
                            if let Some(last_replay) = applied {
                                let state = lease.state.lock();
                                if state.active {
                                    emit(
                                        &signals,
                                        IngestionSignal::ReplayApplied {
                                            source: lease.source.clone(),
                                            replay_seq: last_replay,
                                        },
                                    );
                                }
                            }
                        }
                    }
                    lease.apply(seq, &batch, &signals);
                }
                Err(zmq::Error::EAGAIN) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(e) => {
                    warn!("kv_index sub {} recv: {}", lease.source, e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    });

    SubscriberHandle {
        shutdown_tx,
        lease,
        task: Some(task),
    }
}

/// Shut down on an explicit signal OR the handle (sender) being dropped.
fn is_shutdown(rx: &mut broadcast::Receiver<()>) -> bool {
    matches!(
        rx.try_recv(),
        Ok(()) | Err(broadcast::error::TryRecvError::Closed)
    )
}

/// Advisory signal send: `try_send` so a slow arbiter never stalls the hot loop.
/// A dropped signal under load is covered by the DEGRADED window.
fn emit(tx: &Option<mpsc::Sender<IngestionSignal>>, signal: IngestionSignal) {
    if let Some(tx) = tx {
        if let Err(e) = tx.try_send(signal) {
            debug!("kv_index signal dropped: {}", e);
        }
    }
}

fn read_seq(bytes: &[u8]) -> Option<i64> {
    bytes.try_into().ok().map(i64::from_be_bytes)
}

fn decode(payload: &[u8]) -> Option<KVEventBatch> {
    match decode_batch(payload) {
        Ok(b) => Some(b),
        Err(e) => {
            warn!("kv_index decode batch: {}", e);
            None
        }
    }
}

fn owner_for(source_id: &SourceId, dp_rank: u32, incarnation: u64) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: source_id.clone(),
        dp_rank,
        incarnation,
    }
}

/// Build the owner an event's blocks belong to. A non-empty `ownership` field
/// tags the block as a shared-pool copy (`CacheOwner`); otherwise the worker's
/// own residency (`Worker`).
fn event_owner(
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    ownership: Option<&str>,
) -> ResidencyOwner {
    match ownership {
        Some(pool) if !pool.is_empty() => ResidencyOwner::CacheOwner {
            pool_id: Arc::from(pool),
            source: source_id.clone(),
            dp_rank,
        },
        _ => ResidencyOwner::Worker {
            source: source_id.clone(),
            dp_rank,
            incarnation,
        },
    }
}

/// Tier from the owner domain: CacheOwner → External, Worker → from `medium`.
fn event_tier(owner: &ResidencyOwner, medium: Option<&str>) -> StorageTier {
    match owner {
        ResidencyOwner::CacheOwner { .. } => StorageTier::External,
        ResidencyOwner::Worker { .. } => StorageTier::from_medium(medium),
    }
}

fn hash_str(h: &ExternalBlockHash) -> Arc<str> {
    match h {
        ExternalBlockHash::Int(i) => Arc::from(format!("{}", i)),
        ExternalBlockHash::Bytes(b) => Arc::from(hex(b)),
    }
}

fn apply_batch(
    indexer: &KvBlockIndexer,
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    batch: &KVEventBatch,
) {
    // Batch rank if the 3-element array carries it; else the discovery rank.
    let rank = batch.data_parallel_rank.unwrap_or(dp_rank);
    for ev in &batch.events {
        match ev {
            KVEvent::BlockStored(s) => {
                let group = s.group_idx.unwrap_or(0);
                let owner = event_owner(source_id, rank, incarnation, s.ownership.as_deref());
                let tier = event_tier(&owner, s.medium.as_deref());
                let locals: Vec<Arc<str>> =
                    local_hashes(&s.token_ids, s.block_size, s.lora_name.as_deref())
                        .iter()
                        .map(|b| Arc::from(hex(b)))
                        .collect();
                // vLLM emits one BlockStored per contiguous chain (block[i]'s parent
                // is block[i-1]); zipping seqs with local hashes yields one edge.
                let seqs: Vec<Arc<str>> = s.block_hashes.iter().map(hash_str).collect();
                let blocks: Vec<(Arc<str>, Arc<str>)> = seqs.into_iter().zip(locals).collect();
                let parent_seq = s.parent_block_hash.as_ref().map(hash_str);
                indexer.store(group, owner, tier, parent_seq, &blocks);
            }
            KVEvent::BlockRemoved(r) => {
                let group = r.group_idx.unwrap_or(0);
                let owner = event_owner(source_id, rank, incarnation, r.ownership.as_deref());
                let tier = event_tier(&owner, r.medium.as_deref());
                let hashes: Vec<Arc<str>> = r.block_hashes.iter().map(hash_str).collect();
                indexer.remove(group, &owner, tier, &hashes);
            }
            KVEvent::AllBlocksCleared(_) => {
                indexer.clear(&owner_for(source_id, rank, incarnation), ClearScope::Worker);
            }
        }
    }
}

/// Backfill `(last_seq, ∞)` from the replay ROUTER via a DEALER socket.
/// Frames with `seq <= last_seq` are skipped — deduped so an out-of-order
/// re-send can't re-apply a stale `remove` over a freshly re-`store`d block.
/// Returns the highest applied sequence only when a valid end marker completes
/// a contiguous gap; this is an advisory observation, never global EXACT.
async fn replay(
    endpoint: &str,
    topic: &str,
    last_seq: i64,
    gap_to: i64,
    ctx: &zmq::Context,
    lease: &SubscriberLease,
    signals: &Option<mpsc::Sender<IngestionSignal>>,
) -> Option<i64> {
    let dealer = match ctx.socket(zmq::DEALER) {
        Ok(s) => s,
        Err(e) => {
            warn!("kv_index replay socket for {}: {}", lease.source, e);
            return None;
        }
    };
    if let Err(e) = dealer.set_linger(0).and_then(|()| dealer.connect(endpoint)) {
        warn!("kv_index replay connect {}: {}", endpoint, e);
        return None;
    }
    // Request everything strictly past the live high-water.
    let start_bytes = ((last_seq + 1) as u64).to_be_bytes();
    // DEALER sends [empty delim, start_seq].
    if dealer
        .send_multipart([vec![], start_bytes.to_vec()], zmq::DONTWAIT)
        .is_err()
    {
        warn!("kv_index replay {} send failed", lease.source);
        return None;
    }
    // ponytail: bounded poll loop. Replay is brief and rare; a hard deadline
    // guards against a silent router. Swap for backoff if replay grows large.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut high = last_seq;
    let mut last_applied: Option<i64> = None;
    let mut contiguous = true;
    loop {
        if !lease.state.lock().active || tokio::time::Instant::now() >= deadline {
            return None;
        }
        match dealer.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => {
                // DEALER reply: [empty, exact topic, 8-byte signed seq, payload].
                if frames.len() != 4 || !frames[0].is_empty() || frames[1] != topic.as_bytes() {
                    continue;
                }
                let Some(seq) = read_seq(&frames[2]) else {
                    continue;
                };
                if seq == END_SEQ {
                    // Applied observations do not imply complete recovery. A
                    // missing sequence or timeout must not report a filled gap.
                    return last_applied.filter(|_| contiguous && high >= gap_to);
                }
                if seq < 0 {
                    continue;
                }
                let Some(batch) = decode(&frames[3]) else {
                    continue;
                };
                match lease.apply(seq, &batch, signals) {
                    ApplyResult::Applied => {
                        contiguous &= seq == high + 1;
                        high = seq;
                        last_applied = Some(seq);
                    }
                    ApplyResult::RankConflict => contiguous = false,
                    ApplyResult::Skipped => {}
                }
            }
            Err(zmq::Error::EAGAIN) => {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            Err(e) => {
                warn!("kv_index replay {} recv: {}", lease.source, e);
                return None;
            }
        }
    }
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        write!(s, "{:02x}", byte).unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_index::indexer::MatchQuery;
    use crate::kv_index::types::Locality;
    use crate::kv_index::wire::BlockStored;

    fn coordinated_lease(
        indexer: &Arc<KvBlockIndexer>,
        claims: &RankClaims,
        rank: u32,
    ) -> SubscriberLease {
        let token = Arc::new(());
        assert!(claims.lock().insert(rank, token.clone()).is_none());
        SubscriberLease {
            state: Arc::new(Mutex::new(LeaseState {
                active: true,
                incarnation: 0,
                last_seq: -1,
                last_live_seq: -1,
                applied_ranks: HashSet::new(),
            })),
            indexer: indexer.clone(),
            source: Arc::from("shared-source"),
            dp_rank: rank,
            ownership: Some(RankOwnership {
                claims: claims.clone(),
                token,
            }),
        }
    }

    fn rank_batch(rank: u32) -> KVEventBatch {
        KVEventBatch {
            ts: 1.0,
            events: vec![KVEvent::BlockStored(stored(
                vec![ExternalBlockHash::Int(42)],
                vec![1, 2, 3, 4],
                Some("GPU"),
                None,
            ))],
            data_parallel_rank: Some(rank),
        }
    }

    fn rank_query() -> MatchQuery {
        MatchQuery {
            group_idx: 0,
            local_hashes: local_hashes(&[1, 2, 3, 4], 4, None)
                .iter()
                .map(|hash| Arc::from(hex(hash)))
                .collect(),
            tiers_of_interest: vec![StorageTier::Device],
        }
    }

    #[test]
    fn shared_rank_retirement_and_old_drop_preserve_peer_and_replacement_residency() {
        let index = Arc::new(KvBlockIndexer::new());
        let claims = Arc::new(Mutex::new(HashMap::new()));
        let first = coordinated_lease(&index, &claims, 0);
        let peer = coordinated_lease(&index, &claims, 1);
        assert!(matches!(
            first.apply(0, &rank_batch(0), &None),
            ApplyResult::Applied
        ));
        assert!(matches!(
            peer.apply(0, &rank_batch(1), &None),
            ApplyResult::Applied
        ));
        assert_eq!(index.find_matches(&rank_query()).len(), 2);
        first.retire();
        let hits = index.find_matches(&rank_query());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].target.dp_rank, 1);
        assert!(!claims.lock().contains_key(&0));
        assert!(claims.lock().contains_key(&1));
        let replacement = coordinated_lease(&index, &claims, 0);
        assert!(matches!(
            replacement.apply(0, &rank_batch(0), &None),
            ApplyResult::Applied
        ));
        first.retire(); // The same idempotent path used by an old handle's Drop.
        assert_eq!(index.find_matches(&rank_query()).len(), 2);
        replacement.retire();
        peer.retire();
        assert!(index.find_matches(&rank_query()).is_empty());
        assert!(claims.lock().is_empty());
    }

    #[test]
    fn rank_conflict_and_restart_clear_only_owned_residency_and_retain_reservations() {
        let index = Arc::new(KvBlockIndexer::new());
        let claims = Arc::new(Mutex::new(HashMap::new()));
        let first = coordinated_lease(&index, &claims, 0);
        let peer = coordinated_lease(&index, &claims, 1);
        assert!(matches!(
            first.apply(5, &rank_batch(0), &None),
            ApplyResult::Applied
        ));
        assert!(matches!(
            peer.apply(5, &rank_batch(1), &None),
            ApplyResult::Applied
        ));
        let conflicting = KVEventBatch {
            ts: 1.0,
            events: vec![KVEvent::AllBlocksCleared(Default::default())],
            data_parallel_rank: Some(1),
        };
        assert!(matches!(
            first.apply(6, &conflicting, &None),
            ApplyResult::RankConflict
        ));
        assert_eq!(first.state.lock().last_seq, 5);
        assert_eq!(index.find_matches(&rank_query()).len(), 2);
        assert!(matches!(
            first.apply(6, &rank_batch(3), &None),
            ApplyResult::Applied
        ));
        assert_eq!(index.find_matches(&rank_query()).len(), 3);
        {
            // The production restart's clear operation holds this same lease
            // -> claims -> index fence, without releasing active reservations.
            let mut state = first.state.lock();
            let owned = claims.lock();
            first.clear_worker(&mut state, Some(&owned));
            assert!(owned.contains_key(&0));
            assert!(owned.contains_key(&1));
            assert!(owned.contains_key(&3));
        }
        let hits = index.find_matches(&rank_query());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].target.dp_rank, 1);
        first.retire();
        assert!(!claims.lock().contains_key(&0));
        assert!(!claims.lock().contains_key(&3));
        assert!(claims.lock().contains_key(&1));
        let replacement = coordinated_lease(&index, &claims, 3);
        assert!(matches!(
            replacement.apply(0, &rank_batch(3), &None),
            ApplyResult::Applied
        ));
        first.retire();
        assert_eq!(index.find_matches(&rank_query()).len(), 2);
        replacement.retire();
        peer.retire();
    }

    fn stored(
        block_hashes: Vec<ExternalBlockHash>,
        token_ids: Vec<u32>,
        medium: Option<&str>,
        ownership: Option<&str>,
    ) -> BlockStored {
        BlockStored {
            extra_keys: None,
            block_hashes,
            parent_block_hash: None,
            token_ids,
            block_size: 4,
            lora_id: None,
            medium: medium.map(String::from),
            lora_name: None,
            group_idx: Some(0),
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: ownership.map(String::from),
            session_id: None,
        }
    }

    #[test]
    fn apply_batch_routes_ownership_to_tier_and_rank_from_batch() {
        let idx = Arc::new(KvBlockIndexer::new());
        let batch = KVEventBatch {
            ts: 1.0,
            events: vec![
                // ownership set → CacheOwner → External (not GPU/Device).
                KVEvent::BlockStored(stored(
                    vec![ExternalBlockHash::Int(10)],
                    vec![1, 2, 3, 4],
                    Some("GPU"),
                    Some("mooncake"),
                )),
                // ownership None, medium CPU → Worker → HostPinned.
                KVEvent::BlockStored(stored(
                    vec![ExternalBlockHash::Int(20)],
                    vec![5, 6, 7, 8],
                    Some("CPU"),
                    None,
                )),
            ],
            data_parallel_rank: Some(1),
        };
        apply_batch(&idx, &SourceId::from("w0"), 0, 0, &batch);

        let pool_locals: Vec<Arc<str>> = local_hashes(&[1, 2, 3, 4], 4, None)
            .iter()
            .map(|b| Arc::from(hex(b)))
            .collect();
        let pool_res = idx.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: pool_locals,
            tiers_of_interest: vec![StorageTier::External],
        });
        assert_eq!(pool_res.len(), 1);
        assert_eq!(pool_res[0].tier, StorageTier::External);
        assert_eq!(pool_res[0].locality, Locality::Remote);
        assert_eq!(&*pool_res[0].target.instance_id, "w0");
        assert_eq!(pool_res[0].target.dp_rank, 1);

        let wkr_locals: Vec<Arc<str>> = local_hashes(&[5, 6, 7, 8], 4, None)
            .iter()
            .map(|b| Arc::from(hex(b)))
            .collect();
        let wkr_res = idx.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: wkr_locals,
            tiers_of_interest: vec![StorageTier::HostPinned],
        });
        assert_eq!(wkr_res.len(), 1);
        assert_eq!(wkr_res[0].tier, StorageTier::HostPinned);
        assert_eq!(wkr_res[0].locality, Locality::Local);
        assert_eq!(&*wkr_res[0].target.instance_id, "w0");
        assert_eq!(wkr_res[0].target.dp_rank, 1);
    }

    #[test]
    fn read_seq_big_endian_signed() {
        assert_eq!(read_seq(&(-1i64).to_be_bytes()), Some(-1));
        assert_eq!(read_seq(&42i64.to_be_bytes()), Some(42));
    }

    #[test]
    fn read_seq_requires_exactly_eight_bytes() {
        assert_eq!(read_seq(&[0u8, 1]), None);
        assert_eq!(read_seq(&[0u8; 9]), None);
    }

    #[test]
    fn hex_encodes_bytes() {
        assert_eq!(hex(&[0u8, 255]), "00ff");
    }

    #[test]
    fn hash_str_int_and_bytes() {
        assert_eq!(&*hash_str(&ExternalBlockHash::Int(7)), "7");
        assert_eq!(
            &*hash_str(&ExternalBlockHash::Bytes(vec![0xab, 0xcd])),
            "abcd"
        );
    }

    #[test]
    fn local_hashes_chunks_by_block_size() {
        let ids = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let hashes = local_hashes(&ids, 2, None);
        assert_eq!(hashes.len(), 4);
        assert_eq!(hashes[0].len(), 32);
    }

    #[test]
    fn local_hashes_trailing_partial_block_included() {
        let ids = vec![1u32, 2, 3, 4, 5];
        let hashes = local_hashes(&ids, 2, None);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn local_hashes_block_size_zero_yields_empty() {
        assert!(local_hashes(&[1u32, 2, 3], 0, None).is_empty());
    }

    #[test]
    fn local_hashes_determinism() {
        let ids = vec![10u32, 20, 30, 40];
        assert_eq!(local_hashes(&ids, 2, None), local_hashes(&ids, 2, None));
    }

    #[test]
    fn local_hashes_seed_distinguishes_lora() {
        let ids = vec![1u32, 2, 3, 4];
        let base = local_hashes(&ids, 2, None);
        let lora = local_hashes(&ids, 2, Some("adapter-a"));
        assert_ne!(base, lora);
    }

    #[tokio::test]
    async fn emit_drops_when_no_receiver() {
        let (tx, mut rx) = mpsc::channel::<IngestionSignal>(1);
        emit(
            &Some(tx),
            IngestionSignal::Advance {
                source: SourceId::from("s"),
                last_seq: 1,
            },
        );
        // Drain the one buffered, then the next send must drop (channel full/closed).
        assert!(rx.try_recv().is_ok());
        drop(rx);
        emit(
            &None,
            IngestionSignal::Advance {
                source: SourceId::from("s"),
                last_seq: 2,
            },
        );
    }
}

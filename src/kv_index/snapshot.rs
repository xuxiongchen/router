//! Bounded source-local control acquisition and event consumption.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::binding::SourceLease;
use super::subscriber::apply_batch;
use super::wire::ExternalBlockHash;
use super::{BlockRemoved, KVEvent, KVEventBatch, KvBlockIndexer, SourceId};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct Identity {
    pub cache_id: String,
    pub stream_id: String,
    pub source_id: String,
    pub rank: u32,
    pub namespace: String,
    pub profile: String,
    pub block_size: u32,
    pub lookahead: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Descriptor {
    pub payload: Vec<u8>,
    pub resident: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Record {
    pub seq: i64,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Transcript {
    pub status: String,
    pub schema: u32,
    pub capability: String,
    pub snapshot_required: bool,
    pub identity: Identity,
    pub snapshot_id: String,
    pub watermark: i64,
    pub descriptors: Vec<Descriptor>,
    pub descriptor_count: usize,
    pub digest: String,
    pub lease_id: String,
    pub high_watermark: i64,
    pub retained_from: i64,
    pub replay: Vec<Record>,
}

/// The original capture is producer evidence, not a still-live remote lease.
/// Each proof has a fresh, finite local budget and an actual registration lease.
pub(crate) struct Attempt {
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    lease: SourceLease,
    attempt: u64,
    expected: Identity,
    index_rank: u32,
    deadline: Instant,
}

impl Attempt {
    pub(super) fn new(
        indexer: Arc<KvBlockIndexer>,
        source: SourceId,
        lease: SourceLease,
        attempt: u64,
        expected: Identity,
    ) -> Self {
        Self {
            indexer,
            source,
            lease,
            attempt,
            index_rank: expected.rank,
            expected,
            deadline: Instant::now() + Duration::from_secs(30),
        }
    }

    pub(crate) fn install(&self, transcript: &Transcript) -> Result<bool, &'static str> {
        self.install_before_publish(transcript, || {})
    }

    pub(crate) fn install_before_publish(
        &self,
        transcript: &Transcript,
        before_publish: impl FnOnce(),
    ) -> Result<bool, &'static str> {
        let (mut stores, mut structural_removals, mut suffix) =
            validate(transcript, &self.expected)?;
        // Wire rank is the global Engine rank; the independently addressable
        // DP1 HTTP target has local rank zero. Validate first, then translate
        // only the index target, never the remote identity or journal cursor.
        for batch in stores
            .iter_mut()
            .chain(&mut structural_removals)
            .chain(&mut suffix)
        {
            batch.data_parallel_rank = Some(self.index_rank);
        }
        if Instant::now() >= self.deadline {
            return Err("expired");
        }
        // Same lock order as supervisor promotion/retirement: lease, then index.
        Ok(self
            .lease
            .with_active(|_| {
                if Instant::now() >= self.deadline {
                    return false;
                }
                self.indexer.install_snapshot_if_current(
                    &self.source,
                    self.index_rank,
                    self.attempt,
                    || {
                        if Instant::now() >= self.deadline {
                            return false;
                        }
                        for batch in stores.iter().chain(&structural_removals).chain(&suffix) {
                            apply_batch(
                                &self.indexer,
                                &self.source,
                                self.index_rank,
                                0,
                                batch,
                                Some(self.expected.block_size),
                            )
                            .expect("whole plan was admitted before mutation");
                        }
                        before_publish();
                        Instant::now() < self.deadline
                    },
                )
            })
            .unwrap_or(false))
    }
}

fn batch(payload: &[u8], expected: &Identity) -> Result<KVEventBatch, &'static str> {
    let batch = super::decode_batch(payload).map_err(|_| "invalid wire")?;
    if batch.data_parallel_rank != Some(expected.rank) {
        return Err("rank");
    }
    super::admit::validate_full_attention_batch(&batch, expected.block_size)
        .map_err(|_| "unsupported event")?;
    for event in &batch.events {
        let (group, medium, ownership) = match event {
            KVEvent::BlockStored(store) => {
                if store.lora_id.is_some() || store.lora_name.is_some() {
                    return Err("unsupported adapter");
                }
                (store.group_idx, store.medium.as_deref(), &store.ownership)
            }
            KVEvent::BlockRemoved(remove) => (
                remove.group_idx,
                remove.medium.as_deref(),
                &remove.ownership,
            ),
            KVEvent::AllBlocksCleared(_) => continue,
        };
        if group.unwrap_or(0) != 0
            || medium.is_some_and(|medium| !matches!(medium, "GPU" | "DEVICE"))
            || ownership.is_some()
        {
            return Err("unsupported scope");
        }
    }
    Ok(batch)
}

type Plan = (Vec<KVEventBatch>, Vec<KVEventBatch>, Vec<KVEventBatch>);

#[derive(Default)]
struct DescriptionHistory {
    blocks: HashMap<ExternalBlockHash, (Option<ExternalBlockHash>, Vec<u32>)>,
    token_bytes: usize,
}

impl DescriptionHistory {
    fn check(&mut self, batch: &KVEventBatch, expected: &Identity) -> Result<(), &'static str> {
        for event in &batch.events {
            match event {
                KVEvent::AllBlocksCleared(_) => {
                    self.blocks.clear();
                    self.token_bytes = 0;
                }
                KVEvent::BlockRemoved(_) => {}
                KVEvent::BlockStored(store) => {
                    let width = expected.block_size as usize;
                    if store.token_ids.len() != width * store.block_hashes.len() {
                        return Err("partial descriptor");
                    }
                    let mut parent = store.parent_block_hash.clone();
                    for (hash, tokens) in
                        store.block_hashes.iter().zip(store.token_ids.chunks(width))
                    {
                        if parent
                            .as_ref()
                            .is_some_and(|p| !self.blocks.contains_key(p))
                        {
                            return Err("unknown parent");
                        }
                        let value = (parent, tokens.to_vec());
                        if let Some(previous) = self.blocks.get(hash) {
                            if previous != &value {
                                return Err("conflicting descriptor");
                            }
                        } else {
                            if self.blocks.len() >= 4096
                                || self.token_bytes + tokens.len() * 4 > 1024 * 1024
                            {
                                return Err("descriptor capacity");
                            }
                            self.token_bytes += tokens.len() * 4;
                            self.blocks.insert(hash.clone(), value);
                        }
                        parent = Some(hash.clone());
                    }
                }
            }
        }
        Ok(())
    }
}

pub(super) fn validate(transcript: &Transcript, expected: &Identity) -> Result<Plan, &'static str> {
    if transcript.status != "complete"
        || transcript.schema != 1
        || transcript.capability != "kv-snapshot-v1"
        || !transcript.snapshot_required
    {
        return Err("unsupported capability");
    }
    if &transcript.identity != expected
        || expected.cache_id.is_empty()
        || expected.stream_id.is_empty()
        || expected.source_id.is_empty()
        || expected.namespace.is_empty()
        || expected.profile != "local-full-attention-v1"
        || expected.lookahead != 0
        || expected.block_size == 0
    {
        return Err("identity/profile");
    }
    if transcript.snapshot_id.is_empty()
        || transcript.lease_id.is_empty()
        || transcript.watermark < -1
        || transcript.high_watermark < transcript.watermark
        || transcript.high_watermark == i64::MAX
        || transcript.retained_from > transcript.watermark + 1
        || transcript.descriptors.len() > 4096
        || transcript.replay.len() > 4096
        || transcript.descriptor_count != transcript.descriptors.len()
        || transcript.high_watermark - transcript.watermark != transcript.replay.len() as i64
    {
        return Err("incomplete/range");
    }
    let bytes: usize = transcript
        .descriptors
        .iter()
        .map(|entry| entry.payload.len())
        .chain(transcript.replay.iter().map(|entry| entry.payload.len()))
        .sum();
    if bytes > 1024 * 1024 {
        return Err("capacity");
    }
    let mut digest = Sha256::new();
    let mut seen = DescriptionHistory::default();
    let mut stores = Vec::new();
    let mut removals = Vec::new();
    for descriptor in &transcript.descriptors {
        digest.update([u8::from(descriptor.resident)]);
        digest.update((descriptor.payload.len() as u64).to_be_bytes());
        digest.update(&descriptor.payload);
        let decoded = batch(&descriptor.payload, expected)?;
        let [KVEvent::BlockStored(stored)] = decoded.events.as_slice() else {
            return Err("descriptor event");
        };
        if stored.block_hashes.len() != 1
            || stored.token_ids.len() != expected.block_size as usize
            || stored.group_idx.unwrap_or(0) != 0
            || stored.lora_name.is_some()
            || stored.lora_id.is_some()
            || seen.blocks.contains_key(&stored.block_hashes[0])
        {
            return Err("descriptor ancestry/profile");
        }
        if !descriptor.resident {
            removals.push(KVEventBatch {
                ts: decoded.ts,
                data_parallel_rank: Some(expected.rank),
                events: vec![KVEvent::BlockRemoved(BlockRemoved {
                    block_hashes: stored.block_hashes.clone(),
                    medium: stored.medium.clone(),
                    group_idx: stored.group_idx,
                    locality: stored.locality.clone(),
                    ownership: stored.ownership.clone(),
                })],
            });
        }
        seen.check(&decoded, expected)?;
        stores.push(decoded);
    }
    let actual_digest = format!("{:x}", digest.finalize());
    if actual_digest != transcript.digest {
        return Err("digest");
    }
    let mut suffix = Vec::new();
    for (offset, record) in transcript.replay.iter().enumerate() {
        if record.seq != transcript.watermark + 1 + offset as i64 {
            return Err("gap/conflict");
        }
        let decoded = batch(&record.payload, expected)?;
        seen.check(&decoded, expected)?;
        suffix.push(decoded);
    }
    Ok((stores, removals, suffix))
}

fn validate_open(transcript: &Transcript, expected: &Identity) -> Result<(), &'static str> {
    if transcript.high_watermark != transcript.watermark || !transcript.replay.is_empty() {
        return Err("open cut");
    }
    validate(transcript, expected).map(|_| ())
}

#[derive(Clone)]
pub(super) struct Enrollment {
    pub config: crate::config::types::KvSnapshotSource,
    control_key: reqwest::header::HeaderValue,
    pub bound_source: Option<String>,
}

impl Enrollment {
    pub(super) fn load(config: crate::config::types::KvSnapshotSource) -> Result<Self, String> {
        use std::io::Read;
        if config.namespace.is_empty() || config.namespace.len() > 256 || config.block_size == 0 {
            return Err("invalid snapshot enrollment".into());
        }
        let mut key = String::new();
        std::fs::File::open(&config.control_key_file)
            .and_then(|file| file.take(4097).read_to_string(&mut key))
            .map_err(|_| "cannot read snapshot control credential")?;
        if key.len() > 4096 || key.trim().is_empty() {
            return Err("invalid snapshot control credential".into());
        }
        let mut control_key = reqwest::header::HeaderValue::from_str(key.trim())
            .map_err(|_| "invalid snapshot control credential")?;
        control_key.set_sensitive(true);
        Ok(Self {
            config,
            control_key,
            bound_source: None,
        })
    }
}

#[derive(Deserialize)]
struct Capabilities {
    status: String,
    schema: u32,
    capability: String,
    identity: Identity,
    limits: Limits,
}

fn validate_capabilities(caps: &Capabilities, enrollment: &Enrollment) -> Result<(), &'static str> {
    let expected = &caps.identity;
    if caps.status != "complete"
        || caps.schema != 1
        || caps.capability != "kv-snapshot-v1"
        || expected.rank != enrollment.config.rank
        || expected.namespace != enrollment.config.namespace
        || expected.block_size != enrollment.config.block_size
        || expected.profile != "local-full-attention-v1"
        || expected.lookahead != 0
        || expected.source_id.is_empty()
        || expected.cache_id.is_empty()
        || expected.stream_id.is_empty()
        || !caps.limits.lease_seconds.is_finite()
        || !(2.0..=60.0).contains(&caps.limits.lease_seconds)
    {
        return Err("capability identity");
    }
    Ok(())
}

pub(super) async fn discover_identity(
    worker: &str,
    enrollment: Enrollment,
    api_key: Option<String>,
) -> Result<Identity, &'static str> {
    let control = Control::new(worker, enrollment, api_key)?;
    let caps: Capabilities = control.call("capabilities", None, None, None).await?;
    validate_capabilities(&caps, &control.enrollment)?;
    Ok(caps.identity)
}

#[derive(Deserialize)]
struct Limits {
    lease_seconds: f64,
}

#[derive(Deserialize)]
struct Range {
    status: String,
    identity: Identity,
    lease_id: String,
    high_watermark: i64,
    retained_from: i64,
    replay: Vec<Record>,
}

struct Control {
    client: reqwest::Client,
    url: reqwest::Url,
    enrollment: Enrollment,
    api_key: Option<String>,
}

impl Control {
    fn new(
        worker: &str,
        enrollment: Enrollment,
        api_key: Option<String>,
    ) -> Result<Self, &'static str> {
        let mut url = control_origin(worker)?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err("worker origin");
        }
        // Resolve only below this enrolled origin. Never accept a control URL
        // from metadata, or follow redirects carrying either credential.
        url.set_path(&format!(
            "{}/v1/kv_snapshot",
            url.path().trim_end_matches('/')
        ));
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|_| "control client")?;
        Ok(Self {
            client,
            url,
            enrollment,
            api_key,
        })
    }

    async fn call<T: serde::de::DeserializeOwned>(
        &self,
        operation: &str,
        identity: Option<&Identity>,
        lease: Option<&str>,
        after: Option<i64>,
    ) -> Result<T, &'static str> {
        let request = serde_json::json!({
            "operation": operation, "rank": self.enrollment.config.rank,
            "namespace": self.enrollment.config.namespace,
            "identity": identity, "lease_id": lease, "after": after,
        });
        let mut call = self
            .client
            .post(self.url.clone())
            .header("X-KV-Snapshot-Control", self.enrollment.control_key.clone())
            .json(&request);
        if let Some(key) = &self.api_key {
            call = call.bearer_auth(key);
        }
        let mut response = call.send().await.map_err(|_| "control transport")?;
        if !response.status().is_success() {
            return Err("control rejected");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| "control body")? {
            if bytes.len() + chunk.len() > 8 * 1024 * 1024 {
                return Err("control size");
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(|_| "control schema")
    }
}

pub(super) fn control_origin(worker: &str) -> Result<reqwest::Url, &'static str> {
    let url = reqwest::Url::parse(worker).map_err(|_| "worker URL")?;
    let loopback = match url.host() {
        Some(url::Host::Ipv4(address)) => address.is_loopback(),
        Some(url::Host::Ipv6(address)) => address.is_loopback(),
        _ => false,
    };
    if url.scheme() != "https" && !(url.scheme() == "http" && loopback) {
        return Err("snapshot control requires verified HTTPS or literal loopback HTTP");
    }
    Ok(url)
}

struct OwnedAttempt {
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    rank: u32,
    lease: SourceLease,
    attempt: u64,
}

impl Drop for OwnedAttempt {
    fn drop(&mut self) {
        self.lease.with_active(|_| {
            self.indexer
                .fail_snapshot_if_current(&self.source, self.rank, self.attempt);
        });
    }
}

fn range_batches(
    range: &Range,
    identity: &Identity,
    lease: &str,
    after: i64,
    history: &mut DescriptionHistory,
) -> Result<Vec<KVEventBatch>, &'static str> {
    if range.status != "complete"
        || &range.identity != identity
        || range.lease_id != lease
        || range.high_watermark < after
        || range.high_watermark == i64::MAX
        || range.retained_from > after + 1
        || range.replay.len() > 4096
        || range.high_watermark - after != range.replay.len() as i64
        || range.replay.iter().map(|r| r.payload.len()).sum::<usize>() > 1024 * 1024
    {
        return Err("control range");
    }
    let mut batches = Vec::with_capacity(range.replay.len());
    for (offset, record) in range.replay.iter().enumerate() {
        if record.seq != after + 1 + offset as i64 {
            return Err("control sequence");
        }
        let decoded = batch(&record.payload, identity)?;
        history.check(&decoded, identity)?;
        batches.push(decoded);
    }
    Ok(batches)
}

pub(super) fn spawn(
    worker: String,
    enrollment: Enrollment,
    api_key: Option<String>,
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    lease: SourceLease,
    signals: tokio::sync::mpsc::Sender<super::IngestionSignal>,
) -> super::subscriber::SubscriberHandle {
    let (shutdown_tx, mut shutdown) = tokio::sync::broadcast::channel(1);
    tokio::spawn(async move {
        let mut consecutive_failures = 0;
        loop {
            let rank = 0; // one independently addressable HTTP engine target
                          // Establish the existing source fence before any await.
            let Some((_, attempt)) =
                lease.with_active(|_| indexer.begin_replay(&source, rank, -1, true))
            else {
                return;
            };
            let owned = OwnedAttempt {
                indexer: indexer.clone(),
                source: source.clone(),
                rank,
                lease: lease.clone(),
                attempt,
            };
            let Ok(control) = Control::new(&worker, enrollment.clone(), api_key.clone()) else {
                return;
            };
            // Client-selected IDs make a canceled/lost open response cleanable without
            // retrying an operation whose side effects are unknown.
            let remote_lease = uuid::Uuid::new_v4().to_string();
            let mut identity = None;
            let mut made_progress = false;
            let run = async {
                let caps: Capabilities = control.call("capabilities", None, None, None).await?;
                validate_capabilities(&caps, &control.enrollment)?;
                let expected = &caps.identity;
                if control.enrollment.bound_source.as_deref() != Some(expected.source_id.as_str()) {
                    return Err("source replacement requires rediscovery");
                }
                identity = Some(expected.clone());
                let ttl = Duration::from_secs_f64(caps.limits.lease_seconds);
                let mut snapshot: Transcript = control
                    .call("open", Some(expected), Some(&remote_lease), None)
                    .await?;
                if snapshot.lease_id != remote_lease {
                    return Err("open lease");
                }
                validate_open(&snapshot, expected)?;
                let mut history = DescriptionHistory::default();
                for d in &snapshot.descriptors {
                    history.check(&batch(&d.payload, expected)?, expected)?;
                }
                let catchup: Range = control
                    .call("catch_up", Some(expected), Some(&remote_lease), None)
                    .await?;
                range_batches(
                    &catchup,
                    expected,
                    &remote_lease,
                    snapshot.watermark,
                    &mut history,
                )?;
                // This is a real same-journal live poll, not a legacy ZMQ sequence.
                // Already-received post-H Remove/Clear is folded before first visibility.
                let live_start = tokio::time::Instant::now();
                let first_live: Range = control
                    .call(
                        "live",
                        Some(expected),
                        Some(&remote_lease),
                        Some(catchup.high_watermark),
                    )
                    .await?;
                let mut expires = live_start + ttl;
                range_batches(
                    &first_live,
                    expected,
                    &remote_lease,
                    catchup.high_watermark,
                    &mut history,
                )?;
                snapshot.replay = catchup.replay;
                snapshot.replay.extend(first_live.replay);
                snapshot.high_watermark = first_live.high_watermark;
                let mut install = Attempt::new(
                    owned.indexer.clone(),
                    owned.source.clone(),
                    owned.lease.clone(),
                    attempt,
                    expected.clone(),
                );
                install.index_rank = rank;
                if tokio::time::Instant::now() >= expires {
                    return Err("lease expired");
                }
                install.deadline =
                    Instant::now() + expires.saturating_duration_since(tokio::time::Instant::now());
                if !install.install(&snapshot)? {
                    return Err("stale install");
                }
                let mut cursor = snapshot.high_watermark;
                let period = Duration::from_secs_f64((caps.limits.lease_seconds / 4.0).min(0.25));
                loop {
                    let _ = signals.try_send(super::IngestionSignal::Advance {
                        source: owned.source.clone(),
                        last_seq: cursor,
                    });
                    let started = tokio::time::Instant::now();
                    let range: Range = tokio::time::timeout_at(expires, async {
                        tokio::time::sleep(period).await;
                        control
                            .call("live", Some(expected), Some(&remote_lease), Some(cursor))
                            .await
                    })
                    .await
                    .map_err(|_| "lease expired")??;
                    expires = started + ttl;
                    let mut batches =
                        range_batches(&range, expected, &remote_lease, cursor, &mut history)?;
                    for batch in &mut batches {
                        batch.data_parallel_rank = Some(rank);
                    }
                    if tokio::time::Instant::now() >= expires {
                        return Err("lease expired");
                    }
                    let applied = owned
                        .lease
                        .with_active(|_| {
                            owned.indexer.with_current_snapshot(
                                &owned.source,
                                rank,
                                attempt,
                                Instant::now()
                                    + expires
                                        .saturating_duration_since(tokio::time::Instant::now()),
                                || {
                                    for batch in &batches {
                                        apply_batch(
                                            &owned.indexer,
                                            &owned.source,
                                            rank,
                                            0,
                                            batch,
                                            Some(expected.block_size),
                                        )
                                        .expect("range preflight precedes all mutation");
                                    }
                                },
                            )
                        })
                        .flatten()
                        .is_some();
                    if !applied {
                        return Err("stale live");
                    }
                    cursor = range.high_watermark;
                    made_progress = true; // complete, fenced post-publication poll/ACK cycle
                }
            };
            let mut canceled = false;
            let result: Result<(), &str> = tokio::select! {
                result = run => result,
                _ = shutdown.recv() => { canceled = true; Err("shutdown") },
            };
            // Revoke immediately; do not leave credit visible during network cleanup.
            drop(owned);
            if let Some(identity) = identity {
                let _: Result<serde_json::Value, _> = control
                    .call("release", Some(&identity), Some(&remote_lease), None)
                    .await;
            }
            if let Err(reason) = result {
                tracing::debug!(reason, "KV snapshot source stopped");
            }
            if canceled || !lease.is_active() {
                return;
            }
            // A bounded semantic history is renewed by a new snapshot, never by
            // forgetting hashes while preserving old credit. Healthy ACK progress
            // resets the failure budget; impossible initial snapshots still stop.
            consecutive_failures = next_failure_count(
                consecutive_failures,
                made_progress,
                result == Err("descriptor capacity"),
            );
            if consecutive_failures >= 3 {
                return;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                _ = shutdown.recv() => return,
            }
        }
    });
    super::subscriber::SubscriberHandle { shutdown_tx }
}

fn next_failure_count(previous: u8, progressed: bool, history_pressure: bool) -> u8 {
    if progressed && history_pressure {
        0
    } else if progressed {
        1
    } else {
        previous.saturating_add(1)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::future::IntoFuture;

    #[test]
    fn snapshot_resnapshot_budget_counts_failures_not_healthy_lifetime() {
        assert_eq!(next_failure_count(2, false, true), 3); // cannot open oversized profile
        assert_eq!(next_failure_count(2, false, false), 3);
        assert_eq!(next_failure_count(2, true, false), 1);
        assert_eq!(next_failure_count(2, true, true), 0); // planned bounded re-snapshot
    }

    #[test]
    fn snapshot_control_transport_requires_confidential_origin() {
        for (worker, expected) in [
            ("https://worker.example:8443", true),
            ("http://127.0.0.1:8000", true),
            ("http://[::1]:8000", true),
            ("http://10.0.0.2:8000", false),
            ("http://worker.example", false),
            ("http://localhost:8000", false),
        ] {
            assert_eq!(control_origin(worker).is_ok(), expected, "{worker}");
        }
    }

    #[test]
    fn snapshot_description_identity_and_clear_boundary() {
        let t = super::super::snapshot_proof::fixture("regular_snapshot.json");
        let mut history = DescriptionHistory::default();
        let first = batch(&t.descriptors[0].payload, &t.identity).unwrap();
        history.check(&first, &t.identity).unwrap();
        history.check(&first, &t.identity).unwrap(); // same logical frame is harmless
        let mut conflicting = first.clone();
        let KVEvent::BlockStored(store) = &mut conflicting.events[0] else {
            unreachable!()
        };
        store.token_ids[0] += 1;
        assert_eq!(
            history.check(&conflicting, &t.identity),
            Err("conflicting descriptor")
        );
        let mut reparented = first.clone();
        let KVEvent::BlockStored(store) = &mut reparented.events[0] else {
            unreachable!()
        };
        store.parent_block_hash = Some(store.block_hashes[0].clone());
        assert_eq!(
            history.check(&reparented, &t.identity),
            Err("conflicting descriptor")
        );
        let mut clear = first.clone();
        clear.events = vec![KVEvent::AllBlocksCleared(Default::default())];
        history.check(&clear, &t.identity).unwrap();
        history.check(&conflicting, &t.identity).unwrap();
        assert_eq!(history.blocks.len(), 1);
    }

    #[test]
    fn snapshot_live_range_requires_bound_identity_and_continuity() {
        let t = super::super::snapshot_proof::fixture("regular_snapshot.json");
        for case in [
            "valid",
            "cache",
            "stream",
            "rank",
            "namespace",
            "lease",
            "gap",
            "truncated",
        ] {
            let mut r = Range {
                status: "complete".into(),
                identity: t.identity.clone(),
                lease_id: "own".into(),
                high_watermark: 1,
                retained_from: 1,
                replay: vec![Record {
                    seq: 1,
                    payload: t.descriptors[0].payload.clone(),
                }],
            };
            match case {
                "cache" => r.identity.cache_id.push('x'),
                "stream" => r.identity.stream_id.push('x'),
                "rank" => r.identity.rank += 1,
                "namespace" => r.identity.namespace.push('x'),
                "lease" => r.lease_id.push('x'),
                "gap" => r.replay[0].seq += 1,
                "truncated" => {
                    r.replay.clear();
                }
                _ => {}
            }
            let result = range_batches(
                &r,
                &t.identity,
                "own",
                0,
                &mut DescriptionHistory::default(),
            );
            assert_eq!(result.is_ok(), case == "valid", "{case}");
        }
    }

    #[test]
    fn snapshot_live_and_cleanup_keep_attempt_fence() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let indexer = KvBlockIndexer::new();
        let source = SourceId::from("worker");
        indexer.register_source(&source, 0);
        let (_, first) = indexer.begin_replay(&source, 0, -1, true);
        assert!(indexer.install_snapshot_if_current(&source, 0, first, || true));
        assert_eq!(
            indexer.with_current_snapshot(&source, 0, first, deadline, || 7),
            Some(7)
        );
        let (_, next) = indexer.begin_replay(&source, 0, -1, true);
        assert!(indexer
            .with_current_snapshot(&source, 0, first, deadline, || 8)
            .is_none());
        indexer.fail_snapshot_if_current(&source, 0, first);
        assert!(indexer.install_snapshot_if_current(&source, 0, next, || true));
        assert_eq!(
            indexer.with_current_snapshot(&source, 0, next, deadline, || 9),
            Some(9)
        );
        assert!(indexer
            .with_current_snapshot(&source, 0, next, Instant::now(), || 10)
            .is_none());
    }

    #[test]
    fn snapshot_open_is_validated_before_catch_up() {
        let mut valid = super::super::snapshot_proof::fixture("regular_snapshot.json");
        valid.replay.clear();
        valid.high_watermark = valid.watermark;
        assert!(validate_open(&valid, &valid.identity).is_ok());
        for case in ["high", "retained", "identity", "status", "digest", "replay"] {
            let mut changed = valid.clone();
            match case {
                "high" => changed.high_watermark += 1,
                "retained" => changed.retained_from = changed.watermark + 2,
                "identity" => changed.identity.cache_id.push('x'),
                "status" => changed.status.clear(),
                "digest" => changed.digest.clear(),
                "replay" => changed.replay.push(Record {
                    seq: valid.watermark + 1,
                    payload: vec![],
                }),
                _ => {}
            }
            assert!(validate_open(&changed, &valid.identity).is_err(), "{case}");
        }
    }

    #[tokio::test]
    async fn snapshot_control_never_redirects_credentials() {
        use axum::{routing::post, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let leaked = Arc::new(AtomicUsize::new(0));
        let observed = leaked.clone();
        let target = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let redirect = format!("http://{}/v1/kv_snapshot", target.local_addr().unwrap());
        let target_task = tokio::spawn(
            axum::serve(
                target,
                Router::new().route(
                    "/v1/kv_snapshot",
                    post(move || async move {
                        observed.fetch_add(1, Ordering::SeqCst);
                        axum::Json(serde_json::json!({}))
                    }),
                ),
            )
            .into_future(),
        );
        let source = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker = format!("http://{}", source.local_addr().unwrap());
        let source_task = tokio::spawn(
            axum::serve(
                source,
                Router::new().route(
                    "/v1/kv_snapshot",
                    post(move |headers: axum::http::HeaderMap| async move {
                        assert_eq!(headers["x-kv-snapshot-control"], "control-secret");
                        assert_eq!(headers["authorization"], "Bearer inference-secret");
                        axum::response::Redirect::temporary(&redirect)
                    }),
                ),
            )
            .into_future(),
        );
        let enrollment = Enrollment {
            config: crate::config::types::KvSnapshotSource {
                namespace: "m".into(),
                rank: 0,
                block_size: 16,
                control_key_file: "unused".into(),
            },
            control_key: reqwest::header::HeaderValue::from_static("control-secret"),
            bound_source: None,
        };
        let client = Control::new(&worker, enrollment, Some("inference-secret".into())).unwrap();
        let result: Result<serde_json::Value, _> =
            client.call("capabilities", None, None, None).await;
        source_task.abort();
        target_task.abort();
        assert_eq!(result, Err("control rejected"));
        assert_eq!(leaked.load(Ordering::SeqCst), 0);
    }
}

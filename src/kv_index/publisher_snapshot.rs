//! Experimental #57789 interop: bounded reference normalization, not a second index.
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rmpv::Value;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::binding::SourceLease;
use super::snapshot::{Attempt, Enrollment, Identity};
use super::subscriber::{apply_batch, SubscriberHandle};
use super::wire::ExternalBlockHash;
use super::{BlockRemoved, BlockStored, KVEvent, KVEventBatch, KvBlockIndexer, SourceId};
use crate::config::types::{KvSnapshotSource, PublisherSnapshotSource};

const LIMIT: usize = 4096;
const BYTES: usize = 1024 * 1024;
const LIVE_TIMEOUT: Duration = Duration::from_secs(3);
const STRICT_REQUEST: &[u8] = b"snapshot-full-attention-v1";

pub(super) fn validate_config(config: &PublisherSnapshotSource) -> Result<(), String> {
    for endpoint in [&config.endpoint, &config.snapshot_endpoint] {
        let url = reqwest::Url::parse(endpoint).map_err(|_| "invalid publisher endpoint")?;
        // tcp is a non-special URL scheme: a literal IPv4 host may be exposed
        // as Domain, not Host::Ipv4. Parse the literal without DNS resolution.
        let local = url
            .host_str()
            .and_then(|host| {
                host.trim_start_matches('[')
                    .trim_end_matches(']')
                    .parse::<std::net::IpAddr>()
                    .ok()
            })
            .is_some_and(|ip| ip.is_loopback());
        if url.scheme() != "tcp"
            || !local
            || url.port().is_none_or(|p| p == 0)
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !matches!(url.path(), "" | "/")
        {
            return Err("experimental publisher endpoints require literal loopback TCP".into());
        }
    }
    if config.topic.len() > 256 {
        return Err("publisher topic too large".into());
    }
    Ok(())
}

#[derive(Deserialize)]
struct Proof {
    protocol: String,
    reporting: String,
    rank: u32,
    block_size: u32,
    complete: bool,
    live_blocks: usize,
    live_references: usize,
}

struct Normalizer {
    descriptions: HashMap<ExternalBlockHash, BlockStored>,
    references: HashMap<ExternalBlockHash, usize>,
    bytes: usize,
    total: usize,
    block_size: u32,
    rank: u32,
}

impl Normalizer {
    fn new(config: &KvSnapshotSource) -> Self {
        Self {
            descriptions: HashMap::new(),
            references: HashMap::new(),
            bytes: 0,
            total: 0,
            block_size: config.block_size,
            rank: config.rank,
        }
    }

    fn apply(&mut self, batch: KVEventBatch, snapshot: bool) -> Result<KVEventBatch, &'static str> {
        if batch.data_parallel_rank != Some(self.rank) {
            return Err("publisher rank");
        }
        let mut changes = Vec::new();
        for event in batch.events {
            match event {
                KVEvent::AllBlocksCleared(clear) => {
                    self.descriptions.clear();
                    self.references.clear();
                    self.bytes = 0;
                    self.total = 0;
                    changes.push(KVEvent::AllBlocksCleared(clear));
                }
                KVEvent::BlockStored(store) => {
                    if store.medium.as_deref() != Some("GPU")
                        || store.group_idx != Some(0)
                        || store.ownership.is_some()
                        || store.locality.is_some()
                        || store.lora_id.is_some()
                        || store.lora_name.is_some()
                        || store.block_hashes.is_empty()
                        || store
                            .kv_cache_spec_kind
                            .as_deref()
                            .is_some_and(|kind| kind != "full_attention")
                        || store.kv_cache_spec_sliding_window.is_some()
                        || store.extra_keys.as_ref().is_some_and(|keys| {
                            keys.len() != store.block_hashes.len()
                                || keys.iter().any(|v| !v.is_nil())
                        })
                        || store.block_hashes.iter().collect::<HashSet<_>>().len()
                            != store.block_hashes.len()
                    {
                        return Err("unsupported publisher scope");
                    }
                    let teaching = store.block_size != 0;
                    if teaching && store.kv_cache_spec_kind.as_deref() != Some("full_attention") {
                        return Err("publisher full-attention declaration");
                    }
                    if !snapshot
                        && teaching
                        && store
                            .block_hashes
                            .iter()
                            .any(|hash| self.references.get(hash).is_some_and(|n| *n != 0))
                    {
                        // A full cache report is indistinguishable from another
                        // real reference. Refuse credit rather than inflate it.
                        return Err("publisher repeated full report");
                    }
                    if (teaching
                        && (store.block_size != self.block_size
                            || store.token_ids.len()
                                != store.block_hashes.len() * self.block_size as usize))
                        || (!teaching
                            && (!store.token_ids.is_empty() || store.parent_block_hash.is_some()))
                    {
                        return Err("publisher descriptor shape");
                    }
                    let mut parent = store.parent_block_hash.clone();
                    for (offset, hash) in store.block_hashes.iter().enumerate() {
                        if teaching {
                            if parent
                                .as_ref()
                                .is_some_and(|p| !self.descriptions.contains_key(p))
                            {
                                return Err("unknown publisher parent");
                            }
                            let width = self.block_size as usize;
                            let description = BlockStored {
                                block_hashes: vec![hash.clone()],
                                parent_block_hash: parent,
                                token_ids: store.token_ids[offset * width..(offset + 1) * width]
                                    .to_vec(),
                                block_size: store.block_size,
                                extra_keys: None,
                                medium: store.medium.clone(),
                                group_idx: store.group_idx,
                                locality: None,
                                ownership: None,
                                lora_id: None,
                                lora_name: None,
                                kv_cache_spec_kind: store.kv_cache_spec_kind.clone(),
                                kv_cache_spec_sliding_window: None,
                                session_id: store.session_id.clone(),
                            };
                            if let Some(old) = self.descriptions.get(hash) {
                                if old.parent_block_hash != description.parent_block_hash
                                    || old.token_ids != description.token_ids
                                {
                                    return Err("conflicting publisher descriptor");
                                }
                            } else {
                                self.bytes += description.token_ids.len() * 4;
                                if self.descriptions.len() >= LIMIT || self.bytes > BYTES {
                                    return Err("publisher history capacity");
                                }
                                self.descriptions.insert(hash.clone(), description);
                            }
                            parent = Some(hash.clone());
                        }
                        let description = self.descriptions.get(hash).ok_or("unknown reference")?;
                        let count = self.references.entry(hash.clone()).or_default();
                        if self.total >= LIMIT {
                            return Err("publisher reference capacity");
                        }
                        if *count == 0 {
                            changes.push(KVEvent::BlockStored(description.clone()));
                        }
                        *count += 1;
                        self.total += 1;
                    }
                }
                KVEvent::BlockRemoved(remove) => {
                    if remove.medium.as_deref() != Some("GPU")
                        || remove.group_idx != Some(0)
                        || remove.ownership.is_some()
                        || remove.locality.is_some()
                        || remove.block_hashes.is_empty()
                        || remove.block_hashes.iter().collect::<HashSet<_>>().len()
                            != remove.block_hashes.len()
                    {
                        return Err("unsupported publisher removal");
                    }
                    let mut removed = Vec::new();
                    for hash in &remove.block_hashes {
                        let count = self
                            .references
                            .get_mut(hash)
                            .ok_or("unmatched publisher removal")?;
                        if *count == 0 {
                            return Err("unmatched publisher removal");
                        }
                        *count -= 1;
                        self.total -= 1;
                        if *count == 0 {
                            removed.push(hash.clone());
                        }
                    }
                    if !removed.is_empty() {
                        changes.push(KVEvent::BlockRemoved(BlockRemoved {
                            block_hashes: removed,
                            ..remove
                        }));
                    }
                }
            }
        }
        let normalized = KVEventBatch {
            ts: batch.ts,
            data_parallel_rank: Some(0),
            events: changes,
        };
        super::admit::validate_full_attention_batch(&normalized, self.block_size)
            .map_err(|_| "normalized admission")?;
        Ok(normalized)
    }
}

// Only this opt-in protocol understands the fourth publisher UUID and empty
// reference Store. Generic decoding/admission remains strict and unchanged.
fn decode(
    payload: &[u8],
    identity: Option<&[u8]>,
) -> Result<(KVEventBatch, Option<Vec<u8>>), &'static str> {
    if payload.len() > BYTES {
        return Err("publisher payload capacity");
    }
    let mut syntax = rmp_serde::Deserializer::new(Cursor::new(payload));
    syntax.set_max_depth(16);
    serde::de::IgnoredAny::deserialize(&mut syntax).map_err(|_| "publisher syntax")?;
    let mut cursor = Cursor::new(payload);
    let Value::Array(mut fields) =
        rmpv::decode::read_value_with_max_depth(&mut cursor, 16).map_err(|_| "publisher wire")?
    else {
        return Err("publisher array");
    };
    if cursor.position() as usize != payload.len()
        || fields.len() != if identity.is_some() { 4 } else { 3 }
    {
        return Err("publisher envelope");
    }
    let publisher = if let Some(expected) = identity {
        let Value::Binary(actual) = fields.pop().ok_or("publisher identity")? else {
            return Err("publisher identity");
        };
        if actual.len() != 16 || actual != expected {
            return Err("publisher replacement");
        }
        Some(actual)
    } else {
        None
    };
    let Some(Value::Array(events)) = fields.get_mut(1) else {
        return Err("publisher events");
    };
    for event in events {
        let Value::Map(pairs) = event else {
            return Err("publisher event map");
        };
        if pairs
            .iter()
            .filter(|(key, _)| key.as_str() == Some("lookahead_token_ids"))
            .count()
            > 1
        {
            return Err("duplicate publisher lookahead");
        }
        if let Some((_, ahead)) = pairs
            .iter()
            .find(|(key, _)| key.as_str() == Some("lookahead_token_ids"))
        {
            if !ahead.is_nil() {
                return Err("publisher lookahead");
            }
        }
        pairs.retain(|(key, _)| key.as_str() != Some("lookahead_token_ids"));
    }
    let mut original_contract = Vec::new();
    rmpv::encode::write_value(&mut original_contract, &Value::Array(fields))
        .map_err(|_| "publisher wire")?;
    Ok((
        super::decode_batch(&original_contract).map_err(|_| "publisher event")?,
        publisher,
    ))
}

fn recv(socket: &zmq::Socket, deadline: Instant) -> Result<Vec<Vec<u8>>, &'static str> {
    while Instant::now() < deadline {
        if socket.poll(zmq::POLLIN, 50).map_err(|_| "publisher poll")? == 0 {
            continue;
        }
        let mut parts = Vec::new();
        let mut size = 0;
        loop {
            let frame = socket.recv_bytes(0).map_err(|_| "publisher receive")?;
            size += frame.len();
            if parts.len() >= LIMIT || size > BYTES {
                return Err("publisher multipart capacity");
            }
            parts.push(frame);
            if !socket.get_rcvmore().map_err(|_| "publisher receive")? {
                return Ok(parts);
            }
        }
    }
    Err("publisher deadline")
}

struct Connection {
    _context: zmq::Context,
    live: zmq::Socket,
    snapshot: zmq::Socket,
    publisher: Vec<u8>,
}

impl Connection {
    fn new(config: &PublisherSnapshotSource) -> Result<Self, &'static str> {
        let context = zmq::Context::new();
        let live = context.socket(zmq::SUB).map_err(|_| "publisher socket")?;
        live.set_linger(0).map_err(|_| "publisher socket")?;
        live.set_rcvhwm(LIMIT as i32)
            .map_err(|_| "publisher socket")?;
        live.set_maxmsgsize(BYTES as i64)
            .map_err(|_| "publisher socket")?;
        live.set_subscribe(config.topic.as_bytes())
            .map_err(|_| "publisher socket")?;
        live.connect(&config.endpoint)
            .map_err(|_| "publisher connect")?;
        let first = recv(&live, Instant::now() + LIVE_TIMEOUT)?;
        if first.len() != 3 || first[0] != config.topic.as_bytes() {
            return Err("publisher topic");
        }
        let value: Value = rmpv::decode::read_value_with_max_depth(&mut Cursor::new(&first[2]), 16)
            .map_err(|_| "publisher wire")?;
        let publisher = value
            .as_array()
            .and_then(|a| a.get(3))
            .and_then(|v| {
                if let Value::Binary(id) = v {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .ok_or("missing publisher identity")?;
        decode(&first[2], Some(&publisher))?;
        let snapshot = context.socket(zmq::REQ).map_err(|_| "publisher socket")?;
        snapshot.set_linger(0).map_err(|_| "publisher socket")?;
        snapshot
            .set_sndtimeo(1000)
            .map_err(|_| "publisher socket")?;
        snapshot
            .set_maxmsgsize(BYTES as i64)
            .map_err(|_| "publisher socket")?;
        snapshot
            .connect(&config.snapshot_endpoint)
            .map_err(|_| "snapshot connect")?;
        Ok(Self {
            _context: context,
            live,
            snapshot,
            publisher,
        })
    }

    fn cut(
        &self,
        config: &KvSnapshotSource,
    ) -> Result<(i64, Vec<KVEventBatch>, Normalizer), &'static str> {
        self.snapshot
            .send(STRICT_REQUEST, 0)
            .map_err(|_| "snapshot request")?;
        let parts = recv(&self.snapshot, Instant::now() + Duration::from_secs(2))?;
        stage(&parts, &self.publisher, config)
    }
}

fn stage(
    parts: &[Vec<u8>],
    publisher: &[u8],
    config: &KvSnapshotSource,
) -> Result<(i64, Vec<KVEventBatch>, Normalizer), &'static str> {
    if parts.len() < 3 || parts[0].len() != 8 || parts[1] != publisher {
        return Err("snapshot identity");
    }
    let watermark = i64::from_be_bytes(
        parts[0]
            .as_slice()
            .try_into()
            .map_err(|_| "snapshot sequence")?,
    );
    if watermark < -1 {
        return Err("snapshot unavailable");
    }
    let proof: Proof = serde_json::from_slice(&parts[2]).map_err(|_| "snapshot proof")?;
    if proof.protocol != "full-attention-v1"
        || proof.reporting != "incremental"
        || !proof.complete
        || proof.rank != config.rank
        || proof.block_size != config.block_size
        || proof.live_blocks > LIMIT
        || proof.live_references > LIMIT
    {
        return Err("snapshot incomplete profile");
    }
    let mut state = Normalizer::new(config);
    let mut plan = Vec::new();
    for bytes in &parts[3..] {
        plan.push(state.apply(decode(bytes, None)?.0, true)?);
    }
    if state.references.values().filter(|v| **v > 0).count() != proof.live_blocks
        || state.total != proof.live_references
    {
        return Err("snapshot inventory mismatch");
    }
    Ok((watermark, plan, state))
}
fn identity(publisher: &[u8], config: &KvSnapshotSource) -> Identity {
    let id = publisher
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Identity {
        source_id: id.clone(),
        cache_id: id.clone(),
        stream_id: id,
        rank: config.rank,
        namespace: config.namespace.clone(),
        profile: "local-full-attention-v1".into(),
        block_size: config.block_size,
        lookahead: 0,
    }
}

pub(super) async fn discover(
    protocol: PublisherSnapshotSource,
    config: KvSnapshotSource,
) -> Result<Identity, &'static str> {
    tokio::task::spawn_blocking(move || {
        let connection = Connection::new(&protocol)?;
        connection.cut(&config)?;
        Ok(identity(&connection.publisher, &config))
    })
    .await
    .map_err(|_| "publisher discovery task")?
}

struct Ingestion {
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    lease: SourceLease,
    attempt: u64,
}
impl Drop for Ingestion {
    fn drop(&mut self) {
        self.lease.with_active(|_| {
            self.indexer
                .fail_snapshot_if_current(&self.source, 0, self.attempt)
        });
    }
}

pub(super) fn spawn(
    enrollment: Enrollment,
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    lease: SourceLease,
) -> SubscriberHandle {
    let (shutdown_tx, mut shutdown) = tokio::sync::broadcast::channel(1);
    tokio::task::spawn_blocking(move || {
        let run = || -> Result<(), &'static str> {
            let (_, attempt) = lease
                .with_active(|_| indexer.begin_replay(&source, 0, -1, true))
                .ok_or("retired publisher")?;
            let owned = Ingestion {
                indexer,
                source,
                lease,
                attempt,
            };
            let config = &enrollment.config;
            let protocol = config
                .publisher_snapshot
                .as_ref()
                .ok_or("publisher config")?;
            let connection = Connection::new(protocol)?;
            let expected = identity(&connection.publisher, config);
            if enrollment.bound_source.as_deref() != Some(expected.source_id.as_str()) {
                return Err("publisher replacement");
            }
            let (mut cursor, mut plan, mut state) = connection.cut(config)?;
            // SUB was established first. Consume at least one post-W frame,
            // then the already-received finite suffix before publishing credit.
            let mut frames = 0;
            let mut bytes = 0;
            let deadline = Instant::now() + LIVE_TIMEOUT;
            let cut = cursor;
            let mut seen = HashMap::<i64, [u8; 32]>::new();
            loop {
                let live = recv(&connection.live, deadline)?;
                let (seq, batch) = live_batch(&live, &connection.publisher, protocol)?;
                bytes += live[2].len();
                frames += 1;
                if frames > LIMIT || bytes > BYTES {
                    return Err("publisher buffer capacity");
                }
                if seq <= cut {
                    continue;
                }
                if !sequence(&mut seen, seq, &mut cursor, &live[2])? {
                    continue;
                }
                plan.push(state.apply(batch, false)?);
                if connection
                    .live
                    .get_events()
                    .map_err(|_| "publisher poll")?
                    .contains(zmq::POLLIN)
                {
                    continue;
                }
                break;
            }
            let mut install = Attempt::new(
                owned.indexer.clone(),
                owned.source.clone(),
                owned.lease.clone(),
                attempt,
                expected,
            );
            install.index_rank = 0;
            if !install.install_batches(plan, || {})? {
                return Err("stale publisher install");
            }
            let mut expires = Instant::now() + LIVE_TIMEOUT;
            while shutdown.try_recv() == Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                && owned.lease.is_active()
            {
                let live = recv(&connection.live, expires)?;
                let (seq, batch) = live_batch(&live, &connection.publisher, protocol)?;
                if !sequence(&mut seen, seq, &mut cursor, &live[2])? {
                    continue;
                }
                let normalized = state.apply(batch, false)?;
                let applied = owned
                    .lease
                    .with_active(|_| {
                        owned.indexer.with_current_snapshot(
                            &owned.source,
                            0,
                            attempt,
                            Instant::now() + LIVE_TIMEOUT,
                            || {
                                apply_batch(
                                    &owned.indexer,
                                    &owned.source,
                                    0,
                                    0,
                                    &normalized,
                                    Some(config.block_size),
                                )
                                .expect("normalized batch preflight");
                            },
                        )
                    })
                    .flatten()
                    .is_some();
                if !applied {
                    return Err("stale publisher live");
                }
                expires = Instant::now() + LIVE_TIMEOUT;
            }
            Ok(())
        };
        if let Err(reason) = run() {
            tracing::debug!(reason, "publisher snapshot stopped");
        }
    });
    SubscriberHandle { shutdown_tx }
}

fn sequence(
    seen: &mut HashMap<i64, [u8; 32]>,
    seq: i64,
    cursor: &mut i64,
    payload: &[u8],
) -> Result<bool, &'static str> {
    let fingerprint: [u8; 32] = Sha256::digest(payload).into();
    if seq <= *cursor {
        return if seen.get(&seq) == Some(&fingerprint) {
            Ok(false)
        } else {
            Err("conflicting publisher sequence")
        };
    }
    if seq != *cursor + 1 {
        return Err("publisher gap");
    }
    *cursor = seq;
    seen.insert(seq, fingerprint);
    seen.remove(&(seq - LIMIT as i64));
    Ok(true)
}

fn live_batch(
    parts: &[Vec<u8>],
    identity: &[u8],
    config: &PublisherSnapshotSource,
) -> Result<(i64, KVEventBatch), &'static str> {
    if parts.len() != 3 || parts[0] != config.topic.as_bytes() || parts[1].len() != 8 {
        return Err("publisher frames");
    }
    let seq = i64::from_be_bytes(
        parts[1]
            .as_slice()
            .try_into()
            .map_err(|_| "publisher seq")?,
    );
    if seq < 0 || seq == i64::MAX {
        return Err("publisher seq");
    }
    Ok((seq, decode(&parts[2], Some(identity))?.0))
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    fn config() -> KvSnapshotSource {
        KvSnapshotSource {
            namespace: "actual-public-synthetic-b16".into(),
            rank: 0,
            block_size: 16,
            control_key_file: Default::default(),
            publisher_snapshot: None,
        }
    }

    fn snapshot(name: &str) -> Vec<Vec<u8>> {
        serde_json::from_str(match name {
            "one" => {
                include_str!("../../tests/rust/fixtures/kv_snapshot/publisher-57789-baseline.json")
            }
            "two" => include_str!(
                "../../tests/rust/fixtures/kv_snapshot/publisher-57789-two-references.json"
            ),
            "removed" => {
                include_str!("../../tests/rust/fixtures/kv_snapshot/publisher-57789-removed.json")
            }
            _ => panic!("unknown captured snapshot"),
        })
        .unwrap()
    }

    #[test]
    fn actual_publisher_teaching_references_do_not_evict_residency() {
        // Exact multipart bytes from ZmqEventPublisher/recorder/export, with
        // two public B16 token blocks; no synthetic ideal snapshot envelope.
        let config = config();
        for (name, refs) in [("one", 2), ("two", 4), ("removed", 0)] {
            let parts = snapshot(name);
            let (_, _, state) = stage(&parts, &parts[1], &config).unwrap();
            assert_eq!(state.total, refs, "{name}");
            assert_eq!(
                state.references.values().filter(|n| **n > 0).count(),
                if refs == 0 { 0 } else { 2 }
            );
        }
        let two = snapshot("two");
        let (_, _, mut state) = stage(&two, &two[1], &config).unwrap();
        let live: Vec<Vec<u8>> = serde_json::from_str(include_str!(
            "../../tests/rust/fixtures/kv_snapshot/publisher-57789-remove.json"
        ))
        .unwrap();
        let remove = decode(&live[2], Some(&two[1])).unwrap().0;
        assert!(state
            .apply(remove.clone(), false)
            .unwrap()
            .events
            .is_empty());
        assert_eq!(state.total, 2);
        let final_remove = state.apply(remove, false).unwrap();
        assert!(
            matches!(&final_remove.events[..], [KVEvent::BlockRemoved(event)] if event.block_hashes.len() == 2)
        );
        assert_eq!(state.total, 0);
    }

    #[test]
    fn actual_publisher_proof_and_live_identity_are_fail_closed() {
        let config = config();
        for (endpoint, accepted) in [
            ("tcp://127.0.0.1:5557", true),
            ("tcp://[::1]:5557", true),
            ("tcp://localhost:5557", false),
            ("tcp://192.0.2.1:5557", false),
        ] {
            let source = PublisherSnapshotSource {
                endpoint: endpoint.into(),
                snapshot_endpoint: endpoint.into(),
                topic: "kv".into(),
            };
            assert_eq!(validate_config(&source).is_ok(), accepted, "{endpoint}");
        }
        let parts = snapshot("one");
        for reason in ["missing", "full-report", "subset", "wrong-count"] {
            let mut bad = parts.clone();
            let mut proof: serde_json::Value = serde_json::from_slice(&bad[2]).unwrap();
            match reason {
                "missing" => {
                    bad.remove(2);
                }
                "full-report" => proof["reporting"] = "full".into(),
                "subset" => proof["complete"] = false.into(),
                "wrong-count" => proof["live_references"] = 3.into(),
                _ => unreachable!(),
            }
            if reason != "missing" {
                bad[2] = serde_json::to_vec(&proof).unwrap();
            }
            assert!(stage(&bad, &parts[1], &config).is_err(), "{reason}");
        }
        let seed: Vec<Vec<u8>> = serde_json::from_str(include_str!(
            "../../tests/rust/fixtures/kv_snapshot/publisher-57789-seed.json"
        ))
        .unwrap();
        for case in [
            "missing-group",
            "locality",
            "missing-layout",
            "duplicate-hash",
            "extra-count",
        ] {
            let mut batch = decode(&seed[2], Some(&parts[1])).unwrap().0;
            let KVEvent::BlockStored(store) = &mut batch.events[0] else {
                panic!("actual seed Store");
            };
            match case {
                "missing-group" => store.group_idx = None,
                "locality" => store.locality = Some("LOCAL".into()),
                "missing-layout" => store.kv_cache_spec_kind = None,
                "duplicate-hash" => store.block_hashes[1] = store.block_hashes[0].clone(),
                "extra-count" => store.extra_keys = Some(vec![Value::Nil]),
                _ => unreachable!(),
            }
            assert!(
                Normalizer::new(&config).apply(batch, false).is_err(),
                "{case}"
            );
        }
        let mut wire = rmpv::decode::read_value(&mut Cursor::new(&seed[2])).unwrap();
        let Value::Array(fields) = &mut wire else {
            unreachable!()
        };
        let Value::Array(events) = &mut fields[1] else {
            unreachable!()
        };
        let Value::Map(pairs) = &mut events[0] else {
            unreachable!()
        };
        pairs.retain(|(key, _)| key.as_str() != Some("lookahead_token_ids"));
        pairs.push((
            "lookahead_token_ids".into(),
            Value::Array(vec![Value::Array(vec![]), Value::Array(vec![])]),
        ));
        let mut altered = Vec::new();
        rmpv::encode::write_value(&mut altered, &wire).unwrap();
        assert!(decode(&altered, Some(&parts[1])).is_err());
        assert!(decode(&seed[2], Some(&[0; 16])).is_err());
        let (_, _, mut state) = stage(&parts, &parts[1], &config).unwrap();
        assert_eq!(
            state
                .apply(decode(&seed[2], Some(&parts[1])).unwrap().0, false)
                .unwrap_err(),
            "publisher repeated full report"
        );
        let mut cursor = 4;
        let mut seen = HashMap::new();
        assert!(sequence(&mut seen, 5, &mut cursor, &seed[2]).unwrap());
        assert!(!sequence(&mut seen, 5, &mut cursor, &seed[2]).unwrap());
        assert!(sequence(&mut seen, 5, &mut cursor, b"different").is_err());
        assert!(sequence(&mut seen, 7, &mut cursor, &seed[2]).is_err());
    }
}

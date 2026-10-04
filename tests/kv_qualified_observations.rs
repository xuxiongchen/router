//! CPU integration observations from the real supervisor/index and loopback ZMQ.
//! Fixtures establish their own static contract, not remote GPU capability proof.
use std::{sync::Arc, time::Duration};

use tokio::sync::mpsc;
use vllm_router_rs::kv_index::{
    subscriber::{local_hashes, EventSource},
    CacheKey, HashMode, IngestionSignal, KvIndexSupervisor, MatchQuery, QualifiedSourceCursor,
    StorageTier,
};

const TIMEOUT: Duration = Duration::from_secs(5);
const TOPIC: &[u8] = b"kv";
const URL: &str = "http://worker-a:8000";

struct Publisher {
    _context: zmq::Context,
    socket: zmq::Socket,
    endpoint: String,
}

impl Publisher {
    fn new() -> Self {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::XPUB).unwrap();
        socket.set_linger(0).unwrap();
        socket.set_xpub_verbose(true).unwrap();
        socket.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = socket.get_last_endpoint().unwrap().unwrap();
        Self {
            _context: context,
            socket,
            endpoint,
        }
    }

    fn source(&self) -> EventSource {
        EventSource {
            source: Arc::from("worker-a"),
            dp_rank: 0,
            pub_endpoint: self.endpoint.clone(),
            replay_endpoint: None,
            topic: "kv".into(),
            hwm: None,
        }
    }

    async fn subscribed(&self) {
        tokio::time::timeout(TIMEOUT, async {
            loop {
                if let Ok(frames) = self.socket.recv_multipart(zmq::DONTWAIT) {
                    if frames == vec![vec![1, b'k', b'v']] {
                        return;
                    }
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("XPUB subscription handshake");
    }

    fn send(&self, seq: i64, events: Vec<rmpv::Value>) {
        self.raw(TOPIC, &seq.to_be_bytes(), &batch(events));
    }

    fn raw(&self, topic: &[u8], sequence: &[u8], payload: &[u8]) {
        self.socket
            .send_multipart([topic, sequence, payload], 0)
            .unwrap();
    }
}

fn key() -> CacheKey {
    CacheKey {
        model: Arc::from("fixture-model"),
        hash_mode: HashMode::Sha256,
        block_size: 2,
    }
}

fn query(tokens: &[u32]) -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        local_hashes: local_hashes(tokens, 2, None)
            .iter()
            .map(|hash| {
                Arc::from(
                    hash.iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>(),
                )
            })
            .collect(),
        tiers_of_interest: vec![StorageTier::Device],
    }
}

fn stored() -> rmpv::Value {
    rmpv::Value::Map(vec![
        ("type".into(), "BlockStored".into()),
        (
            "block_hashes".into(),
            rmpv::Value::Array(vec![rmpv::Value::Binary(vec![7; 32])]),
        ),
        ("parent_block_hash".into(), rmpv::Value::Nil),
        (
            "token_ids".into(),
            rmpv::Value::Array(vec![1.into(), 2.into()]),
        ),
        ("block_size".into(), 2.into()),
        ("lora_id".into(), rmpv::Value::Nil),
        ("lora_name".into(), rmpv::Value::Nil),
        ("medium".into(), "GPU".into()),
    ])
}

fn removed() -> rmpv::Value {
    rmpv::Value::Map(vec![
        ("type".into(), "BlockRemoved".into()),
        (
            "block_hashes".into(),
            rmpv::Value::Array(vec![rmpv::Value::Binary(vec![7; 32])]),
        ),
        ("medium".into(), "GPU".into()),
    ])
}

fn clear() -> rmpv::Value {
    rmpv::Value::Map(vec![("type".into(), "AllBlocksCleared".into())])
}

fn batch(events: Vec<rmpv::Value>) -> Vec<u8> {
    let mut bytes = Vec::new();
    rmpv::encode::write_value(
        &mut bytes,
        &rmpv::Value::Array(vec![1.0.into(), rmpv::Value::Array(events), 0.into()]),
    )
    .unwrap();
    bytes
}

async fn advance(rx: &mut mpsc::Receiver<IngestionSignal>, expected: i64) {
    tokio::time::timeout(TIMEOUT, async {
        while let Some(signal) = rx.recv().await {
            if matches!(signal, IngestionSignal::Advance { last_seq, .. } if last_seq == expected) {
                return;
            }
        }
        panic!("subscriber stopped before advance");
    })
    .await
    .expect("subscriber advance");
}

async fn revoked(cursor: &QualifiedSourceCursor) {
    tokio::time::timeout(TIMEOUT, async {
        while cursor
            .with_qualified_matches(&query(&[1, 2]), |_, _| ())
            .is_some()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("synchronous qualification revocation");
}

async fn attach(supervisor: &KvIndexSupervisor, publisher: &Publisher) -> QualifiedSourceCursor {
    supervisor
        .register_static_worker(URL, key(), publisher.source())
        .unwrap();
    let cursor = supervisor.qualified_source(URL, &key()).unwrap();
    publisher.subscribed().await;
    cursor
}

#[tokio::test]
async fn store_cold_miss_remove_and_clear_use_the_actual_partition() {
    let publisher = Publisher::new();
    let (supervisor, mut signals) =
        KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    assert!(supervisor.provider(&key()).is_none());
    let cursor = attach(&supervisor, &publisher).await;
    assert!(cursor
        .with_qualified_matches(&query(&[1, 2]), |_, _| ())
        .is_none());
    publisher.send(0, vec![stored()]);
    advance(&mut signals, 0).await;
    let provider = supervisor.provider(&key()).unwrap();
    assert_eq!(
        provider.find_tiered_matches(&query(&[1, 2]))[0].matched_depth,
        1
    );
    assert_eq!(
        cursor.with_qualified_matches(&query(&[1, 2]), |hits, generation| {
            assert_eq!(generation.last_seq, 0);
            assert_eq!(generation.incarnation, 0);
            hits[0].matched_depth
        }),
        Some(1)
    );
    assert_eq!(
        cursor.with_qualified_matches(&query(&[3, 4]), |hits, _| hits.len()),
        Some(0)
    );
    publisher.send(1, vec![removed()]);
    advance(&mut signals, 1).await;
    assert_eq!(
        cursor.with_qualified_matches(&query(&[1, 2]), |hits, generation| {
            assert_eq!(generation.last_seq, 1);
            hits.len()
        }),
        Some(0)
    );
    publisher.send(2, vec![stored()]);
    advance(&mut signals, 2).await;
    publisher.send(3, vec![clear()]);
    advance(&mut signals, 3).await;
    assert!(provider.find_tiered_matches(&query(&[1, 2])).is_empty());
    assert!(cursor
        .with_qualified_matches(&query(&[1, 2]), |_, _| ())
        .is_none());
    publisher.send(4, vec![stored()]);
    advance(&mut signals, 4).await;
    assert!(provider.find_tiered_matches(&query(&[1, 2])).is_empty());
}

#[tokio::test]
async fn old_cursor_and_provider_do_not_follow_replacement() {
    let first = Publisher::new();
    let (supervisor, mut signals) =
        KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let old = attach(&supervisor, &first).await;
    first.send(0, vec![stored()]);
    advance(&mut signals, 0).await;
    let old_provider = supervisor.provider(&key()).unwrap();
    supervisor.on_worker_removed(URL);
    assert!(old
        .with_qualified_matches(&query(&[1, 2]), |_, _| ())
        .is_none());
    assert!(old_provider.find_tiered_matches(&query(&[1, 2])).is_empty());
    let replacement = Publisher::new();
    let fresh = attach(&supervisor, &replacement).await;
    replacement.send(0, vec![stored()]);
    advance(&mut signals, 0).await;
    assert_eq!(
        fresh.with_qualified_matches(&query(&[1, 2]), |hits, _| hits.len()),
        Some(1)
    );
    assert!(old
        .with_qualified_matches(&query(&[1, 2]), |_, _| ())
        .is_none());
    assert!(old_provider.find_tiered_matches(&query(&[1, 2])).is_empty());
    assert!(supervisor
        .register_static_worker("http://peer:8000", key(), replacement.source())
        .is_err());
    assert_eq!(
        fresh.with_qualified_matches(&query(&[1, 2]), |hits, _| hits.len()),
        Some(1)
    );
}

#[tokio::test]
async fn gap_revokes_even_with_full_or_closed_advisory_queue() {
    for close in [false, true] {
        let publisher = Publisher::new();
        let (supervisor, mut signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let cursor = attach(&supervisor, &publisher).await;
        publisher.send(0, vec![stored()]);
        advance(&mut signals, 0).await;
        if close {
            signals.close();
        } else {
            for seq in 1..=256 {
                publisher.send(seq, vec![stored()]);
            }
            tokio::time::timeout(TIMEOUT, async {
                while cursor
                    .with_qualified_matches(&query(&[1, 2]), |_, generation| generation.last_seq)
                    != Some(256)
                {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
        publisher.send(if close { 2 } else { 258 }, vec![stored()]);
        revoked(&cursor).await;
        assert!(supervisor
            .provider(&key())
            .unwrap()
            .find_tiered_matches(&query(&[1, 2]))
            .is_empty());
        publisher.send(if close { 3 } else { 259 }, vec![stored()]);
        assert!(cursor
            .with_qualified_matches(&query(&[1, 2]), |_, _| ())
            .is_none());
    }
}

#[tokio::test]
async fn initial_nonzero_and_observed_restart_never_restore_qualification() {
    for initial in [0, 4] {
        let publisher = Publisher::new();
        let (supervisor, mut signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let cursor = attach(&supervisor, &publisher).await;
        publisher.send(initial, vec![stored()]);
        advance(&mut signals, initial).await;
        if initial == 0 {
            publisher.send(1, vec![stored()]);
            advance(&mut signals, 1).await;
        }
        publisher.send(0, vec![stored()]);
        advance(&mut signals, 0).await;
        assert!(cursor
            .with_qualified_matches(&query(&[1, 2]), |_, _| ())
            .is_none());
        assert!(supervisor
            .provider(&key())
            .unwrap()
            .find_tiered_matches(&query(&[1, 2]))
            .is_empty());
    }
}

#[tokio::test]
async fn unsupported_device_scope_adapter_and_salt_revoke_without_credit() {
    let cases: Vec<(&str, rmpv::Value)> = vec![
        ("medium", "CPU".into()),
        ("group_idx", 1.into()),
        ("locality", "REMOTE".into()),
        ("ownership", "shared-pool".into()),
        ("lora_name", "adapter".into()),
        ("extra_keys", rmpv::Value::Array(vec!["salt".into()])),
    ];
    for (field, value) in cases {
        let publisher = Publisher::new();
        let (supervisor, mut signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let cursor = attach(&supervisor, &publisher).await;
        publisher.send(0, vec![stored()]);
        advance(&mut signals, 0).await;
        let mut unsupported = stored();
        let rmpv::Value::Map(fields) = &mut unsupported else {
            panic!("stored fixture must be a map");
        };
        fields.retain(|(key, _)| key.as_str() != Some(field));
        fields.push((field.into(), value));
        publisher.send(1, vec![unsupported]);
        revoked(&cursor).await;
        publisher.send(2, vec![stored()]);
        advance(&mut signals, 2).await;
        assert!(
            supervisor
                .provider(&key())
                .unwrap()
                .find_tiered_matches(&query(&[1, 2]))
                .is_empty(),
            "{field}"
        );
    }
}

#[tokio::test]
async fn malformed_own_topic_revokes_but_other_topic_is_ignored() {
    for malformed in 0..3 {
        let publisher = Publisher::new();
        let (supervisor, mut signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let cursor = attach(&supervisor, &publisher).await;
        publisher.send(0, vec![stored()]);
        advance(&mut signals, 0).await;
        // SUB receives topic-prefix matches too, but these are not our source topic.
        publisher.raw(b"kv-other", &[0], &[0xc1]);
        publisher.send(1, vec![stored()]);
        advance(&mut signals, 1).await;
        assert_eq!(
            cursor.with_qualified_matches(&query(&[1, 2]), |hits, _| hits.len()),
            Some(1)
        );
        match malformed {
            0 => publisher.raw(TOPIC, &[0], &batch(vec![stored()])),
            1 => publisher.raw(TOPIC, &2_i64.to_be_bytes(), &[0xc1]),
            _ => publisher
                .socket
                .send_multipart([TOPIC, &2_i64.to_be_bytes()], 0)
                .unwrap(),
        }
        revoked(&cursor).await;
        publisher.send(2, vec![stored()]);
        advance(&mut signals, 2).await;
        assert!(supervisor
            .provider(&key())
            .unwrap()
            .find_tiered_matches(&query(&[1, 2]))
            .is_empty());
    }
}

#[tokio::test]
async fn qualified_query_rejects_other_identity_group_and_tier() {
    let publisher = Publisher::new();
    let (supervisor, mut signals) =
        KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let cursor = attach(&supervisor, &publisher).await;
    publisher.send(0, vec![stored()]);
    advance(&mut signals, 0).await;
    let mut other_key = key();
    other_key.model = Arc::from("other-model");
    assert!(supervisor.provider(&other_key).is_none());
    assert!(supervisor.qualified_source(URL, &other_key).is_none());
    let mut other_query = query(&[1, 2]);
    other_query.group_idx = 1;
    assert!(cursor
        .with_qualified_matches(&other_query, |_, _| ())
        .is_none());
    other_query.group_idx = 0;
    other_query.tiers_of_interest = vec![StorageTier::HostPinned];
    assert!(cursor
        .with_qualified_matches(&other_query, |_, _| ())
        .is_none());
}

#[tokio::test]
async fn replay_partial_or_complete_never_restores_after_gap() {
    for complete in [false, true] {
        let context = zmq::Context::new();
        let replay = context.socket(zmq::ROUTER).unwrap();
        replay.set_linger(0).unwrap();
        replay.bind("tcp://127.0.0.1:*").unwrap();
        let publisher = Publisher::new();
        let (supervisor, mut signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let mut source = publisher.source();
        source.replay_endpoint = Some(replay.get_last_endpoint().unwrap().unwrap());
        supervisor
            .register_static_worker(URL, key(), source)
            .unwrap();
        let cursor = supervisor.qualified_source(URL, &key()).unwrap();
        publisher.subscribed().await;
        publisher.send(0, vec![stored()]);
        advance(&mut signals, 0).await;
        publisher.send(3, vec![stored()]);
        let request = tokio::time::timeout(TIMEOUT, async {
            loop {
                if let Ok(request) = replay.recv_multipart(zmq::DONTWAIT) {
                    return request;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("DEALER replay request");
        assert_eq!(request.len(), 3);
        assert_eq!(request[2], 1_u64.to_be_bytes());
        assert!(cursor
            .with_qualified_matches(&query(&[1, 2]), |_, _| ())
            .is_none());
        assert!(supervisor
            .provider(&key())
            .unwrap()
            .find_tiered_matches(&query(&[1, 2]))
            .is_empty());
        for seq in if complete {
            vec![1_i64, 2]
        } else {
            vec![2_i64]
        } {
            replay
                .send_multipart(
                    vec![
                        request[0].clone(),
                        vec![],
                        TOPIC.to_vec(),
                        seq.to_be_bytes().to_vec(),
                        batch(vec![stored()]),
                    ],
                    0,
                )
                .unwrap();
        }
        // Preserve vLLM's empty-topic terminal schema, including a partial gap.
        replay
            .send_multipart(
                vec![
                    request[0].clone(),
                    vec![],
                    vec![],
                    (-1_i64).to_be_bytes().to_vec(),
                    vec![],
                ],
                0,
            )
            .unwrap();
        advance(&mut signals, 3).await;
        publisher.send(4, vec![stored()]);
        advance(&mut signals, 4).await;
        assert!(cursor
            .with_qualified_matches(&query(&[1, 2]), |_, _| ())
            .is_none());
        assert!(supervisor
            .provider(&key())
            .unwrap()
            .find_tiered_matches(&query(&[1, 2]))
            .is_empty());
    }
}

#[tokio::test]
async fn shared_partition_queries_credit_only_this_source_and_rank() {
    let first = Publisher::new();
    let second = Publisher::new();
    let (supervisor, mut signals) =
        KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let cursor = attach(&supervisor, &first).await;
    let mut other_source = second.source();
    other_source.source = Arc::from("worker-b");
    supervisor
        .register_static_worker("http://worker-b:8000", key(), other_source)
        .unwrap();
    second.subscribed().await;
    first.send(0, vec![stored()]);
    advance(&mut signals, 0).await;
    second.send(0, vec![stored()]);
    advance(&mut signals, 0).await;
    assert_eq!(
        supervisor
            .provider(&key())
            .unwrap()
            .find_tiered_matches(&query(&[1, 2]))
            .len(),
        2
    );
    assert_eq!(
        cursor.with_qualified_matches(&query(&[1, 2]), |hits, _| {
            assert_eq!(&*hits[0].target.instance_id, "worker-a");
            assert_eq!(hits[0].target.dp_rank, 0);
            hits.len()
        }),
        Some(1)
    );
    supervisor.on_worker_removed(URL);
    assert_eq!(
        supervisor
            .provider(&key())
            .unwrap()
            .find_tiered_matches(&query(&[1, 2]))
            .len(),
        1
    );
}

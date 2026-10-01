//! Public, CPU-only lifecycle regressions using real loopback ZMQ and HTTP.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use axum::{routing::get, Json, Router};
use tokio::sync::{mpsc, Notify};
use vllm_router_rs::kv_index::{
    subscriber::{local_hashes, spawn, EventSource, SubscriberHandle},
    HashMode, IngestionSignal, KvBlockIndexer, KvIndexSupervisor, MatchQuery, ResidencyOwner,
    StorageTier,
};

const TOPIC: &[u8] = b"kv";
const TOKENS: &[u32] = &[1, 2];
const TIMEOUT: Duration = Duration::from_secs(5);

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
            source: Arc::from("worker-A"),
            dp_rank: 0,
            pub_endpoint: self.endpoint.clone(),
            replay_endpoint: None,
            topic: String::from_utf8(TOPIC.to_vec()).unwrap(),
            hwm: None,
        }
    }

    async fn subscribed(&self) {
        tokio::time::timeout(TIMEOUT, async {
            loop {
                let message = receive(&self.socket).await;
                if message == vec![[&[1], TOPIC].concat()] {
                    return;
                }
            }
        })
        .await
        .expect("subscription handshake timed out");
    }

    fn send(&self, seq: i64, payload: &[u8]) {
        self.socket
            .send_multipart([TOPIC, &seq.to_be_bytes(), payload], 0)
            .unwrap();
    }
}

async fn receive(socket: &zmq::Socket) -> Vec<Vec<u8>> {
    tokio::time::timeout(TIMEOUT, async {
        loop {
            match socket.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => return frames,
                Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                Err(error) => panic!("loopback ZMQ receive: {error}"),
            }
        }
    })
    .await
    .expect("loopback ZMQ handshake timed out")
}

fn stored() -> Vec<u8> {
    batch(vec![stored_event("GPU", None)], 0)
}

fn stored_event(medium: &str, ownership: Option<&str>) -> rmpv::Value {
    rmpv::Value::Map(vec![
        ("type".into(), "BlockStored".into()),
        ("block_hashes".into(), rmpv::Value::Array(vec![42.into()])),
        ("parent_block_hash".into(), rmpv::Value::Nil),
        (
            "token_ids".into(),
            rmpv::Value::Array(TOKENS.iter().map(|token| (*token).into()).collect()),
        ),
        ("block_size".into(), 2.into()),
        ("lora_id".into(), rmpv::Value::Nil),
        ("lora_name".into(), rmpv::Value::Nil),
        ("medium".into(), medium.into()),
        (
            "ownership".into(),
            ownership.map_or(rmpv::Value::Nil, Into::into),
        ),
    ])
}

fn batch(events: Vec<rmpv::Value>, rank: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    rmpv::encode::write_value(
        &mut payload,
        &rmpv::Value::Array(vec![1.0.into(), rmpv::Value::Array(events), rank.into()]),
    )
    .unwrap();
    payload
}

fn query() -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        local_hashes: local_hashes(TOKENS, 2, None)
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

async fn advance(rx: &mut mpsc::Receiver<IngestionSignal>, expected: i64) -> Vec<IngestionSignal> {
    tokio::time::timeout(TIMEOUT, async {
        let mut signals = Vec::new();
        loop {
            let signal = rx.recv().await.expect("subscriber stopped before advance");
            let done = matches!(&signal, IngestionSignal::Advance { last_seq, .. } if *last_seq == expected);
            signals.push(signal);
            if done {
                return signals;
            }
        }
    })
    .await
    .expect("subscriber advance timed out")
}

#[tokio::test]
async fn drop_retires_worker_residency_synchronously() {
    let publisher = Publisher::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (tx, mut rx) = mpsc::channel(32);
    let handle = spawn(publisher.source(), index.clone(), Some(tx));
    publisher.subscribed().await;
    publisher.send(0, &stored());
    advance(&mut rx, 0).await;
    assert_eq!(index.find_matches(&query()).len(), 1);

    drop(handle);
    assert!(index.find_matches(&query()).is_empty());
}

#[tokio::test]
async fn malformed_live_does_not_consume_the_same_sequence() {
    let publisher = Publisher::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (tx, mut rx) = mpsc::channel(32);
    let handle = spawn(publisher.source(), index.clone(), Some(tx));
    publisher.subscribed().await;
    publisher.send(0, &[0xc1]); // Reserved MessagePack marker, not a batch.
    publisher.send(0, &stored());
    publisher.send(1, &batch(vec![], 0)); // Ordered barrier for both preceding frames.
    advance(&mut rx, 1).await;
    assert_eq!(index.find_matches(&query()).len(), 1);
    handle.shutdown();
}

struct ReplayServer {
    _context: zmq::Context,
    socket: zmq::Socket,
    endpoint: String,
}

impl ReplayServer {
    fn new() -> Self {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::ROUTER).unwrap();
        socket.set_linger(0).unwrap();
        socket.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = socket.get_last_endpoint().unwrap().unwrap();
        Self {
            _context: context,
            socket,
            endpoint,
        }
    }

    fn reply(&self, request: &[Vec<u8>], frames: Vec<Vec<u8>>) {
        let mut routed = vec![request[0].clone()];
        routed.extend(frames);
        self.socket.send_multipart(routed, 0).unwrap();
    }

    fn send(&self, request: &[Vec<u8>], seq: i64, payload: &[u8]) {
        self.reply(
            request,
            vec![
                vec![],
                TOPIC.to_vec(),
                seq.to_be_bytes().to_vec(),
                payload.to_vec(),
            ],
        );
    }
}

async fn replaying(
    publisher: &Publisher,
    replay: &ReplayServer,
    index: Arc<KvBlockIndexer>,
    initial: &[u8],
) -> (
    SubscriberHandle,
    mpsc::Receiver<IngestionSignal>,
    Vec<Vec<u8>>,
) {
    let mut source = publisher.source();
    source.replay_endpoint = Some(replay.endpoint.clone());
    let (tx, mut rx) = mpsc::channel(32);
    let handle = spawn(source, index, Some(tx));
    publisher.subscribed().await;
    publisher.send(0, initial);
    advance(&mut rx, 0).await;
    publisher.send(2, &batch(vec![], 0));
    let request = receive(&replay.socket).await;
    assert_eq!(request.len(), 3);
    assert!(request[1].is_empty());
    assert_eq!(request[2], 1_u64.to_be_bytes());
    (handle, rx, request)
}

fn advances(signals: &[IngestionSignal]) -> Vec<i64> {
    signals
        .iter()
        .filter_map(|signal| match signal {
            IngestionSignal::Advance { last_seq, .. } => Some(*last_seq),
            _ => None,
        })
        .collect()
}

fn replay_applied(signals: &[IngestionSignal]) -> Vec<i64> {
    signals
        .iter()
        .filter_map(|signal| match signal {
            IngestionSignal::ReplayApplied { replay_seq, .. } => Some(*replay_seq),
            _ => None,
        })
        .collect()
}

async fn no_late_publication(rx: &mut mpsc::Receiver<IngestionSignal>) {
    tokio::time::timeout(TIMEOUT, async {
        while let Some(signal) = rx.recv().await {
            assert!(!matches!(
                signal,
                IngestionSignal::Advance { .. } | IngestionSignal::ReplayApplied { .. }
            ));
        }
    })
    .await
    .expect("retired task retained its signal sender");
}

#[tokio::test]
async fn shutdown_fences_blocked_replay_and_waits_for_resources() {
    let publisher = Publisher::new();
    let replay = ReplayServer::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (handle, mut rx, request) = replaying(&publisher, &replay, index.clone(), &stored()).await;
    handle.shutdown();
    assert!(index.find_matches(&query()).is_empty());
    replay.send(&request, 1, &stored());
    replay.send(&request, -1, &[]);
    tokio::time::timeout(TIMEOUT, handle.shutdown_and_wait())
        .await
        .unwrap()
        .unwrap();
    no_late_publication(&mut rx).await;
    assert!(index.find_matches(&query()).is_empty());
}

#[tokio::test]
async fn drop_fences_blocked_replay_and_releases_the_task() {
    let publisher = Publisher::new();
    let replay = ReplayServer::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (handle, mut rx, request) = replaying(&publisher, &replay, index.clone(), &stored()).await;
    drop(handle);
    assert!(index.find_matches(&query()).is_empty());
    replay.send(&request, 1, &stored());
    replay.send(&request, -1, &[]);
    no_late_publication(&mut rx).await;
    assert!(index.find_matches(&query()).is_empty());
}

#[tokio::test]
async fn retired_handle_drop_cannot_clear_its_replacement() {
    let old_publisher = Publisher::new();
    let replay = ReplayServer::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (old, mut old_rx, request) =
        replaying(&old_publisher, &replay, index.clone(), &stored()).await;
    old.shutdown();
    let publisher = Publisher::new();
    let (tx, mut rx) = mpsc::channel(32);
    let replacement = spawn(publisher.source(), index.clone(), Some(tx));
    publisher.subscribed().await;
    publisher.send(0, &stored());
    advance(&mut rx, 0).await;
    replay.send(&request, 1, &stored());
    replay.send(&request, -1, &[]);
    old.shutdown_and_wait().await.unwrap(); // Includes the old handle's Drop.
    no_late_publication(&mut old_rx).await;
    assert_eq!(index.find_matches(&query()).len(), 1);
    replacement.shutdown_and_wait().await.unwrap();
}

#[tokio::test]
async fn malformed_replay_does_not_consume_a_valid_same_sequence() {
    let publisher = Publisher::new();
    let replay = ReplayServer::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (handle, mut rx, request) =
        replaying(&publisher, &replay, index.clone(), &batch(vec![], 0)).await;
    replay.send(&request, 1, &[0xc1]);
    replay.send(&request, 1, &stored());
    replay.send(&request, -1, &[]);
    let signals = advance(&mut rx, 2).await;
    assert_eq!(advances(&signals), [1, 2]);
    assert_eq!(replay_applied(&signals), [1]);
    assert_eq!(index.find_matches(&query()).len(), 1);
    handle.shutdown_and_wait().await.unwrap();
}

fn removed() -> rmpv::Value {
    rmpv::Value::Map(vec![
        ("type".into(), "BlockRemoved".into()),
        ("block_hashes".into(), rmpv::Value::Array(vec![42.into()])),
        ("medium".into(), "GPU".into()),
    ])
}

#[tokio::test]
async fn replay_high_water_deduplicates_queued_live_without_fake_restart() {
    let publisher = Publisher::new();
    let replay = ReplayServer::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (handle, mut rx, request) =
        replaying(&publisher, &replay, index.clone(), &batch(vec![], 0)).await;
    publisher.send(3, &batch(vec![removed()], 0)); // Covered by replay seq 3.
    replay.send(&request, 1, &stored());
    replay.send(&request, 2, &batch(vec![], 0));
    replay.send(&request, 3, &stored());
    replay.send(&request, -1, &[]);
    publisher.send(4, &batch(vec![], 0));
    let signals = advance(&mut rx, 4).await;
    assert_eq!(advances(&signals), [1, 2, 3, 4]);
    assert_eq!(replay_applied(&signals), [3]);
    assert!(!signals
        .iter()
        .any(|signal| matches!(signal, IngestionSignal::IncarnationReset { .. })));
    assert_eq!(index.find_matches(&query()).len(), 1);
    handle.shutdown_and_wait().await.unwrap();
}

#[tokio::test]
async fn malformed_backwards_live_does_not_reset_the_attachment() {
    let publisher = Publisher::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (tx, mut rx) = mpsc::channel(32);
    let handle = spawn(publisher.source(), index.clone(), Some(tx));
    publisher.subscribed().await;
    publisher.send(5, &stored());
    advance(&mut rx, 5).await;
    publisher.send(0, &[0xc1]);
    publisher.send(6, &batch(vec![], 0));
    let signals = advance(&mut rx, 6).await;
    assert_eq!(advances(&signals), [6]);
    assert!(!signals
        .iter()
        .any(|signal| matches!(signal, IngestionSignal::IncarnationReset { .. })));
    assert_eq!(index.find_matches(&query()).len(), 1);
    handle.shutdown_and_wait().await.unwrap();
}

#[tokio::test]
async fn validated_restart_clears_actual_batch_ranks_before_new_incarnation() {
    let publisher = Publisher::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (tx, mut rx) = mpsc::channel(32);
    let handle = spawn(publisher.source(), index.clone(), Some(tx));
    publisher.subscribed().await;
    publisher.send(5, &batch(vec![stored_event("GPU", None)], 3));
    advance(&mut rx, 5).await;
    assert_eq!(index.find_matches(&query())[0].target.dp_rank, 3);
    publisher.send(0, &batch(vec![], 4));
    let signals = advance(&mut rx, 0).await;
    assert!(signals.iter().any(|signal| matches!(
        signal,
        IngestionSignal::IncarnationReset { incarnation: 1, .. }
    )));
    assert!(index.find_matches(&query()).is_empty());
    publisher.send(1, &batch(vec![stored_event("GPU", None)], 4));
    advance(&mut rx, 1).await;
    assert_eq!(index.find_matches(&query())[0].target.dp_rank, 4);
    handle.shutdown_and_wait().await.unwrap();
    assert!(index.find_matches(&query()).is_empty());
}

#[tokio::test]
async fn incomplete_replay_gap_keeps_observations_without_recovery_summary() {
    let publisher = Publisher::new();
    let replay = ReplayServer::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (handle, mut rx, request) =
        replaying(&publisher, &replay, index.clone(), &batch(vec![], 0)).await;
    replay.send(&request, 2, &stored()); // seq 1 remains missing.
    replay.send(&request, -1, &[]);
    publisher.send(3, &batch(vec![], 0));
    let signals = advance(&mut rx, 3).await;
    assert_eq!(advances(&signals), [2, 3]);
    assert!(replay_applied(&signals).is_empty());
    assert_eq!(index.find_matches(&query()).len(), 1);
    handle.shutdown_and_wait().await.unwrap();
}

fn invalid_live_frames() -> Vec<Vec<Vec<u8>>> {
    let payload = stored();
    let seq = 0_i64.to_be_bytes().to_vec();
    let mut long_seq = seq.clone();
    long_seq.push(0);
    let mut trailing_payload = payload.clone();
    trailing_payload.push(0);
    vec![
        vec![b"kv-other".to_vec(), seq.clone(), payload.clone()],
        vec![TOPIC.to_vec(), seq.clone(), payload.clone(), vec![]],
        vec![TOPIC.to_vec(), vec![0; 7], payload.clone()],
        vec![TOPIC.to_vec(), long_seq, payload.clone()],
        vec![
            TOPIC.to_vec(),
            (-2_i64).to_be_bytes().to_vec(),
            payload.clone(),
        ],
        vec![TOPIC.to_vec(), (-1_i64).to_be_bytes().to_vec(), payload],
        vec![TOPIC.to_vec(), seq, trailing_payload],
    ]
}

#[tokio::test]
async fn invalid_pub_envelopes_and_trailing_payload_produce_no_positive_signal() {
    for frames in invalid_live_frames() {
        let publisher = Publisher::new();
        let index = Arc::new(KvBlockIndexer::new());
        let (tx, mut rx) = mpsc::channel(32);
        let handle = spawn(publisher.source(), index.clone(), Some(tx));
        publisher.subscribed().await;
        publisher.socket.send_multipart(frames, 0).unwrap();
        publisher.send(1, &batch(vec![], 0));
        assert_eq!(advances(&advance(&mut rx, 1).await), [1]);
        assert!(index.find_matches(&query()).is_empty());
        handle.shutdown_and_wait().await.unwrap();
    }
}

#[tokio::test]
async fn invalid_replay_envelopes_and_missing_gap_produce_no_recovery_summary() {
    let mut envelopes: Vec<_> = invalid_live_frames()
        .into_iter()
        .map(|mut frames| {
            frames.insert(0, vec![]);
            frames
        })
        .collect();
    envelopes.push(vec![
        b"bad-delimiter".to_vec(),
        TOPIC.to_vec(),
        1_i64.to_be_bytes().to_vec(),
        stored(),
    ]);
    envelopes.push(vec![
        vec![],
        TOPIC.to_vec(),
        1_i64.to_be_bytes().to_vec(),
        vec![0xc1],
    ]);
    for mut frames in envelopes {
        // PUB's negative sentinel is legal only as the properly framed replay
        // end marker; its attached event payload must still never be applied.
        let publisher = Publisher::new();
        let replay = ReplayServer::new();
        let index = Arc::new(KvBlockIndexer::new());
        let (handle, mut rx, request) =
            replaying(&publisher, &replay, index.clone(), &batch(vec![], 0)).await;
        if frames[2] == 0_i64.to_be_bytes() {
            frames[2] = 1_i64.to_be_bytes().to_vec();
        }
        replay.reply(&request, frames);
        replay.send(&request, -1, &[]);
        let signals = advance(&mut rx, 2).await;
        assert_eq!(advances(&signals), [2]);
        assert!(replay_applied(&signals).is_empty());
        assert!(index.find_matches(&query()).is_empty());
        handle.shutdown_and_wait().await.unwrap();
    }
}

#[tokio::test]
async fn multi_tier_nonzero_rank_remove_clear_and_retire_preserve_other_domains() {
    let publisher = Publisher::new();
    let index = Arc::new(KvBlockIndexer::new());
    index.store(
        0,
        ResidencyOwner::Worker {
            source: Arc::from("worker-B"),
            dp_rank: 3,
            incarnation: 0,
        },
        StorageTier::Device,
        None,
        &[(Arc::from("900"), query().local_hashes[0].clone())],
    );
    let (tx, mut rx) = mpsc::channel(32);
    let handle = spawn(publisher.source(), index.clone(), Some(tx));
    publisher.subscribed().await;
    publisher.send(
        0,
        &batch(
            vec![
                stored_event("GPU", None),
                stored_event("CPU", None),
                stored_event("GPU", Some("shared-pool")),
            ],
            3,
        ),
    );
    advance(&mut rx, 0).await;
    let mut all = query();
    all.tiers_of_interest = vec![
        StorageTier::Device,
        StorageTier::HostPinned,
        StorageTier::External,
    ];
    let hits = index.find_matches(&all);
    assert_eq!(hits.len(), 4);
    assert!(hits.iter().all(|hit| hit.target.dp_rank == 3));
    publisher.send(1, &batch(vec![removed()], 3));
    advance(&mut rx, 1).await;
    assert_eq!(index.find_matches(&all).len(), 3);
    publisher.send(
        2,
        &batch(
            vec![rmpv::Value::Map(vec![(
                "type".into(),
                "AllBlocksCleared".into(),
            )])],
            3,
        ),
    );
    advance(&mut rx, 2).await;
    assert_eq!(index.find_matches(&all).len(), 2);
    publisher.send(
        3,
        &batch(
            vec![stored_event("GPU", None), stored_event("CPU", None)],
            3,
        ),
    );
    advance(&mut rx, 3).await;
    assert_eq!(index.find_matches(&all).len(), 4);
    handle.shutdown_and_wait().await.unwrap();
    let hits = index.find_matches(&all);
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().any(
        |hit| hit.target.instance_id.as_ref() == "worker-B" && hit.tier == StorageTier::Device
    ));
    assert!(hits
        .iter()
        .any(|hit| hit.target.instance_id.as_ref() == "worker-A"
            && hit.tier == StorageTier::External));
}

#[tokio::test]
async fn delayed_discovery_cannot_register_after_remove() {
    let publisher = Publisher::new();
    let server = DiscoveryServer::new(&publisher, "model-A", Some("info")).await;
    let (supervisor, _signals) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let supervisor = Arc::new(supervisor);
    let added = tokio::spawn({
        let supervisor = supervisor.clone();
        let url = server.url.clone();
        async move { supervisor.on_worker_added(&url).await }
    });
    server.started().await;
    supervisor.on_worker_removed(&server.url);
    server.release.notify_one();
    let result = tokio::time::timeout(TIMEOUT, added).await.unwrap().unwrap();
    supervisor.shutdown();
    assert!(
        result.is_err(),
        "removed discovery returned a live registration: {result:?}"
    );
}

struct DiscoveryServer {
    url: String,
    entered: Arc<Notify>,
    release: Arc<Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl DiscoveryServer {
    async fn new(publisher: &Publisher, model: &str, delay: Option<&'static str>) -> Self {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let requests = Arc::new(AtomicUsize::new(0));
        let model = model.to_string();
        let app = Router::new()
            .route("/get_server_info", get({
                let entered = entered.clone();
                let release = release.clone();
                let requests = requests.clone();
                move || {
                    let entered = entered.clone();
                    let release = release.clone();
                    let requests = requests.clone();
                    let model = model.clone();
                    async move {
                        if delay == Some("info") && requests.fetch_add(1, Ordering::SeqCst) == 0 {
                            entered.notify_one();
                            release.notified().await;
                        }
                        Json(serde_json::json!({"model_id": model, "instance_id": "worker-A", "kv_block_size": 2}))
                    }
                }
            }))
            .route("/kv_event_sources", get({
                let endpoint = publisher.endpoint.clone();
                let entered = entered.clone();
                let release = release.clone();
                move || {
                    let endpoint = endpoint.clone();
                    let entered = entered.clone();
                    let release = release.clone();
                    let requests = requests.clone();
                    async move {
                        if delay == Some("sources") && requests.fetch_add(1, Ordering::SeqCst) == 0 {
                            entered.notify_one();
                            release.notified().await;
                        }
                        Json(serde_json::json!({"0": {"endpoint": endpoint, "topic": "kv"}}))
                    }
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url,
            entered,
            release,
            task,
        }
    }

    async fn started(&self) {
        tokio::time::timeout(TIMEOUT, self.entered.notified())
            .await
            .unwrap();
    }
}

impl Drop for DiscoveryServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test]
async fn cancellation_at_either_http_await_allows_new_registration() {
    for delay in ["info", "sources"] {
        let publisher = Publisher::new();
        let server = DiscoveryServer::new(&publisher, "model-A", Some(delay)).await;
        let (supervisor, _signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let supervisor = Arc::new(supervisor);
        let added = tokio::spawn({
            let supervisor = supervisor.clone();
            let url = server.url.clone();
            async move { supervisor.on_worker_added(&url).await }
        });
        server.started().await;
        added.abort();
        assert!(added.await.unwrap_err().is_cancelled());
        let info = tokio::time::timeout(TIMEOUT, supervisor.on_worker_added(&server.url))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(info.ranks, [0]);
        publisher.subscribed().await;
        server.release.notify_one();
        supervisor.shutdown();
    }
}

#[tokio::test]
async fn delayed_discovery_cannot_register_after_shutdown() {
    let publisher = Publisher::new();
    let server = DiscoveryServer::new(&publisher, "model-A", Some("sources")).await;
    let (supervisor, _signals) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let supervisor = Arc::new(supervisor);
    let added = tokio::spawn({
        let supervisor = supervisor.clone();
        let url = server.url.clone();
        async move { supervisor.on_worker_added(&url).await }
    });
    server.started().await;
    supervisor.shutdown();
    server.release.notify_one();
    assert!(tokio::time::timeout(TIMEOUT, added)
        .await
        .unwrap()
        .unwrap()
        .is_err());
}

#[tokio::test]
async fn replacement_invalidates_late_discovery_without_disturbing_new_subscription() {
    let publisher = Publisher::new();
    let server = DiscoveryServer::new(&publisher, "model-A", Some("sources")).await;
    let (supervisor, mut signals) =
        KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let supervisor = Arc::new(supervisor);
    let old = tokio::spawn({
        let supervisor = supervisor.clone();
        let url = server.url.clone();
        async move { supervisor.on_worker_added(&url).await }
    });
    server.started().await;
    assert_eq!(
        supervisor.on_worker_added(&server.url).await.unwrap().ranks,
        [0]
    );
    publisher.subscribed().await;
    server.release.notify_one();
    assert!(tokio::time::timeout(TIMEOUT, old)
        .await
        .unwrap()
        .unwrap()
        .is_err());
    publisher.send(0, &stored());
    assert_eq!(advances(&advance(&mut signals, 0).await), [0]);
    supervisor.on_worker_removed(&server.url);
    assert_eq!(
        supervisor.on_worker_added(&server.url).await.unwrap().ranks,
        [0]
    );
    publisher.subscribed().await;
    publisher.send(0, &stored());
    assert_eq!(advances(&advance(&mut signals, 0).await), [0]);
    supervisor.shutdown();
}

#[tokio::test]
async fn overlapping_url_source_is_rejected_only_within_the_same_cache_identity() {
    let first_publisher = Publisher::new();
    let second_publisher = Publisher::new();
    let first = DiscoveryServer::new(&first_publisher, "model-A", None).await;
    let second = DiscoveryServer::new(&second_publisher, "model-A", None).await;
    let different_identity = DiscoveryServer::new(&second_publisher, "model-B", None).await;
    let (supervisor, _signals) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    assert_eq!(
        supervisor.on_worker_added(&first.url).await.unwrap().ranks,
        [0]
    );
    first_publisher.subscribed().await;
    let error = supervisor.on_worker_added(&second.url).await.unwrap_err();
    assert!(error.contains("already registered for this cache identity"));
    assert_eq!(
        supervisor
            .on_worker_added(&different_identity.url)
            .await
            .unwrap()
            .ranks,
        [0]
    );
    second_publisher.subscribed().await;
    supervisor.on_worker_removed(&first.url);
    assert_eq!(
        supervisor.on_worker_added(&second.url).await.unwrap().ranks,
        [0]
    );
    supervisor.shutdown();
}

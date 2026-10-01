//! Public, CPU-only lifecycle regressions using real loopback ZMQ and HTTP.
use std::sync::Arc;
use std::time::Duration;

use axum::{routing::get, Json, Router};
use tokio::sync::{mpsc, Notify};
use vllm_router_rs::kv_index::{
    subscriber::{local_hashes, spawn, EventSource},
    HashMode, IngestionSignal, KvBlockIndexer, KvIndexSupervisor, MatchQuery, StorageTier,
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
        let message = receive(&self.socket).await;
        assert_eq!(message, vec![[&[1], TOPIC].concat()]);
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
    let event = rmpv::Value::Map(vec![
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
        ("medium".into(), "GPU".into()),
    ]);
    batch(vec![event])
}

fn batch(events: Vec<rmpv::Value>) -> Vec<u8> {
    let mut payload = Vec::new();
    rmpv::encode::write_value(
        &mut payload,
        &rmpv::Value::Array(vec![1.0.into(), rmpv::Value::Array(events), 0.into()]),
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

async fn advance(rx: &mut mpsc::Receiver<IngestionSignal>, expected: i64) {
    tokio::time::timeout(TIMEOUT, async {
        loop {
            match rx.recv().await.expect("subscriber stopped before advance") {
                IngestionSignal::Advance { last_seq, .. } if last_seq == expected => return,
                _ => {}
            }
        }
    })
    .await
    .expect("subscriber advance timed out");
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
    publisher.send(1, &batch(vec![])); // Ordered barrier for both preceding frames.
    advance(&mut rx, 1).await;
    assert_eq!(index.find_matches(&query()).len(), 1);
    handle.shutdown();
}

#[tokio::test]
async fn delayed_discovery_cannot_register_after_remove() {
    let publisher = Publisher::new();
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let app =
        Router::new()
            .route(
                "/get_server_info",
                get({
                    let entered = entered.clone();
                    let release = release.clone();
                    move || {
                        let entered = entered.clone();
                        let release = release.clone();
                        async move {
                            entered.notify_one();
                            release.notified().await;
                            Json(serde_json::json!({
                                "model_id": "model-A", "instance_id": "worker-A", "kv_block_size": 2
                            }))
                        }
                    }
                }),
            )
            .route(
                "/kv_event_sources",
                get({
                    let endpoint = publisher.endpoint.clone();
                    move || {
                        let endpoint = endpoint.clone();
                        async move {
                            Json(serde_json::json!({"0": {"endpoint": endpoint, "topic": "kv"}}))
                        }
                    }
                }),
            );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (supervisor, _signals) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let supervisor = Arc::new(supervisor);
    let added = tokio::spawn({
        let supervisor = supervisor.clone();
        let url = url.clone();
        async move { supervisor.on_worker_added(&url).await }
    });
    tokio::time::timeout(TIMEOUT, entered.notified())
        .await
        .unwrap();
    supervisor.on_worker_removed(&url);
    release.notify_one();
    let result = tokio::time::timeout(TIMEOUT, added).await.unwrap().unwrap();
    supervisor.shutdown();
    server.abort();
    assert!(
        result.is_err(),
        "removed discovery returned a live registration: {result:?}"
    );
}

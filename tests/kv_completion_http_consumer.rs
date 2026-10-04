//! Local assembly proof, not an installed production KV policy or Engine oracle.
//! The unique-owner fixture uses the existing selector; it defines no cost model.
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::json;
use vllm_router_rs::{
    backend::{
        completion_activation::{load_completion_input_assets, CompletionInputAssets},
        CompletionPreparationError,
    },
    core::{BasicWorker, Worker, WorkerLoadGuard, WorkerType},
    kv_index::{
        subscriber::{local_hashes, EventSource, QualifiedSourceCursor},
        CacheKey, HashMode, KvIndexSupervisor, MatchQuery, StorageTier,
    },
    policies::{LoadBalancingPolicy, PolicyRequestContext, RoundRobinPolicy},
    protocols::spec::CompletionRequest,
};
use vllm_tokenizer::Tokenizer;

const WAIT: Duration = Duration::from_secs(5);

struct CountingTokenizer {
    inner: Arc<dyn Tokenizer>,
    encodes: AtomicUsize,
}
impl Tokenizer for CountingTokenizer {
    fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        self.encodes.fetch_add(1, Ordering::SeqCst);
        self.inner.encode(text, special)
    }
    fn encode_ordinary(&self, _: &str) -> vllm_tokenizer::Result<Vec<u32>> {
        panic!("not Completion")
    }
    fn decode(&self, ids: &[u32], skip: bool) -> vllm_tokenizer::Result<String> {
        self.inner.decode(ids, skip)
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

#[derive(Clone, Default)]
struct Backend {
    received: Arc<Mutex<Vec<bytes::Bytes>>>,
    fail_once: Arc<AtomicBool>,
}
async fn backend(State(state): State<Backend>, request: Request) -> Response {
    let body = to_bytes(request.into_body(), 4096).await.unwrap();
    let stream = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["stream"] == true;
    state.received.lock().push(body);
    if state.fail_once.swap(false, Ordering::SeqCst) {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    if stream {
        return Response::new(Body::from_stream(
            futures_util::stream::once(async {
                Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"data: token\n\n"))
            })
            .chain(futures_util::stream::pending()),
        ));
    }
    axum::Json(json!({"choices":[{"text":"ok"}]})).into_response()
}

#[derive(Clone)]
struct Consumer {
    assets: Arc<CompletionInputAssets>,
    cursor: QualifiedSourceCursor,
    owner: Arc<dyn Worker>,
    workers: Vec<Arc<dyn Worker>>,
    policy: Arc<RoundRobinPolicy>,
    client: reqwest::Client,
    attempts: Arc<Mutex<Vec<AttemptObservation>>>,
}

#[derive(Clone, Debug, PartialEq)]
struct AttemptObservation {
    token_allocation: Option<usize>,
    observed_owner: bool,
    source_sequence: Option<i64>,
}
async fn ingress(State(state): State<Consumer>, request: Request) -> Response {
    let raw = to_bytes(request.into_body(), 4096).await.unwrap();
    let body: CompletionRequest = serde_json::from_slice(&raw).unwrap();
    let prepared = match state.assets.prepare(&body) {
        Ok(prepared) => Some(prepared),
        Err(CompletionPreparationError::Unsupported(_)) => None,
        Err(CompletionPreparationError::InvalidRequest(_)) => {
            return StatusCode::BAD_REQUEST.into_response()
        }
        Err(CompletionPreparationError::ServiceFailure(_)) => {
            return StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
    };
    // One preparation owner lives outside the attempts; selection borrows it.
    let context = PolicyRequestContext::new(Some("controlled Completion"), None);
    let context = if let Some(prepared) = &prepared {
        context.with_token_ids(prepared.token_ids())
    } else {
        context
    };
    for attempt in 0..2 {
        let observed = context.token_ids.and_then(|tokens| {
            let complete = tokens.len() / 2 * 2;
            let query = MatchQuery {
                group_idx: 0,
                local_hashes: local_hashes(&tokens[..complete], 2, None)
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
            };
            state
                .cursor
                .with_qualified_matches(&query, |matches, generation| {
                    if !matches.iter().any(|hit| hit.matched_depth > 0)
                        || !state.owner.is_available()
                    {
                        return None;
                    }
                    let candidates = [state.owner.clone()];
                    let index = state
                        .policy
                        .select_worker_with_context(&candidates, &context)?;
                    let worker = candidates[index].clone();
                    // Existing guard, reserved under the source fence, exactly once.
                    let guard = WorkerLoadGuard::new_owned(worker.clone());
                    Some((worker, guard, true, Some(generation.last_seq)))
                })
                .flatten()
        });
        let (worker, guard, observed_owner, sequence) = observed.unwrap_or_else(|| {
            let index = state
                .policy
                .select_worker_with_context(&state.workers, &context)
                .unwrap();
            let worker = state.workers[index].clone();
            let guard = WorkerLoadGuard::new_owned(worker.clone());
            (worker, guard, false, None)
        });
        state.attempts.lock().push(AttemptObservation {
            token_allocation: context.token_ids.map(|ids| ids.as_ptr() as usize),
            observed_owner,
            source_sequence: sequence,
        });
        let response = match state
            .client
            .post(format!("{}/v1/completions", worker.url()))
            .header("content-type", "application/json")
            .body(raw.clone())
            .send()
            .await
        {
            Ok(response) => response,
            Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
        };
        if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE && attempt == 0 {
            drop(response);
            drop(guard);
            continue;
        }
        let status = StatusCode::from_u16(response.status().as_u16()).unwrap();
        let stream = futures_util::stream::unfold(
            (guard, response.bytes_stream()),
            |(guard, mut stream)| async move {
                stream.next().await.map(|chunk| (chunk, (guard, stream)))
            },
        );
        return (status, Body::from_stream(stream)).into_response();
    }
    unreachable!()
}

struct Harness {
    _assets: tempfile::TempDir,
    counter: Arc<CountingTokenizer>,
    supervisor: KvIndexSupervisor,
    _context: zmq::Context,
    publisher: zmq::Socket,
    key: CacheKey,
    consumer: Consumer,
    backends: Vec<Backend>,
    url: String,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}
impl Drop for Harness {
    fn drop(&mut self) {
        self.supervisor.shutdown();
        for task in &self.tasks {
            task.abort();
        }
    }
}
impl Harness {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            include_bytes!("fixtures/tokenizer/completion_word_level.json"),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"tokenizer_class":"PreTrainedTokenizerFast","added_tokens_decoder":{}}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            r#"{"vocab_size":16,"is_encoder_decoder":false}"#,
        )
        .unwrap();
        let contract = directory.path().join("contract.json");
        std::fs::write(&contract, json!({"assets_path":directory.path(),"model":"base","aliases":["alias"],"supervised_immutable_workers":true}).to_string()).unwrap();
        let mut assets = load_completion_input_assets(&contract).unwrap();
        let counter = Arc::new(CountingTokenizer {
            inner: assets.tokenizer.clone(),
            encodes: AtomicUsize::new(0),
        });
        assets.tokenizer = counter.clone();
        let mut workers = Vec::<Arc<dyn Worker>>::new();
        let mut backends = Vec::new();
        let mut tasks = Vec::new();
        for _ in 0..2 {
            let state = Backend::default();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            workers.push(Arc::new(BasicWorker::new(
                format!("http://{}", listener.local_addr().unwrap()),
                WorkerType::Regular,
            )));
            let app = Router::new()
                .route("/v1/completions", post(backend))
                .with_state(state.clone());
            tasks.push(tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            }));
            backends.push(state);
        }
        let zmq_context = zmq::Context::new();
        let publisher = zmq_context.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let (supervisor, _advisory) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let key = CacheKey {
            model: Arc::from("base"),
            hash_mode: HashMode::Sha256,
            block_size: 2,
        };
        supervisor
            .register_static_worker(
                workers[1].url(),
                key.clone(),
                EventSource {
                    source: Arc::from("publisher-B"),
                    dp_rank: 0,
                    pub_endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
                    replay_endpoint: None,
                    topic: "kv".into(),
                    hwm: None,
                },
            )
            .unwrap();
        tokio::time::timeout(WAIT, async {
            loop {
                match publisher.recv_multipart(zmq::DONTWAIT) {
                    Ok(message) if message == [b"\x01kv".to_vec()] => break,
                    Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                    other => panic!("subscription handshake: {other:?}"),
                }
            }
        })
        .await
        .unwrap();
        let consumer = Consumer {
            assets: Arc::new(assets),
            cursor: supervisor.qualified_source(workers[1].url(), &key).unwrap(),
            owner: workers[1].clone(),
            workers,
            policy: Arc::new(RoundRobinPolicy::new()),
            client: reqwest::Client::new(),
            attempts: Arc::new(Mutex::new(Vec::new())),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/v1/completions", post(ingress))
            .with_state(consumer.clone());
        tasks.push(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }));
        Self {
            _assets: directory,
            counter,
            supervisor,
            _context: zmq_context,
            publisher,
            key,
            consumer,
            backends,
            url,
            tasks,
        }
    }
    fn send(&self, seq: i64, kind: &str) {
        let hash = rmpv::Value::Binary(vec![42; 32]);
        let event = match kind {
            "store" | "store-two" => rmpv::Value::Map(vec![
                ("type".into(), "BlockStored".into()),
                (
                    "block_hashes".into(),
                    rmpv::Value::Array(if kind == "store-two" {
                        vec![hash, rmpv::Value::Binary(vec![43; 32])]
                    } else {
                        vec![hash]
                    }),
                ),
                ("parent_block_hash".into(), rmpv::Value::Nil),
                (
                    "token_ids".into(),
                    rmpv::Value::Array(if kind == "store-two" {
                        vec![1.into(), 3.into(), 4.into(), 3.into()]
                    } else {
                        vec![1.into(), 3.into()]
                    }),
                ),
                ("block_size".into(), 2.into()),
                ("lora_id".into(), rmpv::Value::Nil),
                ("lora_name".into(), rmpv::Value::Nil),
                ("medium".into(), "GPU".into()),
            ]),
            "remove" => rmpv::Value::Map(vec![
                ("type".into(), "BlockRemoved".into()),
                ("block_hashes".into(), rmpv::Value::Array(vec![hash])),
                ("medium".into(), "GPU".into()),
            ]),
            "clear" => rmpv::Value::Map(vec![("type".into(), "AllBlocksCleared".into())]),
            _ => unreachable!(),
        };
        let mut payload = Vec::new();
        rmpv::encode::write_value(
            &mut payload,
            &rmpv::Value::Array(vec![1.0.into(), rmpv::Value::Array(vec![event]), 0.into()]),
        )
        .unwrap();
        self.publisher
            .send_multipart([b"kv".as_slice(), &seq.to_be_bytes(), &payload], 0)
            .unwrap();
    }
    fn query(&self) -> MatchQuery {
        MatchQuery {
            group_idx: 0,
            local_hashes: local_hashes(&[1, 3], 2, None)
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
    async fn wait_matches(&self, expected: usize) {
        tokio::time::timeout(WAIT, async {
            loop {
                if self
                    .supervisor
                    .provider(&self.key)
                    .unwrap()
                    .find_tiered_matches(&self.query())
                    .len()
                    == expected
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
    async fn request(&self, raw: &'static str) -> reqwest::Response {
        self.consumer
            .client
            .post(&self.url)
            .header("content-type", "application/json")
            .body(raw)
            .send()
            .await
            .unwrap()
    }
    async fn wait_zero(&self) {
        tokio::time::timeout(WAIT, async {
            while self
                .consumer
                .workers
                .iter()
                .any(|worker| worker.load() != 0)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}

#[tokio::test]
async fn shared_store_first_owner_retry_borrow_remove_clear_and_raw_fallback() {
    let h = Harness::new().await;
    h.send(0, "store");
    h.wait_matches(1).await;
    let raw = "{ \"model\":\"alias\", \"prompt\":\"hello world\", \"temperature\":0.2 }";
    h.backends[1].fail_once.store(true, Ordering::SeqCst);
    assert_eq!(h.request(raw).await.status(), StatusCode::OK);
    h.wait_zero().await;
    let attempts = h.consumer.attempts.lock().clone();
    assert_eq!(attempts.len(), 2);
    assert!(attempts
        .iter()
        .all(|attempt| attempt.observed_owner && attempt.source_sequence == Some(0)));
    assert!(attempts[0].token_allocation.is_some());
    assert_eq!(attempts[0].token_allocation, attempts[1].token_allocation);
    assert!(h.backends[0].received.lock().is_empty());
    assert_eq!(
        h.backends[1].received.lock().as_slice(),
        &[
            bytes::Bytes::from_static(raw.as_bytes()),
            bytes::Bytes::from_static(raw.as_bytes())
        ]
    );
    assert_eq!(h.counter.encodes.load(Ordering::SeqCst), 1);
    h.send(1, "remove");
    h.wait_matches(0).await;
    let miss = "{\"prompt\":\"world\"}";
    h.request(miss).await.bytes().await.unwrap();
    assert!(!h.consumer.attempts.lock().last().unwrap().observed_owner);
    h.send(2, "store");
    h.wait_matches(1).await;
    h.send(3, "clear");
    h.wait_matches(0).await;
    let unsupported = "{ \"prompt\":[\"hello\",\"world\"], \"cache_salt\":\"separate\" }";
    h.request(unsupported).await.bytes().await.unwrap();
    assert_eq!(
        h.consumer.attempts.lock().last().unwrap(),
        &AttemptObservation {
            token_allocation: None,
            observed_owner: false,
            source_sequence: None
        }
    );
    assert!(h.backends.iter().any(|backend| backend
        .received
        .lock()
        .iter()
        .any(|body| body.as_ref() == unsupported.as_bytes())));
    h.supervisor.on_worker_removed(h.consumer.owner.url());
    assert!(h
        .consumer
        .cursor
        .with_qualified_matches(&h.query(), |_, _| ())
        .is_none());
    h.request(raw).await.bytes().await.unwrap();
    assert!(!h.consumer.attempts.lock().last().unwrap().observed_owner);
    h.wait_zero().await;
}

#[tokio::test]
async fn ids_no_reencode_invalid_no_dispatch_and_cancel_single_guard() {
    let h = Harness::new().await;
    h.send(0, "store");
    h.wait_matches(1).await;
    let raw = "{\"prompt\":[1,3,4],\"stream\":true}";
    let mut response = h.request(raw).await;
    assert!(response.chunk().await.unwrap().is_some());
    assert_eq!(h.consumer.owner.load(), 1);
    assert_eq!(h.consumer.workers[0].load(), 0);
    assert_eq!(h.counter.encodes.load(Ordering::SeqCst), 0);
    drop(response);
    h.wait_zero().await;
    let before = h.consumer.attempts.lock().len();
    assert_eq!(
        h.request("{\"prompt\":[16]}").await.status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(h.consumer.attempts.lock().len(), before);
    h.wait_zero().await;
}

#[tokio::test]
async fn removed_first_block_with_remaining_residency_is_not_a_prefix_hit() {
    let h = Harness::new().await;
    h.send(0, "store-two");
    h.wait_matches(1).await;
    h.send(1, "remove");
    let mut query = h.query();
    query.local_hashes.push(Arc::from(
        local_hashes(&[4, 3], 2, None)[0]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
    ));
    tokio::time::timeout(WAIT, async {
        loop {
            let hits = h
                .supervisor
                .provider(&h.key)
                .unwrap()
                .find_tiered_matches(&query);
            if hits.len() == 1 && hits[0].matched_depth == 0 {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let raw = "{\"prompt\":\"hello world hello\"}";
    h.request(raw).await.bytes().await.unwrap();
    assert!(!h.consumer.attempts.lock().last().unwrap().observed_owner);
    assert!(h.backends.iter().any(|backend| backend
        .received
        .lock()
        .iter()
        .any(|body| body.as_ref() == raw.as_bytes())));
    h.wait_zero().await;
}

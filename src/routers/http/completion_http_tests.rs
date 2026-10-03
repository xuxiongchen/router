mod completion_http {
    use super::*;
    use axum::extract::State;
    use http_body_util::BodyExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

    struct CountingTokenizer {
        inner: HuggingFaceTokenizer,
        encodes: AtomicUsize,
    }

    impl Tokenizer for CountingTokenizer {
        fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            self.encodes.fetch_add(1, Ordering::SeqCst);
            self.inner.encode(text, special)
        }
        fn encode_ordinary(&self, _: &str) -> vllm_tokenizer::Result<Vec<u32>> {
            panic!("ordinary encoding is not Completion preparation")
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

    #[derive(Debug)]
    struct Observation {
        tokens: Option<Vec<u32>>,
        pointer: Option<usize>,
        text: String,
        session: String,
        urls: Vec<String>,
    }

    #[derive(Debug, Default)]
    struct ObservingPolicy {
        seen: Mutex<Vec<Observation>>,
    }

    impl LoadBalancingPolicy for ObservingPolicy {
        fn select_worker_with_headers(
            &self,
            _: &[Arc<dyn Worker>],
            _: Option<&str>,
            _: Option<&crate::policies::RequestHeaders>,
        ) -> Option<usize> {
            panic!("router must use the actual shared context boundary")
        }
        fn select_worker_with_context(
            &self,
            workers: &[Arc<dyn Worker>],
            context: &PolicyRequestContext<'_>,
        ) -> Option<usize> {
            self.seen.lock().push(Observation {
                tokens: context.token_ids.map(<[u32]>::to_vec),
                pointer: context.token_ids.map(|ids| ids.as_ptr() as usize),
                text: context.request_text.unwrap_or_default().into(),
                session: context
                    .headers
                    .and_then(|h| h.get("x-session-id"))
                    .cloned()
                    .unwrap_or_default(),
                urls: workers.iter().map(|w| w.url().into()).collect(),
            });
            (!workers.is_empty()).then_some(0)
        }
        // Exercise the existing tracked-attempt path, not a production KV policy.
        fn name(&self) -> &'static str {
            "cache_aware"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[derive(Clone)]
    struct Backend {
        mode: &'static str,
        attempts: Arc<AtomicUsize>,
        received: Arc<Mutex<Vec<(bytes::Bytes, HeaderMap)>>>,
        body_error_gate: Arc<tokio::sync::Notify>,
        health_requests: Arc<AtomicUsize>,
        health_request_gate: Arc<tokio::sync::Notify>,
    }

    async fn health(State(state): State<Backend>) -> StatusCode {
        state.health_requests.fetch_add(1, Ordering::SeqCst);
        state.health_request_gate.notify_one();
        StatusCode::OK
    }

    async fn backend(State(state): State<Backend>, request: Request) -> Response {
        let headers = request.headers().clone();
        let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        state.received.lock().push((body, headers));
        let attempt = state.attempts.fetch_add(1, Ordering::SeqCst);
        if state.mode == "headers_pending" {
            return std::future::pending().await;
        }
        if state.mode == "exhausted" || state.mode == "retry" && attempt == 0 {
            return (StatusCode::SERVICE_UNAVAILABLE, "retry").into_response();
        }
        if state.mode == "stream_pending" || state.mode == "done_pending" {
            let chunk = if state.mode == "done_pending" {
                "data: [DONE]\n\n"
            } else {
                "data: token\n\n"
            };
            let stream = futures_util::stream::once(async move {
                Ok::<_, std::io::Error>(bytes::Bytes::from_static(chunk.as_bytes()))
            })
            .chain(futures_util::stream::pending());
            return Response::new(Body::from_stream(stream));
        }
        if state.mode == "body_error" {
            let body_error_gate = state.body_error_gate.clone();
            let stream = futures_util::stream::once(async {
                Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"data: token\n\n"))
            })
            .chain(futures_util::stream::once(async move {
                // The test opens this gate only after consuming the first frame.
                body_error_gate.notified().await;
                Err(std::io::Error::other("controlled backend body failure"))
            }));
            return Response::new(Body::from_stream(stream));
        }
        (
            [("x-backend-result", "preserved")],
            "{\"choices\":[{\"text\":\"ok\"}]}",
        )
            .into_response()
    }

    struct Harness {
        router: Arc<Router>,
        worker: Arc<dyn Worker>,
        policy: Arc<ObservingPolicy>,
        tokenizer: Arc<CountingTokenizer>,
        backend: Backend,
        ingress_url: String,
        tasks: Vec<tokio::task::JoinHandle<()>>,
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            for task in &self.tasks {
                task.abort();
            }
        }
    }

    async fn harness(mode: &'static str, prepare: bool) -> Harness {
        let backend_state = Backend {
            mode,
            attempts: Arc::new(AtomicUsize::new(0)),
            received: Arc::new(Mutex::new(Vec::new())),
            body_error_gate: Arc::new(tokio::sync::Notify::new()),
            health_requests: Arc::new(AtomicUsize::new(0)),
            health_request_gate: Arc::new(tokio::sync::Notify::new()),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new()
            .route("/v1/completions", axum::routing::post(backend))
            .route("/health", axum::routing::get(health))
            .with_state(backend_state.clone());
        let backend_task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let tokenizer_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(if mode == "tokenizer_error" {
                "tests/fixtures/tokenizer/completion_missing_unk.json"
            } else {
                "tests/fixtures/tokenizer/completion_word_level.json"
            });
        let definition: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&tokenizer_path).unwrap()).unwrap();
        crate::backend::validate_completion_tokenizer_definition(&definition).unwrap();
        let tokenizer = Arc::new(CountingTokenizer {
            inner: HuggingFaceTokenizer::new_hf(&tokenizer_path).unwrap(),
            encodes: AtomicUsize::new(0),
        });
        let policy = Arc::new(ObservingPolicy::default());
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(worker_url, WorkerType::Regular));
        // Other in-flight work stays at baseline 1. No health/load monitor runs:
        // existing periodic resets are outside this request-ownership proof.
        worker.increment_load();
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(worker.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::with_default_policy_for_test(policy.clone()));
        router.retry_config = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 1,
            jitter_factor: 0.0,
            ..RetryConfig::default()
        };
        router.completion_frontend = prepare.then(|| CompletionFrontend {
            tokenizer: tokenizer.clone(),
            model_vocab_size: match mode {
                "zero_vocab" => 0,
                "encoded_vocab_error" => 3,
                _ => 16,
            },
            activation: None,
        });
        let router = Arc::new(router);
        let context = Arc::new(
            crate::server::AppContext::new(
                crate::config::RouterConfig::default(),
                Client::new(),
                8,
                None,
                Vec::new(),
            )
            .unwrap(),
        );
        let state = Arc::new(crate::server::AppState {
            router: router.clone(),
            context,
            concurrency_queue_tx: None,
            router_manager: None,
        });
        let app = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(crate::server::v1_completions),
            )
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ingress_url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
        let ingress_task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Harness {
            router,
            worker,
            policy,
            tokenizer,
            backend: backend_state,
            ingress_url,
            tasks: vec![backend_task, ingress_task],
        }
    }

    async fn wait_load(worker: &Arc<dyn Worker>, expected: usize) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while worker.load() != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    async fn prepared_route(h: &Harness, stream: bool) -> Response {
        let raw = bytes::Bytes::from(
            serde_json::json!({"prompt": "hello world", "stream": stream}).to_string(),
        );
        let body = serde_json::from_slice(&raw).unwrap();
        h.router.route_prepared_completion(None, &body, &raw).await
    }

    #[tokio::test]
    async fn ingress_prepares_once_retries_borrowed_tokens_and_preserves_raw_body() {
        let h = harness("retry", true).await;
        let raw = "{ \"temperature\":0.25, \"prompt\":\"ignored\", \"prompt\":\"hello world\", \"max_tokens\":7, \"stream\":false }\n";
        let response = Client::new()
            .post(&h.ingress_url)
            .header("content-type", "application/json")
            .header("x-session-id", "session-a")
            .body(raw)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.headers()["x-backend-result"], "preserved");
        assert_eq!(
            response.text().await.unwrap(),
            "{\"choices\":[{\"text\":\"ok\"}]}"
        );
        assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 1);
        assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 2);
        let received = h.backend.received.lock();
        for (body, headers) in received.iter() {
            assert_eq!(body.as_ref(), raw.as_bytes());
            assert_eq!(headers["x-session-id"], "session-a");
            let value: serde_json::Value = serde_json::from_slice(body).unwrap();
            assert_eq!(value["temperature"], 0.25);
            assert_eq!(value["max_tokens"], 7);
        }
        let seen = h.policy.seen.lock();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].tokens, Some(vec![1, 3, 4]));
        assert_eq!(seen[0].pointer, seen[1].pointer);
        assert_eq!(seen[0].session, "session-a");
        assert_eq!(seen[0].text, "hello world");
        assert_eq!(h.worker.load(), 1);
    }

    #[tokio::test]
    async fn ingress_ids_do_not_encode_and_invalid_inputs_do_not_dispatch() {
        let h = harness("ok", true).await;
        let client = Client::new();
        for (raw, status) in [
            ("{\"prompt\":[0,2,15]}", reqwest::StatusCode::OK),
            ("{\"prompt\":[-1]}", reqwest::StatusCode::BAD_REQUEST),
            ("{\"prompt\":[16]}", reqwest::StatusCode::BAD_REQUEST),
            (
                "{\"prompt\":[-1],\"suffix\":\"world\"}",
                reqwest::StatusCode::BAD_REQUEST,
            ),
            (
                "{\"prompt\":[16],\"lora_path\":\"adapter\"}",
                reqwest::StatusCode::BAD_REQUEST,
            ),
            (
                "{\"prompt\":[-1],\"unknown_extension\":null}",
                reqwest::StatusCode::BAD_REQUEST,
            ),
            (
                "{\"prompt\":[[3],[-1]],\"suffix\":\"world\"}",
                reqwest::StatusCode::BAD_REQUEST,
            ),
            (
                "{\"prompt\":[[16]],\"unknown_extension\":null}",
                reqwest::StatusCode::BAD_REQUEST,
            ),
            (
                "{\"prompt\":[\"hello\"],\"suffix\":\"world\",\"add_special_tokens\":null}",
                reqwest::StatusCode::BAD_REQUEST,
            ),
            ("{\"prompt\":", reqwest::StatusCode::BAD_REQUEST),
            ("{\"prompt\":{}}", reqwest::StatusCode::UNPROCESSABLE_ENTITY),
        ] {
            let response = client
                .post(&h.ingress_url)
                .header("content-type", "application/json")
                .body(raw)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), status, "{raw}");
            let _ = response.bytes().await.unwrap();
        }
        assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 0);
        assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 1);
        assert_eq!(h.policy.seen.lock()[0].tokens, Some(vec![0, 2, 15]));
        assert_eq!(h.worker.load(), 1);
    }

    #[tokio::test]
    async fn ingress_unsupported_inputs_retry_original_bytes_without_tokens_or_encoding() {
        for raw in [
            "{ \"prompt\":\"hello\", \"suffix\":\"world\" }\n",
            "{ \"prompt\":[0,2,15], \"suffix\":\"world\" }\n",
            "{ \"prompt\":[\"hello\",\"world\"] }\n",
            "{ \"prompt\":[[3],[4,15]] }\n",
            "{ \"prompt\":\"hello\", \"unknown_extension\":null }\n",
        ] {
            let h = harness("retry", true).await;
            let response = Client::new()
                .post(&h.ingress_url)
                .header("content-type", "application/json")
                .header("x-session-id", "fallback-session")
                .body(raw)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK, "{raw}");
            assert_eq!(
                response.text().await.unwrap(),
                "{\"choices\":[{\"text\":\"ok\"}]}"
            );
            assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 0);
            assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 2);
            for (body, headers) in h.backend.received.lock().iter() {
                assert_eq!(body.as_ref(), raw.as_bytes());
                assert_eq!(headers["x-session-id"], "fallback-session");
            }
            let seen = h.policy.seen.lock();
            assert_eq!(seen.len(), 2);
            assert!(seen
                .iter()
                .all(|context| context.tokens.is_none() && context.pointer.is_none()));
            assert!(seen
                .iter()
                .all(|context| context.session == "fallback-session"));
            assert_eq!(h.worker.load(), 1);
        }
    }

    #[tokio::test]
    async fn ingress_configuration_failures_never_fall_back() {
        for (mode, raw, encodes) in [
            ("encoded_vocab_error", "{\"prompt\":\"hello world\"}", 1),
            (
                "zero_vocab",
                "{\"prompt\":\"hello\",\"suffix\":\"world\"}",
                0,
            ),
        ] {
            let h = harness(mode, true).await;
            let response = Client::new()
                .post(&h.ingress_url)
                .header("content-type", "application/json")
                .body(raw)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
            let error = response.text().await.unwrap();
            assert!(
                error.contains(if mode == "zero_vocab" {
                    "nonzero"
                } else {
                    "outside"
                }),
                "{error}"
            );
            assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), encodes);
            assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 0);
            assert!(h.policy.seen.lock().is_empty());
            assert_eq!(h.worker.load(), 1);
        }
    }

    #[tokio::test]
    async fn unsupported_completion_cannot_bypass_mixed_or_grpc_pool_failure() {
        for mixed in [true, false] {
            let h = harness("ok", true).await;
            if !mixed {
                h.router
                    .worker_registry
                    .remove_by_url(h.worker.url())
                    .unwrap();
            }
            h.router.worker_registry.register(Arc::new(BasicWorker::new(
                "grpc://127.0.0.1:1".into(),
                WorkerType::Regular,
            )));
            let response = tokio::time::timeout(
                Duration::from_secs(3),
                Client::new()
                    .post(&h.ingress_url)
                    .json(&serde_json::json!({"prompt": "hello", "suffix": "world"}))
                    .send(),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(
                response.status(),
                if mixed {
                    reqwest::StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    reqwest::StatusCode::BAD_REQUEST
                }
            );
            let _ = response.bytes().await.unwrap();
            assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 0);
            assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 0);
            assert!(h.policy.seen.lock().is_empty());
            assert_eq!(h.worker.load(), 1);
        }
    }

    #[tokio::test]
    async fn ingress_tokenizer_execution_failure_is_service_unavailable_without_dispatch() {
        let h = harness("tokenizer_error", true).await;
        let response = Client::new()
            .post(&h.ingress_url)
            .json(&serde_json::json!({"prompt": "unknown token"}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let error = response.text().await.unwrap();
        assert!(
            error.starts_with("completion tokenization failed:"),
            "{error}"
        );
        assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 1);
        assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 0);
        assert!(h.policy.seen.lock().is_empty());
        assert_eq!(h.worker.load(), 1);
    }

    #[tokio::test]
    async fn prepared_ingress_selects_only_available_workers_for_requested_model() {
        let h = harness("ok", true).await;
        let other_backend = harness("ok", false).await;
        h.router
            .worker_registry
            .remove_by_url(h.worker.url())
            .unwrap();
        let wanted: Arc<dyn Worker> = Arc::new(
            BasicWorker::new(h.worker.url().into(), WorkerType::Regular)
                .with_labels(HashMap::from([("model_id".into(), "wanted".into())])),
        );
        let other: Arc<dyn Worker> = Arc::new(
            BasicWorker::new(other_backend.worker.url().into(), WorkerType::Regular)
                .with_labels(HashMap::from([("model_id".into(), "other".into())])),
        );
        wanted.increment_load();
        other.increment_load();
        h.router.worker_registry.register(wanted.clone());
        h.router.worker_registry.register(other.clone());
        let response = Client::new()
            .post(&h.ingress_url)
            .json(&serde_json::json!({"model": "wanted", "prompt": "hello world"}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let _ = response.bytes().await.unwrap();
        assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 1);
        assert_eq!(other_backend.backend.attempts.load(Ordering::SeqCst), 0);
        assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 1);
        {
            let seen = h.policy.seen.lock();
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0].urls, [wanted.url()]);
            assert_eq!(seen[0].tokens, Some(vec![1, 3, 4]));
        }
        assert_eq!(wanted.load(), 1);
        assert_eq!(other.load(), 1);

        let raw = "{ \"model\":\"wanted\", \"prompt\":\"hello\", \"suffix\":\"world\" }\n";
        let response = Client::new()
            .post(&h.ingress_url)
            .header("content-type", "application/json")
            .header("x-session-id", "model-fallback")
            .body(raw)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let _ = response.bytes().await.unwrap();
        assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 2);
        assert_eq!(other_backend.backend.attempts.load(Ordering::SeqCst), 0);
        assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 1);
        let received = h.backend.received.lock();
        assert_eq!(received[1].0.as_ref(), raw.as_bytes());
        assert_eq!(received[1].1["x-session-id"], "model-fallback");
        let seen = h.policy.seen.lock();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].urls, [wanted.url()]);
        assert!(seen[1].tokens.is_none());
        assert_eq!(seen[1].session, "model-fallback");
        assert_eq!(wanted.load(), 1);
        assert_eq!(other.load(), 1);
    }

    #[tokio::test]
    async fn missing_frontend_keeps_legacy_typed_duplicate_validation_and_absent_tokens() {
        let h = harness("ok", false).await;
        let client = Client::new();
        let response = client
            .post(&h.ingress_url)
            .header("content-type", "application/json")
            .body("{\"prompt\":\"hello\",\"prompt\":\"world\"}")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(h.backend.attempts.load(Ordering::SeqCst), 0);
        let response = client
            .post(&h.ingress_url)
            .json(&serde_json::json!({"prompt": "hello world"}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let _ = response.bytes().await.unwrap();
        assert!(h.policy.seen.lock()[0].tokens.is_none());
        let raw = bytes::Bytes::from_static(b"{\"prompt\":\"hello\"}");
        let body = serde_json::from_slice(&raw).unwrap();
        let response = h.router.route_prepared_completion(None, &body, &raw).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(h.tokenizer.encodes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cancellation_before_backend_headers_releases_only_selected_attempt() {
        let h = harness("headers_pending", true).await;
        let router = h.router.clone();
        let task = tokio::spawn(async move {
            let raw = bytes::Bytes::from_static(b"{\"prompt\":\"hello world\"}");
            let body = serde_json::from_slice(&raw).unwrap();
            router.route_prepared_completion(None, &body, &raw).await
        });
        tokio::time::timeout(Duration::from_secs(3), async {
            while h.backend.attempts.load(Ordering::SeqCst) != 1 || h.worker.load() != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        wait_load(&h.worker, 1).await;
    }

    #[tokio::test]
    async fn health_monitor_preserves_two_live_completion_loads() {
        let h = harness("stream_pending", true).await;
        h.worker.decrement_load();
        assert_eq!(h.worker.load(), 0);
        let client = Client::new();
        let mut first = client
            .post(&h.ingress_url)
            .json(&serde_json::json!({"prompt": "hello world", "stream": true}))
            .send()
            .await
            .unwrap();
        let mut second = client
            .post(&h.ingress_url)
            .json(&serde_json::json!({"prompt": "hello world", "stream": true}))
            .send()
            .await
            .unwrap();
        for response in [&mut first, &mut second] {
            let chunk = response.chunk().await.unwrap().unwrap();
            assert!(chunk.windows(11).any(|window| window == b"data: token"));
        }
        assert_eq!(h.worker.load(), 2);
        let checker = h.router.worker_registry.start_health_checker(1);
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                let notified = h.backend.health_request_gate.notified();
                // The eleventh request proves the tenth completed, including
                // its periodic load-reset block in the old implementation.
                if h.backend.health_requests.load(Ordering::SeqCst) >= 11 {
                    break;
                }
                notified.await;
            }
        })
        .await
        .unwrap();
        assert_eq!(h.worker.load(), 2);
        drop(first);
        wait_load(&h.worker, 1).await;
        drop(second);
        wait_load(&h.worker, 0).await;
        checker.shutdown().await;
    }

    #[tokio::test]
    async fn streaming_drop_releases_old_arc_not_same_url_replacement() {
        let h = harness("stream_pending", true).await;
        let mut response = Client::new()
            .post(&h.ingress_url)
            .json(&serde_json::json!({"prompt": "hello world", "stream": true}))
            .send()
            .await
            .unwrap();
        let chunk = response.chunk().await.unwrap().unwrap();
        assert!(chunk.windows(11).any(|window| window == b"data: token"));
        assert_eq!(h.worker.load(), 2);
        let removed = h
            .router
            .worker_registry
            .remove_by_url(h.worker.url())
            .unwrap();
        assert!(Arc::ptr_eq(&removed, &h.worker));
        let replacement: Arc<dyn Worker> =
            Arc::new(BasicWorker::new(h.worker.url().into(), WorkerType::Regular));
        replacement.increment_load();
        h.router.worker_registry.register(replacement.clone());
        drop(response);
        wait_load(&h.worker, 1).await;
        assert_eq!(replacement.load(), 1);
    }

    #[tokio::test]
    async fn stream_eof_error_and_retry_exhaustion_preserve_other_inflight_load() {
        for mode in ["ok", "body_error", "exhausted"] {
            let h = harness(mode, true).await;
            let mut response = prepared_route(&h, true).await;
            if mode == "exhausted" {
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            }
            if mode == "body_error" {
                let chunk = response.body_mut().frame().await.unwrap().unwrap();
                assert!(chunk.data_ref().is_some());
                h.backend.body_error_gate.notify_one();
            }
            let result = to_bytes(response.into_body(), usize::MAX).await;
            if mode == "body_error" {
                assert!(result.is_err());
            }
            wait_load(&h.worker, 1).await;
        }
    }

    #[tokio::test]
    async fn done_marker_does_not_release_before_producer_termination() {
        let h = harness("done_pending", true).await;
        let mut response = prepared_route(&h, true).await;
        let chunk = response.body_mut().frame().await.unwrap().unwrap();
        assert!(chunk
            .data_ref()
            .unwrap()
            .windows(12)
            .any(|window| window == b"data: [DONE]"));
        assert_eq!(h.worker.load(), 2);
        drop(response);
        wait_load(&h.worker, 1).await;
    }

    #[test]
    fn shared_context_selection_keeps_model_scope_and_availability() {
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        let policy = Arc::new(ObservingPolicy::default());
        router.policy_registry =
            Arc::new(PolicyRegistry::with_default_policy_for_test(policy.clone()));
        for (url, model, healthy) in [
            ("http://chosen", "wanted", true),
            ("http://unhealthy", "wanted", false),
            ("http://other", "other", true),
        ] {
            let worker = BasicWorker::new(url.into(), WorkerType::Regular)
                .with_labels(HashMap::from([("model_id".into(), model.into())]));
            worker.set_healthy(healthy);
            router.worker_registry.register(Arc::new(worker));
        }
        let ids = [1, 3];
        let selected = router
            .select_worker_for_model(Some("wanted"), Some("hello"), None, Some(&ids))
            .unwrap();
        assert_eq!(selected.url(), "http://chosen");
        let seen = policy.seen.lock();
        assert_eq!(seen[0].urls, ["http://chosen"]);
        assert_eq!(seen[0].pointer, Some(ids.as_ptr() as usize));
    }
}

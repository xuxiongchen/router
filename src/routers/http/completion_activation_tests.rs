mod completion_activation {
    use super::*;
    use crate::backend::completion_activation::{
        load_completion_input_assets, tests::{assets_fixture, effective_fixture},
    };
    use axum::extract::State;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct ControlWorker {
        tokenizer: Arc<dyn vllm_tokenizer::Tokenizer>,
        invalid: &'static str,
        probes: Arc<AtomicUsize>,
        requests: Arc<Mutex<Vec<bytes::Bytes>>>,
    }

    async fn metadata(
        State(worker): State<ControlWorker>,
        request: Request,
    ) -> Json<serde_json::Value> {
        let result = match request.uri().path() {
            "/version" => {
                serde_json::json!({"version": if worker.invalid == "version" { "0.28.0" } else { "0.29.0" }})
            }
            "/v1/models" => {
                serde_json::json!({"data": [{"id":"base","root":"/mock/base","parent":null},{"id":"alias","root":"/mock/base","parent":null}]})
            }
            "/tokenizer_info" => {
                serde_json::json!({"tokenizer_class":"TokenizerPoolCachedPreTrainedTokenizerFast"})
            }
            "/server_info" => serde_json::json!({"vllm_config": {
                "model_config": {"model":"/mock/base","tokenizer_mode":"hf","skip_tokenizer_init":false,"trust_remote_code":false,"io_processor_plugin":null,"hf_overrides":{}},
                "parallel_config":{"data_parallel_size":1},"lora_config":null,"speculative_config":null},
                "vllm_env":{"VLLM_USE_FASTOKENS":false},
                "system_env":{"pip_packages":if worker.invalid == "transformers" {"transformers==5.16.0"} else {"transformers==5.17.0"}}}),
            _ => unreachable!(),
        };
        Json(result)
    }

    async fn tokenize(
        State(worker): State<ControlWorker>,
        Json(request): Json<serde_json::Value>,
    ) -> Json<serde_json::Value> {
        worker.probes.fetch_add(1, Ordering::SeqCst);
        let mut tokens = worker
            .tokenizer
            .encode(
                request["prompt"].as_str().unwrap(),
                request["add_special_tokens"].as_bool().unwrap_or(true),
            )
            .unwrap();
        if worker.invalid == "tokens" {
            tokens.push(15);
        }
        Json(serde_json::json!({"count":tokens.len(),"tokens":tokens,"max_model_len":4096}))
    }

    async fn completion(State(worker): State<ControlWorker>, request: Request) -> Response {
        let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        let streaming =
            serde_json::from_slice::<serde_json::Value>(&body).unwrap()["stream"] == true;
        worker.requests.lock().push(body);
        if streaming {
            Response::new(Body::from_stream(futures_util::stream::pending::<
                Result<bytes::Bytes, std::io::Error>,
            >()))
        } else {
            Json(serde_json::json!({"choices":[{"text":"ok"}]})).into_response()
        }
    }

    struct Harness {
        context: Arc<crate::server::AppContext>,
        worker: ControlWorker,
        _assets: tempfile::TempDir,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn harness(invalid: &'static str, enabled: bool, effective: bool) -> Harness {
        let (assets, path) = assets_fixture();
        if effective {
            effective_fixture(assets.path(), &path);
        }
        let input = load_completion_input_assets(&path).unwrap();
        let worker = ControlWorker {
            tokenizer: input.tokenizer,
            invalid,
            probes: Arc::new(AtomicUsize::new(0)),
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new()
            .route("/health", axum::routing::get(|| async { StatusCode::OK }))
            .route("/version", axum::routing::get(metadata))
            .route("/v1/models", axum::routing::get(metadata))
            .route("/server_info", axum::routing::get(metadata))
            .route("/tokenizer_info", axum::routing::get(metadata))
            .route("/tokenize", axum::routing::post(tokenize))
            .route("/v1/completions", axum::routing::post(completion))
            .with_state(worker.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let config = crate::config::RouterConfig {
            mode: crate::config::RoutingMode::Regular {
                worker_urls: vec![url],
            },
            completion_input_contract: enabled.then_some(path),
            policy: crate::config::PolicyConfig::CacheAware {
                cache_threshold: 0.3,
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                eviction_interval_secs: 120,
                max_tree_size: 1024,
            },
            disable_retries: true,
            ..Default::default()
        };
        Harness {
            context: Arc::new(
                crate::server::AppContext::new(config, Client::new(), 8, None, vec![]).unwrap(),
            ),
            worker,
            _assets: assets,
            task,
        }
    }

    async fn dispatch(router: &Router, request: serde_json::Value) -> Response {
        let raw = bytes::Bytes::from(request.to_string());
        let body = serde_json::from_value(request).unwrap();
        router.route_prepared_completion(None, &body, &raw).await
    }

    #[tokio::test]
    async fn completion_factory_installs_default_off_or_verified_frontend() {
        use crate::routers::factory::RouterFactory;
        let off = harness("version", false, false).await;
        let router = RouterFactory::create_router(&off.context).await.unwrap();
        assert!(!router
            .as_any()
            .downcast_ref::<Router>()
            .unwrap()
            .has_completion_frontend());
        assert_eq!(off.worker.probes.load(Ordering::SeqCst), 0);
        let on = harness("", true, false).await;
        let router = RouterFactory::create_router(&on.context).await.unwrap();
        let router = router.as_any().downcast_ref::<Router>().unwrap();
        assert!(router.has_completion_frontend());
        assert_eq!(on.worker.probes.load(Ordering::SeqCst), 12);
        let response = dispatch(
            router,
            serde_json::json!({"model":"alias","prompt":"hello world"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        assert_eq!(on.worker.requests.lock().len(), 1);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&on.worker.requests.lock()[0]).unwrap()
                ["model"],
            "alias"
        );
        assert_eq!(on.worker.probes.load(Ordering::SeqCst), 12);
        for body in [
            serde_json::json!({"model":"other","prompt":[999]}),
            serde_json::json!({"model":"alias","prompt":""}),
        ] {
            let response = dispatch(router, body).await;
            assert!(matches!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE | StatusCode::BAD_REQUEST
            ));
        }
        let response = dispatch(
            router,
            serde_json::json!({"model":"alias","prompt":[999],"lora_path":"adapter"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        assert_eq!(on.worker.requests.lock().len(), 2);
    }

    #[tokio::test]
    async fn completion_factory_rejects_cold_worker_mismatch() {
        for invalid in ["version", "tokens"] {
            let h = harness(invalid, true, false).await;
            assert!(
                crate::routers::factory::RouterFactory::create_router(&h.context)
                    .await
                    .is_err()
            );
            assert!(h.context.worker_registry.get_all().is_empty());
            assert!(h.worker.requests.lock().is_empty());
        }
    }

    #[tokio::test]
    async fn completion_factory_effective_contract_keeps_probes_and_raw_bytes() {
        let h = harness("", true, true).await;
        let router = crate::routers::factory::RouterFactory::create_router(&h.context)
            .await
            .unwrap();
        let router = router.as_any().downcast_ref::<Router>().unwrap();
        assert!(router.has_completion_frontend());
        assert_eq!(h.worker.probes.load(Ordering::SeqCst), 12);
        let raw = bytes::Bytes::from_static(b" { \"model\": \"alias\", \"prompt\": \"hello world\", \"max_tokens\": 1 } \n");
        let request = serde_json::from_slice(&raw).unwrap();
        let response = router.route_prepared_completion(None, &request, &raw).await;
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        assert_eq!(h.worker.requests.lock().as_slice(), &[raw]);
        assert_eq!(h.worker.probes.load(Ordering::SeqCst), 12);
        assert_eq!(h.context.worker_registry.get_all()[0].load(), 0);
    }

    #[tokio::test]
    async fn completion_factory_effective_contract_rejects_remote_behavior_or_version() {
        for invalid in ["tokens", "transformers"] {
            let h = harness(invalid, true, true).await;
            assert!(crate::routers::factory::RouterFactory::create_router(&h.context)
                .await.is_err());
            assert!(h.context.worker_registry.get_all().is_empty());
            assert!(h.worker.requests.lock().is_empty());
        }
    }

    #[tokio::test]
    async fn completion_factory_closes_new_admission_without_resetting_active_guard() {
        let h = harness("", true, false).await;
        let router = crate::routers::factory::RouterFactory::create_router(&h.context)
            .await
            .unwrap();
        let router = router.as_any().downcast_ref::<Router>().unwrap();
        let worker = h.context.worker_registry.get_all().pop().unwrap();
        let active = dispatch(
            router,
            serde_json::json!({"model":"base","prompt":"hello","stream":true}),
        )
        .await;
        assert_eq!(active.status(), StatusCode::OK);
        assert_eq!(worker.load(), 1);
        h.context.worker_registry.notify_worker_state_change();
        assert_eq!(
            dispatch(router, serde_json::json!({"model":"base","prompt":"hello"}))
                .await
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(worker.load(), 1);
        assert_eq!(h.worker.requests.lock().len(), 1);
        drop(active);
        tokio::task::yield_now().await;
        assert_eq!(worker.load(), 0);
        assert_eq!(
            dispatch(
                router,
                serde_json::json!({"model":"alias","prompt":"hello","suffix":"x"})
            )
            .await
            .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}

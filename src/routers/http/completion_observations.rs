//! Default-off affinity over shared, attachment-fenced event observations.
//! This is a local proposal, not remote engine admission or an M7 cost model.
//! Observed blocks never grant verified reusable-token or physical cache credit.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Deserialize;

use crate::core::{Worker, WorkerLoadGuard, WorkerRegistry};
use crate::kv_index::{
    subscriber::{local_hashes, EventSource},
    CacheKey, HashMode, KvIndexSupervisor, MatchQuery, ObservationGeneration,
    QualifiedSourceCursor, StorageTier,
};
use crate::policies::PolicyRequestContext;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mapping {
    block_size: u32,
    sources: Vec<SourceMapping>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMapping {
    worker_url: String,
    source_id: String,
    pub_endpoint: String,
    #[serde(default)]
    replay_endpoint: Option<String>,
    topic: String,
}

struct BoundSource {
    worker: Arc<dyn Worker>,
    cursor: QualifiedSourceCursor,
}

/// Owns the existing supervisor; cursors cannot outlive their attachments here.
pub(super) struct CompletionObservations {
    supervisor: KvIndexSupervisor,
    sources: Vec<BoundSource>,
    block_size: u32,
    /// Serializes selection plus the existing load reservation, not HTTP work.
    next_tie: Mutex<usize>,
    diagnostic_task: tokio::task::JoinHandle<()>,
}

impl Drop for CompletionObservations {
    fn drop(&mut self) {
        self.diagnostic_task.abort();
    }
}

impl std::fmt::Debug for CompletionObservations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionObservations")
            .field("sources", &self.sources.len())
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

impl CompletionObservations {
    pub(super) fn load(
        path: &Path,
        model: &str,
        worker_urls: &[String],
        registry: &WorkerRegistry,
        client: reqwest::Client,
    ) -> Result<Self, String> {
        let mapping: Mapping =
            serde_json::from_slice(&std::fs::read(path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        mapping.validate(worker_urls)?;
        let key = CacheKey {
            model: Arc::from(model),
            hash_mode: HashMode::Sha256,
            block_size: mapping.block_size,
        };
        let (supervisor, mut signals) = KvIndexSupervisor::new(client, key.hash_mode);
        let mut sources = Vec::with_capacity(mapping.sources.len());
        for source in mapping.sources {
            let worker = registry
                .get_by_url(&source.worker_url)
                .ok_or("KV observation mapping missing actual registered worker")?;
            supervisor.register_static_worker(
                &source.worker_url,
                key.clone(),
                EventSource {
                    source: Arc::from(source.source_id),
                    dp_rank: 0,
                    pub_endpoint: source.pub_endpoint,
                    replay_endpoint: source.replay_endpoint,
                    topic: source.topic,
                    hwm: None,
                },
            )?;
            let cursor = supervisor
                .qualified_source(&source.worker_url, &key)
                .ok_or("KV observation attachment lacks qualification cursor")?;
            sources.push(BoundSource { worker, cursor });
        }
        // Diagnostics may be dropped; selection safety reads only cursor fences.
        let diagnostic_task = tokio::spawn(async move {
            while let Some(signal) = signals.recv().await {
                tracing::debug!(
                    event = "kv_observation_ingestion",
                    ?signal,
                    "kv_observation_ingestion"
                );
            }
        });
        Ok(Self {
            supervisor,
            sources,
            block_size: mapping.block_size,
            next_tie: Mutex::new(0),
            diagnostic_task,
        })
    }

    /// Retires the shared attachment without disturbing already reserved bodies.
    pub(super) fn retire_worker(&self, worker_url: &str) {
        self.supervisor.on_worker_removed(worker_url);
    }

    /// Evaluates every legal candidate, then rechecks only the winner under its
    /// publication fence. No nested cursor locks and no asynchronous reservation.
    pub(super) fn select(
        &self,
        registry: &WorkerRegistry,
        available: &[Arc<dyn Worker>],
        context: &PolicyRequestContext<'_>,
        fallback: impl FnOnce() -> Option<Arc<dyn Worker>>,
    ) -> Option<(Arc<dyn Worker>, WorkerLoadGuard<'static>)> {
        let mut next_tie = self.next_tie.lock();
        let query = context.token_ids.map(|tokens| {
            let complete = tokens.len() / self.block_size as usize * self.block_size as usize;
            MatchQuery {
                group_idx: 0,
                local_hashes: local_hashes(&tokens[..complete], self.block_size, None)
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
        });
        let mut snapshots: Vec<Option<(u32, ObservationGeneration)>> = self
            .sources
            .iter()
            .map(|source| {
                if !available
                    .iter()
                    .any(|worker| Arc::ptr_eq(worker, &source.worker))
                    || !current_worker(registry, &source.worker)
                {
                    return None;
                }
                source
                    .cursor
                    .with_qualified_matches(query.as_ref()?, |hits, generation| {
                        (
                            hits.iter().map(|hit| hit.matched_depth).max().unwrap_or(0),
                            generation,
                        )
                    })
            })
            .collect();
        if let Some(query) = query.as_ref() {
            // Each failed recheck removes one candidate: bounded by source count.
            while let Some(index) = best_candidate(&snapshots, &self.sources, *next_tie) {
                let (depth, generation) = snapshots[index]?;
                let source = &self.sources[index];
                let reservation = source
                    .cursor
                    .with_qualified_matches(query, |hits, current| {
                        if current != generation
                            || hits.iter().map(|hit| hit.matched_depth).max().unwrap_or(0) != depth
                            || !current_worker(registry, &source.worker)
                        {
                            return None;
                        }
                        Some(WorkerLoadGuard::new_owned(source.worker.clone()))
                    })
                    .flatten();
                if let Some(guard) = reservation {
                    *next_tie = (index + 1) % self.sources.len();
                    diagnostic(context, source.worker.url(), depth, Some(generation));
                    return Some((source.worker.clone(), guard));
                }
                snapshots[index] = None;
            }
        }
        let worker = fallback()?;
        if !current_worker(registry, &worker) {
            return None;
        }
        let guard = WorkerLoadGuard::new_owned(worker.clone());
        diagnostic(context, worker.url(), 0, None);
        Some((worker, guard))
    }
}

fn current_worker(registry: &WorkerRegistry, worker: &Arc<dyn Worker>) -> bool {
    worker.is_available()
        && registry
            .get_by_url(worker.url())
            .is_some_and(|current| Arc::ptr_eq(&current, worker))
}

fn best_candidate(
    snapshots: &[Option<(u32, ObservationGeneration)>],
    sources: &[BoundSource],
    start: usize,
) -> Option<usize> {
    let mut best = None;
    for offset in 0..sources.len() {
        let index = (start + offset) % sources.len();
        let Some((depth, _)) = snapshots[index].filter(|(depth, _)| *depth > 0) else {
            continue;
        };
        let load = sources[index].worker.load();
        if best.is_none_or(|(_, best_depth, best_load)| {
            depth > best_depth || (depth == best_depth && load < best_load)
        }) {
            best = Some((index, depth, load));
        }
    }
    best.map(|(index, _, _)| index)
}

fn diagnostic(
    context: &PolicyRequestContext<'_>,
    worker: &str,
    blocks: u32,
    generation: Option<ObservationGeneration>,
) {
    tracing::debug!(
        event = "completion_kv_observation",
        request_id = context
            .headers
            .and_then(|headers| headers.get("x-request-id"))
            .map(String::as_str)
            .unwrap_or(""),
        selected_worker = worker,
        observed_blocks = blocks,
        local_incarnation = generation.map(|generation| generation.incarnation),
        local_sequence = generation.map(|generation| generation.last_seq),
        exact_cache = false,
        verified_reusable_tokens = 0,
        "completion_kv_observation"
    );
}

impl Mapping {
    fn validate(&self, workers: &[String]) -> Result<(), String> {
        let configured: HashSet<_> = workers.iter().map(String::as_str).collect();
        let mut urls = HashSet::new();
        let mut sources = HashSet::new();
        let mut endpoints = HashSet::new();
        if self.block_size == 0 || self.sources.len() != workers.len() {
            return Err(
                "KV observations require nonzero block_size and one source per worker".into(),
            );
        }
        for source in &self.sources {
            if !configured.contains(source.worker_url.as_str())
                || !urls.insert(source.worker_url.as_str())
                || source.source_id.trim().is_empty()
                || !sources.insert(source.source_id.as_str())
                || !valid_endpoint(&source.pub_endpoint)
                || !endpoints.insert(source.pub_endpoint.as_str())
                || source.replay_endpoint.as_ref().is_some_and(|endpoint| {
                    !valid_endpoint(endpoint) || !endpoints.insert(endpoint.as_str())
                })
            {
                return Err("KV observations require exact worker coverage, unique source IDs and concrete unique TCP endpoints".into());
            }
        }
        Ok(())
    }
}

fn valid_endpoint(endpoint: &str) -> bool {
    reqwest::Url::parse(endpoint).is_ok_and(|url| {
        url.scheme() == "tcp"
            && url.host_str().is_some_and(|host| {
                host != "*"
                    && !host
                        .trim_matches(['[', ']'])
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|address| address.is_unspecified())
            })
            && url.port().is_some_and(|port| port != 0)
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && url.path().is_empty()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use serde_json::json;
    use std::time::Duration;

    fn mapping() -> serde_json::Value {
        json!({"block_size":2,"sources":[
            {"worker_url":"http://worker-a:8000","source_id":"source-a","pub_endpoint":"tcp://127.0.0.1:5557","topic":"kv"},
            {"worker_url":"http://worker-b:8000","source_id":"source-b","pub_endpoint":"tcp://127.0.0.1:5558","topic":"kv"}
        ]})
    }

    #[test]
    fn mapping_requires_exact_coverage_and_unique_identity() {
        let workers = vec!["http://worker-a:8000".into(), "http://worker-b:8000".into()];
        assert!(serde_json::from_value::<Mapping>(mapping())
            .unwrap()
            .validate(&workers)
            .is_ok());
        for (pointer, value) in [
            ("/block_size", json!(0)),
            ("/sources/1/worker_url", json!("http://worker-a:8000")),
            ("/sources/1/worker_url", json!("http://worker-c:8000")),
            ("/sources/1/source_id", json!("source-a")),
            ("/sources/1/source_id", json!(" ")),
            ("/sources/1/pub_endpoint", json!("tcp://127.0.0.1:5557")),
            ("/sources/1/pub_endpoint", json!("tcp://*:5558")),
        ] {
            let mut invalid = mapping();
            *invalid.pointer_mut(pointer).unwrap() = value;
            assert!(
                serde_json::from_value::<Mapping>(invalid)
                    .unwrap()
                    .validate(&workers)
                    .is_err(),
                "accepted {pointer}"
            );
        }
        let mut unknown = mapping();
        unknown["exact_cache"] = json!(true);
        assert!(serde_json::from_value::<Mapping>(unknown).is_err());
    }

    #[test]
    fn mapping_only_accepts_concrete_connect_endpoints() {
        for endpoint in [
            "http://127.0.0.1:5557",
            "tcp://0.0.0.0:5557",
            "tcp://[::]:5557",
            "tcp://[0:0:0:0:0:0:0:0]:5557",
            "tcp://127.0.0.1:0",
            "tcp://user@127.0.0.1:5557",
            "tcp://127.0.0.1:5557/path",
            "tcp://127.0.0.1:5557?x=1",
        ] {
            assert!(!valid_endpoint(endpoint), "accepted {endpoint}");
        }
        assert!(valid_endpoint("tcp://127.0.0.1:5557"));
    }

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

        async fn subscribed(&self) {
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if self
                        .socket
                        .recv_multipart(zmq::DONTWAIT)
                        .is_ok_and(|frames| frames == vec![vec![1, b'k', b'v']])
                    {
                        return;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }

        fn send(&self, seq: i64, depth: usize) {
            let event = rmpv::Value::Map(vec![
                ("type".into(), "BlockStored".into()),
                (
                    "block_hashes".into(),
                    rmpv::Value::Array(
                        (0..depth)
                            .map(|block| rmpv::Value::Binary(vec![block as u8 + 1; 32]))
                            .collect(),
                    ),
                ),
                ("parent_block_hash".into(), rmpv::Value::Nil),
                (
                    "token_ids".into(),
                    rmpv::Value::Array(
                        (1..=depth * 2)
                            .map(|id| rmpv::Value::from(id as u64))
                            .collect(),
                    ),
                ),
                ("block_size".into(), 2.into()),
                ("lora_id".into(), rmpv::Value::Nil),
                ("lora_name".into(), rmpv::Value::Nil),
                ("medium".into(), "GPU".into()),
            ]);
            let batch =
                rmpv::Value::Array(vec![0.0.into(), rmpv::Value::Array(vec![event]), 0.into()]);
            let mut payload = Vec::new();
            rmpv::encode::write_value(&mut payload, &batch).unwrap();
            self.socket
                .send_multipart(
                    [b"kv".as_slice(), seq.to_be_bytes().as_slice(), &payload],
                    0,
                )
                .unwrap();
        }
    }

    async fn fixture() -> (
        CompletionObservations,
        WorkerRegistry,
        Vec<Arc<dyn Worker>>,
        Vec<Publisher>,
    ) {
        let publishers: Vec<_> = (0..3).map(|_| Publisher::new()).collect();
        let registry = WorkerRegistry::new();
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| {
                let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                    format!("http://worker-{index}:8000"),
                    WorkerType::Regular,
                ));
                registry.register(worker.clone());
                worker
            })
            .collect();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sources.json");
        let sources: Vec<_> = workers.iter().zip(&publishers).enumerate().map(|(index, (worker, publisher))| json!({"worker_url":worker.url(),"source_id":format!("source-{index}"),"pub_endpoint":publisher.endpoint,"topic":"kv"})).collect();
        std::fs::write(&path, json!({"block_size":2,"sources":sources}).to_string()).unwrap();
        let urls = workers
            .iter()
            .map(|worker| worker.url().to_string())
            .collect::<Vec<_>>();
        let observations = CompletionObservations::load(
            &path,
            "fixture-model",
            &urls,
            &registry,
            reqwest::Client::new(),
        )
        .unwrap();
        for publisher in &publishers {
            publisher.subscribed().await;
        }
        (observations, registry, workers, publishers)
    }

    async fn wait_depth(observations: &CompletionObservations, source: usize, depth: u32) {
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: local_hashes(&[1, 2, 3, 4, 5, 6], 2, None)
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
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if observations.sources[source]
                    .cursor
                    .with_qualified_matches(&query, |hits, _| {
                        hits.iter().map(|hit| hit.matched_depth).max().unwrap_or(0)
                    })
                    == Some(depth)
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn observation_selector_evaluates_all_candidates_and_reserves_once() {
        let (observations, registry, workers, publishers) = fixture().await;
        for (index, publisher) in publishers.iter().enumerate() {
            publisher.send(0, index + 1);
            wait_depth(&observations, index, (index + 1) as u32).await;
        }
        let context = PolicyRequestContext::new(None, None).with_token_ids(&[1, 2, 3, 4, 5, 6]);
        let (selected, guard) = observations
            .select(&registry, &workers, &context, || {
                panic!("positive observation must not fall back")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&selected, &workers[2]));
        assert_eq!(
            workers
                .iter()
                .map(|worker| worker.load())
                .collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
        drop(guard);
        assert_eq!(workers[2].load(), 0);
        // All equal prefix depths: load wins, then ties rotate rather than fixed owner.
        let short = PolicyRequestContext::new(None, None).with_token_ids(&[1, 2]);
        let busy = WorkerLoadGuard::new_owned(workers[0].clone());
        let (first, first_guard) = observations
            .select(&registry, &workers, &short, || None)
            .unwrap();
        let (second, second_guard) = observations
            .select(&registry, &workers, &short, || None)
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &workers[0]));
        assert!(!Arc::ptr_eq(&first, &second));
        drop((busy, first_guard, second_guard));
        assert!(workers.iter().all(|worker| worker.load() == 0));
    }

    #[tokio::test]
    async fn observation_retirement_revokes_cursor_without_releasing_active_guard() {
        let (observations, registry, workers, publishers) = fixture().await;
        publishers[2].send(0, 3);
        wait_depth(&observations, 2, 3).await;
        let cursor = observations.sources[2].cursor.clone();
        let context = PolicyRequestContext::new(None, None).with_token_ids(&[1, 2, 3, 4, 5, 6]);
        let (worker, guard) = observations
            .select(&registry, &workers, &context, || None)
            .unwrap();
        assert!(Arc::ptr_eq(&worker, &workers[2]));
        assert_eq!(worker.load(), 1);
        registry.remove_by_url(worker.url()).unwrap();
        observations.retire_worker(worker.url());
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: vec![],
            tiers_of_interest: vec![StorageTier::Device],
        };
        assert!(cursor.with_qualified_matches(&query, |_, _| ()).is_none());
        assert_eq!(worker.load(), 1);
        // A captured cursor cannot follow a retired subscriber or late frames.
        publishers[2].send(1, 3);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if publishers[2]
                    .socket
                    .recv_multipart(zmq::DONTWAIT)
                    .is_ok_and(|frames| frames == vec![vec![0, b'k', b'v']])
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(cursor.with_qualified_matches(&query, |_, _| ()).is_none());
        assert_eq!(worker.load(), 1);
        drop(guard);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn observation_selector_rejects_gap_and_same_url_replacement() {
        let (observations, registry, workers, publishers) = fixture().await;
        publishers[2].send(0, 3);
        wait_depth(&observations, 2, 3).await;
        publishers[2].send(2, 3);
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: vec![],
            tiers_of_interest: vec![StorageTier::Device],
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            while observations.sources[2]
                .cursor
                .with_qualified_matches(&query, |_, _| ())
                .is_some()
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let context = PolicyRequestContext::new(None, None).with_token_ids(&[1, 2, 3, 4, 5, 6]);
        let (fallback, guard) = observations
            .select(&registry, &workers, &context, || Some(workers[0].clone()))
            .unwrap();
        assert!(Arc::ptr_eq(&fallback, &workers[0]));
        drop(guard);
        let replacement: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            workers[0].url().to_string(),
            WorkerType::Regular,
        ));
        registry.register(replacement);
        assert!(observations
            .select(&registry, &workers, &context, || Some(workers[0].clone()))
            .is_none());
        assert!(workers.iter().all(|worker| worker.load() == 0));
    }
}

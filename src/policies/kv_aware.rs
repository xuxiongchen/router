//! Real-event prefix affinity, with fair least-load fallback on a cold miss.

use super::exact_history::{
    ExactHistoryStore, HistoricalRoutingRequest, HistoricalSelectionStage, HistoryReservation,
    ProcessTokenizerContractId,
};
use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::config::KvAwareConfig;
use crate::core::Worker;
use crate::kv_index::{BlockKeyGenerator, KVBlockIndex};
use crate::metrics::RouterMetrics;
use crate::prompt_tokens::timing::StageTimer;
use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
pub struct KvAwarePolicy {
    index: Arc<KVBlockIndex>,
    generator: BlockKeyGenerator,
    cursor: AtomicUsize,
    dense_reuse: AtomicBool,
    load_guard: bool,
    history: Option<Arc<ExactHistoryStore>>,
    pub(crate) history_contract: ProcessTokenizerContractId,
}

// Experimental cache-first slack, not a calibrated queueing/cost model. A hot
// worker may retain one more in-flight request than the least-loaded candidate.
const CACHE_LOAD_SLACK: usize = 1;

struct CandidateScore {
    worker_index: usize,
    matched_blocks: usize,
    score: usize,
    load: usize,
}

impl KvAwarePolicy {
    pub fn new(config: &KvAwareConfig) -> Self {
        let index = Arc::new(KVBlockIndex::new(config.index_max_entries));
        let history = config.history.enabled.then(|| {
            Arc::new(
                ExactHistoryStore::new(config.history.store_config())
                    .expect("validated exact-history configuration"),
            )
        });
        if let Some(history) = &history {
            index.attach_history(history.clone());
        }
        Self {
            index,
            history,
            history_contract: ProcessTokenizerContractId::mint(),
            generator: BlockKeyGenerator::new(config.block_size, u64::from(config.hash_seed)),
            cursor: AtomicUsize::new(0),
            dense_reuse: AtomicBool::new(false),
            load_guard: config.load_guard,
        }
    }

    pub fn index(&self) -> Arc<KVBlockIndex> {
        self.index.clone()
    }

    pub(crate) fn history_enabled(&self) -> bool {
        self.history.is_some()
    }

    #[cfg(test)]
    pub(crate) fn history(&self) -> Option<&Arc<ExactHistoryStore>> {
        self.history.as_ref()
    }

    pub(crate) fn load_guard_enabled(&self) -> bool {
        self.load_guard
    }

    /// Enable only after the automatic vLLM capability path has verified Normal,
    /// single-group full attention with equal hash/reuse block units and the
    /// one-token terminal recompute rule. Native and legacy providers retain
    /// their existing advisory stored-block score. This flag does not establish
    /// ownership: the verified pool must still fence and populate the index.
    pub fn enable_dense_reuse(&self) {
        self.dense_reuse.store(true, Ordering::Release);
    }
}

impl KvAwarePolicy {
    /// Router serializes this selection, reservation and the single load lease.
    pub(crate) fn select_attempt(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        history_request: Option<&HistoricalRoutingRequest>,
    ) -> Option<(usize, Option<HistoryReservation>)> {
        let _selector = StageTimer::start("selector_total");
        let hash = StageTimer::start("block_hash");
        let keys = token_ids
            .map(|ids| self.generator.generate_block_keys(ids))
            .unwrap_or_default();
        drop(hash);
        let dense_reuse = self.dense_reuse.load(Ordering::Acquire);
        let query_tokens = token_ids.map_or(0, <[u32]>::len);
        let index = StageTimer::start("index_candidates");
        let candidates: Vec<_> = get_healthy_worker_indices(workers)
            .into_iter()
            // Empty but active generations remain legitimate cold targets.
            // A retired generation must not be revived as a cheap alternative.
            // The opt-out path deliberately preserves legacy cold fallback.
            .filter(|&i| {
                !(self.load_guard || (self.history.is_some() && history_request.is_some()))
                    || self.index.current_generation(workers[i].url()).is_some()
            })
            .map(|i| {
                let matched_blocks = self.index.prefix_score(workers[i].url(), &keys);
                CandidateScore {
                    worker_index: i,
                    matched_blocks,
                    score: if dense_reuse {
                        dense_reusable_tokens(
                            matched_blocks,
                            query_tokens,
                            self.generator.block_size(),
                        )
                    } else {
                        matched_blocks
                    },
                    load: workers[i].load(),
                }
            })
            .collect();
        drop(index);
        let cache_best_score = candidates.iter().map(|candidate| candidate.score).max()?;
        let cache_best_load = candidates
            .iter()
            .filter(|candidate| candidate.score == cache_best_score)
            .map(|candidate| candidate.load)
            .min()?;
        let min_load = candidates.iter().map(|candidate| candidate.load).min()?;
        let load_ceiling = min_load.saturating_add(CACHE_LOAD_SLACK);
        let guarded = self.load_guard && cache_best_load > load_ceiling;
        // Retain the highest genuine reuse among candidates within the slack;
        // do not force least-load routing or invent residency on a cold target.
        let eligible = |candidate: &&CandidateScore| !guarded || candidate.load <= load_ceiling;
        let best_score = candidates
            .iter()
            .filter(eligible)
            .map(|candidate| candidate.score)
            .max()?;
        let least_load = candidates
            .iter()
            .filter(eligible)
            .filter(|candidate| candidate.score == best_score)
            .map(|candidate| candidate.load)
            .min()?;
        let mut tied: Vec<_> = candidates
            .iter()
            .filter(eligible)
            .filter(|candidate| candidate.score == best_score && candidate.load == least_load)
            .map(|candidate| candidate.worker_index)
            .collect();
        tied.sort_unstable_by(|a, b| workers[*a].url().cmp(workers[*b].url()));
        // A unique cache hit must not consume a fallback turn: alternating
        // hot and cold requests would otherwise always send cold requests to
        // the same worker in a two-worker pool.
        let fallback = || {
            if tied.len() == 1 {
                tied[0]
            } else {
                tied[self.cursor.fetch_add(1, Ordering::Relaxed) % tied.len()]
            }
        };
        let mut stage = if best_score > 0 { "hbm" } else { "least_load" };
        let mut matched_history_tokens = 0;
        let (selected, reservation) = if let Some((history, request)) =
            self.history.as_ref().zip(history_request)
        {
            if best_score > 0 {
                let selected = fallback();
                (
                    selected,
                    history.reserve_selected(request, &workers[selected]),
                )
            } else {
                // CL remains authoritative: advisory affinity cannot route above
                // its existing one-request slack when enabled.
                let eligible_workers: Vec<_> = candidates
                    .iter()
                    .filter(|c| !self.load_guard || c.load <= load_ceiling)
                    .map(|c| workers[c.worker_index].clone())
                    .collect();
                let decision = history.select_and_reserve(request, &eligible_workers, || {
                    let selected = fallback();
                    eligible_workers
                        .iter()
                        .position(|w| Arc::ptr_eq(w, &workers[selected]))
                })?;
                stage = match decision.stage {
                    HistoricalSelectionStage::ExactHistory => "exact_history",
                    HistoricalSelectionStage::Session => "session",
                    HistoricalSelectionStage::LeastLoad => "least_load",
                };
                matched_history_tokens = decision.matched_tokens;
                let selected = workers
                    .iter()
                    .position(|w| Arc::ptr_eq(w, &eligible_workers[decision.worker_index]))?;
                (selected, decision.reservation)
            }
        } else {
            (fallback(), None)
        };
        if tracing::enabled!(tracing::Level::DEBUG) {
            // `prefix_blocks` retains its existing raw stored-coverage meaning.
            // Only verified Dense mode has a reusable-token prediction; legacy
            // null is intentional, not a claim of zero backend cache hits.
            let scores: Vec<_> = candidates
                .iter()
                .map(|candidate| {
                    serde_json::json!({
                        "worker": workers[candidate.worker_index].url(),
                        "prefix_blocks": candidate.matched_blocks,
                        "reusable_prefix_tokens": dense_reuse.then_some(candidate.score),
                        "inflight": candidate.load,
                    })
                })
                .collect();
            let selected_candidate = candidates
                .iter()
                .find(|candidate| candidate.worker_index == selected)
                .expect("selected worker must have a score");
            let token_ids_sha256 = token_ids.map(|ids| {
                let mut digest = Sha256::new();
                for id in ids {
                    digest.update(id.to_be_bytes());
                }
                format!("{:x}", digest.finalize())
            });
            let decision = serde_json::json!({"worker": workers[selected].url(),
                "prefix_blocks": selected_candidate.matched_blocks,
                "reusable_prefix_tokens": dense_reuse.then_some(selected_candidate.score),
                "stage": stage, "history_matched_tokens": matched_history_tokens,
                "history_reserved": reservation.is_some(),
                "score_kind": if dense_reuse { "reusable_prefix_tokens" } else { "stored_prefix_blocks" },
                "inflight": selected_candidate.load,
                "cache_best_workers": candidates.iter().filter(|candidate|
                    candidate.score == cache_best_score && candidate.load == cache_best_load)
                    .map(|candidate| workers[candidate.worker_index].url()).collect::<Vec<_>>(),
                "cache_best_inflight": cache_best_load, "minimum_inflight": min_load,
                "load_guard": guarded,
                "load_guard_reason": if guarded { "excess_inflight" } else if self.load_guard { "within_slack" } else { "disabled" },
                "query_tokens": query_tokens, "complete_blocks": keys.len(), "scores": scores,
                "token_ids_sha256": token_ids_sha256});
            tracing::debug!(decision = %decision, "kv_route_decision");
        }
        workers[selected].increment_processed();
        RouterMetrics::record_processed_request(workers[selected].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected].url());
        Some((selected, reservation))
    }
}

/// Pinned Normal Dense semantics, not a general Hybrid/speculative cache rule.
/// The cap also guarantees multiplication cannot exceed `query_tokens - 1`.
fn dense_reusable_tokens(matched_blocks: usize, query_tokens: usize, block_size: usize) -> usize {
    block_size * matched_blocks.min(query_tokens.saturating_sub(1) / block_size)
}

impl LoadBalancingPolicy for KvAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_with_tokens(workers, None, None, headers)
    }

    fn select_worker_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        token_ids: Option<&[u32]>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_attempt(workers, token_ids, None)
            .map(|(selected, _)| selected)
    }

    fn name(&self) -> &'static str {
        "kv_aware"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::exact_history::{
        ExactHistoryRequest, ExactHistoryScope, ModelHistoryScope,
    };

    fn history_policy() -> (KvAwarePolicy, Vec<Arc<dyn Worker>>) {
        let mut config = KvAwareConfig {
            load_guard: true,
            ..Default::default()
        };
        config.history.enabled = true;
        config.history.eviction_interval_secs = 0;
        config.history.cache_threshold = 0.5;
        let policy = KvAwarePolicy::new(&config);
        policy.enable_dense_reuse();
        let (_, workers) = dense_policy(16);
        for worker in &workers {
            policy.index.begin_worker(worker.url());
        }
        (policy, workers)
    }

    fn history_input(policy: &KvAwarePolicy, tokens: &[u32]) -> HistoricalRoutingRequest {
        let model = ModelHistoryScope::new("verified-model-and-contract").unwrap();
        HistoricalRoutingRequest::new(
            model.clone(),
            ExactHistoryRequest::new(
                ExactHistoryScope::new(model, policy.history_contract),
                Arc::from(tokens),
            ),
            None,
        )
        .unwrap()
    }

    #[test]
    fn history_zero_hbm_is_advisory_and_does_not_consume_fair_cursor() {
        let (policy, workers) = history_policy();
        let tokens = [1, 2, 3]; // Shorter than a physical block.
        let request = history_input(&policy, &tokens);
        let (cold, pending) = policy
            .select_attempt(&workers, Some(&tokens), Some(&request))
            .unwrap();
        let (_, repeat) = policy
            .select_attempt(&workers, Some(&tokens), Some(&request))
            .unwrap();
        assert_eq!(policy.cursor.load(Ordering::Relaxed), 1);
        assert_eq!(policy.history().unwrap().stats().exact_lookup_hits, 1);
        assert_eq!(policy.index.ownership_count(), 0);
        assert_eq!(dense_reusable_tokens(0, tokens.len(), 16), 0);
        drop(repeat);
        drop(pending);
        let (next, pending) = policy
            .select_attempt(&workers, Some(&tokens), Some(&request))
            .unwrap();
        assert_ne!(cold, next);
        drop(pending);
        assert_eq!(policy.history().unwrap().stats().reservation_count, 0);
    }

    #[test]
    fn physical_owner_beats_history_and_history_respects_cl_slack() {
        let (policy, workers) = history_policy();
        let tokens: Vec<u32> = (0..49).collect();
        let request = history_input(&policy, &tokens);
        policy
            .history()
            .unwrap()
            .reserve_selected(&request, &workers[0])
            .unwrap()
            .commit();
        let generation = policy.index.current_generation(workers[1].url()).unwrap();
        let keys = policy.generator.generate_block_keys(&tokens);
        policy.index.store(workers[1].url(), generation, &keys);
        let (selected, pending) = policy
            .select_attempt(&workers, Some(&tokens), Some(&request))
            .unwrap();
        assert_eq!(selected, 1);
        drop(pending);
        policy.index.clear(workers[1].url(), generation);
        // History remains only on W0, but cannot bypass CL.
        workers[0].increment_load();
        workers[0].increment_load();
        let (selected, pending) = policy
            .select_attempt(&workers, Some(&tokens), Some(&request))
            .unwrap();
        assert_eq!(selected, 1);
        drop(pending);
        workers[0].decrement_load();
        workers[0].decrement_load();
        assert_eq!(policy.index.ownership_count(), 0);
    }

    #[test]
    fn history_clear_roll_retire_and_replacement_fence_late_commits() {
        for invalidation in 0..4 {
            let (policy, workers) = history_policy();
            let request = history_input(&policy, &[1, 2, 3]);
            let mut pending = policy
                .history()
                .unwrap()
                .reserve_selected(&request, &workers[0])
                .unwrap();
            let generation = policy.index.current_generation(workers[0].url()).unwrap();
            match invalidation {
                0 => {
                    policy.index.clear(workers[0].url(), generation);
                }
                1 => {
                    policy.index.roll_worker(workers[0].url(), generation);
                }
                2 => policy.index.retire_worker(workers[0].url()),
                _ => {
                    policy.index.begin_worker(workers[0].url());
                }
            }
            assert!(!pending.commit());
            assert_eq!(policy.history().unwrap().stats().entry_count, 0);
        }
    }

    #[test]
    fn history_off_preserves_no_store_and_no_learning() {
        let (policy, workers) = dense_policy(16);
        assert!(!policy.history_enabled());
        let request = history_input(&policy, &[1, 2, 3]);
        assert!(policy
            .select_attempt(&workers, Some(&[1, 2, 3]), Some(&request))
            .unwrap()
            .1
            .is_none());
        assert_eq!(policy.index.ownership_count(), 0);
    }

    fn dense_policy(block_size: usize) -> (KvAwarePolicy, Vec<Arc<dyn Worker>>) {
        let policy = KvAwarePolicy::new(&KvAwareConfig {
            block_size,
            ..KvAwareConfig::default()
        });
        policy.enable_dense_reuse();
        let workers = ["http://w0:8000", "http://w1:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        (policy, workers)
    }

    fn guarded_policy() -> (KvAwarePolicy, Vec<Arc<dyn Worker>>, Vec<u32>) {
        let policy = KvAwarePolicy::new(&KvAwareConfig {
            load_guard: true,
            ..KvAwareConfig::default()
        });
        policy.enable_dense_reuse();
        let (_, workers) = dense_policy(16);
        let ids: Vec<u32> = (0..49).collect();
        for worker in &workers {
            policy.index.begin_worker(worker.url());
        }
        let generation = policy.index.current_generation(workers[1].url()).unwrap();
        assert!(policy.index.store(
            workers[1].url(),
            generation,
            &policy.generator.generate_block_keys(&ids),
        ));
        (policy, workers, ids)
    }

    #[test]
    fn load_guard_keeps_idle_and_near_load_cache_owner() {
        let (policy, workers, ids) = guarded_policy();
        for load in 0..=1 {
            assert_eq!(workers[1].load(), load);
            assert_eq!(
                policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
                Some(1),
            );
            workers[1].increment_load();
        }
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(0),
        );
        // Dispatching to a cold worker is not evidence of stored cache blocks.
        assert_eq!(
            policy.index.prefix_score(
                workers[0].url(),
                &policy.generator.generate_block_keys(&ids)
            ),
            0
        );
        assert_eq!(policy.index.ownership_count(), 3);
    }

    #[test]
    fn load_guard_keeps_best_real_reuse_inside_slack_not_forced_least_load() {
        let (policy, mut workers, ids) = guarded_policy();
        let partial = Arc::new(BasicWorker::new(
            "http://w2:8000".into(),
            WorkerType::Regular,
        )) as Arc<dyn Worker>;
        let generation = policy.index.begin_worker(partial.url());
        let keys = policy.generator.generate_block_keys(&ids);
        assert!(policy.index.store(partial.url(), generation, &keys[..2]));
        partial.increment_load();
        workers[1].increment_load();
        workers[1].increment_load();
        workers.push(partial);
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(2),
        );
    }

    #[test]
    fn load_guard_cold_and_equal_reuse_remain_fair() {
        let (policy, workers, ids) = guarded_policy();
        let cold = vec![777; 49];
        let choices: Vec<_> = (0..4)
            .map(|_| policy.select_worker_with_tokens(&workers, None, Some(&cold), None))
            .collect();
        assert_eq!(choices, [Some(0), Some(1), Some(0), Some(1)]);
        let generation = policy.index.current_generation(workers[0].url()).unwrap();
        assert!(policy.index.store(
            workers[0].url(),
            generation,
            &policy.generator.generate_block_keys(&ids)
        ));
        let choices: Vec<_> = (0..4)
            .map(|_| policy.select_worker_with_tokens(&workers, None, Some(&ids), None))
            .collect();
        assert_eq!(choices, [Some(0), Some(1), Some(0), Some(1)]);
    }

    #[test]
    fn load_guard_never_reintroduces_unhealthy_or_retired_candidates() {
        let (policy, workers, ids) = guarded_policy();
        for _ in 0..8 {
            workers[1].increment_load();
        }
        workers[0].set_healthy(false);
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1)
        );
        workers[0].set_healthy(true);
        policy.index.retire_worker(workers[0].url());
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1)
        );
        policy.index.begin_worker(workers[0].url());
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(0)
        );
        policy.index.retire_worker(workers[0].url());
        policy.index.retire_worker(workers[1].url());
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            None
        );
    }

    #[test]
    fn load_guard_all_busy_and_single_eligible_do_not_queue_or_force_balance() {
        let (policy, workers, ids) = guarded_policy();
        for worker in &workers {
            for _ in 0..20 {
                worker.increment_load();
            }
        }
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1)
        );
        workers[0].set_healthy(false);
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1)
        );
        // The opt-out path is still strict cache affinity, regardless of load.
        let (legacy, legacy_workers) = dense_policy(16);
        let generation = legacy.index.begin_worker(legacy_workers[1].url());
        assert!(legacy.index.store(
            legacy_workers[1].url(),
            generation,
            &legacy.generator.generate_block_keys(&ids)
        ));
        for _ in 0..20 {
            legacy_workers[1].increment_load();
        }
        assert_eq!(
            legacy.select_worker_with_tokens(&legacy_workers, None, Some(&ids), None),
            Some(1)
        );
    }

    #[test]
    fn dense_reuse_matches_installed_vllm_source_oracle_fixtures() {
        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            query_tokens: usize,
            block_size: usize,
            cached_blocks: Vec<usize>,
            #[serde(default)]
            removed_blocks: Vec<usize>,
            expected_matched_blocks: usize,
            expected_reusable_tokens: usize,
        }
        #[derive(serde::Deserialize)]
        struct Fixtures {
            schema_version: u32,
            cases: Vec<Case>,
        }
        // Explicit expected values are independently checked by AST-executing
        // the installed vLLM source in test_kv_dense_source_oracle.py, without
        // importing an engine or touching live allocator/LRU methods.
        let fixtures: Fixtures = serde_json::from_str(include_str!(
            "../../tests/fixtures/kv_capabilities/dense_reuse_cases.json"
        ))
        .unwrap();
        assert_eq!(fixtures.schema_version, 1);
        for case in fixtures.cases {
            let (policy, workers) = dense_policy(case.block_size);
            let ids: Vec<u32> = (0..case.query_tokens as u32).collect();
            let keys = policy.generator.generate_block_keys(&ids);
            let generation = policy.index.begin_worker(workers[1].url());
            let cached: Vec<_> = case.cached_blocks.iter().map(|&i| keys[i]).collect();
            assert!(policy.index.store(workers[1].url(), generation, &cached));
            let removed: Vec<_> = case.removed_blocks.iter().map(|&i| keys[i]).collect();
            assert!(policy.index.remove(workers[1].url(), generation, &removed));
            let matched = policy.index.prefix_score(workers[1].url(), &keys);
            assert_eq!(matched, case.expected_matched_blocks, "{}", case.name);
            assert_eq!(
                dense_reusable_tokens(matched, ids.len(), case.block_size),
                case.expected_reusable_tokens,
                "{}",
                case.name,
            );
            // A stored but terminal-only block must not beat an idle cold
            // worker; actual reusable ownership still wins over a lower load.
            workers[1].increment_load();
            let expected_owner = usize::from(case.expected_reusable_tokens > 0);
            assert_eq!(
                policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
                Some(expected_owner),
                "{}",
                case.name,
            );
        }
    }

    #[test]
    fn dense_29_and_28_stored_blocks_tie_at_448_reusable_tokens() {
        let (policy, workers) = dense_policy(16);
        let ids: Vec<u32> = (0..464).collect();
        let keys = policy.generator.generate_block_keys(&ids);
        for (i, count) in [(0, 29), (1, 28)] {
            let generation = policy.index.begin_worker(workers[i].url());
            assert!(policy
                .index
                .store(workers[i].url(), generation, &keys[..count]));
        }
        assert_eq!(dense_reusable_tokens(29, 464, 16), 448);
        assert_eq!(dense_reusable_tokens(28, 464, 16), 448);
        workers[0].increment_load();
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1),
        );
        workers[0].decrement_load();
        let choices: Vec<_> = (0..4)
            .map(|_| policy.select_worker_with_tokens(&workers, None, Some(&ids), None))
            .collect();
        assert_eq!(choices, [Some(0), Some(1), Some(0), Some(1)]);
    }

    #[test]
    fn dense_exact_first_block_rotates_like_cold_and_zero_input_is_safe() {
        let (policy, workers) = dense_policy(16);
        let ids = vec![1; 16];
        let generation = policy.index.begin_worker(workers[1].url());
        assert!(policy.index.store(
            workers[1].url(),
            generation,
            &policy.generator.generate_block_keys(&ids),
        ));
        let choices: Vec<_> = (0..4)
            .map(|_| policy.select_worker_with_tokens(&workers, None, Some(&ids), None))
            .collect();
        assert_eq!(choices, [Some(0), Some(1), Some(0), Some(1)]);
        assert_eq!(dense_reusable_tokens(usize::MAX, 0, 16), 0);
        assert_eq!(dense_reusable_tokens(usize::MAX, 1, 16), 0);
        assert_eq!(
            dense_reusable_tokens(usize::MAX, usize::MAX, 16),
            (usize::MAX - 1) / 16 * 16,
        );
        assert!(policy
            .select_worker_with_tokens(&workers, None, Some(&[]), None)
            .is_some());
        assert!(policy
            .select_worker_with_tokens(&workers, None, None, None)
            .is_some());
    }

    #[test]
    fn dense_reuse_keeps_full_hash_identity_and_generation_fencing() {
        let (policy, workers) = dense_policy(16);
        let ids: Vec<u32> = (0..33).collect();
        let keys = policy.generator.generate_block_keys(&ids);
        let mut different_full_hash = keys[0];
        different_full_hash[31] ^= 1;
        assert_eq!(&different_full_hash[..8], &keys[0][..8]);
        let generation = policy.index.begin_worker(workers[1].url());
        workers[1].increment_load();
        assert!(policy.index.store(
            workers[1].url(),
            generation,
            &[different_full_hash, keys[1]],
        ));
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(0),
        );
        assert!(policy.index.store(workers[1].url(), generation, &keys));
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1),
        );
        let fresh_generation = policy.index.roll_worker(workers[1].url(), generation);
        assert!(fresh_generation.is_some());
        assert!(!policy.index.store(workers[1].url(), generation, &keys));
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(0),
        );
    }

    #[test]
    fn legacy_default_retains_stored_block_ranking() {
        let policy = KvAwarePolicy::new(&KvAwareConfig::default());
        assert!(!policy.dense_reuse.load(Ordering::Acquire));
        let (_, workers) = dense_policy(16);
        let ids = vec![1; 16];
        let generation = policy.index.begin_worker(workers[1].url());
        assert!(policy.index.store(
            workers[1].url(),
            generation,
            &policy.generator.generate_block_keys(&ids),
        ));
        workers[1].increment_load();
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1),
        );
        policy.enable_dense_reuse();
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(0),
        );
    }

    #[test]
    fn real_ownership_wins_and_cold_misses_rotate_without_learning() {
        let policy = KvAwarePolicy::new(&KvAwareConfig::default());
        let workers: Vec<Arc<dyn Worker>> = ["http://w0:8000", "http://w1:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let ids: Vec<_> = (0..32).collect();
        let first = policy
            .select_worker_with_tokens(&workers, None, Some(&ids), None)
            .unwrap();
        let second = policy
            .select_worker_with_tokens(&workers, None, Some(&ids), None)
            .unwrap();
        assert_ne!(first, second);
        assert_eq!(policy.index.ownership_count(), 0);
        let generation = policy.index.begin_worker(workers[1].url());
        policy.index.store(
            workers[1].url(),
            generation,
            &policy.generator.generate_block_keys(&ids),
        );
        workers[1].increment_load();
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1)
        );
        // A legacy string/JSON never becomes guessed prompt tokens.
        assert_eq!(policy.select_worker(&workers, Some("[0,1,2]")), Some(0));
        workers[1].set_healthy(false);
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(0)
        );
    }

    #[test]
    fn unique_hits_do_not_consume_cold_fallback_turns() {
        let policy = KvAwarePolicy::new(&KvAwareConfig::default());
        let workers: Vec<Arc<dyn Worker>> = ["http://w0:8000", "http://w1:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let hot = vec![1; 32];
        let cold = vec![2; 32];
        let generation = policy.index.begin_worker(workers[1].url());
        policy.index.store(
            workers[1].url(),
            generation,
            &policy.generator.generate_block_keys(&hot),
        );
        let mut cold_choices = Vec::new();
        for _ in 0..4 {
            assert_eq!(
                policy.select_worker_with_tokens(&workers, None, Some(&hot), None),
                Some(1)
            );
            cold_choices.push(
                policy
                    .select_worker_with_tokens(&workers, None, Some(&cold), None)
                    .unwrap(),
            );
        }
        assert_eq!(cold_choices, [0, 1, 0, 1]);
    }
}

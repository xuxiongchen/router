//! Publisher enrollment and shared source-local publication fence.
use super::binding::SourceLease;
use super::subscriber::apply_batch;
use super::{KVEventBatch, KvBlockIndexer, SourceId};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

#[derive(Clone)]
pub(super) struct Enrollment {
    pub config: crate::config::types::KvSnapshotSource,
    pub bound_source: Option<String>,
}

impl Enrollment {
    pub(super) fn load(config: crate::config::types::KvSnapshotSource) -> Result<Self, String> {
        if config.namespace.is_empty() || config.namespace.len() > 256 || config.block_size == 0 {
            return Err("invalid snapshot enrollment".into());
        }
        super::publisher_snapshot::validate_config(&config.publisher_snapshot)?;
        Ok(Self {
            config,
            bound_source: None,
        })
    }
}

/// The original capture is producer evidence, not a still-live remote lease.
/// Each proof has a fresh, finite local budget and an actual registration lease.
pub(crate) struct Attempt {
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    lease: SourceLease,
    attempt: u64,
    expected: Identity,
    pub(super) index_rank: u32,
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

    pub(super) fn install_batches(
        &self,
        mut batches: Vec<KVEventBatch>,
        before_publish: impl FnOnce(),
    ) -> Result<bool, &'static str> {
        // Wire rank is the global Engine rank; the independently addressable
        // DP1 HTTP target has local rank zero. Validate first, then translate
        // only the index target, never the remote identity or journal cursor.
        for batch in &mut batches {
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
                        for batch in &batches {
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

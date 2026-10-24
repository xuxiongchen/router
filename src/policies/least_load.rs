//! Lowest-load worker selection.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Selects the healthy worker with the lowest observed load.
#[derive(Debug, Default)]
pub struct LeastLoadPolicy {
    tie_breaker: AtomicUsize,
}

impl LoadBalancingPolicy for LeastLoadPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        let min_load = healthy.iter().map(|&index| workers[index].load()).min()?;
        let least_loaded: Vec<_> = healthy
            .into_iter()
            .filter(|&index| workers[index].load() == min_load)
            .collect();
        let next = self.tie_breaker.fetch_add(1, Ordering::Relaxed);
        least_loaded.get(next % least_loaded.len()).copied()
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn selects_lowest_loaded_healthy_worker() {
        let policy = LeastLoadPolicy::default();
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        assert_eq!(policy.select_worker(&workers, None), Some(2));
    }

    #[test]
    fn rotates_across_workers_with_equal_load() {
        let policy = LeastLoadPolicy::default();
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://w{i}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();

        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }
}

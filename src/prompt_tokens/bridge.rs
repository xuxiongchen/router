//! Bounded, single-threaded bridge to an already initialized Python facade.
//!
//! No Python call runs on a Tokio network worker. A timed-out synchronous call
//! cannot be interrupted safely: its input and token reservation remain charged
//! until it really returns. Shutdown never finalizes Python or joins indefinitely.
//! A Python non-daemon lifetime waiter keeps normal interpreter shutdown from
//! finalizing while the execution thread is active. It does not render or own an
//! event loop. A bounded shutdown may return while this waiter still holds the
//! process alive; a GIL deadlock/native crash has no hard recovery guarantee.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict, PyList},
};
use tokio::sync::{oneshot, watch};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderContract {
    pub id: String,
    pub epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestKind {
    Completion,
    Chat,
}

impl RequestKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completion => "completion",
            Self::Chat => "chat",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedTokens {
    pub token_ids: Arc<[u32]>,
    pub contract: RenderContract,
    pub cache_eligible: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeadlineStage {
    Queue,
    Execution,
}

/// Deliberately excludes exception text, which may contain request content.
#[derive(Clone, Debug)]
pub enum PreparedResult {
    Exact(PreparedTokens),
    Unsupported,
    Invalid { http_status: u16 },
    Unavailable,
    Busy,
    Deadline(DeadlineStage),
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct BridgeLimits {
    /// Includes the active job, not just waiting jobs.
    pub max_pending_jobs: usize,
    /// Aggregate original request bytes retained by admitted jobs.
    pub max_input_bytes: usize,
    pub max_tokens_per_request: usize,
    /// Each admitted job reserves `max_tokens_per_request` before rendering.
    pub max_reserved_tokens: usize,
    pub queue_timeout: Duration,
    pub execution_timeout: Duration,
}

impl Default for BridgeLimits {
    fn default() -> Self {
        Self {
            max_pending_jobs: 16,
            max_input_bytes: 4 * 1024 * 1024,
            max_tokens_per_request: 65_536,
            max_reserved_tokens: 4 * 65_536,
            queue_timeout: Duration::from_millis(250),
            execution_timeout: Duration::from_secs(10),
        }
    }
}

impl BridgeLimits {
    fn validate(&self) -> Result<(), String> {
        if self.max_pending_jobs == 0
            || self.max_input_bytes == 0
            || self.max_tokens_per_request == 0
            || self.max_reserved_tokens < self.max_tokens_per_request
            || self.queue_timeout.is_zero()
            || self.execution_timeout.is_zero()
            || Instant::now().checked_add(self.queue_timeout).is_none()
            || Instant::now().checked_add(self.execution_timeout).is_none()
        {
            return Err("invalid render bridge limits".into());
        }
        Ok(())
    }
}

#[derive(Default)]
struct Budget {
    jobs: usize,
    bytes: usize,
    tokens: usize,
}

struct Shared {
    accepting: AtomicBool,
    stopping: AtomicBool,
    budget: Mutex<Budget>,
    contract: RenderContract,
    limits: BridgeLimits,
}

impl Shared {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        let mut budget = self.budget.lock();
        let tokens = self.limits.max_tokens_per_request;
        if !self.accepting.load(Ordering::Acquire)
            || budget.jobs >= self.limits.max_pending_jobs
            || bytes > self.limits.max_input_bytes.saturating_sub(budget.bytes)
            || tokens
                > self
                    .limits
                    .max_reserved_tokens
                    .saturating_sub(budget.tokens)
        {
            return None;
        }
        budget.jobs += 1;
        budget.bytes += bytes;
        budget.tokens += tokens;
        Some(Reservation {
            shared: Arc::clone(self),
            bytes,
            tokens,
        })
    }
}

struct Reservation {
    shared: Arc<Shared>,
    bytes: usize,
    tokens: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut budget = self.shared.budget.lock();
        budget.jobs -= 1;
        budget.bytes -= self.bytes;
        budget.tokens -= self.tokens;
    }
}

struct Job {
    kind: RequestKind,
    raw: Arc<[u8]>,
    cancelled: Arc<AtomicBool>,
    queue_deadline: Instant,
    started: oneshot::Sender<Instant>,
    result: oneshot::Sender<PreparedResult>,
    _reservation: Reservation,
}

/// Dropping the request future cancels queued work without spawning cleanup work.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Kept private so there is one production transport, not a plugin framework.
trait Executor: Send + 'static {
    fn startup(&mut self) -> Result<(), ()> {
        Ok(())
    }
    fn render(&mut self, kind: RequestKind, raw: &[u8], limit: usize) -> PreparedResult;
    fn is_invalidated(&self) -> bool {
        false
    }
    fn close(&mut self) {}
}

struct PythonExecutor {
    facade: Option<Py<PyAny>>,
    contract: RenderContract,
    invalidated: bool,
    lifetime: PythonLifetime,
}

/// Python's normal interpreter shutdown joins non-daemon Python threads, but
/// does not know about this Rust-created execution thread. Bridge that lifecycle
/// gap with one GIL-releasing Event waiter, not a second rendering executor.
struct PythonLifetime {
    event: Option<Py<PyAny>>,
}

impl PythonLifetime {
    fn new() -> PyResult<Self> {
        Python::attach(|py| {
            let threading = py.import("threading")?;
            let event = threading.getattr("Event")?.call0()?;
            let lifetime = Self {
                event: Some(event.clone().unbind()),
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("target", event.getattr("wait")?)?;
            kwargs.set_item("name", "cmb-render-lifetime")?;
            kwargs.set_item("daemon", false)?;
            let thread = threading.getattr("Thread")?.call((), Some(&kwargs))?;
            // Thread.start returns only after Python registered the thread;
            // after this point normal finalization must wait for event.set().
            thread.call_method0("start")?;
            Ok(lifetime)
        })
    }

    fn release(&mut self) {
        if let Some(event) = self.event.take() {
            Python::attach(|py| {
                let _ = event.bind(py).call_method0("set");
                drop(event.into_bound(py));
            });
        }
    }
}

impl Drop for PythonLifetime {
    fn drop(&mut self) {
        self.release();
    }
}

enum PythonReply {
    Prepared(PreparedResult),
    Invalidated,
}

fn client_error_status(status: Option<u16>) -> u16 {
    status
        .filter(|status| (400..500).contains(status))
        .unwrap_or(400)
}

impl Executor for PythonExecutor {
    fn startup(&mut self) -> Result<(), ()> {
        Python::attach(|py| {
            self.facade
                .as_ref()
                .ok_or(())?
                .bind(py)
                .call_method0("startup")
                .map(|_| ())
                .map_err(|_| ())
        })
    }

    fn render(&mut self, kind: RequestKind, raw: &[u8], limit: usize) -> PreparedResult {
        let reply = Python::attach(|py| {
            let Some(facade) = self.facade.as_ref() else {
                return PythonReply::Prepared(PreparedResult::Unavailable);
            };
            // One safe copy into Python and one bounded copy back into Rust.
            let result = facade
                .bind(py)
                .call_method1("render", (kind.as_str(), PyBytes::new(py, raw)))
                .and_then(|result| parse_python_result(&result, limit, &self.contract));
            result.unwrap_or(PythonReply::Prepared(PreparedResult::Unavailable))
        });
        match reply {
            PythonReply::Prepared(result) => result,
            PythonReply::Invalidated => {
                self.invalidated = true;
                PreparedResult::Unavailable
            }
        }
    }

    fn is_invalidated(&self) -> bool {
        self.invalidated
    }

    fn close(&mut self) {
        if let Some(facade) = self.facade.take() {
            Python::attach(|py| {
                // Do not print exceptions: their messages/tracebacks can expose
                // original messages, template contents, or local secret paths.
                let _ = facade.bind(py).call_method0("close");
                drop(facade.into_bound(py));
            });
        }
        // A callback has returned (or unwound) before Executor can be dropped.
        // Release the lifetime waiter only after disposing our facade reference.
        self.lifetime.release();
    }
}

impl Drop for PythonExecutor {
    fn drop(&mut self) {
        // Also covers OS-thread spawn failure and an unwound executor thread.
        // There is no join here; a still-running callback owns this object and
        // therefore cannot run this destructor prematurely.
        self.close();
    }
}

fn parse_python_result(
    result: &Bound<'_, PyAny>,
    limit: usize,
    contract: &RenderContract,
) -> PyResult<PythonReply> {
    let result = result.cast::<PyDict>()?;
    // Identity evidence applies to every status, including an unavailable or
    // unsupported reply. It must not be converted into safe fair fallback.
    if let Some(id) = result.get_item("contract_id")? {
        if id.extract::<&str>().map_or(true, |id| id != contract.id) {
            return Ok(PythonReply::Invalidated);
        }
    }
    if let Some(epoch) = result.get_item("epoch")? {
        if epoch
            .extract::<u64>()
            .map_or(true, |epoch| epoch != contract.epoch)
        {
            return Ok(PythonReply::Invalidated);
        }
    }
    let Some(status) = result.get_item("status")? else {
        return Ok(PythonReply::Prepared(PreparedResult::Unavailable));
    };
    let prepared = match status.extract::<&str>()? {
        "invalidated" => return Ok(PythonReply::Invalidated),
        "unsupported" => PreparedResult::Unsupported,
        "invalid" => PreparedResult::Invalid {
            http_status: client_error_status(
                result
                    .get_item("http_status")?
                    .and_then(|status| status.extract().ok()),
            ),
        },
        "unavailable" => PreparedResult::Unavailable,
        "exact" => {
            let (Some(tokens), Some(id), Some(epoch), Some(eligible)) = (
                result.get_item("token_ids")?,
                result.get_item("contract_id")?,
                result.get_item("epoch")?,
                result.get_item("cache_eligible")?,
            ) else {
                return Ok(PythonReply::Prepared(PreparedResult::Unavailable));
            };
            let tokens = tokens.cast::<PyList>()?;
            let id = id.extract::<&str>()?;
            if tokens.is_empty() || tokens.len() > limit || id.is_empty() || id.len() > 256 {
                return Ok(PythonReply::Prepared(PreparedResult::Unsupported));
            }
            let token_ids = tokens
                .iter()
                .map(|token| token.extract::<u32>())
                .collect::<PyResult<Vec<_>>>()?;
            PreparedResult::Exact(PreparedTokens {
                token_ids: token_ids.into(),
                contract: RenderContract {
                    id: id.to_owned(),
                    epoch: epoch.extract()?,
                },
                cache_eligible: eligible.extract()?,
            })
        }
        _ => PreparedResult::Unavailable,
    };
    Ok(PythonReply::Prepared(prepared))
}

pub struct RenderBridge {
    /// Optional, verified control-plane input for the separate Dense event path.
    pub capability_cohort: Option<crate::kv_capabilities::CapabilityCohort>,
    shared: Arc<Shared>,
    sender: SyncSender<Job>,
    closed: watch::Receiver<bool>,
    ready: watch::Receiver<Readiness>,
}

#[derive(Clone, Copy)]
enum Readiness {
    Pending,
    Ready,
    Failed,
}

struct ExecutorFinished {
    shared: Arc<Shared>,
    closed: watch::Sender<bool>,
}

impl Drop for ExecutorFinished {
    fn drop(&mut self) {
        // This guard belongs to the outer thread frame: run_executor and its
        // executor/Python references have already returned or unwound first.
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.accepting.store(false, Ordering::Release);
        let _ = self.closed.send(true);
    }
}

impl std::fmt::Debug for RenderBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderBridge")
            .field("contract", &self.shared.contract)
            .field("accepting", &self.shared.accepting.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl RenderBridge {
    pub fn with_capabilities(
        mut self,
        cohort: Option<crate::kv_capabilities::CapabilityCohort>,
    ) -> Self {
        self.capability_cohort = cohort;
        self
    }

    #[cfg(test)]
    pub(crate) fn for_test<F>(contract: RenderContract, limits: BridgeLimits, render: F) -> Self
    where
        F: FnMut(RequestKind, &[u8]) -> PreparedResult + Send + 'static,
    {
        struct ClosureExecutor<F>(F);

        impl<F> Executor for ClosureExecutor<F>
        where
            F: FnMut(RequestKind, &[u8]) -> PreparedResult + Send + 'static,
        {
            fn render(&mut self, kind: RequestKind, raw: &[u8], _: usize) -> PreparedResult {
                (self.0)(kind, raw)
            }
        }

        Self::with_executor(Box::new(ClosureExecutor(render)), contract, limits).unwrap()
    }

    /// `facade.startup()` verifies deployment/worker conformance on the execution
    /// thread and owns its persistent event loop. Await `wait_ready` before the
    /// server accepts any traffic; failed startup is not a fallback condition.
    /// Construction starts one Python lifetime waiter in the existing
    /// interpreter; it never initializes an interpreter or invokes rendering.
    pub fn new(
        facade: Py<PyAny>,
        contract: RenderContract,
        limits: BridgeLimits,
    ) -> Result<Self, String> {
        // Do not create a Python lifetime thread for invalid configuration.
        limits.validate()?;
        if contract.id.is_empty() || contract.id.len() > 256 {
            return Err("invalid render contract identity".into());
        }
        let lifetime = PythonLifetime::new()
            .map_err(|_| "could not start Python render lifetime guard".to_string())?;
        Self::with_executor(
            Box::new(PythonExecutor {
                facade: Some(facade),
                contract: contract.clone(),
                invalidated: false,
                lifetime,
            }),
            contract,
            limits,
        )
    }

    fn with_executor(
        executor: Box<dyn Executor>,
        contract: RenderContract,
        limits: BridgeLimits,
    ) -> Result<Self, String> {
        limits.validate()?;
        if contract.id.is_empty() || contract.id.len() > 256 {
            return Err("invalid render contract identity".into());
        }
        let (sender, receiver) = mpsc::sync_channel(limits.max_pending_jobs);
        let (closed_sender, closed) = watch::channel(false);
        let (ready_sender, ready) = watch::channel(Readiness::Pending);
        let shared = Arc::new(Shared {
            accepting: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            budget: Mutex::new(Budget::default()),
            contract,
            limits,
        });
        let worker_shared = Arc::clone(&shared);
        // Intentionally detach the handle. A Python computation cannot be killed
        // safely; callers may explicitly await bounded shutdown acknowledgment.
        std::thread::Builder::new()
            .name("vllm-render-bridge".into())
            .spawn(move || {
                let _finished = ExecutorFinished {
                    shared: Arc::clone(&worker_shared),
                    closed: closed_sender.clone(),
                };
                run_executor(
                    executor,
                    receiver,
                    worker_shared,
                    closed_sender,
                    ready_sender,
                )
            })
            .map_err(|_| "could not start render bridge thread".to_string())?;
        Ok(Self {
            capability_cohort: None,
            shared,
            sender,
            closed,
            ready,
        })
    }

    /// A startup timeout permanently fences this instance, including a late
    /// successful conformance result. It never silently enables fallback.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), String> {
        let mut ready = self.ready.clone();
        let result = tokio::time::timeout(timeout, async move {
            loop {
                match *ready.borrow_and_update() {
                    Readiness::Ready => return Ok(()),
                    Readiness::Failed => return Err("render bridge startup failed".to_string()),
                    Readiness::Pending => {}
                }
                if ready.changed().await.is_err() {
                    return Err("render bridge startup stopped".to_string());
                }
            }
        })
        .await
        .unwrap_or_else(|_| Err("render bridge startup deadline exceeded".into()));
        let result = result.and_then(|()| {
            if self.shared.accepting.load(Ordering::Acquire) {
                Ok(())
            } else {
                Err("render bridge is no longer available".into())
            }
        });
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    pub fn current_contract(&self) -> Option<RenderContract> {
        self.shared
            .accepting
            .load(Ordering::Acquire)
            .then(|| self.shared.contract.clone())
    }

    pub fn is_current(&self, contract: &RenderContract) -> bool {
        self.shared.accepting.load(Ordering::Acquire) && contract == &self.shared.contract
    }

    /// Epoch/identity changes require a new verified facade/bridge. Never revive
    /// an old provider or reuse its prepared tokens after invalidation.
    pub fn invalidate(&self) {
        let _budget = self.shared.budget.lock();
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.accepting.store(false, Ordering::Release);
    }

    pub fn shutdown(&self) {
        self.invalidate();
    }

    /// `false` means Python may still be executing: it is not safe-finalization
    /// evidence. This does not terminate the computation or release its budget.
    pub async fn wait_closed(&self, timeout: Duration) -> bool {
        let mut closed = self.closed.clone();
        tokio::time::timeout(timeout, async move {
            loop {
                if *closed.borrow_and_update() {
                    return true;
                }
                if closed.changed().await.is_err() {
                    return false;
                }
            }
        })
        .await
        .unwrap_or(false)
    }

    pub async fn prepare(&self, kind: RequestKind, raw: Arc<[u8]>) -> PreparedResult {
        let reservation = match self.reserve_input(raw.len()) {
            Ok(reservation) => reservation,
            Err(result) => return result,
        };
        self.prepare_reserved(kind, raw, reservation).await
    }

    /// HTTP ingress borrows its original body until all admission budgets have
    /// been reserved. Oversized/busy requests never allocate a bridge copy.
    /// The job reservation includes active and not-yet-enqueued jobs, so there
    /// is always a bounded queue slot for a successfully admitted request.
    pub async fn prepare_bytes(&self, kind: RequestKind, raw: &[u8]) -> PreparedResult {
        let reservation = match self.reserve_input(raw.len()) {
            Ok(reservation) => reservation,
            Err(result) => return result,
        };
        self.prepare_reserved(kind, Arc::from(raw), reservation)
            .await
    }

    fn reserve_input(&self, bytes: usize) -> Result<Reservation, PreparedResult> {
        if !self.shared.accepting.load(Ordering::Acquire) {
            return Err(PreparedResult::Unavailable);
        }
        self.shared.reserve(bytes).ok_or_else(|| {
            if self.shared.accepting.load(Ordering::Acquire) {
                PreparedResult::Busy
            } else {
                PreparedResult::Unavailable
            }
        })
    }

    async fn prepare_reserved(
        &self,
        kind: RequestKind,
        raw: Arc<[u8]>,
        reservation: Reservation,
    ) -> PreparedResult {
        let cancelled = Arc::new(AtomicBool::new(false));
        let _cancel = CancelOnDrop(Arc::clone(&cancelled));
        let (started_sender, mut started_receiver) = oneshot::channel();
        let (result_sender, mut result_receiver) = oneshot::channel();
        let queue_deadline = Instant::now() + self.shared.limits.queue_timeout;
        let job = Job {
            kind,
            raw,
            cancelled,
            queue_deadline,
            started: started_sender,
            result: result_sender,
            _reservation: reservation,
        };
        if let Err(error) = self.sender.try_send(job) {
            return match error {
                mpsc::TrySendError::Full(_) => PreparedResult::Busy,
                mpsc::TrySendError::Disconnected(_) => PreparedResult::Unavailable,
            };
        }
        let execution_deadline = tokio::select! {
            biased;
            result = &mut result_receiver => return result.unwrap_or(PreparedResult::Unavailable),
            started = &mut started_receiver => match started {
                Ok(deadline) => deadline,
                Err(_) => return result_receiver.await.unwrap_or(PreparedResult::Unavailable),
            },
            _ = tokio::time::sleep_until(queue_deadline.into()) => {
                return PreparedResult::Deadline(DeadlineStage::Queue);
            }
        };
        tokio::select! {
            biased;
            result = &mut result_receiver => result.unwrap_or(PreparedResult::Unavailable),
            _ = tokio::time::sleep_until(execution_deadline.into()) => {
                PreparedResult::Deadline(DeadlineStage::Execution)
            }
        }
    }
}

impl Drop for RenderBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_executor(
    mut executor: Box<dyn Executor>,
    receiver: Receiver<Job>,
    shared: Arc<Shared>,
    closed: watch::Sender<bool>,
    ready: watch::Sender<Readiness>,
) {
    let startup = executor.startup();
    {
        let _budget = shared.budget.lock();
        if startup.is_ok() && !shared.stopping.load(Ordering::Acquire) {
            shared.accepting.store(true, Ordering::Release);
            let _ = ready.send(Readiness::Ready);
        } else {
            shared.stopping.store(true, Ordering::Release);
            let _ = ready.send(Readiness::Failed);
        }
    }
    while shared.accepting.load(Ordering::Acquire) {
        let job = match receiver.recv_timeout(Duration::from_millis(25)) {
            Ok(job) => job,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let result = if !shared.accepting.load(Ordering::Acquire) {
            PreparedResult::Unavailable
        } else if job.cancelled.load(Ordering::Acquire) {
            PreparedResult::Cancelled
        } else if Instant::now() >= job.queue_deadline {
            PreparedResult::Deadline(DeadlineStage::Queue)
        } else {
            let deadline = Instant::now() + shared.limits.execution_timeout;
            if job.started.send(deadline).is_err() || job.cancelled.load(Ordering::Acquire) {
                PreparedResult::Cancelled
            } else {
                let result =
                    executor.render(job.kind, &job.raw, shared.limits.max_tokens_per_request);
                if executor.is_invalidated() {
                    shared.stopping.store(true, Ordering::Release);
                    shared.accepting.store(false, Ordering::Release);
                }
                let result = match result {
                    PreparedResult::Invalid { http_status } => PreparedResult::Invalid {
                        http_status: client_error_status(Some(http_status)),
                    },
                    PreparedResult::Exact(tokens) if tokens.contract != shared.contract => {
                        shared.accepting.store(false, Ordering::Release);
                        PreparedResult::Unavailable
                    }
                    PreparedResult::Exact(tokens)
                        if tokens.token_ids.is_empty()
                            || tokens.token_ids.len() > shared.limits.max_tokens_per_request =>
                    {
                        PreparedResult::Unsupported
                    }
                    result => result,
                };
                if !shared.accepting.load(Ordering::Acquire) {
                    PreparedResult::Unavailable
                } else if job.cancelled.load(Ordering::Acquire) {
                    PreparedResult::Cancelled
                } else if Instant::now() >= deadline {
                    PreparedResult::Deadline(DeadlineStage::Execution)
                } else {
                    result
                }
            }
        };
        let _ = job.result.send(result);
        // All reservations are dropped only after the actual call returned.
    }
    shared.accepting.store(false, Ordering::Release);
    for job in receiver.try_iter() {
        let _ = job.result.send(PreparedResult::Unavailable);
    }
    executor.close();
    drop(executor);
    let _ = closed.send(true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn invalid_status_is_limited_to_client_errors() {
        for status in [
            None,
            Some(0),
            Some(200),
            Some(399),
            Some(500),
            Some(599),
            Some(u16::MAX),
        ] {
            assert_eq!(client_error_status(status), 400);
        }
        for status in [400, 401, 403, 404, 422, 429, 499] {
            assert_eq!(client_error_status(Some(status)), status);
        }
    }

    struct ControlledExecutor {
        entered: tokio::sync::mpsc::UnboundedSender<()>,
        release: Receiver<()>,
        calls: Arc<AtomicUsize>,
        was_closed: Arc<AtomicBool>,
        result: PreparedResult,
        invalidated: bool,
    }

    impl Executor for ControlledExecutor {
        fn render(&mut self, _: RequestKind, _: &[u8], _: usize) -> PreparedResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _ = self.entered.send(());
            self.release.recv_timeout(Duration::from_secs(3)).unwrap();
            self.result.clone()
        }

        fn close(&mut self) {
            self.was_closed.store(true, Ordering::SeqCst);
        }

        fn is_invalidated(&self) -> bool {
            self.invalidated
        }
    }

    struct Harness {
        bridge: Arc<RenderBridge>,
        entered: tokio::sync::mpsc::UnboundedReceiver<()>,
        release: mpsc::Sender<()>,
        calls: Arc<AtomicUsize>,
        was_closed: Arc<AtomicBool>,
    }

    fn contract() -> RenderContract {
        RenderContract {
            id: "verified-test-contract".into(),
            epoch: 7,
        }
    }

    fn exact() -> PreparedResult {
        PreparedResult::Exact(PreparedTokens {
            token_ids: Arc::from([1, 2, 3]),
            contract: contract(),
            cache_eligible: true,
        })
    }

    fn harness(limits: BridgeLimits, result: PreparedResult) -> Harness {
        harness_with_invalidation(limits, result, false)
    }

    fn harness_with_invalidation(
        limits: BridgeLimits,
        result: PreparedResult,
        invalidated: bool,
    ) -> Harness {
        let (entered_sender, entered) = tokio::sync::mpsc::unbounded_channel();
        let (release, release_receiver) = mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let was_closed = Arc::new(AtomicBool::new(false));
        let bridge = RenderBridge::with_executor(
            Box::new(ControlledExecutor {
                entered: entered_sender,
                release: release_receiver,
                calls: Arc::clone(&calls),
                was_closed: Arc::clone(&was_closed),
                result,
                invalidated,
            }),
            contract(),
            limits,
        )
        .unwrap();
        Harness {
            bridge: Arc::new(bridge),
            entered,
            release,
            calls,
            was_closed,
        }
    }

    impl Harness {
        fn request(&self, bytes: usize) -> tokio::task::JoinHandle<PreparedResult> {
            let bridge = Arc::clone(&self.bridge);
            tokio::spawn(async move {
                bridge.wait_ready(Duration::from_secs(1)).await.unwrap();
                bridge
                    .prepare(RequestKind::Chat, vec![b'x'; bytes].into())
                    .await
            })
        }

        async fn entered(&mut self) {
            tokio::time::timeout(Duration::from_secs(1), self.entered.recv())
                .await
                .unwrap()
                .unwrap();
        }

        async fn close(&self) {
            self.bridge.shutdown();
            assert!(self.bridge.wait_closed(Duration::from_secs(1)).await);
            assert!(self.was_closed.load(Ordering::SeqCst));
            let budget = self.bridge.shared.budget.lock();
            assert_eq!((budget.jobs, budget.bytes, budget.tokens), (0, 0, 0));
        }
    }

    #[tokio::test]
    async fn owned_exact_result_and_stable_contract() {
        let mut h = harness(BridgeLimits::default(), exact());
        let request = h.request(20);
        h.entered().await;
        h.release.send(()).unwrap();
        let PreparedResult::Exact(tokens) = request.await.unwrap() else {
            panic!("expected exact tokens");
        };
        assert_eq!(&*tokens.token_ids, &[1, 2, 3]);
        assert!(h.bridge.is_current(&tokens.contract));
        h.close().await;
        assert!(!h.bridge.is_current(&tokens.contract));
    }

    #[tokio::test]
    async fn overload_accounts_for_jobs_bytes_and_token_reservations() {
        for limits in [
            BridgeLimits {
                max_pending_jobs: 1,
                ..BridgeLimits::default()
            },
            BridgeLimits {
                max_input_bytes: 20,
                ..BridgeLimits::default()
            },
            BridgeLimits {
                max_reserved_tokens: 65_536,
                ..BridgeLimits::default()
            },
        ] {
            let mut h = harness(limits, exact());
            let request = h.request(20);
            h.entered().await;
            assert!(matches!(
                h.bridge.prepare(RequestKind::Chat, Arc::from([b'y'])).await,
                PreparedResult::Busy
            ));
            assert_eq!(h.calls.load(Ordering::SeqCst), 1);
            h.release.send(()).unwrap();
            assert!(matches!(request.await.unwrap(), PreparedResult::Exact(_)));
            h.close().await;
        }
    }

    #[tokio::test]
    async fn borrowed_ingress_rejects_before_admission_and_reserves_once() {
        let mut h = harness(
            BridgeLimits {
                max_input_bytes: 16,
                ..BridgeLimits::default()
            },
            exact(),
        );
        h.bridge.wait_ready(Duration::from_secs(1)).await.unwrap();
        assert!(matches!(
            h.bridge.prepare_bytes(RequestKind::Chat, &[b'x'; 17]).await,
            PreparedResult::Busy
        ));
        assert_eq!(h.bridge.shared.budget.lock().jobs, 0);
        assert_eq!(h.calls.load(Ordering::SeqCst), 0);
        let bridge = Arc::clone(&h.bridge);
        let request =
            tokio::spawn(async move { bridge.prepare_bytes(RequestKind::Chat, b"admitted").await });
        h.entered().await;
        {
            let budget = h.bridge.shared.budget.lock();
            assert_eq!((budget.jobs, budget.bytes, budget.tokens), (1, 8, 65_536));
        }
        h.release.send(()).unwrap();
        assert!(matches!(request.await.unwrap(), PreparedResult::Exact(_)));
        h.close().await;
    }

    #[tokio::test]
    async fn active_timeout_does_not_release_capacity_or_deliver_late_result() {
        let mut h = harness(
            BridgeLimits {
                max_pending_jobs: 1,
                execution_timeout: Duration::from_millis(20),
                ..BridgeLimits::default()
            },
            exact(),
        );
        let request = h.request(20);
        h.entered().await;
        assert!(matches!(
            request.await.unwrap(),
            PreparedResult::Deadline(DeadlineStage::Execution)
        ));
        assert_eq!(h.bridge.shared.budget.lock().jobs, 1);
        assert!(matches!(
            h.bridge.prepare(RequestKind::Chat, Arc::from([b'y'])).await,
            PreparedResult::Busy
        ));
        h.bridge.shutdown();
        assert!(!h.bridge.wait_closed(Duration::from_millis(10)).await);
        assert_eq!(h.bridge.shared.budget.lock().jobs, 1);
        h.release.send(()).unwrap();
        h.close().await;
    }

    #[tokio::test]
    async fn cancelled_active_and_queued_requests_do_not_spawn_replacements() {
        let mut h = harness(BridgeLimits::default(), exact());
        let active = h.request(20);
        h.entered().await;
        let queued = h.request(30);
        tokio::time::timeout(Duration::from_secs(1), async {
            while h.bridge.shared.budget.lock().jobs != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        queued.abort();
        assert!(queued.await.unwrap_err().is_cancelled());
        active.abort();
        assert!(active.await.unwrap_err().is_cancelled());
        assert_eq!(h.bridge.shared.budget.lock().jobs, 2);
        h.release.send(()).unwrap();
        h.close().await;
        assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn queue_deadline_does_not_execute_expired_work() {
        let mut h = harness(
            BridgeLimits {
                queue_timeout: Duration::from_millis(20),
                ..BridgeLimits::default()
            },
            exact(),
        );
        let active = h.request(20);
        h.entered().await;
        let queued = h.request(20);
        assert!(matches!(
            queued.await.unwrap(),
            PreparedResult::Deadline(DeadlineStage::Queue)
        ));
        h.release.send(()).unwrap();
        assert!(matches!(active.await.unwrap(), PreparedResult::Exact(_)));
        h.close().await;
        assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn changed_epoch_permanently_disables_provider() {
        let PreparedResult::Exact(mut tokens) = exact() else {
            unreachable!();
        };
        tokens.contract.epoch += 1;
        let mut h = harness(BridgeLimits::default(), PreparedResult::Exact(tokens));
        let request = h.request(20);
        h.entered().await;
        h.release.send(()).unwrap();
        assert!(matches!(
            request.await.unwrap(),
            PreparedResult::Unavailable
        ));
        assert!(h.bridge.current_contract().is_none());
        assert!(matches!(
            h.bridge.prepare(RequestKind::Chat, Arc::from([b'y'])).await,
            PreparedResult::Unavailable
        ));
        h.close().await;
    }

    #[tokio::test]
    async fn provider_failure_releases_budget_and_shutdown_closes_executor() {
        let mut h = harness(BridgeLimits::default(), PreparedResult::Unavailable);
        let request = h.request(20);
        h.entered().await;
        h.release.send(()).unwrap();
        assert!(matches!(
            request.await.unwrap(),
            PreparedResult::Unavailable
        ));
        h.close().await;
    }

    #[tokio::test]
    async fn explicit_invalidation_cannot_become_an_unavailable_fallback() {
        let mut h =
            harness_with_invalidation(BridgeLimits::default(), PreparedResult::Unavailable, true);
        let request = h.request(20);
        h.entered().await;
        h.release.send(()).unwrap();
        assert!(matches!(
            request.await.unwrap(),
            PreparedResult::Unavailable
        ));
        assert!(h.bridge.current_contract().is_none());
        assert!(!h.bridge.is_current(&contract()));
        h.close().await;
    }

    #[tokio::test]
    async fn shutdown_rejects_active_result_and_waiting_work() {
        let mut h = harness(BridgeLimits::default(), exact());
        let active = h.request(20);
        h.entered().await;
        let queued = h.request(20);
        tokio::time::timeout(Duration::from_secs(1), async {
            while h.bridge.shared.budget.lock().jobs != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        h.bridge.shutdown();
        h.release.send(()).unwrap();
        assert!(matches!(active.await.unwrap(), PreparedResult::Unavailable));
        assert!(matches!(queued.await.unwrap(), PreparedResult::Unavailable));
        h.close().await;
        assert_eq!(h.calls.load(Ordering::SeqCst), 1);
    }

    struct StartupExecutor {
        entered: Option<oneshot::Sender<()>>,
        release: Receiver<bool>,
        was_closed: Arc<AtomicBool>,
    }

    impl Executor for StartupExecutor {
        fn startup(&mut self) -> Result<(), ()> {
            let _ = self.entered.take().unwrap().send(());
            if self.release.recv_timeout(Duration::from_secs(3)).unwrap() {
                Ok(())
            } else {
                Err(())
            }
        }

        fn render(&mut self, _: RequestKind, _: &[u8], _: usize) -> PreparedResult {
            panic!("unverified startup must never render");
        }

        fn close(&mut self) {
            self.was_closed.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn failed_or_late_startup_never_activates_provider() {
        for timeout in [false, true] {
            let (entered_sender, entered) = oneshot::channel();
            let (release, release_receiver) = mpsc::channel();
            let was_closed = Arc::new(AtomicBool::new(false));
            let bridge = RenderBridge::with_executor(
                Box::new(StartupExecutor {
                    entered: Some(entered_sender),
                    release: release_receiver,
                    was_closed: Arc::clone(&was_closed),
                }),
                contract(),
                BridgeLimits::default(),
            )
            .unwrap();
            tokio::time::timeout(Duration::from_secs(1), entered)
                .await
                .unwrap()
                .unwrap();
            assert!(bridge.current_contract().is_none());
            assert!(matches!(
                bridge.prepare(RequestKind::Chat, Arc::from([b'y'])).await,
                PreparedResult::Unavailable
            ));
            if timeout {
                assert!(bridge.wait_ready(Duration::from_millis(10)).await.is_err());
                release.send(true).unwrap();
            } else {
                release.send(false).unwrap();
                assert!(bridge.wait_ready(Duration::from_secs(1)).await.is_err());
            }
            assert!(bridge.wait_closed(Duration::from_secs(1)).await);
            assert!(bridge.current_contract().is_none());
            assert!(was_closed.load(Ordering::SeqCst));
        }
    }

    #[tokio::test]
    async fn executor_panic_fences_contract_and_acknowledges_disposal() {
        let bridge = RenderBridge::for_test(contract(), BridgeLimits::default(), |_, _| {
            panic!("synthetic executor panic");
        });
        bridge.wait_ready(Duration::from_secs(1)).await.unwrap();
        assert!(matches!(
            bridge.prepare_bytes(RequestKind::Chat, b"synthetic").await,
            PreparedResult::Unavailable
        ));
        assert!(bridge.wait_closed(Duration::from_secs(1)).await);
        assert!(bridge.current_contract().is_none());
        assert!(!bridge.is_current(&contract()));
        let budget = bridge.shared.budget.lock();
        assert_eq!((budget.jobs, budget.bytes, budget.tokens), (0, 0, 0));
    }
}

//! Automatic-mode subscriber. Metadata gates observed-subset ownership, never
//! synthesizes it. This pool is deliberately separate from the unchanged native
//! subscriber and requires the linked epoch-topic Worker export proposal.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};
use tracing::warn;

use crate::{
    kv_capabilities::{CapabilityClient, CapabilityCohort, Descriptor, FetchError},
    kv_index::{KVBlockIndex, OwnershipEvent},
    prompt_tokens::bridge::RenderBridge,
};

use super::{
    decoder::{decode_batch, MAX_PAYLOAD_BYTES},
    resolve_endpoints,
};

const REFRESH_INTERVAL: Duration = Duration::from_secs(30);
const RETRY_INTERVAL: Duration = Duration::from_secs(2);

pub struct CapabilityEventPool {
    index: Arc<KVBlockIndex>,
    workers: Vec<String>,
    stopping: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for CapabilityEventPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapabilityEventPool")
            .field("workers", &self.workers)
            .finish_non_exhaustive()
    }
}

impl CapabilityEventPool {
    pub fn start(
        endpoints: Vec<(String, String)>,
        index: Arc<KVBlockIndex>,
        cohort: CapabilityCohort,
        bridge: Arc<RenderBridge>,
    ) -> Result<Self, String> {
        let initial = cohort
            .workers
            .values()
            .next()
            .ok_or("empty capability cohort")?;
        let model = initial
            .namespace
            .served_model_names
            .first()
            .ok_or("empty served namespace")?;
        cohort.validate(
            &endpoints,
            initial.hash.block_tokens,
            initial.hash.seed,
            model,
        )?;
        let urls: Vec<_> = endpoints.iter().map(|(worker, _)| worker.clone()).collect();
        let endpoints = resolve_endpoints(&urls, &endpoints, 5557)?;
        let mut pool = Self {
            index,
            workers: Vec::new(),
            stopping: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
        };
        for (ordinal, (worker, endpoint)) in endpoints.into_iter().enumerate() {
            let initial = cohort.workers[&worker].clone();
            let generation = pool.index.begin_worker(&worker);
            pool.workers.push(worker.clone());
            let index = pool.index.clone();
            let stopping = pool.stopping.clone();
            let bridge = bridge.clone();
            let key_env = cohort.api_key_env.clone();
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            pool.threads.push(
                thread::Builder::new()
                    .name(format!("kv-capabilities-{ordinal}"))
                    .spawn(move || {
                        subscriber(
                            worker, endpoint, initial, generation, index, stopping, bridge,
                            key_env, ready_tx,
                        );
                    })
                    .map_err(|error| format!("cannot start capability subscriber: {error}"))?,
            );
            ready_rx
                .recv_timeout(Duration::from_secs(5))
                .map_err(|_| "capability subscriber startup deadline")??;
        }
        Ok(pool)
    }
}

impl Drop for CapabilityEventPool {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        for worker in &self.workers {
            self.index.retire_worker(worker);
        }
        for handle in self.threads.drain(..) {
            handle.thread().unpark();
            if handle.join().is_err() {
                warn!("capability subscriber panicked during shutdown");
            }
        }
    }
}

struct Subscription {
    socket: zmq::Socket,
    monitor: zmq::Socket,
}

impl Subscription {
    fn open(context: &zmq::Context, endpoint: &str, topic: &str) -> Result<Self, zmq::Error> {
        static NEXT_MONITOR: AtomicU64 = AtomicU64::new(0);
        let address = format!(
            "inproc://kv-capability-monitor-{}",
            NEXT_MONITOR.fetch_add(1, Ordering::Relaxed)
        );
        let socket = context.socket(zmq::SUB)?;
        socket.set_linger(0)?;
        socket.set_ipv6(true)?;
        socket.set_rcvhwm(1024)?;
        socket.set_maxmsgsize(MAX_PAYLOAD_BYTES as i64)?;
        socket.set_subscribe(topic.as_bytes())?;
        socket.monitor(&address, zmq::SocketEvent::DISCONNECTED as i32)?;
        let monitor = context.socket(zmq::PAIR)?;
        monitor.set_linger(0)?;
        monitor.connect(&address)?;
        socket.connect(endpoint)?;
        Ok(Self { socket, monitor })
    }
}

struct Session {
    descriptor: Descriptor,
    latest_metadata_watermark: u64,
    previous: Option<(u64, [u8; 32])>,
}

#[derive(Debug, PartialEq, Eq)]
enum Received {
    Ignore,
    Rebootstrap,
    Events(Vec<OwnershipEvent>),
}

impl Session {
    fn receive(&mut self, frames: &[Vec<u8>]) -> Received {
        if frames.len() != 3
            || frames[0] != self.descriptor.events.topic.as_bytes()
            || frames[1].len() != 8
        {
            return Received::Rebootstrap;
        }
        let sequence = u64::from_be_bytes(frames[1].as_slice().try_into().unwrap());
        // Metadata's next-sequence watermark rejects old queued data even
        // within the same boot epoch. A new epoch has a different exact topic.
        if sequence < self.descriptor.events.next_sequence {
            return Received::Ignore;
        }
        let identity = (sequence, Sha256::digest(&frames[2]).into());
        if self.previous == Some(identity) {
            return Received::Ignore;
        }
        if self
            .previous
            .is_some_and(|previous| previous.0.checked_add(1) != Some(sequence))
        {
            return Received::Rebootstrap;
        }
        match decode_batch(&frames[2], self.descriptor.hash.block_tokens) {
            Ok(events) if !events.contains(&OwnershipEvent::Clear) => {
                self.previous = Some(identity);
                Received::Events(events)
            }
            // AllBlocksCleared revokes the current observation generation.
            // Discard any same-batch stores and re-read metadata, rather than
            // treating a successful refresh as permission to revive old data.
            _ => Received::Rebootstrap,
        }
    }

    fn watermark_valid(&self, next: &Descriptor) -> bool {
        next.events.next_sequence >= self.latest_metadata_watermark
            && self
                .previous
                .is_none_or(|(sequence, _)| next.events.next_sequence > sequence)
    }
}

fn revoke(index: &KVBlockIndex, worker: &str, generation: &mut u64) {
    if let Some(next) = index.roll_worker(worker, *generation) {
        *generation = next;
    }
}

#[allow(clippy::too_many_arguments)]
fn subscriber(
    worker: String,
    endpoint: String,
    initial: Descriptor,
    mut generation: u64,
    index: Arc<KVBlockIndex>,
    stopping: Arc<AtomicBool>,
    bridge: Arc<RenderBridge>,
    key_env: Option<String>,
    ready: mpsc::SyncSender<Result<(), String>>,
) {
    struct Retire(Arc<KVBlockIndex>, String);
    impl Drop for Retire {
        fn drop(&mut self) {
            self.0.retire_worker(&self.1);
        }
    }
    let _retire = Retire(index.clone(), worker.clone());
    // Client construction, requests, body reads and destruction all occur on
    // this owned OS thread, outside Tokio and outside the Python executor.
    let client = match CapabilityClient::new(key_env.as_deref()) {
        Ok(client) => client,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let context = zmq::Context::new();
    let mut subscription = None;
    let mut session: Option<Session> = None;
    let mut ready = Some(ready);
    let mut next_refresh = Instant::now();
    while !stopping.load(Ordering::Acquire) {
        if bridge.current_contract().is_none() {
            break;
        }
        let Some(current) = index.current_generation(&worker) else {
            // Only the health/registration owner can reactivate a Worker.
            subscription = None;
            session = None;
            next_refresh = Instant::now();
            thread::park_timeout(Duration::from_millis(100));
            continue;
        };
        if current != generation {
            generation = current;
            subscription = None;
            session = None;
            next_refresh = Instant::now();
        }
        if Instant::now() >= next_refresh {
            let expected_generation = generation;
            let fetched = client.fetch(&worker);
            if stopping.load(Ordering::Acquire) {
                break;
            }
            // A late reply after removal/health recovery must not bootstrap a
            // replacement generation, invalidate its bridge or apply events.
            if index.current_generation(&worker) != Some(expected_generation) {
                continue;
            }
            let descriptor = match fetched {
                Ok(descriptor) => descriptor,
                Err(error) => {
                    revoke(&index, &worker, &mut generation);
                    subscription = None;
                    session = None;
                    if error == FetchError::Unsupported {
                        index.with_current_generation(&worker, generation, || bridge.invalidate());
                    }
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Err(format!("Worker capability export: {error}")));
                        return;
                    }
                    warn!(worker, %error, "Worker capability unavailable; affinity revoked");
                    next_refresh = Instant::now() + RETRY_INTERVAL;
                    continue;
                }
            };
            // A replacement boot may load different tokenizer/template files
            // under the same path. Namespace equality alone cannot renew the
            // old input conformance; require Router restart and new probes.
            if !descriptor.compatible_with(&initial) || !descriptor.same_publisher(&initial) {
                revoke(&index, &worker, &mut generation);
                index.with_current_generation(&worker, generation, || bridge.invalidate());
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Err(
                        "Worker capability namespace/mechanism/boot changed; restart Router".into(),
                    ));
                }
                break;
            }
            let validation = descriptor.validate(
                &endpoint,
                initial.hash.block_tokens,
                initial.hash.seed,
                &initial.namespace.served_model_names[0],
            );
            if let Err(error) = validation {
                revoke(&index, &worker, &mut generation);
                subscription = None;
                session = None;
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Err(error));
                    return;
                }
                warn!(worker, %error, "invalid Worker capability; affinity revoked");
                next_refresh = Instant::now() + RETRY_INTERVAL;
                continue;
            }
            let unchanged = session
                .as_ref()
                .is_some_and(|previous| previous.descriptor.same_publisher(&descriptor));
            if unchanged {
                if !session.as_ref().unwrap().watermark_valid(&descriptor) {
                    revoke(&index, &worker, &mut generation);
                    subscription = None;
                    session = None;
                    next_refresh = Instant::now() + RETRY_INTERVAL;
                    continue;
                }
                // Do not advance this live session's bootstrap watermark: it
                // may have queued contiguous events below the new snapshot.
                // Nor does refresh populate or restore any ownership entry.
                session.as_mut().unwrap().latest_metadata_watermark =
                    descriptor.events.next_sequence;
            } else {
                if session.is_some() {
                    revoke(&index, &worker, &mut generation);
                }
                subscription = None;
                session = None;
                match Subscription::open(&context, &endpoint, &descriptor.events.topic) {
                    Ok(socket) => {
                        subscription = Some(socket);
                        session = Some(Session {
                            latest_metadata_watermark: descriptor.events.next_sequence,
                            descriptor,
                            previous: None,
                        });
                    }
                    Err(error) => {
                        revoke(&index, &worker, &mut generation);
                        if let Some(ready) = ready.take() {
                            let _ = ready.send(Err(format!("capability subscription: {error}")));
                            return;
                        }
                        next_refresh = Instant::now() + RETRY_INTERVAL;
                        continue;
                    }
                }
            }
            next_refresh = Instant::now() + REFRESH_INTERVAL;
            if let Some(ready) = ready.take() {
                let _ = ready.send(Ok(()));
            }
        }
        let Some(active) = subscription.as_ref() else {
            thread::park_timeout(Duration::from_millis(100));
            continue;
        };
        let mut items = [
            active.monitor.as_poll_item(zmq::POLLIN),
            active.socket.as_poll_item(zmq::POLLIN),
        ];
        let outcome = match zmq::poll(&mut items, 100) {
            Err(zmq::Error::EINTR) => continue,
            Err(_) => Received::Rebootstrap,
            Ok(_) if items[0].is_readable() => {
                let _ = active.monitor.recv_multipart(zmq::DONTWAIT);
                Received::Rebootstrap
            }
            Ok(_) if items[1].is_readable() => match active.socket.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => session.as_mut().unwrap().receive(&frames),
                Err(zmq::Error::EAGAIN) => continue,
                Err(_) => Received::Rebootstrap,
            },
            Ok(_) => continue,
        };
        match outcome {
            Received::Ignore => {}
            Received::Events(events) => {
                // This write-lock transaction is the final generation fence.
                // A health callback racing decode cannot restore stale blocks.
                index.apply_batch(&worker, generation, &events);
            }
            Received::Rebootstrap => {
                revoke(&index, &worker, &mut generation);
                subscription = None;
                session = None;
                next_refresh = Instant::now();
            }
        }
    }
    if let Some(ready) = ready.take() {
        let _ = ready.send(Err("capability subscriber stopped during startup".into()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_capabilities::tests::descriptor;
    use rmpv::Value;

    fn frame(descriptor: &Descriptor, sequence: u64, events: Vec<Value>) -> Vec<Vec<u8>> {
        let mut payload = Vec::new();
        rmpv::encode::write_value(
            &mut payload,
            &Value::Array(vec![1.0.into(), Value::Array(events), 0.into()]),
        )
        .unwrap();
        vec![
            descriptor.events.topic.as_bytes().to_vec(),
            sequence.to_be_bytes().to_vec(),
            payload,
        ]
    }

    fn store() -> Value {
        Value::Map(vec![
            ("type".into(), "BlockStored".into()),
            (
                "block_hashes".into(),
                Value::Array(vec![Value::Binary(vec![7; 32])]),
            ),
            ("parent_block_hash".into(), Value::Nil),
            ("token_ids".into(), Value::Array(vec![1.into(); 16])),
            ("block_size".into(), 16.into()),
            ("lora_id".into(), Value::Nil),
            ("lora_name".into(), Value::Nil),
            ("medium".into(), "GPU".into()),
            ("group_idx".into(), 0.into()),
            ("kv_cache_spec_kind".into(), "full_attention".into()),
        ])
    }

    #[test]
    fn kv_capability_bootstrap_watermark_epoch_duplicates_and_gaps() {
        let mut descriptor = descriptor();
        descriptor.events.next_sequence = 10;
        let mut session = Session {
            latest_metadata_watermark: descriptor.events.next_sequence,
            descriptor: descriptor.clone(),
            previous: None,
        };
        assert_eq!(
            session.receive(&frame(&descriptor, 9, vec![store()])),
            Received::Ignore
        );
        let accepted = frame(&descriptor, 11, vec![store()]);
        assert_eq!(
            session.receive(&accepted),
            Received::Events(vec![OwnershipEvent::Store(vec![[7; 32]])])
        );
        assert_eq!(session.receive(&accepted), Received::Ignore);
        assert_eq!(
            session.receive(&frame(&descriptor, 13, vec![store()])),
            Received::Rebootstrap
        );
        let mut old_epoch = accepted;
        old_epoch[0] = b"kv.fedcba9876543210fedcba9876543210".to_vec();
        assert_eq!(session.receive(&old_epoch), Received::Rebootstrap);
        assert!(!session.watermark_valid(&descriptor));
        descriptor.events.next_sequence = 12;
        assert!(session.watermark_valid(&descriptor));
    }

    #[test]
    fn kv_capability_clear_empty_hash_and_stale_generation_cannot_revive() {
        let descriptor = descriptor();
        let index = KVBlockIndex::new(8);
        let mut generation = index.begin_worker("w");
        index.store("w", generation, &[[7; 32]]);
        let old = generation;
        let mut session = Session {
            latest_metadata_watermark: descriptor.events.next_sequence,
            descriptor: descriptor.clone(),
            previous: None,
        };
        let clear = Value::Map(vec![("type".into(), "AllBlocksCleared".into())]);
        assert_eq!(
            session.receive(&frame(&descriptor, 1, vec![clear, store()])),
            Received::Rebootstrap
        );
        revoke(&index, "w", &mut generation);
        assert_eq!(index.ownership_count(), 0);
        assert!(!index.store("w", old, &[[7; 32]]));
        index.retire_worker("w");
        revoke(&index, "w", &mut generation);
        assert_eq!(index.current_generation("w"), None);
        let mut empty = store();
        let Value::Map(fields) = &mut empty else {
            unreachable!()
        };
        for (key, value) in fields {
            if key.as_str() == Some("block_hashes") {
                *value = Value::Array(vec![]);
            }
        }
        assert_eq!(
            session.receive(&frame(&descriptor, 2, vec![empty])),
            Received::Rebootstrap
        );
    }

    #[test]
    fn kv_capability_successful_metadata_does_not_restore_blocks() {
        let index = KVBlockIndex::new(8);
        let old = index.begin_worker("w");
        index.store("w", old, &[[7; 32]]);
        let current = index.roll_worker("w", old).unwrap();
        let fetched_for_old_generation = descriptor();
        assert_ne!(index.current_generation("w"), Some(old));
        assert!(fetched_for_old_generation
            .validate("tcp://worker:5557", 16, 0, "model")
            .is_ok());
        assert_eq!(index.prefix_score("w", &[[7; 32]]), 0);
        assert!(!index.apply_batch("w", old, &[OwnershipEvent::Store(vec![[7; 32]])]));
        assert!(index.apply_batch("w", current, &[OwnershipEvent::Store(vec![[8; 32]])]));
        assert_eq!(index.prefix_score("w", &[[7; 32]]), 0);
    }

    /// Public local transport test: real bounded HTTP adapter and real ZMQ
    /// subscriber, with an explicitly synthetic Worker capability publisher.
    #[test]
    fn kv_capability_http_zmq_gap_revalidation_and_boot_change() {
        use crate::prompt_tokens::bridge::{BridgeLimits, PreparedResult, RenderContract};
        use std::{
            collections::HashMap,
            io::{Read, Write},
            net::TcpListener,
            sync::{atomic::AtomicUsize, Mutex},
        };

        fn wait_for(mut condition: impl FnMut() -> bool) {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !condition() {
                assert!(Instant::now() < deadline, "transport condition timed out");
                thread::sleep(Duration::from_millis(1));
            }
        }
        fn subscribed(publisher: &zmq::Socket, topic: &str) {
            wait_for(|| {
                if publisher.poll(zmq::POLLIN, 10).unwrap() == 0 {
                    return false;
                }
                let frame = publisher.recv_bytes(0).unwrap();
                frame.first() == Some(&1) && frame.get(1..) == Some(topic.as_bytes())
            });
        }
        fn publish(publisher: &zmq::Socket, descriptor: &Descriptor, sequence: u64) {
            publisher
                .send_multipart(frame(descriptor, sequence, vec![store()]), 0)
                .unwrap();
        }
        struct HttpServer {
            stopped: Arc<AtomicBool>,
            handle: Option<JoinHandle<()>>,
        }
        impl Drop for HttpServer {
            fn drop(&mut self) {
                self.stopped.store(true, Ordering::Release);
                if let Some(handle) = self.handle.take() {
                    let _ = handle.join();
                }
            }
        }

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.set_xpub_verbose(true).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let mut initial = descriptor();
        initial.events.configured_endpoint = "tcp://127.0.0.1:*".into();
        initial.events.resolved_endpoint = endpoint.clone();
        let advertised = Arc::new(Mutex::new(initial.clone()));
        let calls = Arc::new(AtomicUsize::new(0));
        let stopped = Arc::new(AtomicBool::new(false));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let worker = format!("http://{}", listener.local_addr().unwrap());
        let server = {
            let advertised = advertised.clone();
            let calls = calls.clone();
            let stopped = stopped.clone();
            thread::spawn(move || {
                while !stopped.load(Ordering::Acquire) {
                    let (mut stream, _) = match listener.accept() {
                        Ok(connection) => connection,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(1));
                            continue;
                        }
                        Err(error) => panic!("{error}"),
                    };
                    stream
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .unwrap();
                    let mut request = Vec::new();
                    while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                        let mut bytes = [0; 1024];
                        let n = stream.read(&mut bytes).unwrap();
                        assert!(n > 0 && request.len() + n <= 8192);
                        request.extend_from_slice(&bytes[..n]);
                    }
                    assert!(request.starts_with(b"GET /v1/kv-cache/capabilities HTTP/1.1\r\n"));
                    calls.fetch_add(1, Ordering::AcqRel);
                    let body = serde_json::to_string(&*advertised.lock().unwrap()).unwrap();
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .unwrap();
                }
            })
        };
        let server = HttpServer {
            stopped,
            handle: Some(server),
        };
        let bridge = Arc::new(RenderBridge::for_test(
            RenderContract {
                id: "transport-test".into(),
                epoch: 1,
            },
            BridgeLimits::default(),
            |_, _| PreparedResult::Unsupported,
        ));
        wait_for(|| bridge.current_contract().is_some());
        let index = Arc::new(KVBlockIndex::new(8));
        let cohort = CapabilityCohort {
            workers: HashMap::from([(worker.clone(), initial.clone())]),
            api_key_env: None,
        };
        let pool = CapabilityEventPool::start(
            vec![(worker.clone(), endpoint)],
            index.clone(),
            cohort,
            bridge.clone(),
        )
        .unwrap();
        subscribed(&publisher, &initial.events.topic);
        assert_eq!(calls.load(Ordering::Acquire), 1);
        let old_generation = index.current_generation(&worker).unwrap();
        publish(&publisher, &initial, 10);
        wait_for(|| index.ownership_count() == 1);
        for _ in 0..1000 {
            assert_eq!(index.prefix_score(&worker, &[[7; 32]]), 1);
        }
        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "index reads must not issue metadata requests"
        );

        advertised.lock().unwrap().events.next_sequence = 13;
        publish(&publisher, &initial, 12); // Gap: discard this batch and purge.
        subscribed(&publisher, &initial.events.topic);
        assert!(calls.load(Ordering::Acquire) >= 2);
        assert_eq!(index.ownership_count(), 0);
        assert!(!index.store(&worker, old_generation, &[[7; 32]]));
        publish(&publisher, &initial, 11); // Delayed same-epoch pre-watermark data.
        publish(&publisher, &initial, 13);
        wait_for(|| index.ownership_count() == 1);

        // An epoch change is not input-conformance proof even if the model
        // namespace is identical. A gap makes revalidation immediate here.
        {
            let mut next = advertised.lock().unwrap();
            next.events.epoch = "fedcba9876543210fedcba9876543210".into();
            next.events.topic = format!("kv.{}", next.events.epoch);
            next.events.next_sequence = 0;
        }
        publish(&publisher, &initial, 15);
        wait_for(|| bridge.current_contract().is_none());
        assert_eq!(index.ownership_count(), 0);
        drop(pool);
        drop(server);
        bridge.shutdown();
    }
}

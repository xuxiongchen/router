"""Public CPU-only checks for the finite performance experiment contract.

Run: python -m unittest discover -s py_test -p test_kv_performance_harness.py
Uses only the standard library and an owned loopback HTTP fixture, never GPU.
"""

from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import kv_capabilities_performance as perf  # noqa: E402


def stream(ids=None, text="generated", completion=2, done=True, finish=True):
    choice = {"text": text}
    if finish:
        choice["finish_reason"] = "length"
    if ids is not None:
        choice["prompt_token_ids"] = ids
    events = [{"id": "public-fixture", "choices": [choice]},
              {"usage": {"prompt_tokens": 3, "completion_tokens": completion}}]
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return body + (b"data: [DONE]\n\n" if done else b"")


class FakeResponse(BytesIO):
    status = 200

    def getheader(self, _name, _default):
        return "text/event-stream"


class FakeConnection:
    sock = None

    def __init__(self, body):
        self.body, self.closed = body, False

    def request(self, *_args, **_kwargs):
        pass

    def getresponse(self):
        return FakeResponse(self.body)

    def close(self):
        self.closed = True


@contextmanager
def server(*, slow=False, connection_close=False):
    records = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            request = json.loads(raw)
            records.append((self.client_address, request, self.headers.get("X-Request-Id")))
            body = stream(completion=request["max_tokens"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if slow:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(body)))
            if connection_close:
                self.send_header("Connection", "close")
            self.end_headers()
            try:
                if slow:
                    self.wfile.write(f"{len(body):x}\r\n".encode())
                    for value in body:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(0.01)
                    self.wfile.write(b"\r\n0\r\n\r\n")
                else:
                    self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", records
    finally:
        http.shutdown()
        http.server_close()
        thread.join(2)


class PerformanceContractTests(unittest.TestCase):
    def test_feature_and_effective_mode_are_required(self):
        good = {"enabled": True, "modes": ["shared_rr", "render_rr", "render_kv"],
                "selected_mode": "render_rr", "environment_variable": perf.BENCHMARK_ENV,
                "loopback_only": True}
        self.assertEqual(perf.validate_benchmark_capabilities(good, "B"), good)
        for changes in ({"enabled": False}, {"selected_mode": None}, {"modes": []},
                        {"loopback_only": False}, {"environment_variable": "OTHER"}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                perf.validate_benchmark_capabilities({**good, **changes}, "B")

    def test_real_product_baseline_cannot_inherit_ablation(self):
        self.assertEqual(perf.ARMS["product_rr"], ("round_robin", None))
        ordinary = {"enabled": False, "modes": [], "selected_mode": None}
        perf.validate_benchmark_capabilities(ordinary, "product_rr")
        with self.assertRaises(RuntimeError):
            perf.validate_benchmark_capabilities({**ordinary, "selected_mode": "shared_rr"}, "product_rr")

    def test_arm_order_repeats_without_cherry_picking(self):
        arms = ["product_rr", "A", "B", "C"]
        orders = [perf.arm_order(arms, index, 0) for index in range(3)]
        self.assertEqual(len({tuple(order) for order in orders}), 3)
        self.assertTrue(all(Counter(order) == Counter(arms) for order in orders))

    def test_same_seed_bytes_and_no_timed_ids(self):
        args = SimpleNamespace(groups=4, requests=16, prefix_tokens=48, input_tokens=64,
                               block_size=16, model="public-test", output_tokens=2,
                               prompt_format="text", trace_order="burst")
        first = perf.make_trace(args, list(range(256)), "locality", "same", "same")
        second = perf.make_trace(args, list(range(256)), "locality", "same", "same")
        self.assertEqual(first, second)
        self.assertTrue(all(not request["return_token_ids"] for request in first[0]))
        distinct = perf.make_trace(args, list(range(256)), "locality", "different", "same")
        self.assertNotEqual(first[0], distinct[0])
        self.assertEqual(first[2], distinct[2])

    def test_shared_prefix_warmup_and_interleaved_order(self):
        args = SimpleNamespace(groups=4, requests=16, prefix_tokens=48, input_tokens=64,
                               block_size=16, model="public-test", output_tokens=2,
                               prompt_format="text", trace_order="interleaved")
        requests, warm, order = perf.make_trace(args, list(range(256)), "shared", "same", "same")
        self.assertEqual(order, [0, 1, 2, 3] * 4)
        self.assertEqual([owner for owner, _ in warm], [0, 1] * 4)
        self.assertEqual(len(requests), 16)
        for group in range(4):
            self.assertEqual(warm[group * 2][1]["prompt"].split("\n")[0],
                             warm[group * 2 + 1][1]["prompt"].split("\n")[0])

    def call_stream(self, body, **kwargs):
        connection = FakeConnection(body)
        payload = {"prompt": "fixture", "max_tokens": kwargs.pop("max_tokens", 2),
                   "return_token_ids": kwargs.pop("return_token_ids", False)}
        row = perf.streamed_request("http://127.0.0.1:1", payload, 1, connection=connection,
                                    expected_tokens=[1, 2, 3], **kwargs)
        return row, connection

    def test_timed_no_ids_has_all_timing_boundaries(self):
        row, connection = self.call_stream(stream(), require_token_ids=False)
        self.assertEqual(row["status"], "PASS")
        self.assertFalse(connection.closed)
        self.assertLessEqual(row["headers_ms"], row["first_sse_ms"])
        self.assertLessEqual(row["first_sse_ms"], row["ttft_ms"])
        self.assertLessEqual(row["ttft_ms"], row["end_to_end_ms"])
        self.assertIsNone(row["first_reasoning_ms"])
        self.assertTrue(row["correlation_id"].startswith("cmb-perf-"))

    def test_timed_diagnostic_ids_are_rejected(self):
        row, connection = self.call_stream(stream(ids=[1, 2, 3]), require_token_ids=False)
        self.assertEqual(row["status"], "ERROR")
        self.assertIn("diagnostic prompt IDs", row["error"])
        self.assertTrue(connection.closed)

    def test_oracle_still_checks_full_ids(self):
        row, _ = self.call_stream(stream(ids=[1, 2, 4]), require_token_ids=True, return_token_ids=True)
        self.assertEqual(row["status"], "ERROR")
        self.assertIn("prompt IDs differ", row["error"])

    def test_empty_unmeasured_oracle_not_empty_timed_ttft(self):
        row, _ = self.call_stream(stream(ids=[1, 2, 3], text="", completion=1),
                                 require_token_ids=True, return_token_ids=True,
                                 max_tokens=1, measure_ttft=False)
        self.assertEqual(row["status"], "PASS")
        self.assertIsNone(row["ttft_ms"])
        row, _ = self.call_stream(stream(text=""), require_token_ids=False)
        self.assertEqual(row["status"], "ERROR")

    def test_usage_done_and_trailing_body_are_not_relaxed(self):
        for body in (stream(completion=1), stream(done=False), stream(finish=False), stream() + b"unexpected"):
            with self.subTest(body=body):
                row, connection = self.call_stream(body, require_token_ids=False)
                self.assertEqual(row["status"], "ERROR")
                self.assertTrue(connection.closed)

    def test_actual_loopback_keepalive_reuses_two_connections(self):
        with server() as (url, records):
            clients = perf.KeepAliveClients(url, 2)
            try:
                rows = [clients.request({"prompt": "fixture", "max_tokens": 2,
                                         "return_token_ids": False}, index, [1, 2, 3])
                        for index in range(4)]
            finally:
                clients.close()
        self.assertTrue(all(row["status"] == "PASS" and row["connection_reused"] for row in rows))
        self.assertEqual(sorted(Counter(address for address, _, _ in records).values()), [2, 2])
        self.assertEqual(len({correlation for _, _, correlation in records}), 4)
        self.assertTrue(all(not payload["return_token_ids"] for _, payload, _ in records))

    def test_keepalive_drop_is_error_not_automatic_reconnect(self):
        with server(connection_close=True) as (url, records):
            clients = perf.KeepAliveClients(url, 1)
            try:
                first = clients.request({"prompt": "fixture", "max_tokens": 2,
                                         "return_token_ids": False}, 0, [1, 2, 3])
                second = clients.request({"prompt": "fixture", "max_tokens": 2,
                                          "return_token_ids": False}, 1, [1, 2, 3])
            finally:
                clients.close()
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(second["status"], "ERROR")
        self.assertIn("automatic reconnect", second["error"])
        self.assertEqual(len(records), 1)

    def test_absolute_deadline_interrupts_slow_fragmented_sse(self):
        with server(slow=True) as (url, _records):
            started = time.monotonic()
            row = perf.streamed_request(url, {"prompt": "fixture", "max_tokens": 2,
                                              "return_token_ids": False}, 0,
                                        timeout=0.08, expected_tokens=[1, 2, 3], require_token_ids=False)
            elapsed = time.monotonic() - started
        self.assertEqual(row["status"], "ERROR")
        self.assertTrue(row["deadline_exceeded"])
        self.assertLess(elapsed, 0.5)

    def test_product_rr_load_unknown_not_zero(self):
        workers = ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]
        body = json.dumps({"workers": [{"url": worker, "load": 0} for worker in workers]})
        values = {("vllm:num_requests_running", ""): 2.0,
                  ("vllm:num_requests_waiting", ""): 1.0}
        sampler = perf.LoadSampler("http://127.0.0.1:1", workers, router_load_known=False)
        with patch.object(perf.prior, "request", return_value=(200, {}, body)), \
                patch.object(perf, "metric_snapshot", return_value=("raw", values)), \
                patch.object(sampler.stop, "wait", side_effect=lambda _interval: sampler.stop.set()):
            sampler.run()
        self.assertEqual(len(sampler.samples), 1)
        self.assertIsNone(sampler.samples[0]["router_loads"])
        self.assertEqual(sampler.samples[0]["worker_running"], [2.0, 2.0])
        self.assertEqual(sampler.samples[0]["worker_waiting"], [1.0, 1.0])
        self.assertIsNone(perf.optional_count({}, "vllm:num_requests_waiting"))

    def test_reset_never_accepts_false_success(self):
        perf.validate_reset_response({"success": True})
        for value in ({"success": False}, {"success": 1}, {}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                perf.validate_reset_response(value)

    def test_reset_checks_workers_before_router_exists(self):
        workers = ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]
        with patch.object(perf, "idle") as idle, patch.object(perf, "save"), \
                patch.object(perf.prior, "request", return_value=(200, {}, '{"success":true}')):
            perf.reset_exact_test_workers(SimpleNamespace(allow_test_worker_cache_reset=True),
                                           {"workers": workers}, Path("/evidence/reset"))
        idle.assert_called_once_with(workers)

    def test_cohort_hook_requires_fresh_explicit_authority(self):
        args = SimpleNamespace(cache_state="fresh-cohort", cohort_preparation_hook="/owned/hook",
                               allow_cohort_preparation=False, cohort_timeout=180)
        with patch.object(perf.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "authority"):
                perf.validate_cohort_options(args)
            execute.assert_not_called()
        args.cache_state, args.cohort_preparation_hook = "namespaced", None
        perf.validate_cohort_options(args)

    def test_old_epoch_or_engine_process_rejected(self):
        fixture = json.loads((SCRIPTS.parent / "tests/fixtures/kv_capabilities/descriptor.json").read_text())
        before = {"w0": json.loads(json.dumps(fixture)), "w1": json.loads(json.dumps(fixture))}
        after = json.loads(json.dumps(before))
        for value in after.values():
            value["events"].update(epoch="f" * 32, topic="kv." + "f" * 32)
        old_processes = [{"pid": index, "start_ticks": 10} for index in range(1, 5)]
        new_processes = [{"pid": index, "start_ticks": 20} for index in range(5, 9)]
        perf.validate_fresh_cohort(before, after, old_processes, new_processes)
        with self.assertRaisesRegex(RuntimeError, "old Worker event epoch"):
            perf.validate_fresh_cohort(before, before, old_processes, new_processes)
        with self.assertRaisesRegex(RuntimeError, "old HTTP or EngineCore"):
            perf.validate_fresh_cohort(before, after, old_processes,
                                       new_processes[:3] + old_processes[-1:])
        after["w0"]["namespace"]["dtype"] = "changed_dtype"
        with self.assertRaisesRegex(RuntimeError, "semantic Worker capability contract"):
            perf.validate_fresh_cohort(before, after, old_processes, new_processes)

    def test_invalid_hook_result_retained_and_not_accepted(self):
        args = SimpleNamespace(cache_state="fresh-cohort", cohort_preparation_hook="/owned/hook",
                               allow_cohort_preparation=True, cohort_timeout=180)
        common = {"workers": ["w0", "w1"], "capabilities": {}, "worker_processes": []}
        for stdout in ('{"status":"PASS"}', '{"status":"FAIL","state":"fresh_empty_cache"}', 'bad json'):
            with self.subTest(stdout=stdout), \
                    patch.object(perf, "validate_cohort_options"), \
                    patch.object(perf.prior, "sha256", return_value="source-sha"), \
                    patch.object(perf, "save") as save, \
                    patch.object(perf.subprocess, "run", return_value=SimpleNamespace(
                        returncode=0, stdout=stdout, stderr="diagnostic")), \
                    patch.object(perf.acceptance, "worker_capabilities") as fetch:
                with self.assertRaises((RuntimeError, ValueError)):
                    perf.prepare_fresh_cohort(args, common, Path("/evidence/phase"))
                self.assertEqual(save.call_args.args[1]["status"], "FAIL")
                self.assertEqual(save.call_args.args[1]["script_sha256"], "source-sha")
                fetch.assert_not_called()

    def test_oracle_failure_keeps_partial_evidence(self):
        with patch.object(perf, "streamed_request", return_value={"status": "ERROR", "error": "fixture"}), \
                patch.object(perf, "save") as save:
            with self.assertRaisesRegex(RuntimeError, "token oracle failed"):
                perf.token_oracle_after_timing({"workers": ["w0", "w1"]},
                    [{"prompt": "fixture", "max_tokens": 2}], [[1, 2, 3]], Path("/evidence/oracle.json"))
            self.assertEqual(len(save.call_args.args[1]), 1)
            self.assertEqual(save.call_args.args[1][0]["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()

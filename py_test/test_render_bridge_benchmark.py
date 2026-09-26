"""Dependency-free harness checks, not CPU performance/equivalence evidence."""

import asyncio
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_render_bridge_paths as benchmark


class _FakeFacade:
    def __init__(self):
        self.contract_id, self.epoch = "test-contract", 1
        self.effective_config = {"test_double": True}
        self._runtime = SimpleNamespace(renderer=SimpleNamespace(
            model_config=SimpleNamespace(renderer_num_workers=1)), capture=object())
        self._loop = None
        self.owner = None
        self.closed = False

    async def _render(self, kind, raw):
        return self.render(kind, raw)

    def render(self, kind, raw):
        if threading.get_ident() != self.owner:
            raise AssertionError("wrong test owner")
        return {"status": "exact", "cache_eligible": True, "token_ids": json.loads(raw),
                "contract_id": self.contract_id, "epoch": self.epoch}

    def startup(self):
        self.owner = threading.get_ident()
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._verify_workers())

    def close(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("wrong close owner")
        if self._loop is not None:
            self._loop.close()
        self.closed = True


class FacadeBenchmarkHarnessTests(unittest.TestCase):
    def case(self, ids=(3, 7)):
        return {"name": "test-only", "kind": "completion", "target_tokens": len(ids),
                "raw": json.dumps(ids).encode(), "ids": list(ids)}

    def test_independent_owner_threads_close_their_own_facades(self):
        created = []
        def create(_):
            facade = _FakeFacade()
            created.append(facade)
            return facade
        module = SimpleNamespace(create_facade=create)
        cases = [self.case([index]) for index in range(6)]
        with benchmark._FacadePool(module, "unused", cases, 4) as pool:
            replies = [(case, pool.submit(case["kind"], case["raw"])) for case in cases]
            for case, future in replies:
                benchmark._validate_facade_reply(future.result(timeout=2)[0], case, pool.identity)
            self.assertEqual(len({facade.owner for facade in created}), 4)
        self.assertTrue(all(facade.closed for facade in created))
        self.assertTrue(all(not thread.is_alive() for thread in pool.threads))

    def test_shared_oracle_mismatch_refuses_startup_and_closes(self):
        created = []
        def create(_):
            facade = _FakeFacade()
            created.append(facade)
            return facade
        case = self.case()
        case["ids"] = [99]
        with self.assertRaisesRegex(RuntimeError, "shared_oracle_mismatch"):
            with benchmark._FacadePool(SimpleNamespace(create_facade=create), "unused", [case], 1):
                self.fail("oracle failure admitted")
        self.assertTrue(created[0].closed)

    def test_mixed_probe_checks_each_shape_every_round(self):
        module = SimpleNamespace(create_facade=lambda _: _FakeFacade())
        cases = [{**self.case([index]), "name": f"case-{index}"} for index in range(6)]
        with benchmark._FacadePool(module, "unused", cases, 4) as pool:
            result = benchmark._mixed_facade_probe(pool, cases)
        self.assertEqual(result["full_array_and_epoch_checks"], 18)
        self.assertEqual(result["case_counts"], {f"case-{index}": 3 for index in range(6)})
        self.assertEqual(result["client_concurrency"], 4)
        self.assertIs(result["timed"], False)

    def test_full_arrays_not_only_length_and_epoch_are_checked(self):
        case = self.case()
        identity = {"contract_id": "contract", "epoch": 2}
        reply = {"status": "exact", "cache_eligible": True, "token_ids": case["ids"], **identity}
        benchmark._validate_facade_reply(reply, case, identity)
        for replacement in ({"token_ids": [7, 3]}, {"epoch": 3}, {"contract_id": "changed"},
                            {"cache_eligible": False}, {"status": "unsupported"}):
            with self.subTest(replacement=replacement), self.assertRaises(RuntimeError):
                benchmark._validate_facade_reply({**reply, **replacement}, case, identity)

    def test_cell_has_complete_checks_and_explicit_non_rust_scope(self):
        case = self.case()
        module = SimpleNamespace(create_facade=lambda _: _FakeFacade())
        resource = {"pid": 1, "cpu_seconds": 1.0, "rss_bytes": 1000, "threads": 2}
        with benchmark._FacadePool(module, "unused", [case], 2) as pool, \
             patch.object(benchmark, "resources", return_value=[resource]):
            result = benchmark._facade_cell(pool, case, 2, 10, False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["full_array_checks"], 14)
        self.assertEqual(result["contract_epoch_checks"], 14)
        self.assertEqual(len(result["samples_ns"]), 10)
        self.assertEqual(result["stage_ms"], {})
        self.assertEqual(result["mode"], "facade_only_independent_instances")

    def test_missing_requested_observer_refuses_measurement(self):
        case = self.case()
        module = SimpleNamespace(create_facade=lambda _: _FakeFacade())
        resource = {"pid": 1, "cpu_seconds": 1.0, "rss_bytes": 1000, "threads": 2}
        with benchmark._FacadePool(module, "unused", [case], 1) as pool, \
             patch.object(benchmark, "resources", return_value=[resource]), \
             self.assertRaisesRegex(RuntimeError, "observer_mode_mismatch"):
            benchmark._facade_cell(pool, case, 1, 10, True)

    def test_stage_metrics_are_filtered_without_relabeling_aggregate(self):
        body = (b'# HELP ignored help\n'
                b'vllm_router_kv_stage_duration_seconds_sum{stage="queue_wait"} 0.1\n'
                b'vllm_router_kv_stage_duration_seconds_count{stage="queue_wait"} 4\n'
                b'vllm_router_kv_bridge_usage{resource="active"} 0\n'
                b'unrelated_total 29\n')
        response = SimpleNamespace(status=200, read=lambda: body)
        connection = SimpleNamespace(request=lambda *_: None, getresponse=lambda: response, close=lambda: None)
        with patch.object(benchmark.http.client, "HTTPConnection", return_value=connection):
            values = benchmark._stage_metrics(1234)
        self.assertEqual(len(values), 3)
        self.assertEqual(values['vllm_router_kv_stage_duration_seconds_count{stage="queue_wait"}'], 4)
        self.assertEqual(benchmark._stage_metrics(None), {})

    def test_metric_window_only_differences_cumulative_series(self):
        prefix = "vllm_router_kv_stage_duration_seconds"
        total = prefix + '_sum{stage="queue_wait"}'
        count = prefix + '_count{stage="queue_wait"}'
        quantile = prefix + '{stage="queue_wait",quantile="0.95"}'
        gauge = 'vllm_router_kv_bridge_usage{resource="active"}'
        before = {total: 1.0, count: 4, quantile: 0.2, gauge: 2}
        after = {total: 1.5, count: 6, quantile: 0.1, gauge: 0}
        result = benchmark._metric_window(before, after)
        self.assertEqual(result["cumulative_delta"], {total: 0.5, count: 2})
        self.assertEqual(result["duration_mean_seconds_from_sum_count"], {total: 0.25})
        self.assertEqual(result["after"][quantile], 0.1)
        with self.assertRaisesRegex(RuntimeError, "metric_reset"):
            benchmark._metric_window(after, before)


if __name__ == "__main__":
    unittest.main()

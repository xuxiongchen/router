"""Execute the installed vLLM 0.29 Dense lookup source against inert doubles.

This is an independent source oracle, not a second implementation of Router's
reusable-prefix formula. No vLLM import, engine construction, allocator or LRU is
run. Only selected AST function bodies execute; the fake pool has a read-only
mapping and no allocation/touch/eviction methods. The source files are recorded
with hashes for reproducibility, not used as a production model/package gate.

Run with unittest in the isolated Linux environment. The source oracle uses an
installed vLLM distribution without importing it, or CMB_VLLM_SOURCE_ROOT naming
the vllm package directory in an audited source checkout. Without that optional
dependency the source-backed cases are SKIPPED, never reported as oracle PASS.
"""

import ast
from collections.abc import Sequence
import copy
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/kv_capabilities/dense_reuse_cases.json"
)
SOURCE_FILES = {
    "manager": "v1/core/kv_cache_manager.py",
    "coordinator": "v1/core/kv_cache_coordinator.py",
    "full_attention": "v1/core/single_type_kv_cache_manager.py",
    "hash_view": "v1/core/kv_cache_utils.py",
    "request": "v1/request.py",
    "scheduler": "v1/core/sched/scheduler.py",
}


def source_root():
    explicit = os.environ.get("CMB_VLLM_SOURCE_ROOT")
    if explicit:
        root = Path(explicit).resolve()
    else:
        try:
            distribution = importlib.metadata.distribution("vllm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise unittest.SkipTest("vLLM 0.29 source is not installed") from exc
        if distribution.version.split("+", 1)[0] != "0.29.0":
            raise AssertionError(
                f"oracle requires audited vLLM 0.29.0, got {distribution.version}"
            )
        root = Path(distribution.locate_file("vllm")).resolve()
    for relative in SOURCE_FILES.values():
        if not (root / relative).is_file():
            raise AssertionError(f"oracle source missing: {root / relative}")
    return root


def extract_function(root, relative, class_name, method_name):
    """Select one real function body; no source-module top-level code executes."""
    path = root / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    container = tree
    if class_name is not None:
        container = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    method = next(
        node for node in container.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return copy.deepcopy(method)


class FullAttentionSpecDouble:
    """Structural input only; not evidence of runtime adapter type admission."""

    def __init__(self, block_size):
        self.block_size = block_size


class ChunkedLocalAttentionSpecDouble:
    pass


class UnsupportedHashView:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("oracle is restricted to equal Dense block units")


class ReadOnlyBlockPool:
    def __init__(self, block_size, hashes):
        self.hash_block_size = block_size
        self.blocks = MappingProxyType({value: (value,) for value in hashes})
        self.lookups = []

    def get_cached_block(self, block_hash, group_ids):
        if group_ids != [0]:
            raise AssertionError("oracle only models the complete single group")
        self.lookups.append((block_hash, tuple(group_ids)))
        return self.blocks.get(block_hash)


class SourceOracle:
    def __init__(self, root):
        self.provenance = {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in SOURCE_FILES.values()
        }
        declarations = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        declarations.append(extract_function(
            root, SOURCE_FILES["hash_view"], None, "resolve_block_hashes"
        ))
        methods = {
            "KVCacheManager": (
                "manager", ["prefix_cache_lookup_enabled", "get_computed_blocks"]
            ),
            "UnitaryKVCacheCoordinator": (
                "coordinator", ["find_longest_cache_hit"]
            ),
            "FullAttentionManager": (
                "full_attention", ["find_longest_cache_hit"]
            ),
            "Request": ("request", ["num_tokens"]),
            "Scheduler": ("scheduler", ["_get_local_prefix_cache_hit"]),
        }
        for class_name, (source, names) in methods.items():
            declarations.append(ast.ClassDef(
                name=class_name, bases=[], keywords=[], decorator_list=[],
                body=[extract_function(
                    root, SOURCE_FILES[source], class_name, name
                ) for name in names],
            ))
        namespace = {
            "itertools": itertools,
            "Sequence": Sequence,
            "cdiv": lambda numerator, denominator: -(-numerator // denominator),
            "FullAttentionSpec": FullAttentionSpecDouble,
            "ChunkedLocalAttentionSpec": ChunkedLocalAttentionSpecDouble,
            "BlockHashListWithBlockSize": UnsupportedHashView,
        }
        module = ast.fix_missing_locations(ast.Module(
            body=declarations, type_ignores=[]
        ))
        exec(compile(module, "<selected-vllm-0.29-source-oracle>", "exec"), namespace)
        self.types = namespace
        self.types["FullAttentionManager"].supports_fine_grained_hash_lookup = True

    def lookup(self, token_count, block_size, query_hashes, cached_hashes,
               *, enabled=True, skip_read=False):
        pool = ReadOnlyBlockPool(block_size, cached_hashes)
        coordinator = self.types["UnitaryKVCacheCoordinator"]()
        coordinator.single_type_managers = [self.types["FullAttentionManager"]]
        coordinator.block_pool = pool
        coordinator.kv_cache_spec = FullAttentionSpecDouble(block_size)
        coordinator.eagle_group_ids = set()
        coordinator.block_size = block_size
        coordinator.dcp_world_size = coordinator.pcp_world_size = 1
        manager = self.types["KVCacheManager"]()
        manager.enable_caching = enabled
        manager.enable_kv_cache_events = False
        manager.coordinator = coordinator
        manager.empty_kv_cache_blocks = ((),)
        manager.create_kv_cache_blocks = lambda groups: groups
        request = self.types["Request"]()
        # Fresh prepared text generation: no appended output/speculative tokens.
        request._all_token_ids = list(range(token_count))
        request.block_hashes = query_hashes
        request.skip_reading_prefix_cache = skip_read
        request.kv_cache_report_mode = "incremental"
        scheduler = self.types["Scheduler"]()
        scheduler.connector = None
        scheduler.kv_cache_manager = manager
        before = dict(pool.blocks)
        result = scheduler._get_local_prefix_cache_hit(request)
        assert dict(pool.blocks) == before
        return result, pool.lookups


def block_hash(index):
    # Controlled full-width identities, not a replacement for vLLM hash tests.
    return hashlib.sha256(f"oracle-block-{index}".encode()).digest()


def fixture_cases():
    document = json.loads(FIXTURE.read_text())
    assert document["schema_version"] == 1
    return document["cases"]


class DenseFixtureShapeTests(unittest.TestCase):
    def test_fixture_is_explicit_and_covers_required_boundaries(self):
        cases = fixture_cases()
        names = [case["name"] for case in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue({15, 16, 17, 31, 32, 33, 464}.issubset(
            {case["query_tokens"] for case in cases if case["block_size"] == 16}
        ))
        for case in cases:
            with self.subTest(case=case["name"]):
                self.assertGreater(case["query_tokens"], 0)
                self.assertGreater(case["block_size"], 0)
                self.assertEqual(
                    case["expected_reusable_tokens"] % case["block_size"], 0
                )
                self.assertLess(
                    case["expected_reusable_tokens"], case["query_tokens"]
                )


class DenseInstalledSourceOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = SourceOracle(source_root())

    def test_explicit_boundary_fixtures_against_actual_source(self):
        for case in fixture_cases():
            with self.subTest(case=case["name"]):
                count, size = case["query_tokens"], case["block_size"]
                query = [block_hash(i) for i in range(count // size)]
                removed = set(case.get("removed_blocks", []))
                cached = {block_hash(i) for i in case["cached_blocks"] if i not in removed}
                # Raw contiguous stored coverage is independent of the engine
                # terminal recompute cap; a hole stops it even with later keys.
                matched = sum(1 for _ in itertools.takewhile(
                    lambda value: value in cached, query
                ))
                self.assertEqual(matched, case["expected_matched_blocks"])
                (blocks, reused, shared_boundary, divergent), lookups = self.oracle.lookup(
                    count, size, query, cached
                )
                self.assertEqual(reused, case["expected_reusable_tokens"])
                self.assertEqual(len(blocks[0]) * size, reused)
                self.assertEqual(shared_boundary, 0)
                self.assertFalse(divergent)
                self.assertTrue(all(group_ids == (0,) for _, group_ids in lookups))

    def test_cache_read_exclusions_do_not_probe_the_pool(self):
        query = [block_hash(0), block_hash(1)]
        for enabled, skip_read in [(False, False), (True, True), (False, True)]:
            with self.subTest(enabled=enabled, skip_read=skip_read):
                (blocks, reused, _, _), lookups = self.oracle.lookup(
                    33, 16, query, query, enabled=enabled, skip_read=skip_read
                )
                self.assertEqual((blocks, reused, lookups), (((),), 0, []))

    def test_empty_hash_inventory_cannot_produce_ownership(self):
        (_, reused, _, _), lookups = self.oracle.lookup(33, 16, [], [block_hash(0)])
        self.assertEqual((reused, lookups), (0, []))

    def test_hash_identity_is_not_the_first_u64(self):
        query_hash = b"same-u64" + b"a" * 24
        other_hash = b"same-u64" + b"b" * 24
        self.assertEqual(query_hash[:8], other_hash[:8])
        (_, reused, _, _), _ = self.oracle.lookup(17, 16, [query_hash], [other_hash])
        self.assertEqual(reused, 0)

    def test_provenance_identifies_every_executed_source_file(self):
        self.assertEqual(set(self.oracle.provenance), set(SOURCE_FILES.values()))
        self.assertTrue(all(len(value) == 64 for value in self.oracle.provenance.values()))


if __name__ == "__main__":
    unittest.main()

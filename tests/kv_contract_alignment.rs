//! Regression contributions to PR326, not a second index implementation.
use std::sync::Arc;
use vllm_router_rs::kv_index::{
    ClearScope, KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier, TieredMatchProvider,
};

fn owner(name: &str) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(name),
        dp_rank: 0,
        incarnation: 7,
    }
}

fn query(hashes: &[&str]) -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        local_hashes: hashes.iter().map(|hash| Arc::from(*hash)).collect(),
        tiers_of_interest: vec![StorageTier::Device],
    }
}

fn depth(index: &dyn TieredMatchProvider, hashes: &[&str], worker: &str) -> u32 {
    index
        .find_tiered_matches(&query(hashes))
        .iter()
        .find(|hit| hit.target.instance_id.as_ref() == worker)
        .map_or(0, |hit| hit.matched_depth)
}

#[test]
fn equal_local_suffixes_keep_distinct_parent_paths() {
    let index = KvBlockIndexer::new();
    index.store(
        0,
        owner("a"),
        StorageTier::Device,
        None,
        &[
            (Arc::from("seq-a"), Arc::from("prefix-a")),
            (Arc::from("seq-a-x"), Arc::from("suffix")),
        ],
    );
    index.store(
        0,
        owner("b"),
        StorageTier::Device,
        None,
        &[
            (Arc::from("seq-b"), Arc::from("prefix-b")),
            (Arc::from("seq-b-x"), Arc::from("suffix")),
        ],
    );
    assert_eq!(depth(&index, &["prefix-a", "suffix"], "a"), 2);
    assert_eq!(depth(&index, &["prefix-b", "suffix"], "b"), 2);
}

#[test]
fn owning_a_root_does_not_credit_a_suffix_under_another_parent() {
    let index = KvBlockIndexer::new();
    index.store(
        0,
        owner("a"),
        StorageTier::Device,
        None,
        &[
            (Arc::from("a-0"), Arc::from("prefix-a")),
            (Arc::from("a-1"), Arc::from("suffix")),
        ],
    );
    index.store(
        0,
        owner("b"),
        StorageTier::Device,
        None,
        &[(Arc::from("b-a"), Arc::from("prefix-a"))],
    );
    index.store(
        0,
        owner("b"),
        StorageTier::Device,
        None,
        &[
            (Arc::from("b-0"), Arc::from("prefix-b")),
            (Arc::from("b-1"), Arc::from("suffix")),
        ],
    );
    assert_eq!(depth(&index, &["prefix-a", "suffix"], "b"), 1);
    assert_eq!(depth(&index, &["prefix-b", "suffix"], "b"), 2);
}

#[test]
fn repeated_local_blocks_are_distinct_positions() {
    let index = KvBlockIndexer::new();
    index.store(
        0,
        owner("a"),
        StorageTier::Device,
        None,
        &[
            (Arc::from("seq-0"), Arc::from("same")),
            (Arc::from("seq-1"), Arc::from("same")),
        ],
    );
    assert_eq!(depth(&index, &["same", "same"], "a"), 2);
}

#[test]
fn an_unknown_parent_does_not_turn_a_suffix_into_a_root() {
    let index = KvBlockIndexer::new();
    index.store(
        0,
        owner("a"),
        StorageTier::Device,
        Some(Arc::from("missing-parent")),
        &[(Arc::from("seq-suffix"), Arc::from("suffix"))],
    );
    assert_eq!(depth(&index, &["suffix"], "a"), 0);
}

#[test]
fn sequence_lookup_is_scoped_to_the_announcing_owner() {
    let index = KvBlockIndexer::new();
    index.store(
        0,
        owner("a"),
        StorageTier::Device,
        None,
        &[(Arc::from("opaque-seq"), Arc::from("a-token-block"))],
    );
    index.store(
        0,
        owner("b"),
        StorageTier::Device,
        None,
        &[(Arc::from("opaque-seq"), Arc::from("b-token-block"))],
    );
    index.remove(
        0,
        &owner("a"),
        StorageTier::Device,
        &[Arc::from("opaque-seq")],
    );
    assert_eq!(depth(&index, &["a-token-block"], "a"), 0);
    assert_eq!(depth(&index, &["b-token-block"], "b"), 1);
}

#[test]
fn independent_namespace_instances_do_not_share_observations() {
    let first = KvBlockIndexer::new();
    let second = KvBlockIndexer::new();
    first.store(
        0,
        owner("a"),
        StorageTier::Device,
        None,
        &[(Arc::from("sequence"), Arc::from("local"))],
    );
    assert_eq!(depth(&first, &["local"], "a"), 1);
    assert_eq!(depth(&second, &["local"], "a"), 0);
    // This establishes instance isolation, not a complete CacheIdentity API.
}

#[test]
fn worker_clear_keeps_shared_pool_and_unrelated_worker_claims() {
    let index = KvBlockIndexer::new();
    let blocks = [(Arc::from("seq"), Arc::from("local"))];
    let pool = ResidencyOwner::CacheOwner {
        pool_id: Arc::from("pool"),
        source: Arc::from("a"),
        dp_rank: 0,
    };
    index.store(0, owner("a"), StorageTier::Device, None, &blocks);
    index.store(0, owner("b"), StorageTier::Device, None, &blocks);
    index.store(0, pool, StorageTier::External, None, &blocks);
    index.clear(&owner("a"), ClearScope::Worker);
    assert_eq!(depth(&index, &["local"], "a"), 0);
    assert_eq!(depth(&index, &["local"], "b"), 1);
    let mut query = query(&["local"]);
    query.tiers_of_interest = vec![StorageTier::External];
    assert_eq!(index.find_tiered_matches(&query)[0].matched_depth, 1);
}

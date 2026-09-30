//! Static-review counterexamples for PR326 head
//! 5fb50b6a4a588ab75a1ad140669e7f4598e75eb1.
//! These tests have NOT been compiled or run by this review.
//! Add only to an authorized, isolated review worktree, then reproduce before fixing.

use std::sync::Arc;
use vllm_router_rs::kv_index::{
    KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier, TieredMatchProvider,
};

fn owner() -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from("w0"),
        dp_rank: 0,
        incarnation: 0,
    }
}

fn store(index: &KvBlockIndexer, parent: Option<&str>, blocks: &[(&str, &str)]) {
    let blocks: Vec<_> = blocks
        .iter()
        .map(|(sequence, local)| (Arc::from(*sequence), Arc::from(*local)))
        .collect();
    index.store(0, owner(), StorageTier::Device, parent.map(Arc::from), &blocks);
}

fn depth(index: &KvBlockIndexer, locals: &[&str]) -> u32 {
    index
        .find_tiered_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: locals.iter().map(|local| Arc::from(*local)).collect(),
            tiers_of_interest: vec![StorageTier::Device],
        })
        .into_iter()
        .find(|hit| hit.target.instance_id.as_ref() == "w0" && hit.target.dp_rank == 0)
        .map_or(0, |hit| hit.matched_depth)
}

#[test]
fn compressed_edge_mismatch_must_not_skip_to_a_child() {
    let index = KvBlockIndexer::new();
    store(&index, None, &[("seq-a", "a"), ("seq-b", "b")]);
    store(&index, Some("seq-b"), &[("seq-c", "c")]);

    assert_eq!(depth(&index, &["a", "b", "c"]), 3);
    // The index contains a -> b -> c, not a -> c. Only a is a valid prefix.
    assert_eq!(depth(&index, &["a", "c"]), 1);
}

#[test]
fn parent_inside_a_compressed_edge_must_not_be_treated_as_its_tail() {
    let index = KvBlockIndexer::new();
    store(&index, None, &[("seq-a", "a"), ("seq-b", "b")]);
    store(&index, Some("seq-a"), &[("seq-c-after-a", "c")]);

    assert_eq!(depth(&index, &["a", "b"]), 2);
    // The second store announces a -> c; it does not announce a -> b -> c.
    assert_eq!(depth(&index, &["a", "b", "c"]), 2);
    assert_eq!(depth(&index, &["a", "c"]), 2);
}

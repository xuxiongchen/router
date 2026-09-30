//! Regression coverage for PR326's compressed radix index.

use std::sync::Arc;
use vllm_router_rs::kv_index::{
    KvBlockIndexer, MatchQuery, ResidencyOwner, StorageTier, TieredMatchProvider,
};

fn owner() -> ResidencyOwner {
    named_owner("w0")
}

fn named_owner(name: &str) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(name),
        dp_rank: 0,
        incarnation: 0,
    }
}

fn store(index: &KvBlockIndexer, parent: Option<&str>, blocks: &[(&str, &str)]) {
    let blocks: Vec<_> = blocks
        .iter()
        .map(|(sequence, local)| (Arc::from(*sequence), Arc::from(*local)))
        .collect();
    index.store(
        0,
        owner(),
        StorageTier::Device,
        parent.map(Arc::from),
        &blocks,
    );
}

fn depth(index: &KvBlockIndexer, locals: &[&str]) -> u32 {
    owner_depth(index, locals, "w0")
}

fn owner_depth(index: &KvBlockIndexer, locals: &[&str], name: &str) -> u32 {
    index
        .find_tiered_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: locals.iter().map(|local| Arc::from(*local)).collect(),
            tiers_of_interest: vec![StorageTier::Device],
        })
        .into_iter()
        .find(|hit| hit.target.instance_id.as_ref() == name && hit.target.dp_rank == 0)
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

#[test]
fn shared_path_preserves_owner_aliases_across_splits_and_removal() {
    let index = KvBlockIndexer::new();
    store(
        &index,
        None,
        &[("seq-a", "a"), ("seq-b", "b"), ("seq-c", "c")],
    );
    let other = named_owner("w1");
    index.store(
        0,
        other.clone(),
        StorageTier::Device,
        None,
        &[
            (Arc::from("other-a"), Arc::from("a")),
            (Arc::from("other-b"), Arc::from("b")),
            (Arc::from("other-c"), Arc::from("c")),
        ],
    );
    // Splitting at w0's interior parent must also relocate w1's aliases.
    store(&index, Some("seq-a"), &[("seq-x", "x")]);
    index.store(
        0,
        other.clone(),
        StorageTier::Device,
        Some(Arc::from("other-b")),
        &[(Arc::from("other-y"), Arc::from("y"))],
    );
    assert_eq!(owner_depth(&index, &["a", "b", "y"], "w1"), 3);
    index.remove(0, &other, StorageTier::Device, &[Arc::from("other-b")]);
    assert_eq!(owner_depth(&index, &["a", "b", "c"], "w1"), 1);
    assert_eq!(owner_depth(&index, &["a", "b", "y"], "w1"), 1);
    assert_eq!(depth(&index, &["a", "b", "c"]), 3);
    index.clear(&other, vllm_router_rs::kv_index::ClearScope::Worker);
    assert_eq!(owner_depth(&index, &["a"], "w1"), 0);
    assert_eq!(depth(&index, &["a", "b", "c"]), 3);
}

#[test]
fn parent_sequence_resolution_does_not_cross_owners() {
    let index = KvBlockIndexer::new();
    store(&index, None, &[("same-sequence", "a")]);
    let other = named_owner("w1");
    index.store(
        0,
        other,
        StorageTier::Device,
        None,
        &[(Arc::from("same-sequence"), Arc::from("b"))],
    );
    store(&index, Some("same-sequence"), &[("seq-c", "c")]);
    assert_eq!(depth(&index, &["a", "c"]), 2);
    assert_eq!(depth(&index, &["b", "c"]), 0);
}

#[test]
fn concurrent_stores_do_not_overwrite_a_shared_child() {
    use std::sync::Barrier;
    use std::thread;

    for _ in 0..32 {
        let index = Arc::new(KvBlockIndexer::new());
        let barrier = Arc::new(Barrier::new(8));
        let writers: Vec<_> = (0..8)
            .map(|worker| {
                let index = index.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let name: Arc<str> = Arc::from(format!("w{worker}"));
                    let owner = ResidencyOwner::Worker {
                        source: name,
                        dp_rank: 0,
                        incarnation: 0,
                    };
                    let blocks: Vec<_> = ["a", "b", "c"]
                        .into_iter()
                        .map(|local| (Arc::from(format!("w{worker}-{local}")), Arc::from(local)))
                        .collect();
                    barrier.wait();
                    index.store(0, owner, StorageTier::Device, None, &blocks);
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        let matches = index.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: ["a", "b", "c"].into_iter().map(Arc::from).collect(),
            tiers_of_interest: vec![StorageTier::Device],
        });
        assert_eq!(matches.len(), 8);
        assert!(matches.iter().all(|hit| hit.matched_depth == 3));
    }
}

#[test]
fn concurrent_parent_splits_and_queries_preserve_exact_paths() {
    use std::sync::Barrier;
    use std::thread;

    for _ in 0..64 {
        let index = Arc::new(KvBlockIndexer::new());
        store(
            &index,
            None,
            &[("seq-a", "a"), ("seq-b", "b"), ("seq-c", "c")],
        );
        let barrier = Arc::new(Barrier::new(3));
        let writers: Vec<_> = [("seq-a", "seq-x", "x"), ("seq-b", "seq-y", "y")]
            .into_iter()
            .map(|(parent, sequence, local)| {
                let index = index.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    store(&index, Some(parent), &[(sequence, local)]);
                })
            })
            .collect();
        barrier.wait();
        for _ in 0..32 {
            assert_eq!(depth(&index, &["a", "b", "c"]), 3);
            assert_eq!(depth(&index, &["a", "c"]), 1);
            std::thread::yield_now();
        }
        for writer in writers {
            writer.join().unwrap();
        }
        assert_eq!(depth(&index, &["a", "b", "c"]), 3);
        assert_eq!(depth(&index, &["a", "x"]), 2);
        assert_eq!(depth(&index, &["a", "b", "y"]), 3);
        assert_eq!(depth(&index, &["a", "b", "c", "y"]), 3);
    }
}

#[test]
fn concurrent_split_and_alias_removal_keep_other_claims() {
    use std::sync::Barrier;
    use std::thread;

    for _ in 0..64 {
        let index = Arc::new(KvBlockIndexer::new());
        store(
            &index,
            None,
            &[("seq-a", "a"), ("seq-b", "b"), ("seq-c", "c")],
        );
        index.store(
            0,
            named_owner("w1"),
            StorageTier::Device,
            None,
            &[
                (Arc::from("other-a"), Arc::from("a")),
                (Arc::from("other-b"), Arc::from("b")),
                (Arc::from("other-c"), Arc::from("c")),
            ],
        );
        let barrier = Arc::new(Barrier::new(2));
        let split = {
            let index = index.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                store(&index, Some("seq-a"), &[("seq-x", "x")]);
            })
        };
        barrier.wait();
        index.remove(
            0,
            &named_owner("w1"),
            StorageTier::Device,
            &[Arc::from("other-b")],
        );
        split.join().unwrap();
        assert_eq!(owner_depth(&index, &["a", "b", "c"], "w1"), 1);
        assert_eq!(depth(&index, &["a", "b", "c"]), 3);
        assert_eq!(depth(&index, &["a", "x"]), 2);
    }
}

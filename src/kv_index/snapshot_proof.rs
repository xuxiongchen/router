//! Historical CPU capture controls; production acquisition lives in snapshot.
#![cfg(all(test, feature = "cpu", feature = "ready"))]
use super::snapshot::validate;
pub(crate) use super::snapshot::{Attempt, Identity, Transcript};
use super::subscriber::apply_batch;
use super::{KvBlockIndexer, SourceId};
use sha2::{Digest, Sha256};

const REGULAR_SNAPSHOT: &str =
    include_str!("../../tests/rust/fixtures/kv_snapshot/regular_snapshot.json");

pub(crate) fn fixture(name: &str) -> Transcript {
    let json = match name {
        "regular_snapshot.json" => REGULAR_SNAPSHOT,
        "removed_snapshot.json" => {
            include_str!("../../tests/rust/fixtures/kv_snapshot/removed_snapshot.json")
        }
        "replacement_snapshot.json" => {
            include_str!("../../tests/rust/fixtures/kv_snapshot/replacement_snapshot.json")
        }
        "healthy_b_snapshot.json" => {
            include_str!("../../tests/rust/fixtures/kv_snapshot/healthy_b_snapshot.json")
        }
        "structure_snapshot.json" => {
            include_str!("../../tests/rust/fixtures/kv_snapshot/structure_snapshot.json")
        }
        _ => panic!("unknown snapshot fixture: {name}"),
    };
    serde_json::from_str(json).expect("producer transcript schema")
}

#[test]
fn snapshot_transcript_rejects_incomplete_or_conflicting_inputs() {
    let valid = fixture("regular_snapshot.json");
    let expected = valid.identity.clone();
    assert!(validate(&valid, &expected).is_ok());
    for case in [
        "status",
        "capability",
        "required",
        "profile",
        "identity",
        "count",
        "missing descriptor",
        "digest",
        "gap",
        "conflicting seq",
    ] {
        let mut invalid = valid.clone();
        match case {
            "status" => invalid.status = "unavailable".into(),
            "capability" => invalid.capability.clear(),
            "required" => invalid.snapshot_required = false,
            "profile" => invalid.identity.lookahead = 1,
            "identity" => invalid.identity.stream_id.push_str("-replacement"),
            "count" => invalid.descriptor_count += 1,
            "missing descriptor" => {
                invalid.descriptors.remove(0);
                invalid.descriptor_count = invalid.descriptors.len();
                let mut digest = Sha256::new();
                for descriptor in &invalid.descriptors {
                    digest.update([u8::from(descriptor.resident)]);
                    digest.update((descriptor.payload.len() as u64).to_be_bytes());
                    digest.update(&descriptor.payload);
                }
                invalid.digest = format!("{:x}", digest.finalize());
            }
            "digest" => invalid.digest.push('0'),
            "gap" => {
                assert!(
                    !invalid.replay.is_empty(),
                    "producer supplies finite suffix"
                );
                invalid.replay[0].seq += 1;
            }
            "conflicting seq" => {
                let mut conflicting = invalid.replay[0].clone();
                conflicting.payload = invalid.descriptors[0].payload.clone();
                assert_ne!(conflicting.payload, invalid.replay[0].payload);
                invalid.replay.push(conflicting);
                invalid.high_watermark += 1;
            }
            _ => unreachable!(),
        }
        assert!(validate(&invalid, &expected).is_err(), "accepted {case}");
    }
    let mut missing: serde_json::Value = serde_json::from_str(REGULAR_SNAPSHOT).unwrap();
    missing.as_object_mut().unwrap().remove("capability");
    assert!(serde_json::from_value::<Transcript>(missing).is_err());
}

#[test]
fn official_versions_preserve_legacy_store_remove_clear_and_replay_bytes() {
    use super::{MatchQuery, StorageTier};

    fn unhex(value: &serde_json::Value) -> Vec<u8> {
        value
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    let artifact: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/rust/fixtures/kv_snapshot/official-wire.json"
    ))
    .unwrap();
    let rows = artifact["results"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        let version = row["version"].as_str().unwrap();
        assert_eq!(row["snapshot_capability_in_official_module"], false);
        let wire: Vec<_> = row["serialized_batches_hex"]
            .as_array()
            .unwrap()
            .iter()
            .map(unhex)
            .collect();
        let replay: Vec<_> = row["replay_payloads_hex"]
            .as_array()
            .unwrap()
            .iter()
            .map(unhex)
            .collect();
        assert_eq!(replay, wire[1..]);
        assert_eq!(row["replay_sequences"], serde_json::json!([1, 2]));
        assert_eq!(
            row["terminal"],
            serde_json::json!({"topic_hex":"", "seq":-1,"payload_hex":""})
        );
        let decoded: Vec<_> = wire
            .iter()
            .map(|payload| super::decode_batch(payload).unwrap())
            .collect();
        let indexer = KvBlockIndexer::new();
        let source = SourceId::from(format!("official-{version}"));
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: super::local_block_hash_strings(&(0..16).collect::<Vec<_>>(), 16, None),
            tiers_of_interest: vec![StorageTier::Device],
        };
        for (batch, present) in [
            (&decoded[0], true),
            (&decoded[1], false),
            (&decoded[0], true),
            (&decoded[2], false),
        ] {
            apply_batch(&indexer, &source, 0, 0, batch, Some(16)).unwrap();
            let hits = indexer.find_matches(&query);
            if present {
                assert_eq!(hits.len(), 1, "{version} seeded Store");
                assert_eq!(hits[0].target.instance_id, source);
                assert_eq!(hits[0].target.dp_rank, 0);
                assert_eq!(hits[0].tier, StorageTier::Device);
                assert_eq!(hits[0].segments, [(0, 1)]);
            } else {
                assert!(
                    hits.is_empty(),
                    "{version} actual Remove/Clear after positive seed"
                );
            }
        }
        if !row["lookahead_wire_hex"].is_null() {
            let error = super::decode_batch(&unhex(&row["lookahead_wire_hex"])).unwrap_err();
            assert_eq!(error, "unsupported KV event field: lookahead_token_ids");
        }
        eprintln!("official {version}: actual serializer bytes admitted; positive Store, replay Remove, reseed/Clear; no snapshot capability");
    }
}

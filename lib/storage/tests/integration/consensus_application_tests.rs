use std::collections::HashMap;
use std::sync::Arc;

use collection::operations::cluster_ops::ReshardingDirection;
use collection::operations::types::PeerMetadata;
use collection::operations::verification::new_unchecked_verification_pass;
use collection::shards::replica_set::replica_set_state::ReplicaState;
use collection::shards::resharding::{ReshardState, ReshardingStage};
use collection::shards::transfer::{ShardTransfer, ShardTransferMethod};
use raft::eraftpb::{
    ConfChangeSingle, ConfChangeType, ConfChangeV2, ConfState, Entry, EntryType, Snapshot,
    SnapshotMetadata,
};
use raft::storage::MemStorage;
use raft::{Config, RawNode};
use serde_json::json;
use storage::content_manager::CollectionContainer;
use storage::content_manager::collection_meta_ops::*;
use storage::content_manager::consensus_manager::{ApplyMode, ConsensusManager, SnapshotData};
use storage::content_manager::consensus_ops::ConsensusOperations;
use storage::content_manager::consensus_shadow::{ShadowMode, read_cluster_state};
use storage::content_manager::consensus_state_machine::{Action, ClusterState};
use storage::content_manager::errors::StorageError;
use storage::content_manager::shard_distribution::ShardDistributionProposal;
use storage::content_manager::toc::TableOfContent;

use super::alias_tests::{FULL_ACCESS, new_dispatcher};
use super::consensus_action_applier_tests::consensus_manager;

#[test]
fn collection_operations_match_legacy_handlers() {
    let (legacy_dir, _legacy_handle, legacy_dispatcher) = new_dispatcher();
    let (csm_dir, _csm_handle, csm_dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let legacy = consensus_manager(
        &legacy_dir,
        legacy_dispatcher.toc(&FULL_ACCESS, &pass).clone(),
    )
    .with_shadow(ShadowMode::Panic);
    let csm = consensus_manager(&csm_dir, csm_dispatcher.toc(&FULL_ACCESS, &pass).clone())
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    let operations = vec![
        create_collection("books", false),
        meta(CollectionMetaOperations::ChangeAliases(
            ChangeAliasesOperation {
                actions: vec![
                    CreateAlias {
                        collection_name: "books".into(),
                        alias_name: "novels".into(),
                    }
                    .into(),
                ],
            },
        )),
        ConsensusOperations::initialize_replica("books".into(), 0, 0),
        meta(CollectionMetaOperations::UpdateCollection(
            UpdateCollectionOperation::new(
                "novels".into(),
                serde_json::from_value(
                    json!({"hnsw_config": {"m": 24}, "metadata": {"owner": "reader"}}),
                )
                .unwrap(),
            )
            .unwrap(),
        )),
        meta(CollectionMetaOperations::CreatePayloadIndex(
            CreatePayloadIndex {
                collection_name: "novels".into(),
                field_name: "city".parse().unwrap(),
                field_schema: segment::types::PayloadSchemaType::Keyword.into(),
            },
        )),
        meta(CollectionMetaOperations::DropPayloadIndex(
            DropPayloadIndex {
                collection_name: "books".into(),
                field_name: "city".parse().unwrap(),
            },
        )),
        // User errors must leave state unchanged and keep their error variants
        meta(CollectionMetaOperations::UpdateCollection(
            UpdateCollectionOperation::new(
                "books".into(),
                serde_json::from_value(
                    json!({"hnsw_config": {"m": 32}, "vectors": {"missing": {"on_disk": true}}}),
                )
                .unwrap(),
            )
            .unwrap(),
        )),
        meta(CollectionMetaOperations::CreateNamedVector(
            CreateNamedVector {
                collection_name: "books".into(),
                vector_name: "text".into(),
                config: serde_json::from_value(json!({"dense": {"size": 8, "distance": "Dot"}}))
                    .unwrap(),
            },
        )),
        meta(CollectionMetaOperations::DeleteNamedVector(
            DeleteNamedVector {
                collection_name: "books".into(),
                vector_name: "text".into(),
            },
        )),
        create_collection("custom", true),
        meta(CollectionMetaOperations::CreatePayloadIndex(
            CreatePayloadIndex {
                collection_name: "custom".into(),
                field_name: "city".parse().unwrap(),
                field_schema: segment::types::PayloadSchemaType::Keyword.into(),
            },
        )),
        meta(CollectionMetaOperations::CreateShardKey(CreateShardKey {
            collection_name: "custom".into(),
            shard_key: "tenant".to_string().into(),
            placement: vec![vec![0]],
            initial_state: Some(ReplicaState::Active),
        })),
        meta(CollectionMetaOperations::DropShardKey(DropShardKey {
            collection_name: "custom".into(),
            shard_key: "tenant".to_string().into(),
        })),
        ConsensusOperations::UpdatePeerMetadata {
            peer_id: 0,
            metadata: PeerMetadata::current(),
        },
        ConsensusOperations::UpdateClusterMetadata {
            key: "owner".into(),
            value: json!("reader"),
        },
        ConsensusOperations::UpdateClusterMetadata {
            key: "owner".into(),
            value: json!(null),
        },
        ConsensusOperations::SetQuotaConfig(Default::default()),
        meta(CollectionMetaOperations::DeleteCollection(
            DeleteCollectionOperation("books".into()),
        )),
        meta(CollectionMetaOperations::DeleteCollection(
            DeleteCollectionOperation("books".into()),
        )),
    ];

    for operation in operations {
        let expected = legacy.apply_normal_entry(&entry(&operation));
        let actual = csm.apply_normal_entry(&entry(&operation));

        match (expected, actual) {
            (Ok(expected), Ok(actual)) => assert_eq!(actual, expected, "{operation:?}"),
            (Err(expected), Err(actual)) => assert_eq!(
                std::mem::discriminant(&actual),
                std::mem::discriminant(&expected),
                "{operation:?}: {actual} != {expected}"
            ),
            results => panic!("different results for {operation:?}: {results:?}"),
        }
        assert_eq!(
            state(&csm, csm_dispatcher.toc(&FULL_ACCESS, &pass)),
            state(&legacy, legacy_dispatcher.toc(&FULL_ACCESS, &pass)),
            "{operation:?}"
        );
    }
}

#[test]
fn alias_action_failure_reloads_before_replay() {
    let (dir, _handle, dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let toc = dispatcher.toc(&FULL_ACCESS, &pass);
    let csm = consensus_manager(&dir, toc.clone())
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    csm.apply_normal_entry(&entry(&create_collection("books", false)))
        .unwrap();
    let alias = meta(CollectionMetaOperations::ChangeAliases(
        ChangeAliasesOperation {
            actions: vec![
                CreateAlias {
                    collection_name: "books".into(),
                    alias_name: "old".into(),
                }
                .into(),
            ],
        },
    ));
    csm.apply_normal_entry(&entry(&alias)).unwrap();
    let rename = meta(CollectionMetaOperations::ChangeAliases(
        ChangeAliasesOperation {
            actions: vec![
                RenameAlias {
                    old_alias_name: "old".into(),
                    new_alias_name: "new".into(),
                }
                .into(),
            ],
        },
    ));
    let path = dir
        .path()
        .join(storage::content_manager::toc::ALIASES_PATH)
        .join("data.json");
    let backup = path.with_extension("backup");
    fs_err::rename(&path, &backup).unwrap();
    fs_err::create_dir(&path).unwrap();

    let error = csm.apply_normal_entry(&entry(&rename)).unwrap_err();

    assert!(matches!(error, StorageError::ServiceError { .. }));
    fs_err::remove_dir(&path).unwrap();
    fs_err::rename(backup, path).unwrap();

    csm.apply_normal_entry(&entry(&rename)).unwrap();

    assert_eq!(
        toc.alias_mapping().get("new").map(String::as_str),
        Some("books")
    );
    assert!(toc.alias_mapping().get("old").is_none());
}

#[test]
fn delete_collection_replays_after_aliases_and_collection_removal() {
    let (dir, _handle, dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let toc = dispatcher.toc(&FULL_ACCESS, &pass);
    let manager = consensus_manager(&dir, toc.clone())
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    manager
        .apply_normal_entry(&entry(&create_collection("books", false)))
        .unwrap();
    let alias = meta(CollectionMetaOperations::ChangeAliases(
        ChangeAliasesOperation {
            actions: vec![
                CreateAlias {
                    collection_name: "books".into(),
                    alias_name: "novels".into(),
                }
                .into(),
            ],
        },
    ));
    manager.apply_normal_entry(&entry(&alias)).unwrap();
    let delete = entry_from_meta(CollectionMetaOperations::DeleteCollection(
        DeleteCollectionOperation("books".into()),
    ));
    let deleted_path = dir.path().join(".deleted");
    fs_err::write(&deleted_path, "block collection deletion").unwrap();

    let error = manager.apply_normal_entry(&delete).unwrap_err();

    assert!(matches!(error, StorageError::ServiceError { .. }));
    assert!(toc.alias_mapping().get("novels").is_none());
    assert!(toc.collection_state("books").is_none());
    assert!(dir.path().join("collections/books").exists());
    fs_err::remove_file(deleted_path).unwrap();

    assert!(!manager.apply_normal_entry(&delete).unwrap());

    assert!(!dir.path().join("collections/books").exists());
}

#[test]
fn configuration_batch_rejection_preserves_membership_and_prior_changes() {
    let (dir, handle, dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let toc = dispatcher.toc(&FULL_ACCESS, &pass);
    let manager = consensus_manager(&dir, toc.clone());
    manager
        .apply_normal_entry(&entry(&create_collection("books", false)))
        .unwrap();
    for shard_id in 0..2 {
        toc.apply_action(Action::SetReplicaState {
            collection: "books".into(),
            shard_id,
            peer_id: 0,
            state: ReplicaState::Active,
        })
        .unwrap();
    }
    toc.apply_action(Action::SetReplicaState {
        collection: "books".into(),
        shard_id: 0,
        peer_id: 1,
        state: ReplicaState::ReshardingScaleDown,
    })
    .unwrap();
    let resharding = ReshardState {
        uuid: uuid::Uuid::nil(),
        direction: ReshardingDirection::Down,
        peer_id: 0,
        shard_id: 1,
        shard_key: None,
        stage: ReshardingStage::ReadHashRingCommitted,
    };
    toc.apply_action(Action::SetReshardingState {
        collection: "books".into(),
        state: Some(resharding),
    })
    .unwrap();
    toc.apply_action(Action::RegisterTransfer {
        collection: "books".into(),
        transfer: ShardTransfer {
            shard_id: 1,
            to_shard_id: Some(0),
            from: 0,
            to: 1,
            sync: true,
            method: Some(ShardTransferMethod::ReshardingStreamRecords),
            filter: None,
        },
    })
    .unwrap();
    manager
        .add_peer(1, "http://127.0.0.1:6336".parse().unwrap())
        .unwrap();
    let manager = manager
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    let mut raw = raw_node();
    let before = state(&manager, toc);
    let conf_before = raw.raft.prs().conf().to_conf_state();
    let persisted_conf = manager.conf_state();
    let entry = conf_entry(
        &[
            (ConfChangeType::AddLearnerNode, 2),
            (ConfChangeType::RemoveNode, 1),
        ],
        "http://127.0.0.1:6337",
    );

    assert!(!manager.apply_conf_change_entry(&entry, &mut raw).unwrap());

    assert_eq!(raw.raft.prs().conf().to_conf_state(), conf_before);
    assert_eq!(manager.conf_state(), persisted_conf);
    assert_eq!(state(&manager, toc), before);
    // The rejected temporary AddPeer must not appear in the next plan either
    manager
        .apply_normal_entry(&entry_from_meta(CollectionMetaOperations::Nop { token: 0 }))
        .unwrap();
    drop(handle);
}

#[test]
fn configuration_changes_apply_and_self_removal_stops_consensus() {
    let (dir, _handle, dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let toc = dispatcher.toc(&FULL_ACCESS, &pass);
    let manager = consensus_manager(&dir, toc.clone())
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    let mut raw = raw_node();

    assert!(
        !manager
            .apply_conf_change_entry(
                &conf_entry(
                    &[(ConfChangeType::AddLearnerNode, 1)],
                    "http://127.0.0.1:6336"
                ),
                &mut raw
            )
            .unwrap()
    );
    assert!(manager.peer_address_by_id().contains_key(&1));
    assert_eq!(manager.conf_state().learners, vec![1]);
    assert!(
        !manager
            .apply_conf_change_entry(
                &conf_entry(&[(ConfChangeType::RemoveNode, 1)], ""),
                &mut raw
            )
            .unwrap()
    );
    assert!(!manager.peer_address_by_id().contains_key(&1));

    // The fixture's ToC peer ID is 0; Raft uses 10 because 0 is reserved by Raft
    let change = ConfState {
        voters: vec![10, 11],
        ..Default::default()
    };
    raw = raw_node_with_conf(change);
    assert!(
        manager
            .apply_conf_change_entry(
                &conf_entry(&[(ConfChangeType::RemoveNode, 0)], ""),
                &mut raw
            )
            .unwrap()
    );
    assert!(manager.peer_address_by_id().is_empty());
}

#[test]
fn joint_configuration_save_failure_replays_both_transitions() {
    let (dir, _handle, dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let toc = dispatcher.toc(&FULL_ACCESS, &pass);
    let manager = consensus_manager(&dir, toc.clone())
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    let mut raw = raw_node();
    let enter = conf_entry(
        &[
            (ConfChangeType::AddLearnerNode, 1),
            (ConfChangeType::AddLearnerNode, 2),
        ],
        "http://127.0.0.1:6336",
    );
    let leave = conf_entry(&[], "");

    for entry in [enter, leave] {
        let path = manager.persistent.read().path.clone();
        let backup = path.with_extension("backup");
        fs_err::rename(&path, &backup).unwrap();
        fs_err::create_dir(&path).unwrap();

        let error = manager
            .apply_conf_change_entry(&entry, &mut raw)
            .unwrap_err();

        assert!(matches!(error, StorageError::ServiceError { .. }));
        fs_err::remove_dir(&path).unwrap();
        fs_err::rename(backup, path).unwrap();
        // A consensus-thread restart initializes Raft from the changed in-memory membership
        raw = raw_node_with_conf(manager.conf_state());

        assert!(!manager.apply_conf_change_entry(&entry, &mut raw).unwrap());
    }

    assert!(manager.conf_state().voters_outgoing.is_empty());
    assert_eq!(manager.conf_state().learners, vec![1, 2]);
    assert!(manager.peer_address_by_id().contains_key(&1));
    assert!(manager.peer_address_by_id().contains_key(&2));
}

#[test]
fn discovery_address_updates_invalidate_application_state() {
    let (dir, _handle, dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let toc = dispatcher.toc(&FULL_ACCESS, &pass);
    let manager = consensus_manager(&dir, toc.clone())
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    let nop = entry_from_meta(CollectionMetaOperations::Nop { token: 0 });
    manager.apply_normal_entry(&nop).unwrap();

    manager
        .add_peer(1, "http://127.0.0.1:6336".parse().unwrap())
        .unwrap();
    manager.apply_normal_entry(&nop).unwrap();

    assert!(manager.peer_address_by_id().contains_key(&1));
}

#[test]
fn snapshot_invalidates_application_state() {
    let (dir, _handle, dispatcher) = new_dispatcher();
    let pass = new_unchecked_verification_pass();
    let toc = dispatcher.toc(&FULL_ACCESS, &pass);
    let manager = consensus_manager(&dir, toc.clone())
        .with_apply_mode(ApplyMode::StateMachine)
        .with_shadow(ShadowMode::Panic);
    manager
        .apply_normal_entry(&entry(&create_collection("books", false)))
        .unwrap();
    let snapshot = Snapshot {
        data: serde_cbor::to_vec(&SnapshotData {
            collections_data: Default::default(),
            address_by_id: Default::default(),
            metadata_by_id: Default::default(),
            cluster_metadata: HashMap::from([("snapshot".into(), json!(true))]),
            quota_config: None,
        })
        .unwrap(),
        metadata: Some(SnapshotMetadata {
            conf_state: Some(ConfState::default()),
            index: 1,
            term: 1,
        }),
    };

    manager.apply_snapshot(&snapshot).unwrap().unwrap();
    manager
        .apply_normal_entry(&entry(&create_collection("books", false)))
        .unwrap();

    assert!(toc.collection_state("books").is_some());
    assert_eq!(
        manager.persistent.read().cluster_metadata["snapshot"],
        json!(true)
    );
}

fn meta(operation: CollectionMetaOperations) -> ConsensusOperations {
    ConsensusOperations::CollectionMeta(Box::new(operation))
}

fn create_collection(name: &str, custom: bool) -> ConsensusOperations {
    let mut operation = CreateCollectionOperation::new(
        name.into(),
        serde_json::from_value(json!({
            "vectors": {"size": 4, "distance": "Cosine"}, "shard_number": 2,
            "sharding_method": if custom { "custom" } else { "auto" },
            "uuid": uuid::Uuid::nil(),
        }))
        .unwrap(),
    )
    .unwrap();
    operation.set_distribution(ShardDistributionProposal {
        distribution: if custom {
            Vec::new()
        } else {
            vec![(0, vec![0]), (1, vec![0])]
        },
    });
    meta(CollectionMetaOperations::CreateCollection(operation))
}

fn entry(operation: &ConsensusOperations) -> Entry {
    Entry {
        data: serde_cbor::to_vec(operation).unwrap(),
        ..Default::default()
    }
}

fn entry_from_meta(operation: CollectionMetaOperations) -> Entry {
    entry(&meta(operation))
}

fn conf_entry(changes: &[(ConfChangeType, u64)], uri: &str) -> Entry {
    let change = ConfChangeV2 {
        changes: changes
            .iter()
            .map(|&(change_type, node_id)| ConfChangeSingle {
                change_type: change_type as i32,
                node_id,
            })
            .collect(),
        ..Default::default()
    };
    Entry {
        entry_type: EntryType::EntryConfChangeV2 as i32,
        data: prost_for_raft::Message::encode_to_vec(&change),
        context: uri.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn raw_node() -> RawNode<MemStorage> {
    raw_node_with_conf(ConfState {
        voters: vec![10],
        ..Default::default()
    })
}

fn raw_node_with_conf(conf: ConfState) -> RawNode<MemStorage> {
    let storage = MemStorage::new_with_conf_state(conf);
    RawNode::new(
        &Config {
            id: 10,
            ..Default::default()
        },
        storage,
        &slog::Logger::root(slog::Discard, slog::o!()),
    )
    .unwrap()
}

fn state(manager: &ConsensusManager<TableOfContent>, toc: &Arc<TableOfContent>) -> ClusterState {
    read_cluster_state(toc.as_ref(), &manager.persistent.read())
}

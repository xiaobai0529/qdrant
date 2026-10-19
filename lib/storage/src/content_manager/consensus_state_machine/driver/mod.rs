//! Plan consensus operations and compare predicted state with applied state

pub mod diff;

#[cfg(test)]
mod tests;

use collection::shards::CollectionId;

use self::diff::ShallowState;
use crate::content_manager::CollectionContainer;
use crate::content_manager::collection_meta_ops::CollectionMetaOperations;
use crate::content_manager::consensus::persistent::Persistent;
use crate::content_manager::consensus_manager::CollectionsSnapshot;
use crate::content_manager::consensus_ops::ConsensusOperations;
use crate::content_manager::consensus_state_machine::{
    Action, ApplyOutcome, ClusterState, ConsensusStateMachine,
};
use crate::content_manager::errors::{StorageError, StorageResult};

/// Consensus state machine initialized from applied state and used to plan operations
pub struct StateMachineDriver {
    state_machine: ConsensusStateMachine,
    /// Discovery runs in async code and cannot synchronously read collections.
    /// Defer reloading until the next consensus operation is planned.
    needs_reload: bool,
}

impl StateMachineDriver {
    pub fn new(toc: &impl CollectionContainer, persistent: &Persistent) -> Self {
        let state = read_cluster_state(toc, persistent);
        let context = toc.node_context();
        log::info!(
            "Consensus state machine initialized: peer_id={}, collections={}, peers={}",
            context.peer_id,
            state.collections.len(),
            state.peer_address_by_id.len(),
        );
        let state_machine = ConsensusStateMachine::new(state, context);

        Self {
            state_machine,
            needs_reload: false,
        }
    }

    pub fn invalidate(&mut self) {
        if !self.needs_reload {
            log::debug!("Consensus state machine invalidated");
        }

        self.needs_reload = true;
    }

    pub fn apply(
        &mut self,
        toc: &impl CollectionContainer,
        persistent: &Persistent,
        operation: &ConsensusOperations,
    ) -> ApplyOutcome {
        self.prepare(toc, persistent);
        let outcome = self.state_machine.apply(operation);
        log::debug!("Consensus state machine operation {operation:?}: {outcome:?}");

        outcome
    }

    /// Plan every operation against temporary state before changing the applied state
    pub(crate) fn apply_batch(
        &mut self,
        toc: &impl CollectionContainer,
        persistent: &Persistent,
        operations: &[ConsensusOperations],
    ) -> StorageResult<Vec<Vec<Action>>> {
        self.prepare(toc, persistent);
        let mut machine = self.state_machine.clone();
        let mut batch = Vec::with_capacity(operations.len());

        for operation in operations {
            match machine.apply(operation) {
                ApplyOutcome::Accepted(actions) => batch.push(actions),
                ApplyOutcome::Rejected(error) => return Err(error),
                ApplyOutcome::NotCovered => {
                    return Err(StorageError::service_error(format!(
                        "Consensus operation is not modeled: {operation:?}",
                    )));
                }
            }
        }

        self.state_machine = machine;
        Ok(batch)
    }

    fn prepare(&mut self, toc: &impl CollectionContainer, persistent: &Persistent) {
        // Partial snapshot recovery can change payload index schema without a consensus operation.
        // Refresh affected collections so state machine applies next operation to recovered state.
        let dirty_collections = toc.take_dirty_collections();
        if self.needs_reload {
            *self = Self::new(toc, persistent);
        } else {
            self.resync(toc, dirty_collections);
        }
    }

    pub fn compare<T>(
        &mut self,
        toc: &impl CollectionContainer,
        persistent: &Persistent,
        operation: &ConsensusOperations,
        outcome: &ApplyOutcome,
        result: &StorageResult<T>,
    ) {
        let Some(report) = self.diff(toc, persistent, operation, outcome, result) else {
            return;
        };

        if cfg!(any(test, debug_assertions)) {
            panic!(
                "Consensus state machine diverged from applied state: {report}; \
                 operation={operation:?}; outcome={outcome:?}; handler_error={:?}",
                result.as_ref().err(),
            );
        }

        log::error!(
            "Consensus state machine diverged from applied state: {report}; \
             operation={operation:?}; outcome={outcome:?}; handler_error={:?}",
            result.as_ref().err(),
        );
    }

    fn diff<T>(
        &mut self,
        toc: &impl CollectionContainer,
        persistent: &Persistent,
        operation: &ConsensusOperations,
        outcome: &ApplyOutcome,
        result: &StorageResult<T>,
    ) -> Option<String> {
        if self.needs_reload {
            return None;
        }

        // Handler may have persisted only part of operation before returning a service error,
        // while consensus state machine applied it completely. Invalidate instead of comparing.
        if matches!(result, Err(StorageError::ServiceError { .. })) {
            log::debug!(
                "Consensus state machine comparison skipped after handler service error: \
                 operation={operation:?}; error={:?}",
                result.as_ref().err(),
            );
            self.invalidate();
            return None;
        }

        let collections = target_collections(operation, self.state_machine.state());

        // State machine cannot apply an uncovered operation.
        // Resync collections the operation may have changed.
        //
        // If operation does not name a collection, it may have changed any of them,
        // so invalidate the entire consensus state machine.
        if matches!(outcome, ApplyOutcome::NotCovered) {
            log::debug!(
                "Consensus state machine comparison skipped for uncovered operation: {operation:?}"
            );
            if collections.is_empty() {
                self.invalidate();
            } else {
                self.resync(toc, collections);
            }

            return None;
        }

        // Partial snapshot recovery can change payload index schema
        // after state machine applies an operation but before comparison.
        //
        // Refresh affected collections so recovered state is not reported as a divergence.
        self.resync(toc, toc.take_dirty_collections());

        let state_machine = &self.state_machine;

        let applied = read_shallow_state(toc, persistent);

        let mut report = Vec::from_iter(diff::outcome(outcome, result));
        report.extend(diff::cluster(state_machine.state(), &applied));

        // Collection state is expensive to read, so limit it to possible changes
        for collection in collections {
            let machine_collection = state_machine.state().collection(&collection);
            let applied_collection = toc.collection_state(&collection);

            if let (Some(machine), Some(applied)) = (machine_collection, applied_collection) {
                report.extend(diff::collection(&collection, machine, &applied));
            }
        }

        if report.is_empty() {
            log::debug!("Consensus state machine matches applied state");
            return None;
        }

        // Consensus state machine has diverged from operation handlers.
        // Invalidate it so it reinitializes from applied state on next operation.
        self.invalidate();

        let report = report.join(", ");
        Some(report)
    }

    fn resync(
        &mut self,
        toc: &impl CollectionContainer,
        collections: impl IntoIterator<Item = CollectionId>,
    ) {
        for collection in collections {
            let state = toc.collection_state(&collection);
            log::debug!(
                "Consensus state machine resynchronizing collection {collection:?}: present={}",
                state.is_some(),
            );
            self.state_machine.resync_collection(&collection, state);
        }
    }
}

/// Read complete applied state to initialize consensus state machine
pub fn read_cluster_state(toc: &impl CollectionContainer, persistent: &Persistent) -> ClusterState {
    let CollectionsSnapshot {
        collections,
        aliases,
    } = toc.collections_snapshot();

    ClusterState {
        collections,
        aliases,
        peer_address_by_id: persistent.peer_address_by_id.read().clone(),
        peer_metadata_by_id: persistent.peer_metadata_by_id.read().clone(),
        cluster_metadata: persistent.cluster_metadata.clone(),
        quota_config: toc.quota_config(),
    }
}

/// Read applied state without loading full collection state
pub fn read_shallow_state(toc: &impl CollectionContainer, persistent: &Persistent) -> ShallowState {
    ShallowState {
        collections: toc.collection_names(),
        aliases: toc.alias_mapping(),
        peer_address_by_id: persistent.peer_address_by_id.read().clone(),
        peer_metadata_by_id: persistent.peer_metadata_by_id.read().clone(),
        cluster_metadata: persistent.cluster_metadata.clone(),
        quota_config: toc.quota_config(),
    }
}

/// Collections an operation might change, including any alias target
fn target_collections(operation: &ConsensusOperations, state: &ClusterState) -> Vec<CollectionId> {
    let operation = match operation {
        ConsensusOperations::RemovePeer(_) => {
            let mut collections: Vec<_> = state.collections.keys().cloned().collect();
            collections.sort();
            return collections;
        }
        ConsensusOperations::CollectionMeta(operation) => operation,
        ConsensusOperations::AddPeer { .. }
        | ConsensusOperations::UpdatePeerMetadata { .. }
        | ConsensusOperations::UpdateClusterMetadata { .. }
        | ConsensusOperations::SetQuotaConfig(_)
        | ConsensusOperations::RequestSnapshot
        | ConsensusOperations::ReportSnapshot { .. } => return Vec::new(),
    };

    let collection = match operation.as_ref() {
        CollectionMetaOperations::CreateCollection(op) => &op.collection_name,
        CollectionMetaOperations::UpdateCollection(op) => &op.collection_name,
        CollectionMetaOperations::DeleteCollection(op) => &op.0,
        CollectionMetaOperations::CreateShardKey(op) => &op.collection_name,
        CollectionMetaOperations::DropShardKey(op) => &op.collection_name,
        CollectionMetaOperations::SetShardReplicaState(op) => &op.collection_name,
        CollectionMetaOperations::Resharding(collection, _) => collection,
        CollectionMetaOperations::TransferShard(collection, _) => collection,
        CollectionMetaOperations::CreateNamedVector(op) => &op.collection_name,
        CollectionMetaOperations::DeleteNamedVector(op) => &op.collection_name,
        CollectionMetaOperations::CreatePayloadIndex(op) => &op.collection_name,
        CollectionMetaOperations::DropPayloadIndex(op) => &op.collection_name,

        CollectionMetaOperations::Nop { .. } | CollectionMetaOperations::ChangeAliases(_) => {
            return Vec::new();
        }

        #[cfg(feature = "staging")]
        CollectionMetaOperations::TestSlowDown(_)
        | CollectionMetaOperations::TestTransientError(_) => return Vec::new(),
    };

    match state.aliases.get(collection) {
        Some(resolved) if collection != resolved => {
            vec![collection.clone(), resolved.clone()]
        }

        _ => vec![collection.clone()],
    }
}

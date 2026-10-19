use raft::eraftpb::{ConfChangeType, ConfChangeV2, Entry as RaftEntry};
use raft::{RawNode, Storage};
use tonic::transport::Uri;

use super::*;
use crate::content_manager::collection_meta_ops::CollectionMetaOperations;

impl<C: CollectionContainer> ConsensusManager<C> {
    pub(super) fn apply_with_state_machine(
        &self,
        operation: &ConsensusOperations,
    ) -> StorageResult<bool> {
        // DropCollection also cleans up incomplete directories when the collection is absent.
        // Its presence in the action list therefore cannot determine the API result.
        let status = if let ConsensusOperations::CollectionMeta(operation) = operation
            && let CollectionMetaOperations::DeleteCollection(operation) = operation.as_ref()
        {
            self.toc.collection_names().contains(&operation.0)
        } else {
            true
        };
        let outcome = self.plan_operation(operation);
        let result = match &outcome {
            ApplyOutcome::Accepted(actions) => actions
                .iter()
                .try_for_each(|action| self.apply_action(action.clone()))
                .map(|()| status),
            ApplyOutcome::Rejected(error) => Err(error.clone()),
            ApplyOutcome::NotCovered => Err(StorageError::service_error(format!(
                "Consensus operation is not modeled: {operation:?}",
            ))),
        };

        let removes_self = matches!(operation, ConsensusOperations::RemovePeer(peer_id) if *peer_id == self.this_peer_id());
        if removes_self {
            // Self-removal clears all outbound addresses and stops consensus application
            self.invalidate_state_machine();
        } else if matches!(result, Err(StorageError::ServiceError { .. })) {
            // The interpreter has advanced past any writes that failed. Replay must start
            // from the state those writes actually left in storage.
            self.invalidate_state_machine();
        } else if Self::should_check_state_machine() {
            self.compare_state_machine(operation, &outcome, &result);
        }

        result
    }

    pub(super) fn plan_operation(&self, operation: &ConsensusOperations) -> ApplyOutcome {
        let persistent = self.persistent.read();
        self.state_machine
            .lock()
            .apply(self.toc.as_ref(), &persistent, operation)
    }

    fn should_check_state_machine() -> bool {
        // Check every entry in debug builds so tests catch divergences deterministically.
        // Collection reads are expensive, so release builds check a 1% sample.
        cfg!(any(test, debug_assertions)) || rand::random_ratio(1, 100)
    }

    pub(super) fn notify_operation(
        &self,
        operation: &ConsensusOperations,
        result: StorageResult<bool>,
    ) {
        let on_apply = self.on_consensus_op_apply.lock().remove(operation);
        if let Some(on_apply) = on_apply
            && on_apply.send(result).is_err()
        {
            log::debug!("Failed to notify subscribers on consensus operation completion");
        }
    }

    pub(super) fn apply_conf_change_with_state_machine<T: Storage>(
        &self,
        entry: &RaftEntry,
        change: &ConfChangeV2,
        raw_node: &mut RawNode<T>,
    ) -> StorageResult<bool> {
        let result = self.apply_conf_change_actions(entry, change, raw_node);
        if result.is_err() {
            self.invalidate_state_machine();
        }

        result
    }

    fn apply_conf_change_actions<T: Storage>(
        &self,
        entry: &RaftEntry,
        change: &ConfChangeV2,
        raw_node: &mut RawNode<T>,
    ) -> StorageResult<bool> {
        let mut operations = Vec::with_capacity(change.changes.len());
        for single_change in &change.changes {
            let operation = match single_change.change_type() {
                ConfChangeType::AddNode | ConfChangeType::AddLearnerNode => {
                    if entry.context.is_empty() {
                        // Promotions carry no address. Older learner additions may also omit it.
                        continue;
                    }

                    let uri: Uri = std::str::from_utf8(&entry.context)
                        .map_err(|error| {
                            StorageError::service_error(format!(
                                "Failed to parse peer URI: {error}"
                            ))
                        })?
                        .parse()
                        .map_err(|error| {
                            StorageError::service_error(format!(
                                "Failed to parse peer URI: {error}"
                            ))
                        })?;
                    ConsensusOperations::AddPeer {
                        peer_id: single_change.node_id,
                        uri: uri.to_string(),
                    }
                }
                ConfChangeType::RemoveNode => {
                    ConsensusOperations::RemovePeer(single_change.node_id)
                }
            };
            operations.push(operation);
        }

        let batch = {
            let persistent = self.persistent.read();
            self.state_machine
                .lock()
                .apply_batch(self.toc.as_ref(), &persistent, &operations)
        };
        let batch = match batch {
            Ok(batch) => batch,
            Err(error @ StorageError::ServiceError { .. }) => return Err(error),
            Err(error) => {
                // Raft permits rejecting a committed configuration change by not applying it.
                // No address, collection or membership writes have run at this point.
                for operation in &operations {
                    self.notify_operation(operation, Err(error.clone()));
                }
                log::warn!("Rejected consensus configuration change: {error}");
                return Ok(false);
            }
        };

        let conf_state = apply_raft_conf_change(raw_node, change)?;
        self.persistent
            .write()
            .apply_state_update(|state| state.conf_state = conf_state)?;

        for actions in &batch {
            for action in actions {
                self.apply_action(action.clone())?;
            }
        }

        let stop_consensus = operations.iter().any(|operation| {
            matches!(operation, ConsensusOperations::RemovePeer(peer_id) if *peer_id == self.this_peer_id())
        });
        if stop_consensus {
            self.invalidate_state_machine();
        } else if Self::should_check_state_machine() {
            // Planning advanced to the end of the batch, so compare only after every action ran
            for (operation, actions) in operations.iter().zip(batch) {
                self.compare_state_machine(operation, &ApplyOutcome::Accepted(actions), &Ok(true));
            }
        }

        for operation in &operations {
            self.notify_operation(operation, Ok(true));
        }

        Ok(stop_consensus)
    }
}

/// Retry configuration bookkeeping after membership changed but an action or save failed
fn apply_raft_conf_change<T: Storage>(
    raw_node: &mut RawNode<T>,
    change: &ConfChangeV2,
) -> StorageResult<raft::eraftpb::ConfState> {
    let current = raw_node.raft.prs().conf().to_conf_state();
    let joint = !current.voters_outgoing.is_empty();

    // Raft rejects entering an already joint configuration and leaving a non-joint one.
    // Membership is saved before actions, so either transition can already be complete
    // when consensus restarts to retry the same unapplied entry.
    if change.leave_joint() && !joint {
        return Ok(current);
    }

    if let Some(auto_leave) = change.enter_joint()
        && joint
        && current.auto_leave == auto_leave
    {
        let mut checked = HashSet::new();
        let applied = change.changes.iter().rev().all(|single| {
            // A batch may change the same peer more than once; its final change wins
            if !checked.insert(single.node_id) {
                return true;
            }

            let voter = current.voters.contains(&single.node_id);
            let learner = current.learners.contains(&single.node_id)
                || current.learners_next.contains(&single.node_id);
            match single.change_type() {
                ConfChangeType::AddNode => voter && !learner,
                ConfChangeType::AddLearnerNode => learner && !voter,
                ConfChangeType::RemoveNode => !voter && !learner,
            }
        });
        if applied {
            return Ok(current);
        }
    }

    Ok(raw_node.apply_conf_change(change)?)
}

use super::*;

/// Dispatch over the semantic marker-kind rules shared with chat rebuild.
/// Responses markers use the strongly bound typed-tail path; builtin markers
/// fall through to the builtin reducer; unknown kinds fail closed in
/// [`dispatch_compaction_marker`].
pub(super) fn try_replay_responses(
    updates_path: &Path,
    session_dir: &Path,
    live_checkpoint: Option<&ConversationItem>,
    target_prompt_index: usize,
) -> io::Result<Option<ReplayResult>> {
    validate_persisted_compaction_marker_kinds(updates_path)?;
    let Some(iter) = UpdatesIterator::open(updates_path)? else {
        return Ok(None);
    };
    let updates =
        crate::session::storage::filter_rewind_updates(iter.filter_map(Result::ok).collect());
    match dispatch_compaction_marker(&updates)? {
        CompactionMarkerDispatch::Responses {
            marker_index,
            marker,
        } => Ok(Some(replay_responses(
            &updates,
            marker_index,
            &marker,
            live_checkpoint,
            session_dir,
            target_prompt_index,
            false,
        )?)),
        CompactionMarkerDispatch::Builtin { marker, .. }
            if target_prompt_index < marker.prompt_index_at_compaction =>
        {
            let historical = updates
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, update)| {
                    let SessionUpdate::Xai(notification) = update else {
                        return None;
                    };
                    let XaiSessionUpdate::CompactionCheckpoint(marker) = &notification.update
                    else {
                        return None;
                    };
                    (marker.kind == CompactionCheckpointKind::ResponsesServer
                        && marker.prompt_index_at_compaction <= target_prompt_index)
                        .then_some((index, marker.as_ref()))
                });
            match historical {
                Some((marker_index, marker)) => Ok(Some(replay_responses(
                    &updates,
                    marker_index,
                    marker,
                    live_checkpoint,
                    session_dir,
                    target_prompt_index,
                    true,
                )?)),
                None => Ok(None),
            }
        }
        CompactionMarkerDispatch::None | CompactionMarkerDispatch::Builtin { .. } => Ok(None),
    }
}

/// Replay a Responses marker through its strongly bound sidecar and
/// committed typed-tail journal. Any mismatch fails closed; this function
/// never falls back to the builtin reducer.
fn replay_responses(
    updates: &[SessionUpdate],
    marker_index: usize,
    marker: &CompactionCheckpointInfo,
    live_checkpoint: Option<&ConversationItem>,
    session_dir: &Path,
    target_prompt_index: usize,
    allow_historical_marker: bool,
) -> io::Result<ReplayResult> {
    let matching_live_wrapper = live_checkpoint
        .and_then(ConversationItem::as_responses_checkpoint)
        .filter(|wrapper| wrapper.checkpoint_id == marker.checkpoint_id)
        .cloned();
    let (checkpoint, wrapper) = if let Some(wrapper) = matching_live_wrapper {
        let checkpoint =
            crate::session::storage::responses_compaction::read_checkpoint_for_wrapper(
                session_dir,
                &wrapper,
            )?;
        (checkpoint, wrapper)
    } else if allow_historical_marker {
        let digest = marker.portable_history_sha256.as_deref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "historical Responses marker has no portable-history digest",
            )
        })?;
        let checkpoint = crate::session::storage::responses_compaction::read_checkpoint(
            session_dir,
            &marker.checkpoint_file,
            &marker.checkpoint_id,
            marker.prompt_index_at_compaction,
            digest,
        )?;
        let mut wrapper = checkpoint.wrapper.clone();
        wrapper.branch_id = marker.branch_id.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "historical Responses marker has no active branch",
            )
        })?;
        (checkpoint, wrapper)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Responses marker requires the current checkpoint wrapper \
             at the head of live history",
        ));
    };
    crate::session::storage::responses_compaction::validate_marker_for_wrapper(
        marker,
        &checkpoint.wrapper,
    )?;

    if target_prompt_index < marker.prompt_index_at_compaction {
        let mut conversation = checkpoint.portable_history;
        let keep = xai_grok_sampling_types::conversation_truncate_for_prompt(
            &conversation,
            target_prompt_index,
        );
        conversation.truncate(keep);
        return Ok(ReplayResult {
            conversation,
            prompt_index_reached: target_prompt_index,
            original_user_info: checkpoint.original_user_info,
            last_compaction_prompt_index: None,
        });
    }

    use crate::session::storage::responses_compaction::{PersistedChatEntry, TailJournalRecord};
    let records = updates[marker_index + 1..]
        .iter()
        .filter_map(|update| {
            let SessionUpdate::Xai(notification) = update else {
                return None;
            };
            match &notification.update {
                XaiSessionUpdate::ConversationAppendPrepared(prepared) => {
                    Some(TailJournalRecord::Prepared((**prepared).clone()))
                }
                XaiSessionUpdate::ConversationAppendCommitted(committed) => {
                    Some(TailJournalRecord::Committed(committed.clone()))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    // The sidecar carries the branch at compaction time. The selected replay
    // wrapper carries either the live branch or the strongly bound historical
    // marker branch, so journal filtering and the rebuilt head use it.
    let entries = crate::session::storage::responses_compaction::rebuild_updates_only_entries(
        wrapper, &records,
    )?;
    let mut kept = Vec::with_capacity(entries.len());
    for entry in entries {
        match &entry {
            PersistedChatEntry::Item(_) => kept.push(entry),
            PersistedChatEntry::Tail(tail) if tail.prompt_index <= target_prompt_index => {
                kept.push(entry);
            }
            PersistedChatEntry::Tail(_) => break,
        }
    }
    let conversation =
        crate::session::storage::responses_compaction::recover_history_entries(kept)?.conversation;
    Ok(ReplayResult {
        conversation,
        prompt_index_reached: target_prompt_index,
        original_user_info: checkpoint.original_user_info,
        last_compaction_prompt_index: Some(marker.prompt_index_at_compaction),
    })
}

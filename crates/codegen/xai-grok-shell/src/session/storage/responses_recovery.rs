use super::*;

/// Dispatch using the shared semantic marker rules. Builtin checkpoints
/// stay with the builtin reducer; Responses checkpoints use the strongly
/// bound typed-tail rebuild; unknown kinds fail closed before any write.
pub(super) fn try_rebuild_responses(dir: &Path, updates_path: &Path) -> io::Result<Option<usize>> {
    crate::session::helpers::replay::validate_persisted_compaction_marker_kinds(updates_path)?;
    let Some(iter) = UpdatesIterator::open(updates_path)? else {
        return Ok(None);
    };
    let updates = super::filter_rewind_updates(iter.filter_map(Result::ok).collect());
    match crate::session::helpers::replay::dispatch_compaction_marker(&updates)? {
        crate::session::helpers::replay::CompactionMarkerDispatch::Responses {
            marker_index,
            marker,
        } => Ok(Some(rebuild_responses(
            dir,
            &updates,
            marker_index,
            &marker,
        )?)),
        crate::session::helpers::replay::CompactionMarkerDispatch::None
        | crate::session::helpers::replay::CompactionMarkerDispatch::Builtin { .. } => Ok(None),
    }
}

/// Rebuild the current typed history from the durable wrapper, its bound
/// sidecar, and committed tail journal records.
fn rebuild_responses(
    dir: &Path,
    updates: &[super::SessionUpdate],
    marker_index: usize,
    marker: &crate::extensions::notification::CompactionCheckpointInfo,
) -> io::Result<usize> {
    let chat_path = dir.join(CHAT_HISTORY_FILE);
    let wrapper = read_live_wrapper(&chat_path)?;
    let checkpoint = super::responses_compaction::read_checkpoint_for_wrapper(dir, &wrapper)?;
    super::responses_compaction::validate_marker_for_wrapper(marker, &checkpoint.wrapper)?;

    let records = updates[marker_index + 1..]
        .iter()
        .filter_map(|update| {
            let super::SessionUpdate::Xai(notification) = update else {
                return None;
            };
            match &notification.update {
                crate::extensions::notification::SessionUpdate::ConversationAppendPrepared(
                    prepared,
                ) => Some(super::responses_compaction::TailJournalRecord::Prepared(
                    (**prepared).clone(),
                )),
                crate::extensions::notification::SessionUpdate::ConversationAppendCommitted(
                    committed,
                ) => Some(super::responses_compaction::TailJournalRecord::Committed(
                    committed.clone(),
                )),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    // The sidecar stores the branch at compaction time, while the live
    // wrapper may carry a valid rewind/fork rotation. Rebuild against the
    // live branch so abandoned journal records cannot re-enter history.
    let entries = super::responses_compaction::rebuild_updates_only_entries(wrapper, &records)?;
    let count = entries.len();
    super::responses_compaction::write_history_durable(&chat_path, &entries)?;
    Ok(count)
}

/// Read the one supported live wrapper from the history head.
fn read_live_wrapper(
    chat_path: &Path,
) -> io::Result<xai_grok_sampling_types::ServerResponsesCheckpoint> {
    use std::io::BufRead;

    let file = std::fs::File::open(chat_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "Responses marker present but {} is missing: \
                 cannot bind the live wrapper to the sidecar",
                chat_path.display()
            ),
        )
    })?;
    let mut reader = std::io::BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "chat history is empty; Responses marker has no live wrapper to bind",
            ));
        }
        if !line.trim().is_empty() {
            break;
        }
    }
    let item: ConversationItem = serde_json::from_str(line.trim()).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "cannot parse the live wrapper from {}: {error}",
                chat_path.display()
            ),
        )
    })?;
    item.as_responses_checkpoint().cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Responses marker requires the current checkpoint wrapper \
             at the head of chat_history.jsonl",
        )
    })
}

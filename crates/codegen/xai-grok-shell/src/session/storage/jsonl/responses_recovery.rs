use super::*;

impl JsonlStorageAdapter {
    fn append_recovery_update_sync(
        &self,
        info: &Info,
        update: crate::extensions::notification::SessionUpdate,
    ) -> io::Result<()> {
        let update = super::SessionUpdate::Xai(Box::new(
            crate::extensions::notification::SessionNotification {
                session_id: info.id.clone(),
                update,
                meta: None,
            },
        ));
        let envelope = SessionUpdateEnvelope::from_update(&update)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let mut line = serde_json::to_vec(&envelope)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        line.push(b'\n');
        Self::append_jsonl_line_sync(&self.updates_file(info), line, AppendDurability::Durable)
    }

    pub(super) fn repair_responses_recovery_sync(
        &self,
        info: &Info,
        conversation: &[ConversationItem],
    ) -> io::Result<()> {
        crate::session::helpers::replay::validate_persisted_compaction_marker_kinds(
            &self.updates_file(info),
        )?;
        let Some(checkpoint) = conversation
            .first()
            .and_then(ConversationItem::as_responses_checkpoint)
        else {
            return Ok(());
        };
        self.repair_responses_marker_sync(
            info,
            |marker| {
                marker.branch_id.as_deref() == Some(checkpoint.branch_id.as_str())
                    && super::responses_compaction::validate_marker_for_wrapper(marker, checkpoint)
                        .is_ok()
            },
            || super::responses_compaction::marker_for_wrapper(checkpoint),
            super::responses_compaction::read_history(&self.chat_file(info))?,
        )?;
        if checkpoint.mode.name == "segments" {
            let session_dir = self.session_dir(info);
            let staging = super::responses_compaction::read_segment_staging_for_wrapper(
                &session_dir,
                checkpoint,
            )?;
            let published = match staging.as_ref() {
                Some(staging) => {
                    super::responses_compaction::publish_staged_compaction_segment_durable(
                        &session_dir,
                        &checkpoint.checkpoint_id,
                        &staging.operation_id,
                        &staging.wrapper_digest,
                    )
                }
                None => super::responses_compaction::publish_staged_compaction_segment_durable(
                    &session_dir,
                    &checkpoint.checkpoint_id,
                    &checkpoint.operation_id,
                    &checkpoint.wrapper_digest(),
                ),
            };
            match published {
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    // Forks may intentionally omit the auxiliary segment
                    // archive. The strongly bound checkpoint/replay sidecar is
                    // still complete, so missing segment presentation data
                    // must not make the session unloadable.
                    tracing::warn!(
                        checkpoint_id = %checkpoint.checkpoint_id,
                        "Responses segment staging/archive is absent; continuing without transcript segment"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Append a missing marker and missing prepared/committed tail journal
    /// records so a crash between CAS and journal persistence is repaired
    /// from the live wrapper and authoritative history.
    pub(super) fn repair_responses_marker_sync(
        &self,
        info: &Info,
        marker_matches: impl Fn(&crate::extensions::notification::CompactionCheckpointInfo) -> bool,
        build_marker: impl Fn() -> crate::extensions::notification::CompactionCheckpointInfo,
        history: super::responses_compaction::RecoveredHistory,
    ) -> io::Result<()> {
        let updates = self.read_updates_jsonl(self.updates_file(info))?;
        crate::session::helpers::replay::validate_compaction_marker_kinds(&updates)?;
        // Replay always dispatches from the latest marker. If that marker is
        // absent or belongs to an abandoned branch, append the active marker
        // first and treat only records after it as the recovery journal.
        let latest_marker = updates
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, update)| {
                let super::SessionUpdate::Xai(notification) = update else {
                    return None;
                };
                let crate::extensions::notification::SessionUpdate::CompactionCheckpoint(marker) =
                    &notification.update
                else {
                    return None;
                };
                Some((index, marker.as_ref()))
            });
        let append_active_marker = || -> io::Result<usize> {
            self.append_recovery_update_sync(
                info,
                crate::extensions::notification::SessionUpdate::CompactionCheckpoint(Box::new(
                    build_marker(),
                )),
            )?;
            Ok(updates.len())
        };
        let marker_index = match latest_marker {
            Some((index, marker))
                if marker.kind
                    == crate::extensions::notification::CompactionCheckpointKind::ResponsesServer
                    && marker_matches(marker) =>
            {
                index
            }
            Some((_, marker))
                if marker.kind
                    == crate::extensions::notification::CompactionCheckpointKind::ResponsesServer =>
            {
                append_active_marker()?
            }
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "live Responses wrapper conflicts with a non-Responses latest marker",
                ));
            }
            None => append_active_marker()?,
        };
        let journal = updates.get(marker_index + 1..).unwrap_or(&[]);
        let prepared = journal
            .iter()
            .filter_map(|update| {
                let super::SessionUpdate::Xai(notification) = update else {
                    return None;
                };
                let crate::extensions::notification::SessionUpdate::ConversationAppendPrepared(
                    prepared,
                ) = &notification.update
                else {
                    return None;
                };
                Some(prepared.as_ref())
            })
            .collect::<Vec<_>>();
        let committed = journal
            .iter()
            .filter_map(|update| {
                let super::SessionUpdate::Xai(notification) = update else {
                    return None;
                };
                let crate::extensions::notification::SessionUpdate::ConversationAppendCommitted(
                    committed,
                ) = &notification.update
                else {
                    return None;
                };
                Some(committed)
            })
            .collect::<std::collections::BTreeSet<_>>();
        for (prepared_repair, committed_repair) in history
            .prepared_repairs
            .into_iter()
            .zip(history.committed_repairs)
        {
            if !prepared.contains(&&prepared_repair) {
                self.append_recovery_update_sync(
                    info,
                    crate::extensions::notification::SessionUpdate::ConversationAppendPrepared(
                        Box::new(prepared_repair),
                    ),
                )?;
            }
            if !committed.contains(&committed_repair) {
                self.append_recovery_update_sync(
                    info,
                    crate::extensions::notification::SessionUpdate::ConversationAppendCommitted(
                        committed_repair,
                    ),
                )?;
            }
        }
        Ok(())
    }
}

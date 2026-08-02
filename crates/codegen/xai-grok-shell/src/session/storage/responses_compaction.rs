use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Seek, Write};
use std::path::{Component, Path, PathBuf};

use fs2::FileExt as _;
use serde::{Deserialize, Serialize};
use xai_grok_sampling_types::{
    CheckpointReplayMaterial, ConversationItem, RESPONSES_COMPACTION_CONTRACT,
    ServerResponsesCheckpoint,
};

use crate::extensions::notification::CompactionCheckpointKind;

/// Responses server-compaction sidecar.
///
/// Replay verification recomputes every digest from persisted data instead of
/// trusting the sidecar wholesale.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactionCheckpointFile {
    pub kind: CompactionCheckpointKind,
    pub checkpoint_id: String,
    pub operation_id: String,
    pub prompt_index_at_compaction: usize,
    pub wrapper: ServerResponsesCheckpoint,
    pub replay_material: CheckpointReplayMaterial,
    pub portable_history: Vec<ConversationItem>,
    pub portable_history_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_user_info: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reread_file_paths: Vec<String>,
    pub created_at: String,
}

impl CompactionCheckpointFile {
    pub fn new(
        mut wrapper: ServerResponsesCheckpoint,
        replay_material: CheckpointReplayMaterial,
        portable_history: Vec<ConversationItem>,
        original_user_info: Option<String>,
        reread_file_paths: Vec<String>,
    ) -> io::Result<Self> {
        let digest = portable_history_digest(&portable_history)?;
        let portable_bytes = portable_history_bytes(&portable_history)?.len() as u64;
        if wrapper.portable_history_sha256 != digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "wrapper portable history digest does not match sidecar history",
            ));
        }
        wrapper.portable_history_bytes = portable_bytes;
        if replay_material.portable_history_sha256() != digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay material portable history digest does not match sidecar history",
            ));
        }
        if replay_material.wrapper_digest() != wrapper.wrapper_digest() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay material wrapper digest does not match the wrapper",
            ));
        }
        if replay_material.branch_id() != wrapper.branch_id
            || replay_material.prior_checkpoint_id() != wrapper.prior_checkpoint_id.as_deref()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay material branch or prior checkpoint does not match the wrapper",
            ));
        }
        Ok(Self {
            kind: CompactionCheckpointKind::ResponsesServer,
            checkpoint_id: wrapper.checkpoint_id.clone(),
            operation_id: wrapper.operation_id.clone(),
            prompt_index_at_compaction: wrapper.prompt_index,
            created_at: wrapper.created_at.to_rfc3339(),
            wrapper,
            replay_material,
            portable_history,
            portable_history_sha256: digest,
            original_user_info,
            reread_file_paths,
        })
    }
}

/// Staged (unpublished) Responses compaction segment.
///
/// The payload binds through the checkpoint operation, branch, and wrapper
/// digest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponsesCompactionSegmentStaging {
    pub kind: CompactionCheckpointKind,
    pub checkpoint_id: String,
    pub operation_id: String,
    pub branch_id: String,
    pub wrapper_digest: String,
    pub items: Vec<ConversationItem>,
    pub summary: String,
    pub detail: String,
    pub timestamp: String,
}

impl ResponsesCompactionSegmentStaging {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        checkpoint_id: impl Into<String>,
        operation_id: impl Into<String>,
        branch_id: impl Into<String>,
        wrapper_digest: impl Into<String>,
        items: Vec<ConversationItem>,
        summary: impl Into<String>,
        detail: xai_chat_state::CompactionDetail,
        timestamp: impl Into<String>,
    ) -> io::Result<Self> {
        let staging = Self {
            kind: CompactionCheckpointKind::ResponsesServer,
            checkpoint_id: checkpoint_id.into(),
            operation_id: operation_id.into(),
            branch_id: branch_id.into(),
            wrapper_digest: wrapper_digest.into(),
            items,
            summary: summary.into(),
            detail: detail.to_string(),
            timestamp: timestamp.into(),
        };
        validate_segment_staging(&staging)?;
        Ok(staging)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishedCompactionSegment {
    pub index: u64,
    pub newly_published: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedTail {
    pub operation_id: String,
    pub checkpoint_id: String,
    pub branch_id: String,
    pub sequence: u64,
    pub prompt_index: usize,
    pub item: ConversationItem,
}

#[derive(Debug, Clone)]
pub enum PersistedChatEntry {
    Item(ConversationItem),
    Tail(PersistedTail),
}

impl Serialize for PersistedChatEntry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Item(item) => item.serialize(serializer),
            Self::Tail(tail) => {
                let mut value = serde_json::to_value(tail).map_err(serde::ser::Error::custom)?;
                value
                    .as_object_mut()
                    .expect("PersistedTail serializes as an object")
                    .insert(
                        "persisted_entry".into(),
                        serde_json::Value::String("tail".into()),
                    );
                value.serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for PersistedChatEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut value = serde_json::Value::deserialize(deserializer)?;
        if value
            .get("persisted_entry")
            .and_then(serde_json::Value::as_str)
            == Some("tail")
        {
            value
                .as_object_mut()
                .expect("tagged tail is an object")
                .remove("persisted_entry");
            return serde_json::from_value(value)
                .map(Self::Tail)
                .map_err(serde::de::Error::custom);
        }
        serde_json::from_value(value)
            .map(Self::Item)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationAppendPrepared {
    pub operation_id: String,
    pub checkpoint_id: String,
    pub branch_id: String,
    pub sequence: u64,
    pub prompt_index: usize,
    pub item: ConversationItem,
}

impl PartialEq for ConversationAppendPrepared {
    fn eq(&self, other: &Self) -> bool {
        self.operation_id == other.operation_id
            && self.checkpoint_id == other.checkpoint_id
            && self.branch_id == other.branch_id
            && self.sequence == other.sequence
            && self.prompt_index == other.prompt_index
            && serde_json::to_value(&self.item).ok() == serde_json::to_value(&other.item).ok()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConversationAppendCommitted {
    pub operation_id: String,
    pub checkpoint_id: String,
    pub branch_id: String,
    pub sequence: u64,
    pub prompt_index: usize,
}

impl From<&ConversationAppendPrepared> for ConversationAppendCommitted {
    fn from(prepared: &ConversationAppendPrepared) -> Self {
        Self {
            operation_id: prepared.operation_id.clone(),
            checkpoint_id: prepared.checkpoint_id.clone(),
            branch_id: prepared.branch_id.clone(),
            sequence: prepared.sequence,
            prompt_index: prepared.prompt_index,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TailJournalRecord {
    Prepared(ConversationAppendPrepared),
    Committed(ConversationAppendCommitted),
}

#[derive(Debug, Clone)]
pub struct RecoveredHistory {
    pub conversation: Vec<ConversationItem>,
    pub prepared_repairs: Vec<ConversationAppendPrepared>,
    pub committed_repairs: Vec<ConversationAppendCommitted>,
}

pub fn portable_history_bytes(history: &[ConversationItem]) -> io::Result<Vec<u8>> {
    xai_grok_sampling_types::portable_history_bytes(history)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn portable_history_digest(history: &[ConversationItem]) -> io::Result<String> {
    xai_grok_sampling_types::portable_history_digest(history)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn write_checkpoint_durable(
    session_dir: &Path,
    relative_path: &str,
    checkpoint: &CompactionCheckpointFile,
) -> io::Result<()> {
    validate_relative_checkpoint_path(relative_path)?;
    if checkpoint.kind != CompactionCheckpointKind::ResponsesServer {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Responses checkpoint kind",
        ));
    }
    validate_checkpoint(checkpoint, None, None, None)?;
    let path = safe_join_for_write(session_dir, relative_path)?;
    let bytes = serde_json::to_vec(checkpoint)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_bytes_durable(&path, &bytes)
}

pub fn read_checkpoint(
    session_dir: &Path,
    relative_path: &str,
    expected_checkpoint_id: &str,
    expected_prompt_index: usize,
    expected_digest: &str,
) -> io::Result<CompactionCheckpointFile> {
    let path = safe_join_for_read(session_dir, relative_path)?;
    let bytes = std::fs::read(path)?;
    let checkpoint: CompactionCheckpointFile = serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    validate_checkpoint(
        &checkpoint,
        Some(expected_checkpoint_id),
        Some(expected_prompt_index),
        Some(expected_digest),
    )?;
    if checkpoint.wrapper.portable_history_path != relative_path {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "wrapper checkpoint path mismatch",
        ));
    }
    Ok(checkpoint)
}

/// Read and strongly bind the current sidecar to a live wrapper.
///
/// Rewind and fork may rotate only the active branch. Replacing the stored
/// branch with the live branch must reconcile the wrapper digest; every other
/// immutable binding field must match.
pub fn read_checkpoint_for_wrapper(
    session_dir: &Path,
    wrapper: &ServerResponsesCheckpoint,
) -> io::Result<CompactionCheckpointFile> {
    let checkpoint = read_checkpoint(
        session_dir,
        &wrapper.portable_history_path,
        &wrapper.checkpoint_id,
        wrapper.prompt_index,
        &wrapper.portable_history_sha256,
    )?;
    let stored = &checkpoint.wrapper;
    let rotated = stored.branch_id != wrapper.branch_id;
    let mut rebound = stored.clone();
    rebound.branch_id = wrapper.branch_id.clone();
    if &rebound != wrapper {
        let detail = if rotated {
            "rotated branch does not reconcile the wrapper digest"
        } else {
            "active wrapper does not match checkpoint sidecar"
        };
        return Err(io::Error::new(io::ErrorKind::InvalidData, detail));
    }
    Ok(checkpoint)
}

fn validate_checkpoint(
    checkpoint: &CompactionCheckpointFile,
    expected_checkpoint_id: Option<&str>,
    expected_prompt_index: Option<usize>,
    expected_digest: Option<&str>,
) -> io::Result<()> {
    if checkpoint.kind != CompactionCheckpointKind::ResponsesServer {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Responses checkpoint kind",
        ));
    }
    if expected_checkpoint_id.is_some_and(|id| checkpoint.checkpoint_id != id)
        || checkpoint.wrapper.checkpoint_id != checkpoint.checkpoint_id
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint id mismatch",
        ));
    }
    if expected_prompt_index.is_some_and(|index| checkpoint.prompt_index_at_compaction != index)
        || checkpoint.wrapper.prompt_index != checkpoint.prompt_index_at_compaction
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint prompt index mismatch",
        ));
    }
    if checkpoint.wrapper.operation_id != checkpoint.operation_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint operation id mismatch",
        ));
    }
    let digest = portable_history_digest(&checkpoint.portable_history)?;
    if digest != checkpoint.portable_history_sha256
        || digest != checkpoint.wrapper.portable_history_sha256
        || digest != checkpoint.replay_material.portable_history_sha256()
        || expected_digest.is_some_and(|expected| digest != expected)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "portable history digest mismatch",
        ));
    }
    if checkpoint.wrapper.portable_history_bytes
        != portable_history_bytes(&checkpoint.portable_history)?.len() as u64
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "portable history byte count mismatch",
        ));
    }
    if checkpoint.wrapper.wrapper_digest() != checkpoint.replay_material.wrapper_digest() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "wrapper digest mismatch",
        ));
    }
    if checkpoint.wrapper.output.is_empty()
        || checkpoint.wrapper.server_output_item_count != checkpoint.wrapper.output.len()
        || checkpoint.wrapper.checkpoint_token_seed == 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid checkpoint output or token seed",
        ));
    }
    if checkpoint.replay_material.branch_id() != checkpoint.wrapper.branch_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay material branch mismatch",
        ));
    }
    if checkpoint.replay_material.prior_checkpoint_id()
        != checkpoint.wrapper.prior_checkpoint_id.as_deref()
        || checkpoint.wrapper.identity.prior_checkpoint_id != checkpoint.wrapper.prior_checkpoint_id
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint prior chain mismatch",
        ));
    }
    if checkpoint.replay_material.contract_version() != RESPONSES_COMPACTION_CONTRACT
        || checkpoint.wrapper.identity.contract_version != RESPONSES_COMPACTION_CONTRACT
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint contract mismatch",
        ));
    }
    if checkpoint.replay_material.memory_revision() != checkpoint.wrapper.memory_revision {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint memory revision mismatch",
        ));
    }
    Ok(())
}

pub fn marker_for_wrapper(
    wrapper: &ServerResponsesCheckpoint,
) -> crate::extensions::notification::CompactionCheckpointInfo {
    crate::extensions::notification::CompactionCheckpointInfo {
        checkpoint_id: wrapper.checkpoint_id.clone(),
        prompt_index_at_compaction: wrapper.prompt_index,
        kind: CompactionCheckpointKind::ResponsesServer,
        checkpoint_file: wrapper.portable_history_path.clone(),
        auto_continue: None,
        operation_id: Some(wrapper.operation_id.clone()),
        branch_id: Some(wrapper.branch_id.clone()),
        portable_history_sha256: Some(wrapper.portable_history_sha256.clone()),
        responses_mode: Some(wrapper.mode.clone()),
        responses_auto_continue: Some(wrapper.auto_continue),
        created_at: wrapper.created_at.to_rfc3339(),
        wrapper_digest: Some(wrapper.wrapper_digest()),
        prior_checkpoint_id: wrapper.prior_checkpoint_id.clone(),
    }
}

/// Validate stable marker bindings. Creation time is diagnostic. A rotated
/// branch is accepted only when the marker digest recomputes from that exact
/// branch and every other wrapper field remains unchanged.
pub fn validate_marker_for_wrapper(
    marker: &crate::extensions::notification::CompactionCheckpointInfo,
    wrapper: &ServerResponsesCheckpoint,
) -> io::Result<()> {
    if marker.kind != CompactionCheckpointKind::ResponsesServer {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Responses marker kind",
        ));
    }
    if marker.checkpoint_id != wrapper.checkpoint_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker checkpoint id mismatch",
        ));
    }
    if marker.prompt_index_at_compaction != wrapper.prompt_index {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker prompt index mismatch",
        ));
    }
    if marker.checkpoint_file != wrapper.portable_history_path {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker checkpoint file mismatch",
        ));
    }
    if marker.operation_id.as_deref() != Some(wrapper.operation_id.as_str()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker operation id mismatch",
        ));
    }
    if marker.portable_history_sha256.as_deref() != Some(wrapper.portable_history_sha256.as_str()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker portable history digest mismatch",
        ));
    }
    let marker_branch = marker.branch_id.as_deref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker branch is missing",
        )
    })?;
    let wrapper_digest = wrapper.wrapper_digest_for_branch(marker_branch);
    if marker.wrapper_digest.as_deref() != Some(wrapper_digest.as_str()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker wrapper digest mismatch",
        ));
    }
    if marker.prior_checkpoint_id != wrapper.prior_checkpoint_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker prior checkpoint mismatch",
        ));
    }
    if marker.responses_auto_continue != Some(wrapper.auto_continue) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker auto-continue mismatch",
        ));
    }
    if marker.responses_mode.as_ref() != Some(&wrapper.mode) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker mode mismatch",
        ));
    }
    Ok(())
}

pub fn persisted_entries_for_replacement(
    operation_id: &str,
    messages: &[ConversationItem],
) -> io::Result<Vec<PersistedChatEntry>> {
    let Some(ConversationItem::ResponsesCompactionCheckpoint(checkpoint)) = messages.first() else {
        return Ok(messages
            .iter()
            .cloned()
            .map(PersistedChatEntry::Item)
            .collect());
    };
    let request = xai_grok_sampling_types::ConversationRequest {
        items: messages.to_vec(),
        ..Default::default()
    };
    request
        .validate_for_backend(&xai_grok_sampling_types::ApiBackend::Responses)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut entries = vec![PersistedChatEntry::Item(messages[0].clone())];
    let mut prompt_index = checkpoint.prompt_index;
    for (index, item) in messages.iter().skip(1).cloned().enumerate() {
        if matches!(item, ConversationItem::User(_)) {
            prompt_index = prompt_index.saturating_add(1);
        }
        let sequence = (index + 1) as u64;
        entries.push(PersistedChatEntry::Tail(PersistedTail {
            operation_id: format!("{operation_id}-{sequence}"),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: checkpoint.branch_id.clone(),
            sequence,
            prompt_index,
            item,
        }));
    }
    Ok(entries)
}

/// Derive the Prepared/Committed baseline that must follow the active marker
/// for every typed tail item installed by a full history replacement.
pub fn tail_journal_repairs_for_replacement(
    operation_id: &str,
    messages: &[ConversationItem],
) -> io::Result<Vec<(ConversationAppendPrepared, ConversationAppendCommitted)>> {
    persisted_entries_for_replacement(operation_id, messages).map(|entries| {
        entries
            .into_iter()
            .filter_map(|entry| {
                let PersistedChatEntry::Tail(tail) = entry else {
                    return None;
                };
                let prepared = ConversationAppendPrepared {
                    operation_id: tail.operation_id,
                    checkpoint_id: tail.checkpoint_id,
                    branch_id: tail.branch_id,
                    sequence: tail.sequence,
                    prompt_index: tail.prompt_index,
                    item: tail.item,
                };
                let committed = ConversationAppendCommitted::from(&prepared);
                Some((prepared, committed))
            })
            .collect()
    })
}

pub fn write_history_durable(path: &Path, entries: &[PersistedChatEntry]) -> io::Result<()> {
    let mut bytes = Vec::new();
    for entry in entries {
        serde_json::to_writer(&mut bytes, entry)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        bytes.push(b'\n');
    }
    write_bytes_durable(path, &bytes)
}

pub fn read_history(path: &Path) -> io::Result<RecoveredHistory> {
    recover_history_entries(read_persisted_entries(path)?)
}

fn read_persisted_entries(path: &Path) -> io::Result<Vec<PersistedChatEntry>> {
    let bytes = std::fs::read(path)?;
    let final_part = bytes.split(|byte| *byte == b'\n').count().saturating_sub(1);
    let has_terminated_tail = bytes.last().is_none_or(|byte| *byte == b'\n');
    let mut entries = Vec::new();
    for (index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        match serde_json::from_slice::<PersistedChatEntry>(line) {
            Ok(entry) => entries.push(entry),
            Err(error) if index == final_part && !has_terminated_tail => {
                // A process kill/ENOSPC may leave only the final append torn.
                // Prepared without Committed is not authoritative, so ignore
                // this one unterminated fragment; the next append truncates it
                // under the append lock before writing.
                tracing::warn!(
                    path = %path.display(),
                    line = index + 1,
                    %error,
                    "ignoring torn final typed-history record"
                );
            }
            Err(error) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid chat history line {}: {error}", index + 1),
                ));
            }
        }
    }
    Ok(entries)
}

fn lock_history_append(path: &Path) -> io::Result<std::fs::File> {
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path.with_extension("jsonl.lock"))?;
    lock.lock_exclusive()?;
    Ok(lock)
}

/// Repair only an unterminated final append. A complete JSON value merely
/// receives its missing newline; an incomplete fragment is truncated back to
/// the last durable record. Corruption in any terminated line still fails
/// closed in `read_persisted_entries`.
fn heal_torn_history_tail(path: &Path) -> io::Result<()> {
    let bytes = std::fs::read(path)?;
    if bytes.last().is_none_or(|byte| *byte == b'\n') {
        return Ok(());
    }
    let start = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |index| index + 1);
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    if serde_json::from_slice::<PersistedChatEntry>(&bytes[start..]).is_ok() {
        file.seek(io::SeekFrom::End(0))?;
        file.write_all(b"\n")?;
    } else {
        file.set_len(start as u64)?;
    }
    file.flush()?;
    super::sync_file_durable(&file)
}

/// Append one typed tail line and fsync it before returning. Repeating the
/// exact operation is idempotent; conflicts and sequence gaps fail closed.
pub fn append_history_tail_durable(path: &Path, tail: &PersistedTail) -> io::Result<bool> {
    let lock = lock_history_append(path)?;
    let result = (|| {
        let metadata = std::fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "chat history is not a regular file",
            ));
        }
        heal_torn_history_tail(path)?;
        let entries = read_persisted_entries(path)?;
        recover_history_entries(entries.clone())?;
        let Some(PersistedChatEntry::Item(ConversationItem::ResponsesCompactionCheckpoint(
            checkpoint,
        ))) = entries.first()
        else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "typed tail requires an active checkpoint wrapper",
            ));
        };
        if tail.checkpoint_id != checkpoint.checkpoint_id || tail.branch_id != checkpoint.branch_id
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "typed tail checkpoint boundary mismatch",
            ));
        }
        let existing_tail_count = entries.len().saturating_sub(1) as u64;
        if tail.sequence <= existing_tail_count {
            let existing = entries
                .get(tail.sequence as usize)
                .and_then(|entry| match entry {
                    PersistedChatEntry::Tail(existing) => Some(existing),
                    PersistedChatEntry::Item(_) => None,
                });
            return if existing.is_some_and(|existing| {
                existing.operation_id == tail.operation_id
                    && existing.checkpoint_id == tail.checkpoint_id
                    && existing.branch_id == tail.branch_id
                    && existing.sequence == tail.sequence
                    && existing.prompt_index == tail.prompt_index
                    && serde_json::to_value(&existing.item).ok()
                        == serde_json::to_value(&tail.item).ok()
            }) {
                Ok(false)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "typed tail sequence conflicts with authoritative history",
                ))
            };
        }
        if tail.sequence != existing_tail_count.saturating_add(1) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "typed tail sequence is not contiguous",
            ));
        }

        let mut line = serde_json::to_vec(&PersistedChatEntry::Tail(tail.clone()))
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        line.push(b'\n');
        let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
        file.write_all(&line)?;
        file.flush()?;
        super::sync_file_durable(&file)?;
        Ok(true)
    })();
    let _ = fs2::FileExt::unlock(&lock);
    result
}

pub fn recover_history_entries(
    mut entries: Vec<PersistedChatEntry>,
) -> io::Result<RecoveredHistory> {
    let checkpoint = entries.first().and_then(|entry| match entry {
        PersistedChatEntry::Item(ConversationItem::ResponsesCompactionCheckpoint(checkpoint)) => {
            Some(checkpoint.clone())
        }
        _ => None,
    });
    let checkpoint_count = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry,
                PersistedChatEntry::Item(ConversationItem::ResponsesCompactionCheckpoint(_))
            )
        })
        .count();
    if checkpoint_count > 0 && (checkpoint_count != 1 || checkpoint.is_none()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint wrapper must be unique at index 0",
        ));
    }
    let Some(checkpoint) = checkpoint else {
        let mut conversation = Vec::with_capacity(entries.len());
        for entry in entries {
            match entry {
                PersistedChatEntry::Item(item) => conversation.push(item),
                PersistedChatEntry::Tail(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "typed tail has no checkpoint wrapper",
                    ));
                }
            }
        }
        return Ok(RecoveredHistory {
            conversation,
            prepared_repairs: Vec::new(),
            committed_repairs: Vec::new(),
        });
    };

    entries.remove(0);
    let mut conversation = vec![ConversationItem::ResponsesCompactionCheckpoint(
        checkpoint.clone(),
    )];
    let mut prepared_repairs = Vec::new();
    let mut committed_repairs = Vec::new();
    let mut expected_sequence = 1;
    for entry in entries {
        match entry {
            PersistedChatEntry::Tail(tail) => {
                if tail.checkpoint_id != checkpoint.checkpoint_id
                    || tail.branch_id != checkpoint.branch_id
                    || tail.sequence != expected_sequence
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "typed tail boundary mismatch",
                    ));
                }
                expected_sequence += 1;
                let prepared = ConversationAppendPrepared {
                    operation_id: tail.operation_id,
                    checkpoint_id: tail.checkpoint_id,
                    branch_id: tail.branch_id,
                    sequence: tail.sequence,
                    prompt_index: tail.prompt_index,
                    item: tail.item,
                };
                committed_repairs.push(ConversationAppendCommitted::from(&prepared));
                conversation.push(prepared.item.clone());
                prepared_repairs.push(prepared);
            }
            PersistedChatEntry::Item(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "untyped item after checkpoint wrapper",
                ));
            }
        }
    }
    Ok(RecoveredHistory {
        conversation,
        prepared_repairs,
        committed_repairs,
    })
}

pub fn rebuild_updates_only_entries(
    checkpoint: ServerResponsesCheckpoint,
    records: &[TailJournalRecord],
) -> io::Result<Vec<PersistedChatEntry>> {
    let mut prepared = BTreeMap::new();
    let mut committed = BTreeSet::new();
    for record in records {
        match record {
            TailJournalRecord::Prepared(record)
                if record.checkpoint_id == checkpoint.checkpoint_id
                    && record.branch_id == checkpoint.branch_id =>
            {
                let key = (
                    record.operation_id.clone(),
                    record.checkpoint_id.clone(),
                    record.branch_id.clone(),
                    record.sequence,
                    record.prompt_index,
                );
                if prepared.insert(key, record.item.clone()).is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate prepared tail record",
                    ));
                }
            }
            TailJournalRecord::Committed(record)
                if record.checkpoint_id == checkpoint.checkpoint_id
                    && record.branch_id == checkpoint.branch_id =>
            {
                committed.insert((
                    record.operation_id.clone(),
                    record.checkpoint_id.clone(),
                    record.branch_id.clone(),
                    record.sequence,
                    record.prompt_index,
                ));
            }
            _ => {}
        }
    }

    let mut complete_by_sequence = BTreeMap::new();
    for (key, item) in prepared {
        if committed.contains(&key) {
            let tail = PersistedTail {
                operation_id: key.0,
                checkpoint_id: key.1,
                branch_id: key.2,
                sequence: key.3,
                prompt_index: key.4,
                item,
            };
            if complete_by_sequence.insert(tail.sequence, tail).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "duplicate committed tail sequence",
                ));
            }
        }
    }
    let mut entries = vec![PersistedChatEntry::Item(
        ConversationItem::ResponsesCompactionCheckpoint(Box::new(checkpoint)),
    )];
    let mut expected = 1;
    for (sequence, tail) in complete_by_sequence {
        if sequence != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "committed tail sequence gap",
            ));
        }
        entries.push(PersistedChatEntry::Tail(tail));
        expected += 1;
    }
    Ok(entries)
}

pub fn rebuild_updates_only(
    checkpoint: ServerResponsesCheckpoint,
    records: &[TailJournalRecord],
) -> io::Result<Vec<ConversationItem>> {
    recover_history_entries(rebuild_updates_only_entries(checkpoint, records)?)
        .map(|history| history.conversation)
}

/// Build authoritative history for a rewind and rotate the active branch so
/// records from the abandoned future can never replay into it.
pub fn rewind_history_entries(
    checkpoint_file: &CompactionCheckpointFile,
    active_entries: Vec<PersistedChatEntry>,
    target_prompt_index: usize,
    new_branch_id: &str,
) -> io::Result<Vec<PersistedChatEntry>> {
    if new_branch_id.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "rewind branch id is empty",
        ));
    }
    let active = recover_history_entries(active_entries.clone())?;
    let Some(ConversationItem::ResponsesCompactionCheckpoint(active_wrapper)) =
        active.conversation.first()
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rewind has no active checkpoint wrapper",
        ));
    };
    if active_wrapper.checkpoint_id != checkpoint_file.checkpoint_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rewind checkpoint id mismatch",
        ));
    }

    if target_prompt_index < checkpoint_file.prompt_index_at_compaction {
        let mut portable = checkpoint_file.portable_history.clone();
        let keep = xai_grok_sampling_types::conversation_truncate_for_prompt(
            &portable,
            target_prompt_index,
        );
        portable.truncate(keep);
        return Ok(portable.into_iter().map(PersistedChatEntry::Item).collect());
    }

    let mut wrapper = active_wrapper.clone();
    wrapper.branch_id = new_branch_id.to_string();
    let mut result = vec![PersistedChatEntry::Item(
        ConversationItem::ResponsesCompactionCheckpoint(wrapper),
    )];
    for entry in active_entries.into_iter().skip(1) {
        let PersistedChatEntry::Tail(mut tail) = entry else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "plain item after active checkpoint",
            ));
        };
        if tail.prompt_index > target_prompt_index {
            break;
        }
        let sequence = result.len() as u64;
        tail.operation_id = format!("rewind-{new_branch_id}-{sequence}");
        tail.branch_id = new_branch_id.to_string();
        tail.sequence = sequence;
        result.push(PersistedChatEntry::Tail(tail));
    }
    recover_history_entries(result.clone())?;
    Ok(result)
}

const RESPONSES_SEGMENT_MARKER_PREFIX: &str = "responses-compaction-checkpoint:";

pub fn stage_compaction_segment_durable(
    session_dir: &Path,
    staging: &ResponsesCompactionSegmentStaging,
) -> io::Result<()> {
    validate_segment_staging(staging)?;
    let path = segment_staging_path(session_dir, &staging.checkpoint_id)?;
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "staging path has no parent"))?;
    std::fs::create_dir_all(parent)?;
    reject_symlink_components(session_dir, parent)?;
    let bytes = serde_json::to_vec(staging)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if let Ok(existing) = std::fs::read(&path) {
        return if existing == bytes {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "conflicting Responses segment staging payload",
            ))
        };
    }
    write_bytes_durable(&path, &bytes)
}

/// Read staging by its strong binding keys. Missing staging is not an error.
pub fn read_segment_staging(
    session_dir: &Path,
    checkpoint_id: &str,
    operation_id: &str,
    wrapper_digest: &str,
) -> io::Result<Option<ResponsesCompactionSegmentStaging>> {
    let path = segment_staging_path(session_dir, checkpoint_id)?;
    match std::fs::symlink_metadata(&path) {
        Ok(_) => {
            let staging = read_segment_staging_payload(&path, checkpoint_id)?;
            validate_staging_binding(&staging, operation_id, wrapper_digest)?;
            Ok(Some(staging))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Read staging for a live wrapper. Branch-only rotation is accepted when the
/// staging digest reconciles against the stored branch.
pub fn read_segment_staging_for_wrapper(
    session_dir: &Path,
    wrapper: &ServerResponsesCheckpoint,
) -> io::Result<Option<ResponsesCompactionSegmentStaging>> {
    let path = segment_staging_path(session_dir, &wrapper.checkpoint_id)?;
    match std::fs::symlink_metadata(&path) {
        Ok(_) => {
            let staging = read_segment_staging_payload(&path, &wrapper.checkpoint_id)?;
            if staging.operation_id != wrapper.operation_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "segment staging operation mismatch",
                ));
            }
            let digest_matches = staging.wrapper_digest == wrapper.wrapper_digest();
            let rotation_ok = !digest_matches
                && staging.branch_id != wrapper.branch_id
                && wrapper.wrapper_digest_for_branch(&staging.branch_id) == staging.wrapper_digest;
            if !digest_matches && !rotation_ok {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "segment staging wrapper digest mismatch",
                ));
            }
            Ok(Some(staging))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn validate_segment_staging(staging: &ResponsesCompactionSegmentStaging) -> io::Result<()> {
    if staging.kind != CompactionCheckpointKind::ResponsesServer
        || staging.items.is_empty()
        || staging.operation_id.is_empty()
        || staging.branch_id.is_empty()
        || staging.wrapper_digest.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Responses segment staging payload",
        ));
    }
    validate_checkpoint_component(&staging.checkpoint_id)?;
    if xai_chat_state::CompactionDetail::parse(&staging.detail).is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Responses segment staging detail",
        ));
    }
    Ok(())
}

fn read_segment_staging_payload(
    path: &Path,
    expected_checkpoint_id: &str,
) -> io::Result<ResponsesCompactionSegmentStaging> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment staging payload is not a regular file",
        ));
    }
    let staging: ResponsesCompactionSegmentStaging = serde_json::from_slice(&std::fs::read(path)?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    validate_segment_staging(&staging)?;
    if staging.checkpoint_id != expected_checkpoint_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment staging checkpoint mismatch",
        ));
    }
    Ok(staging)
}

fn validate_staging_binding(
    staging: &ResponsesCompactionSegmentStaging,
    operation_id: &str,
    wrapper_digest: &str,
) -> io::Result<()> {
    if staging.operation_id != operation_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment staging operation mismatch",
        ));
    }
    if staging.wrapper_digest != wrapper_digest {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment staging wrapper digest mismatch",
        ));
    }
    Ok(())
}

/// Publish committed staging idempotently after validating its operation and
/// wrapper digest.
pub fn publish_staged_compaction_segment_durable(
    session_dir: &Path,
    checkpoint_id: &str,
    operation_id: &str,
    wrapper_digest: &str,
) -> io::Result<PublishedCompactionSegment> {
    validate_checkpoint_component(checkpoint_id)?;
    let compaction_dir = session_dir.join(xai_chat_state::compaction_transcript::COMPACTION_DIR);
    std::fs::create_dir_all(&compaction_dir)?;
    reject_symlink_components(session_dir, &compaction_dir)?;
    let staging_path = segment_staging_path(session_dir, checkpoint_id)?;
    let existing = find_published_segment(&compaction_dir, checkpoint_id)?;
    if let Some((index, markdown)) = existing {
        if staging_path.exists() {
            let staging = read_segment_staging_payload(&staging_path, checkpoint_id)?;
            validate_staging_binding(&staging, operation_id, wrapper_digest)?;
            ensure_segment_index(
                &compaction_dir,
                index,
                &markdown,
                &staging.summary,
                staging.items.len(),
            )?;
            remove_segment_staging(&staging_path)?;
        }
        return Ok(PublishedCompactionSegment {
            index,
            newly_published: false,
        });
    }

    let staging = read_segment_staging_payload(&staging_path, checkpoint_id)?;
    validate_staging_binding(&staging, operation_id, wrapper_digest)?;
    let detail = xai_chat_state::CompactionDetail::parse(&staging.detail).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "invalid staged segment detail")
    })?;
    let index = next_compaction_segment_index(&compaction_dir)?;
    let rendered = xai_chat_state::compaction_transcript::render_segment_md(
        &staging.items,
        &staging.summary,
        index,
        detail,
        &staging.timestamp,
    );
    let markdown = format!("<!-- {RESPONSES_SEGMENT_MARKER_PREFIX}{checkpoint_id} -->\n{rendered}");
    let segment_path = compaction_dir.join(
        xai_chat_state::compaction_transcript::segment_filename(index),
    );
    write_bytes_durable(&segment_path, markdown.as_bytes())?;
    ensure_segment_index(
        &compaction_dir,
        index,
        &markdown,
        &staging.summary,
        staging.items.len(),
    )?;
    remove_segment_staging(&staging_path)?;
    Ok(PublishedCompactionSegment {
        index,
        newly_published: true,
    })
}

fn validate_checkpoint_component(checkpoint_id: &str) -> io::Result<()> {
    if checkpoint_id.is_empty()
        || !checkpoint_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unsafe checkpoint id",
        ));
    }
    Ok(())
}

fn segment_staging_path(session_dir: &Path, checkpoint_id: &str) -> io::Result<PathBuf> {
    validate_checkpoint_component(checkpoint_id)?;
    Ok(session_dir
        .join(xai_chat_state::compaction_transcript::COMPACTION_DIR)
        .join("staging")
        .join(format!("{checkpoint_id}.json")))
}

fn find_published_segment(
    compaction_dir: &Path,
    checkpoint_id: &str,
) -> io::Result<Option<(u64, String)>> {
    let marker = format!("<!-- {RESPONSES_SEGMENT_MARKER_PREFIX}{checkpoint_id} -->");
    let mut found = None;
    for entry in std::fs::read_dir(compaction_dir)? {
        let entry = entry?;
        let Some(index) = entry
            .file_name()
            .to_str()
            .and_then(xai_chat_state::compaction_transcript::parse_segment_index)
        else {
            continue;
        };
        let metadata = entry.file_type()?;
        if metadata.is_symlink() || !metadata.is_file() {
            continue;
        }
        let markdown = std::fs::read_to_string(entry.path())?;
        if markdown.lines().next() == Some(marker.as_str()) {
            if found.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "checkpoint was published as multiple compaction segments",
                ));
            }
            found = Some((index, markdown));
        }
    }
    Ok(found)
}

fn next_compaction_segment_index(compaction_dir: &Path) -> io::Result<u64> {
    let mut next = 0;
    for entry in std::fs::read_dir(compaction_dir)? {
        let entry = entry?;
        if let Some(index) = entry
            .file_name()
            .to_str()
            .and_then(xai_chat_state::compaction_transcript::parse_segment_index)
        {
            next = next.max(index.saturating_add(1));
        }
    }
    Ok(next)
}

fn ensure_segment_index(
    compaction_dir: &Path,
    index: u64,
    markdown: &str,
    summary: &str,
    items_len: usize,
) -> io::Result<()> {
    let index_path = compaction_dir.join(xai_chat_state::compaction_transcript::INDEX_FILE);
    if !index_path.exists() {
        write_bytes_durable(
            &index_path,
            xai_chat_state::compaction_transcript::INDEX_HEADER.as_bytes(),
        )?;
    }
    let metadata = std::fs::symlink_metadata(&index_path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compaction segment index is not a regular file",
        ));
    }
    let keywords = xai_chat_state::compaction_transcript::extract_keywords(summary);
    let row = xai_chat_state::compaction_transcript::render_index_row(
        index,
        items_len,
        markdown.len(),
        &keywords,
    );
    let existing = std::fs::read_to_string(&index_path)?;
    if existing.lines().any(|line| line == row.trim_end()) {
        return Ok(());
    }
    let mut file = std::fs::OpenOptions::new().append(true).open(&index_path)?;
    file.write_all(row.as_bytes())?;
    file.flush()?;
    super::sync_file_durable(&file)
}

fn remove_segment_staging(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => sync_parent_directory(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn validate_relative_checkpoint_path(relative_path: &str) -> io::Result<()> {
    let path = Path::new(relative_path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || path
            .components()
            .next()
            .and_then(|component| match component {
                Component::Normal(value) => value.to_str(),
                _ => None,
            })
            != Some("compaction_checkpoints")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unsafe checkpoint path",
        ));
    }
    Ok(())
}

fn safe_join_for_write(session_dir: &Path, relative_path: &str) -> io::Result<PathBuf> {
    validate_relative_checkpoint_path(relative_path)?;
    let path = session_dir.join(relative_path);
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "checkpoint has no parent"))?;
    std::fs::create_dir_all(parent)?;
    reject_symlink_components(session_dir, parent)?;
    if let Ok(metadata) = std::fs::symlink_metadata(&path)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint target is not a regular file",
        ));
    }
    Ok(path)
}

fn safe_join_for_read(session_dir: &Path, relative_path: &str) -> io::Result<PathBuf> {
    validate_relative_checkpoint_path(relative_path)?;
    let path = session_dir.join(relative_path);
    reject_symlink_components(session_dir, &path)?;
    let metadata = std::fs::symlink_metadata(&path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint is not a regular file",
        ));
    }
    Ok(path)
}

fn reject_symlink_components(base: &Path, path: &Path) -> io::Result<()> {
    let relative = path
        .strip_prefix(base)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path escaped session"))?;
    let mut current = base.to_path_buf();
    for component in relative.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "checkpoint path contains symlink",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn write_bytes_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    std::fs::create_dir_all(parent)?;
    let mut temp_name = path.as_os_str().to_owned();
    temp_name.push(format!(".{}.tmp", uuid::Uuid::now_v7()));
    let temp = PathBuf::from(temp_name);
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.flush()?;
        super::sync_file_durable(&file)?;
        drop(file);
        std::fs::rename(&temp, path)?;
        sync_parent_directory(path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    std::fs::File::open(parent)?.sync_all()
}

#[cfg(windows)]
fn sync_parent_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn sync_parent_directory(_path: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "directory sync is unsupported",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_grok_sampling_types::{
        CheckpointIdentity, CheckpointReplayMaterial, ResponsesCompactionMode, TokenSeedSource,
        TrustedPromptEnvelope,
    };

    const RELATIVE_PATH: &str = "compaction_checkpoints/checkpoint-current.json";

    fn portable_fixture() -> Vec<ConversationItem> {
        vec![
            ConversationItem::base_instructions("base prompt"),
            ConversationItem::user("first"),
            ConversationItem::assistant("answer"),
        ]
    }

    fn identity_fixture(prior: Option<&str>) -> CheckpointIdentity {
        CheckpointIdentity {
            provider_id: "xai".into(),
            api: "responses".into(),
            endpoint_fingerprint: "endpoint".into(),
            model: "grok-test".into(),
            auth_principal_fingerprint: "principal".into(),
            contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
            prompt_envelope_fingerprint: "envelope-fp".into(),
            base_instructions_sha256: "base-hash".into(),
            prior_checkpoint_id: prior.map(str::to_owned),
            cache_route_fingerprint: None,
        }
    }

    fn envelope_fixture() -> TrustedPromptEnvelope {
        TrustedPromptEnvelope {
            base_instructions_sha256: "base-hash".into(),
            memory_revision: Some(3),
            envelope_fingerprint: "envelope-fp".into(),
            wire_prompt_sha256: "wire-hash".into(),
        }
    }

    fn wrapper_fixture(
        portable: &[ConversationItem],
        prior: Option<&str>,
    ) -> ServerResponsesCheckpoint {
        let digest = portable_history_digest(portable).unwrap();
        ServerResponsesCheckpoint {
            checkpoint_id: "checkpoint-current".into(),
            operation_id: "operation-current".into(),
            prompt_index: 5,
            created_at: chrono::Utc::now(),
            auto_continue: false,
            mode: ResponsesCompactionMode {
                name: "default".into(),
                detail: None,
            },
            branch_id: "branch-1".into(),
            identity: identity_fixture(prior),
            output: vec![serde_json::json!({
                "type": "compaction",
                "encrypted_content": "opaque"
            })],
            portable_history_path: RELATIVE_PATH.into(),
            portable_history_sha256: digest,
            portable_history_bytes: 128,
            checkpoint_token_seed: 42,
            token_seed_source: TokenSeedSource::UsageOutputTokens,
            server_output_item_count: 1,
            prior_checkpoint_id: prior.map(str::to_owned),
            memory_revision: Some(3),
        }
    }

    fn sidecar_fixture(
        portable: Vec<ConversationItem>,
        prior: Option<&str>,
    ) -> (CompactionCheckpointFile, ServerResponsesCheckpoint) {
        let wrapper = wrapper_fixture(&portable, prior);
        let material =
            CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
        let sidecar = CompactionCheckpointFile::new(
            wrapper,
            material,
            portable,
            Some("original user".into()),
            vec!["rel/readme.md".into()],
        )
        .unwrap();
        let authoritative_wrapper = sidecar.wrapper.clone();
        (sidecar, authoritative_wrapper)
    }

    fn write_fixture_sidecar(
        session_dir: &Path,
    ) -> (CompactionCheckpointFile, ServerResponsesCheckpoint) {
        let (sidecar, wrapper) = sidecar_fixture(portable_fixture(), None);
        write_checkpoint_durable(session_dir, RELATIVE_PATH, &sidecar).unwrap();
        (sidecar, wrapper)
    }

    fn rewrite_sidecar_on_disk(session_dir: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
        let path = session_dir.join(RELATIVE_PATH);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        mutate(&mut value);
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    }

    #[test]
    fn current_checkpoint_round_trip_and_rotation_binding() {
        let dir = tempfile::tempdir().unwrap();
        let (sidecar, wrapper) = write_fixture_sidecar(dir.path());
        assert_eq!(sidecar.kind, CompactionCheckpointKind::ResponsesServer);
        let serialized = serde_json::to_value(&sidecar).unwrap();
        assert!(serialized.get("schema_version").is_none());
        assert!(serialized["wrapper"].get("schema_version").is_none());

        let read = read_checkpoint(
            dir.path(),
            RELATIVE_PATH,
            &wrapper.checkpoint_id,
            5,
            &wrapper.portable_history_sha256,
        )
        .unwrap();
        assert_eq!(read.operation_id, wrapper.operation_id);
        assert_eq!(
            read.replay_material.wrapper_digest(),
            wrapper.wrapper_digest()
        );

        let mut rotated = wrapper.clone();
        rotated.branch_id = "branch-rotated".into();
        assert!(read_checkpoint_for_wrapper(dir.path(), &rotated).is_ok());
        let rotated_marker = marker_for_wrapper(&rotated);
        validate_marker_for_wrapper(&rotated_marker, &sidecar.wrapper).unwrap();

        let mut forged_output = rotated.clone();
        forged_output.output = vec![serde_json::json!({
            "type": "compaction",
            "encrypted_content": "forged"
        })];
        assert!(read_checkpoint_for_wrapper(dir.path(), &forged_output).is_err());
        let mut forged_seed = rotated.clone();
        forged_seed.checkpoint_token_seed += 1;
        assert!(read_checkpoint_for_wrapper(dir.path(), &forged_seed).is_err());
        rotated.operation_id = "operation-forged".into();
        assert!(read_checkpoint_for_wrapper(dir.path(), &rotated).is_err());
    }

    #[test]
    fn current_checkpoint_rejects_tampered_data_and_chain() {
        let dir = tempfile::tempdir().unwrap();
        let (_, wrapper) = write_fixture_sidecar(dir.path());
        rewrite_sidecar_on_disk(dir.path(), |value| {
            value["portable_history"] = serde_json::json!([
                {"type": "user", "content": [{"type": "text", "text": "tampered"}]}
            ]);
        });
        assert!(
            read_checkpoint(
                dir.path(),
                RELATIVE_PATH,
                &wrapper.checkpoint_id,
                5,
                &wrapper.portable_history_sha256,
            )
            .is_err()
        );

        write_fixture_sidecar(dir.path());
        rewrite_sidecar_on_disk(dir.path(), |value| {
            value["replay_material"]["prior_checkpoint_id"] = serde_json::json!("forged");
        });
        assert!(
            read_checkpoint(
                dir.path(),
                RELATIVE_PATH,
                &wrapper.checkpoint_id,
                5,
                &wrapper.portable_history_sha256,
            )
            .is_err()
        );
    }

    #[test]
    fn responses_marker_validates_kind_and_stable_fields() {
        let wrapper = wrapper_fixture(&portable_fixture(), Some("checkpoint-prior"));
        let marker = marker_for_wrapper(&wrapper);
        assert_eq!(marker.kind, CompactionCheckpointKind::ResponsesServer);
        assert!(
            serde_json::to_value(&marker)
                .unwrap()
                .get("schema_version")
                .is_none()
        );
        validate_marker_for_wrapper(&marker, &wrapper).unwrap();

        let mut versioned = serde_json::to_value(&marker).unwrap();
        versioned["schema_version"] = serde_json::json!(3);
        let versioned: crate::extensions::notification::CompactionCheckpointInfo =
            serde_json::from_value(versioned).unwrap();
        assert_eq!(
            versioned.kind,
            crate::extensions::notification::CompactionCheckpointKind::Unknown
        );
        assert!(validate_marker_for_wrapper(&versioned, &wrapper).is_err());

        let mut repaired = marker.clone();
        repaired.branch_id = Some("branch-rotated".into());
        repaired.wrapper_digest = Some(wrapper.wrapper_digest_for_branch("branch-rotated"));
        repaired.created_at = "2000-01-01T00:00:00Z".into();
        validate_marker_for_wrapper(&repaired, &wrapper).unwrap();

        let mut wrong_kind = marker;
        wrong_kind.kind = CompactionCheckpointKind::Builtin;
        assert!(validate_marker_for_wrapper(&wrong_kind, &wrapper).is_err());
    }

    #[test]
    fn persisted_tail_uses_only_current_serde_tag() {
        let wrapper = wrapper_fixture(&portable_fixture(), None);
        let tail = PersistedTail {
            operation_id: "tail-1".into(),
            checkpoint_id: wrapper.checkpoint_id,
            branch_id: wrapper.branch_id,
            sequence: 1,
            prompt_index: 6,
            item: ConversationItem::user("next"),
        };
        let value = serde_json::to_value(PersistedChatEntry::Tail(tail)).unwrap();
        assert_eq!(value["persisted_entry"], "tail");

        let mut removed_tag = value.clone();
        removed_tag["persisted_entry"] = serde_json::json!("tail_v2");
        assert!(serde_json::from_value::<PersistedChatEntry>(removed_tag).is_err());
    }

    #[test]
    fn replacement_tail_journal_forms_a_complete_rebuild_baseline() {
        let wrapper = wrapper_fixture(&portable_fixture(), None);
        let messages = vec![
            ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper.clone())),
            ConversationItem::system_reminder("mode hint"),
            ConversationItem::user("next"),
        ];
        let repairs = tail_journal_repairs_for_replacement("replace-op", &messages).unwrap();
        assert_eq!(repairs.len(), 2);
        assert_eq!(repairs[0].0.operation_id, "replace-op-1");
        assert_eq!(repairs[1].0.sequence, 2);
        let records = repairs
            .into_iter()
            .flat_map(|(prepared, committed)| {
                [
                    TailJournalRecord::Prepared(prepared),
                    TailJournalRecord::Committed(committed),
                ]
            })
            .collect::<Vec<_>>();
        let rebuilt = rebuild_updates_only(wrapper, &records).unwrap();
        assert_eq!(
            serde_json::to_value(rebuilt).unwrap(),
            serde_json::to_value(messages).unwrap()
        );
    }

    #[test]
    fn history_tail_is_durable_contiguous_and_rebuildable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chat_history.jsonl");
        let wrapper = wrapper_fixture(&portable_fixture(), None);
        write_history_durable(
            &path,
            &[PersistedChatEntry::Item(
                ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper.clone())),
            )],
        )
        .unwrap();
        let tail = PersistedTail {
            operation_id: "tail-1".into(),
            checkpoint_id: wrapper.checkpoint_id.clone(),
            branch_id: wrapper.branch_id.clone(),
            sequence: 1,
            prompt_index: 6,
            item: ConversationItem::user("next"),
        };
        assert!(append_history_tail_durable(&path, &tail).unwrap());
        assert!(!append_history_tail_durable(&path, &tail).unwrap());

        // A complete record whose final newline was torn is retained and
        // terminated before the next append.
        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        std::fs::write(&path, bytes).unwrap();
        let next = PersistedTail {
            operation_id: "tail-2".into(),
            sequence: 2,
            item: ConversationItem::assistant("answer"),
            ..tail.clone()
        };
        assert!(append_history_tail_durable(&path, &next).unwrap());
        let recovered = read_history(&path).unwrap();
        assert_eq!(recovered.conversation.len(), 3);
        assert_eq!(recovered.prepared_repairs.len(), 2);

        let prepared = recovered.prepared_repairs[0].clone();
        let records = vec![
            TailJournalRecord::Prepared(prepared.clone()),
            TailJournalRecord::Committed(ConversationAppendCommitted::from(&prepared)),
        ];
        let rebuilt = rebuild_updates_only(wrapper, &records).unwrap();
        assert_eq!(rebuilt.len(), 2);
    }

    #[test]
    fn invalid_torn_tail_is_dropped_before_retry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chat_history.jsonl");
        let wrapper = wrapper_fixture(&portable_fixture(), None);
        write_history_durable(
            &path,
            &[PersistedChatEntry::Item(
                ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper.clone())),
            )],
        )
        .unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"{\"persisted_entry\":\"tail\",\"operation_id\":")
            .unwrap();
        drop(file);

        // Resume can ignore only this final unterminated fragment.
        assert_eq!(read_history(&path).unwrap().conversation.len(), 1);
        let tail = PersistedTail {
            operation_id: "tail-1".into(),
            checkpoint_id: wrapper.checkpoint_id,
            branch_id: wrapper.branch_id,
            sequence: 1,
            prompt_index: 6,
            item: ConversationItem::user("retry"),
        };
        assert!(append_history_tail_durable(&path, &tail).unwrap());
        assert_eq!(read_history(&path).unwrap().conversation.len(), 2);
    }

    fn staging_fixture(wrapper: &ServerResponsesCheckpoint) -> ResponsesCompactionSegmentStaging {
        ResponsesCompactionSegmentStaging::new(
            wrapper.checkpoint_id.clone(),
            wrapper.operation_id.clone(),
            wrapper.branch_id.clone(),
            wrapper.wrapper_digest(),
            vec![ConversationItem::user("segmented turn")],
            "summary",
            xai_chat_state::CompactionDetail::Balanced,
            "2026-01-01T00:00:00Z",
        )
        .unwrap()
    }

    #[test]
    fn staging_strong_binding_and_publish() {
        let dir = tempfile::tempdir().unwrap();
        let wrapper = wrapper_fixture(&portable_fixture(), None);
        let staging = staging_fixture(&wrapper);
        stage_compaction_segment_durable(dir.path(), &staging).unwrap();
        stage_compaction_segment_durable(dir.path(), &staging).unwrap();

        let read = read_segment_staging_for_wrapper(dir.path(), &wrapper)
            .unwrap()
            .unwrap();
        assert_eq!(read.kind, CompactionCheckpointKind::ResponsesServer);
        assert!(
            serde_json::to_value(&read)
                .unwrap()
                .get("schema_version")
                .is_none()
        );
        assert!(
            read_segment_staging(
                dir.path(),
                &wrapper.checkpoint_id,
                &wrapper.operation_id,
                &wrapper.wrapper_digest(),
            )
            .unwrap()
            .is_some()
        );

        let published = publish_staged_compaction_segment_durable(
            dir.path(),
            &wrapper.checkpoint_id,
            &wrapper.operation_id,
            &wrapper.wrapper_digest(),
        )
        .unwrap();
        assert!(published.newly_published);
        let again = publish_staged_compaction_segment_durable(
            dir.path(),
            &wrapper.checkpoint_id,
            &wrapper.operation_id,
            &wrapper.wrapper_digest(),
        )
        .unwrap();
        assert!(!again.newly_published);
        assert_eq!(again.index, published.index);
    }

    #[test]
    fn staging_rotation_is_tolerated_but_forgery_fails() {
        let dir = tempfile::tempdir().unwrap();
        let wrapper = wrapper_fixture(&portable_fixture(), None);
        stage_compaction_segment_durable(dir.path(), &staging_fixture(&wrapper)).unwrap();

        let mut rotated = wrapper.clone();
        rotated.branch_id = "branch-rotated".into();
        let staging = read_segment_staging_for_wrapper(dir.path(), &rotated)
            .unwrap()
            .expect("rotated staging");
        let mut forged = rotated.clone();
        forged.operation_id = "operation-forged".into();
        assert!(read_segment_staging_for_wrapper(dir.path(), &forged).is_err());

        publish_staged_compaction_segment_durable(
            dir.path(),
            &rotated.checkpoint_id,
            &staging.operation_id,
            &staging.wrapper_digest,
        )
        .unwrap();
    }
}

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use xai_grok_sampling_types::{ConversationItem, ServerResponsesCheckpointV1};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionCheckpointFileV2 {
    pub schema_version: u32,
    pub kind: String,
    pub checkpoint_id: String,
    pub prompt_index_at_compaction: usize,
    pub wrapper: ServerResponsesCheckpointV1,
    pub portable_history: Vec<ConversationItem>,
    pub portable_history_sha256: String,
    pub mode_repair: xai_grok_sampling_types::ResponsesCompactionModeV1,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_user_info: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reread_file_paths: Vec<String>,
}

impl CompactionCheckpointFileV2 {
    pub fn new(
        mut wrapper: ServerResponsesCheckpointV1,
        portable_history: Vec<ConversationItem>,
        original_user_info: Option<String>,
        reread_file_paths: Vec<String>,
    ) -> io::Result<Self> {
        let digest = portable_history_digest(&portable_history)?;
        let portable_bytes = portable_history_bytes(&portable_history)?;
        wrapper.portable_history_sha256 = digest.clone();
        wrapper.portable_history_bytes = portable_bytes.len() as u64;
        Ok(Self {
            schema_version: 2,
            kind: "responses_server".into(),
            checkpoint_id: wrapper.checkpoint_id.clone(),
            prompt_index_at_compaction: wrapper.prompt_index,
            mode_repair: wrapper.mode.clone(),
            created_at: wrapper.created_at.to_rfc3339(),
            wrapper,
            portable_history,
            portable_history_sha256: digest,
            original_user_info,
            reread_file_paths,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesCompactionSegmentStagingV1 {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub items: Vec<ConversationItem>,
    pub summary: String,
    pub detail: String,
    pub timestamp: String,
}

impl ResponsesCompactionSegmentStagingV1 {
    pub fn new(
        checkpoint_id: impl Into<String>,
        items: Vec<ConversationItem>,
        summary: impl Into<String>,
        detail: xai_chat_state::CompactionDetail,
        timestamp: impl Into<String>,
    ) -> io::Result<Self> {
        let staging = Self {
            schema_version: 1,
            checkpoint_id: checkpoint_id.into(),
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
pub struct TailV2 {
    pub operation_id: String,
    pub checkpoint_id: String,
    pub branch_id: String,
    pub sequence: u64,
    pub prompt_index: usize,
    pub item: ConversationItem,
}

#[derive(Debug, Clone)]
pub enum PersistedChatEntry {
    Legacy(ConversationItem),
    TailV2(TailV2),
}

impl Serialize for PersistedChatEntry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Legacy(item) => item.serialize(serializer),
            Self::TailV2(tail) => {
                let mut value = serde_json::to_value(tail).map_err(serde::ser::Error::custom)?;
                value
                    .as_object_mut()
                    .expect("TailV2 serializes as an object")
                    .insert(
                        "persisted_entry".into(),
                        serde_json::Value::String("tail_v2".into()),
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
            == Some("tail_v2")
        {
            value
                .as_object_mut()
                .expect("tagged tail is an object")
                .remove("persisted_entry");
            return serde_json::from_value(value)
                .map(Self::TailV2)
                .map_err(serde::de::Error::custom);
        }
        serde_json::from_value(value)
            .map(Self::Legacy)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationAppendPreparedV2 {
    pub operation_id: String,
    pub checkpoint_id: String,
    pub branch_id: String,
    pub sequence: u64,
    pub prompt_index: usize,
    pub item: ConversationItem,
}

impl PartialEq for ConversationAppendPreparedV2 {
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
pub struct ConversationAppendCommittedV2 {
    pub operation_id: String,
    pub checkpoint_id: String,
    pub branch_id: String,
    pub sequence: u64,
    pub prompt_index: usize,
}

impl From<&ConversationAppendPreparedV2> for ConversationAppendCommittedV2 {
    fn from(prepared: &ConversationAppendPreparedV2) -> Self {
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
    Prepared(ConversationAppendPreparedV2),
    Committed(ConversationAppendCommittedV2),
}

#[derive(Debug, Clone)]
pub struct RecoveredHistoryV2 {
    pub conversation: Vec<ConversationItem>,
    pub prepared_repairs: Vec<ConversationAppendPreparedV2>,
    pub committed_repairs: Vec<ConversationAppendCommittedV2>,
}

pub fn portable_history_bytes(history: &[ConversationItem]) -> io::Result<Vec<u8>> {
    let value = serde_json::to_value(history)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    xai_grok_sampling_types::canonical_json_bytes(&value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn portable_history_digest(history: &[ConversationItem]) -> io::Result<String> {
    Ok(format!(
        "{:x}",
        sha2::Sha256::digest(portable_history_bytes(history)?)
    ))
}

pub fn write_checkpoint_v2_durable(
    session_dir: &Path,
    relative_path: &str,
    checkpoint: &CompactionCheckpointFileV2,
) -> io::Result<()> {
    validate_relative_checkpoint_path(relative_path)?;
    if checkpoint.schema_version != 2 || checkpoint.kind != "responses_server" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid responses checkpoint schema",
        ));
    }
    validate_checkpoint(checkpoint, None, None, None)?;
    let path = safe_join_for_write(session_dir, relative_path)?;
    let bytes = serde_json::to_vec(checkpoint)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_bytes_durable(&path, &bytes)
}

pub fn read_checkpoint_v2(
    session_dir: &Path,
    relative_path: &str,
    expected_checkpoint_id: &str,
    expected_prompt_index: usize,
    expected_digest: &str,
) -> io::Result<CompactionCheckpointFileV2> {
    let path = safe_join_for_read(session_dir, relative_path)?;
    let bytes = std::fs::read(path)?;
    let checkpoint: CompactionCheckpointFileV2 = serde_json::from_slice(&bytes)
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

pub fn read_checkpoint_for_wrapper(
    session_dir: &Path,
    wrapper: &ServerResponsesCheckpointV1,
) -> io::Result<CompactionCheckpointFileV2> {
    if wrapper.schema_version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported responses checkpoint wrapper schema",
        ));
    }
    let checkpoint = read_checkpoint_v2(
        session_dir,
        &wrapper.portable_history_path,
        &wrapper.checkpoint_id,
        wrapper.prompt_index,
        &wrapper.portable_history_sha256,
    )?;
    // Rewind/fork may rotate only the active branch. Every immutable replay
    // field must still match the sidecar's wrapper copy.
    let mut stored = checkpoint.wrapper.clone();
    stored.branch_id = wrapper.branch_id.clone();
    let expected = serde_json::to_value(wrapper)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let actual = serde_json::to_value(stored)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if actual != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "active wrapper does not match checkpoint sidecar",
        ));
    }
    Ok(checkpoint)
}

pub fn marker_for_wrapper(
    wrapper: &ServerResponsesCheckpointV1,
) -> crate::extensions::notification::CompactionCheckpointInfo {
    crate::extensions::notification::CompactionCheckpointInfo {
        checkpoint_id: wrapper.checkpoint_id.clone(),
        prompt_index_at_compaction: wrapper.prompt_index,
        checkpoint_file: wrapper.portable_history_path.clone(),
        auto_continue: None,
        schema_version: 2,
        operation_id: Some(wrapper.operation_id.clone()),
        branch_id: Some(wrapper.branch_id.clone()),
        portable_history_sha256: Some(wrapper.portable_history_sha256.clone()),
        responses_mode: Some(wrapper.mode.clone()),
        responses_auto_continue: Some(wrapper.auto_continue),
        created_at: wrapper.created_at.to_rfc3339(),
    }
}

pub fn validate_marker_for_wrapper(
    marker: &crate::extensions::notification::CompactionCheckpointInfo,
    wrapper: &ServerResponsesCheckpointV1,
) -> io::Result<()> {
    let expected = marker_for_wrapper(wrapper);
    if marker != &expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses checkpoint marker does not match active wrapper",
        ));
    }
    Ok(())
}

fn validate_checkpoint(
    checkpoint: &CompactionCheckpointFileV2,
    expected_checkpoint_id: Option<&str>,
    expected_prompt_index: Option<usize>,
    expected_digest: Option<&str>,
) -> io::Result<()> {
    if checkpoint.schema_version != 2 || checkpoint.kind != "responses_server" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported checkpoint schema",
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
    let digest = portable_history_digest(&checkpoint.portable_history)?;
    if digest != checkpoint.portable_history_sha256
        || digest != checkpoint.wrapper.portable_history_sha256
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
            .map(PersistedChatEntry::Legacy)
            .collect());
    };
    let request = xai_grok_sampling_types::ConversationRequest {
        items: messages.to_vec(),
        ..Default::default()
    };
    request
        .validate_for_backend(&xai_grok_sampling_types::ApiBackend::Responses)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut entries = vec![PersistedChatEntry::Legacy(messages[0].clone())];
    let mut prompt_index = checkpoint.prompt_index;
    for (index, item) in messages.iter().skip(1).cloned().enumerate() {
        if matches!(item, ConversationItem::User(_)) {
            prompt_index = prompt_index.saturating_add(1);
        }
        let sequence = (index + 1) as u64;
        entries.push(PersistedChatEntry::TailV2(TailV2 {
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

pub fn write_history_v2_durable(path: &Path, entries: &[PersistedChatEntry]) -> io::Result<()> {
    let mut bytes = Vec::new();
    for entry in entries {
        serde_json::to_writer(&mut bytes, entry)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        bytes.push(b'\n');
    }
    write_bytes_durable(path, &bytes)
}

pub fn read_history_v2(path: &Path) -> io::Result<RecoveredHistoryV2> {
    recover_history_entries(read_persisted_entries(path)?)
}

fn read_persisted_entries(path: &Path) -> io::Result<Vec<PersistedChatEntry>> {
    let bytes = std::fs::read(path)?;
    let mut entries = Vec::new();
    for (index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        entries.push(
            serde_json::from_slice::<PersistedChatEntry>(line).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid chat history line {}: {error}", index + 1),
                )
            })?,
        );
    }
    Ok(entries)
}

/// Append one typed tail line and fsync it before returning. Repeating the
/// exact operation is idempotent; conflicting or non-contiguous sequences fail closed.
pub fn append_history_tail_v2_durable(path: &Path, tail: &TailV2) -> io::Result<bool> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "chat history is not a regular file",
        ));
    }
    let entries = read_persisted_entries(path)?;
    recover_history_entries(entries.clone())?;
    let Some(PersistedChatEntry::Legacy(ConversationItem::ResponsesCompactionCheckpoint(
        checkpoint,
    ))) = entries.first()
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "typed tail requires an active checkpoint wrapper",
        ));
    };
    if tail.checkpoint_id != checkpoint.checkpoint_id || tail.branch_id != checkpoint.branch_id {
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
                PersistedChatEntry::TailV2(existing) => Some(existing),
                PersistedChatEntry::Legacy(_) => None,
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

    let mut line = serde_json::to_vec(&PersistedChatEntry::TailV2(tail.clone()))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    line.push(b'\n');
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    file.write_all(&line)?;
    file.flush()?;
    super::sync_file_durable(&file)?;
    Ok(true)
}

pub fn recover_history_entries(entries: Vec<PersistedChatEntry>) -> io::Result<RecoveredHistoryV2> {
    let checkpoint = entries.first().and_then(|entry| match entry {
        PersistedChatEntry::Legacy(ConversationItem::ResponsesCompactionCheckpoint(checkpoint)) => {
            Some(checkpoint.clone())
        }
        _ => None,
    });
    let checkpoint_count = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry,
                PersistedChatEntry::Legacy(ConversationItem::ResponsesCompactionCheckpoint(_))
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
                PersistedChatEntry::Legacy(item) => conversation.push(item),
                PersistedChatEntry::TailV2(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "typed tail has no checkpoint wrapper",
                    ));
                }
            }
        }
        return Ok(RecoveredHistoryV2 {
            conversation,
            prepared_repairs: Vec::new(),
            committed_repairs: Vec::new(),
        });
    };

    let mut conversation = vec![ConversationItem::ResponsesCompactionCheckpoint(
        checkpoint.clone(),
    )];
    let mut prepared_repairs = Vec::new();
    let mut committed_repairs = Vec::new();
    let mut expected_sequence = 1;
    for entry in entries.into_iter().skip(1) {
        let PersistedChatEntry::TailV2(tail) = entry else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "legacy tail after v2 checkpoint",
            ));
        };
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
        let prepared = ConversationAppendPreparedV2 {
            operation_id: tail.operation_id,
            checkpoint_id: tail.checkpoint_id,
            branch_id: tail.branch_id,
            sequence: tail.sequence,
            prompt_index: tail.prompt_index,
            item: tail.item,
        };
        committed_repairs.push(ConversationAppendCommittedV2::from(&prepared));
        conversation.push(prepared.item.clone());
        prepared_repairs.push(prepared);
    }
    Ok(RecoveredHistoryV2 {
        conversation,
        prepared_repairs,
        committed_repairs,
    })
}

pub fn rebuild_updates_only_entries_v2(
    checkpoint: ServerResponsesCheckpointV1,
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
            let tail = TailV2 {
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
    let mut entries = vec![PersistedChatEntry::Legacy(
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
        entries.push(PersistedChatEntry::TailV2(tail));
        expected += 1;
    }
    Ok(entries)
}

pub fn rebuild_updates_only_v2(
    checkpoint: ServerResponsesCheckpointV1,
    records: &[TailJournalRecord],
) -> io::Result<Vec<ConversationItem>> {
    recover_history_entries(rebuild_updates_only_entries_v2(checkpoint, records)?)
        .map(|history| history.conversation)
}

/// Build the authoritative history for a rewind and rotate the active branch
/// so records from the abandoned future can never replay into it.
pub fn rewind_history_entries_v2(
    checkpoint_file: &CompactionCheckpointFileV2,
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
        return Ok(portable
            .into_iter()
            .map(PersistedChatEntry::Legacy)
            .collect());
    }

    let mut wrapper = active_wrapper.clone();
    wrapper.branch_id = new_branch_id.to_string();
    let mut result = vec![PersistedChatEntry::Legacy(
        ConversationItem::ResponsesCompactionCheckpoint(wrapper),
    )];
    for entry in active_entries.into_iter().skip(1) {
        let PersistedChatEntry::TailV2(mut tail) = entry else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "legacy tail after active checkpoint",
            ));
        };
        if tail.prompt_index > target_prompt_index {
            break;
        }
        let sequence = result.len() as u64;
        tail.operation_id = format!("rewind-{new_branch_id}-{sequence}");
        tail.branch_id = new_branch_id.to_string();
        tail.sequence = sequence;
        result.push(PersistedChatEntry::TailV2(tail));
    }
    recover_history_entries(result.clone())?;
    Ok(result)
}

const RESPONSES_SEGMENT_MARKER_PREFIX: &str = "responses-compaction-checkpoint:";

pub fn stage_compaction_segment_durable(
    session_dir: &Path,
    staging: &ResponsesCompactionSegmentStagingV1,
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

pub fn publish_staged_compaction_segment_durable(
    session_dir: &Path,
    checkpoint_id: &str,
) -> io::Result<PublishedCompactionSegment> {
    validate_checkpoint_component(checkpoint_id)?;
    let compaction_dir = session_dir.join(xai_chat_state::compaction_transcript::COMPACTION_DIR);
    std::fs::create_dir_all(&compaction_dir)?;
    reject_symlink_components(session_dir, &compaction_dir)?;
    let staging_path = segment_staging_path(session_dir, checkpoint_id)?;
    let existing = find_published_segment(&compaction_dir, checkpoint_id)?;
    if let Some((index, markdown)) = existing {
        if staging_path.exists() {
            let staging = read_segment_staging(&staging_path, checkpoint_id)?;
            ensure_segment_index(&compaction_dir, index, &markdown, &staging)?;
            remove_segment_staging(&staging_path)?;
        }
        return Ok(PublishedCompactionSegment {
            index,
            newly_published: false,
        });
    }

    let staging = read_segment_staging(&staging_path, checkpoint_id)?;
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
    ensure_segment_index(&compaction_dir, index, &markdown, &staging)?;
    remove_segment_staging(&staging_path)?;
    Ok(PublishedCompactionSegment {
        index,
        newly_published: true,
    })
}

fn validate_segment_staging(staging: &ResponsesCompactionSegmentStagingV1) -> io::Result<()> {
    if staging.schema_version != 1 || staging.items.is_empty() {
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

fn read_segment_staging(
    path: &Path,
    expected_checkpoint_id: &str,
) -> io::Result<ResponsesCompactionSegmentStagingV1> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment staging payload is not a regular file",
        ));
    }
    let staging: ResponsesCompactionSegmentStagingV1 =
        serde_json::from_slice(&std::fs::read(path)?)
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
    staging: &ResponsesCompactionSegmentStagingV1,
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
    let keywords = xai_chat_state::compaction_transcript::extract_keywords(&staging.summary);
    let row = xai_chat_state::compaction_transcript::render_index_row(
        index,
        staging.items.len(),
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

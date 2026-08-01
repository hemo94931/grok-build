use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use xai_grok_sampling_types::{
    CheckpointReplayMaterialV2, ConversationItem, RESPONSES_CHECKPOINT_SCHEMA_V2,
    ServerResponsesCheckpointV1, ServerResponsesCheckpointV2, wrapper_digest_v2,
};

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

/// V3 Responses server-compaction sidecar file (V2 wrapper contract).
///
/// V3 pairs the [`ServerResponsesCheckpointV2`] wrapper with its
/// [`CheckpointReplayMaterialV2`] so replay verification recomputes every
/// digest from actual data instead of trusting the sidecar wholesale. All
/// three views of the portable history (sidecar digest, wrapper digest,
/// replay-material digest) must agree at construction and at every read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionCheckpointFileV3 {
    pub schema_version: u32,
    pub kind: String,
    pub checkpoint_id: String,
    pub operation_id: String,
    pub prompt_index_at_compaction: usize,
    pub wrapper: ServerResponsesCheckpointV2,
    pub replay_material: CheckpointReplayMaterialV2,
    pub portable_history: Vec<ConversationItem>,
    pub portable_history_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_user_info: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reread_file_paths: Vec<String>,
    pub created_at: String,
}

impl CompactionCheckpointFileV3 {
    /// Build a V3 sidecar, computing the portable digest and proving that
    /// the wrapper, the replay material and the portable history all agree.
    pub fn new(
        mut wrapper: ServerResponsesCheckpointV2,
        replay_material: CheckpointReplayMaterialV2,
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
        // Informational mirror of the V2 sidecar: keep the wrapper's byte
        // count truthful for checkpoint-bytes telemetry.
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
        if wrapper.schema_version != RESPONSES_CHECKPOINT_SCHEMA_V2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported responses checkpoint wrapper schema",
            ));
        }
        Ok(Self {
            schema_version: 3,
            kind: "responses_server_v2".into(),
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

/// V2 staged (unpublished) compaction segment for the V2/V3 server
/// contract. Unlike V1 it carries the operation id, branch and wrapper
/// digest so crash recovery can bind the staging to a live V2 wrapper
/// strongly instead of by checkpoint id and portable digest alone.
///
/// The file lives at the same `compaction/staging/{checkpoint_id}.json`
/// path as V1 staging so crash recovery discovers it regardless of which
/// schema version wrote it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesCompactionSegmentStagingV2 {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub operation_id: String,
    pub branch_id: String,
    pub wrapper_digest: String,
    pub items: Vec<ConversationItem>,
    pub summary: String,
    pub detail: String,
    pub timestamp: String,
}

impl ResponsesCompactionSegmentStagingV2 {
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
            schema_version: 2,
            checkpoint_id: checkpoint_id.into(),
            operation_id: operation_id.into(),
            branch_id: branch_id.into(),
            wrapper_digest: wrapper_digest.into(),
            items,
            summary: summary.into(),
            detail: detail.to_string(),
            timestamp: timestamp.into(),
        };
        validate_segment_staging_v2(&staging)?;
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
    xai_grok_sampling_types::portable_history_bytes(history)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn portable_history_digest(history: &[ConversationItem]) -> io::Result<String> {
    xai_grok_sampling_types::portable_history_digest(history)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
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

pub fn write_checkpoint_v3_durable(
    session_dir: &Path,
    relative_path: &str,
    checkpoint: &CompactionCheckpointFileV3,
) -> io::Result<()> {
    validate_relative_checkpoint_path(relative_path)?;
    if checkpoint.schema_version != 3 || checkpoint.kind != "responses_server_v2" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid responses checkpoint schema",
        ));
    }
    validate_checkpoint_v3(checkpoint, None, None, None)?;
    let path = safe_join_for_write(session_dir, relative_path)?;
    let bytes = serde_json::to_vec(checkpoint)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_bytes_durable(&path, &bytes)
}

pub fn read_checkpoint_v3(
    session_dir: &Path,
    relative_path: &str,
    expected_checkpoint_id: &str,
    expected_prompt_index: usize,
    expected_digest: &str,
) -> io::Result<CompactionCheckpointFileV3> {
    let path = safe_join_for_read(session_dir, relative_path)?;
    let bytes = std::fs::read(path)?;
    let checkpoint: CompactionCheckpointFileV3 = serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    validate_checkpoint_v3(
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

/// Read and bind the V3 sidecar for a live V2 wrapper.
///
/// Security rationale: the sidecar's wrapper copy is not trusted wholesale.
/// Every immutable binding field (checkpoint/operation id, prompt index,
/// portable digest, prior checkpoint, identity) must equal the live wrapper,
/// and the wrapper digest must hold. Rewind/fork may rotate only the active
/// branch, so a branch-only difference is accepted when replacing the stored
/// branch with the live one makes the digests agree.
pub fn read_checkpoint_for_wrapper_v2(
    session_dir: &Path,
    wrapper: &ServerResponsesCheckpointV2,
) -> io::Result<CompactionCheckpointFileV3> {
    if wrapper.schema_version != RESPONSES_CHECKPOINT_SCHEMA_V2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported responses checkpoint wrapper schema",
        ));
    }
    let checkpoint = read_checkpoint_v3(
        session_dir,
        &wrapper.portable_history_path,
        &wrapper.checkpoint_id,
        wrapper.prompt_index,
        &wrapper.portable_history_sha256,
    )?;
    // Rotation-aware wrapper binding: only the branch may differ, and only
    // when substituting the live branch reconciles the wrapper digest.
    let stored = &checkpoint.wrapper;
    let rotated = stored.branch_id != wrapper.branch_id;
    let mut rebound = stored.clone();
    rebound.branch_id = wrapper.branch_id.clone();
    let digest_ok = rebound.wrapper_digest() == wrapper.wrapper_digest();
    if !digest_ok
        || stored.checkpoint_id != wrapper.checkpoint_id
        || stored.operation_id != wrapper.operation_id
        || stored.prompt_index != wrapper.prompt_index
        || stored.portable_history_sha256 != wrapper.portable_history_sha256
        || stored.prior_checkpoint_id != wrapper.prior_checkpoint_id
        || stored.identity != wrapper.identity
    {
        let detail = if rotated {
            "rotated branch does not reconcile the wrapper digest"
        } else {
            "active wrapper does not match checkpoint sidecar"
        };
        return Err(io::Error::new(io::ErrorKind::InvalidData, detail));
    }
    Ok(checkpoint)
}

/// Full validation of a V3 sidecar: schema/kind, internal id consistency,
/// expected id/index/digest, and every digest recomputed from the data.
fn validate_checkpoint_v3(
    checkpoint: &CompactionCheckpointFileV3,
    expected_checkpoint_id: Option<&str>,
    expected_prompt_index: Option<usize>,
    expected_digest: Option<&str>,
) -> io::Result<()> {
    if checkpoint.schema_version != 3 || checkpoint.kind != "responses_server_v2" {
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
    if checkpoint.wrapper.wrapper_digest() != checkpoint.replay_material.wrapper_digest() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "wrapper digest mismatch",
        ));
    }
    Ok(())
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
        wrapper_digest: None,
        prior_checkpoint_id: None,
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

/// Build the schema-3 marker for a live V2 wrapper.
///
/// Unlike schema 2, the marker binds through stable identifiers and the
/// wrapper digest instead of relying on whole-JSON equality: branch rotation
/// and marker repair must not invalidate an otherwise identical checkpoint.
pub fn marker_for_wrapper_v3(
    wrapper: &ServerResponsesCheckpointV2,
) -> crate::extensions::notification::CompactionCheckpointInfo {
    crate::extensions::notification::CompactionCheckpointInfo {
        checkpoint_id: wrapper.checkpoint_id.clone(),
        prompt_index_at_compaction: wrapper.prompt_index,
        checkpoint_file: wrapper.portable_history_path.clone(),
        auto_continue: None,
        schema_version: 3,
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

/// Validate a schema-3 marker against a live V2 wrapper by comparing only
/// stable binding fields: schema, checkpoint id, prompt index, checkpoint
/// file, operation id, portable digest, wrapper digest, prior checkpoint,
/// mode and auto-continue flag.
///
/// Deliberately **not** compared: `branch_id` (rewind/fork may rotate the
/// active branch) and `created_at` (repair tolerance). A mismatch means the
/// marker references a different checkpoint than the live wrapper and must
/// fail closed.
pub fn validate_marker_for_wrapper_v3(
    marker: &crate::extensions::notification::CompactionCheckpointInfo,
    wrapper: &ServerResponsesCheckpointV2,
) -> io::Result<()> {
    if marker.schema_version != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported responses marker schema",
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
    if marker.operation_id != Some(wrapper.operation_id.clone()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker operation id mismatch",
        ));
    }
    if marker.portable_history_sha256 != Some(wrapper.portable_history_sha256.clone()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker portable history digest mismatch",
        ));
    }
    if marker.wrapper_digest != Some(wrapper.wrapper_digest()) {
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
    if marker.responses_mode != Some(wrapper.mode.clone()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "responses marker mode mismatch",
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

/// Read a staged (unpublished) V1 compaction segment for checkpoint recovery.
///
/// Returns `Ok(None)` when no staging payload exists for `checkpoint_id`.
/// Callers must anchor the result on a live wrapper: the V1 staging schema
/// carries no operation ID, so only the checkpoint ID and a recomputed
/// portable digest bind it to a checkpoint. Orphan staging left behind by a
/// cancelled/failed/superseded operation must be ignored by matching the
/// live wrapper's checkpoint ID and portable digest before use.
pub fn read_segment_staging_for_recovery(
    session_dir: &Path,
    checkpoint_id: &str,
) -> io::Result<Option<ResponsesCompactionSegmentStagingV1>> {
    let path = segment_staging_path(session_dir, checkpoint_id)?;
    match std::fs::symlink_metadata(&path) {
        Ok(_) => read_segment_staging(&path, checkpoint_id).map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

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

/// Durable-write a V2 staged segment to the same `compaction/staging/`
/// location V1 uses, so crash recovery sees it regardless of schema.
pub fn stage_compaction_segment_v2_durable(
    session_dir: &Path,
    staging: &ResponsesCompactionSegmentStagingV2,
) -> io::Result<()> {
    validate_segment_staging_v2(staging)?;
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

/// Read the V2 staging payload for a checkpoint by its strong binding keys.
///
/// Returns `Ok(None)` when no staging file exists. An existing file that
/// fails the schema/checkpoint/operation/digest binding is corruption or
/// tampering and fails closed.
pub fn read_segment_staging_v2(
    session_dir: &Path,
    checkpoint_id: &str,
    operation_id: &str,
    wrapper_digest: &str,
) -> io::Result<Option<ResponsesCompactionSegmentStagingV2>> {
    let path = segment_staging_path(session_dir, checkpoint_id)?;
    match std::fs::symlink_metadata(&path) {
        Ok(_) => {
            let staging = read_segment_staging_v2_payload(&path, checkpoint_id)?;
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
            Ok(Some(staging))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Read and strongly bind the V2 staging payload for a live V2 wrapper.
///
/// Returns `Ok(None)` when no staging file exists. When one exists, the
/// checkpoint id, operation id and wrapper digest must all match the live
/// wrapper. Rewind/fork may rotate only the active branch after staging was
/// written, so a branch-only difference is accepted when the stored wrapper
/// digest (computed over the stored branch) reconciles with the live
/// wrapper's other binding fields.
pub fn read_segment_staging_v2_for_wrapper(
    session_dir: &Path,
    wrapper: &ServerResponsesCheckpointV2,
) -> io::Result<Option<ResponsesCompactionSegmentStagingV2>> {
    let path = segment_staging_path(session_dir, &wrapper.checkpoint_id)?;
    match std::fs::symlink_metadata(&path) {
        Ok(_) => {
            let staging = read_segment_staging_v2_payload(&path, &wrapper.checkpoint_id)?;
            if staging.operation_id != wrapper.operation_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "segment staging operation mismatch",
                ));
            }
            let digest_matches = staging.wrapper_digest == wrapper.wrapper_digest();
            let rotation_ok = !digest_matches
                && staging.branch_id != wrapper.branch_id
                && wrapper_digest_v2(
                    &wrapper.checkpoint_id,
                    &wrapper.operation_id,
                    wrapper.prompt_index,
                    &staging.branch_id,
                    &wrapper.identity,
                    &wrapper.portable_history_sha256,
                    wrapper.prior_checkpoint_id.as_deref(),
                ) == staging.wrapper_digest;
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

fn validate_segment_staging_v2(staging: &ResponsesCompactionSegmentStagingV2) -> io::Result<()> {
    if staging.schema_version != 2
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

fn read_segment_staging_v2_payload(
    path: &Path,
    expected_checkpoint_id: &str,
) -> io::Result<ResponsesCompactionSegmentStagingV2> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment staging payload is not a regular file",
        ));
    }
    let staging: ResponsesCompactionSegmentStagingV2 =
        serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    validate_segment_staging_v2(&staging)?;
    if staging.checkpoint_id != expected_checkpoint_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment staging checkpoint mismatch",
        ));
    }
    Ok(staging)
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

/// Publish a committed V2-contract staged segment idempotently, binding the
/// staging to the live wrapper through the operation id and wrapper digest
/// (the strong-binding upgrade over the V1 publish, which only knows the
/// checkpoint id). Same segment layout and index semantics as the V1
/// publish; a V2 staging file at the shared `compaction/staging/` path is
/// consumed and removed exactly like V1 staging.
pub fn publish_staged_compaction_segment_durable_v2(
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
            let staging = read_segment_staging_v2_payload(&staging_path, checkpoint_id)?;
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

    let staging = read_segment_staging_v2_payload(&staging_path, checkpoint_id)?;
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
        CheckpointIdentityV2, CheckpointReplayMaterialV2, RESPONSES_COMPACTION_CONTRACT_V2,
        ResponsesCompactionModeV1, TrustedPromptEnvelopeV2,
    };

    fn envelope_fixture() -> TrustedPromptEnvelopeV2 {
        TrustedPromptEnvelopeV2 {
            base_instructions_sha256: "base-hash".into(),
            memory_revision: Some(3),
            envelope_fingerprint: "envelope-fp".into(),
            wire_prompt_sha256: "wire-hash".into(),
        }
    }

    fn identity_fixture(prior: Option<&str>) -> CheckpointIdentityV2 {
        CheckpointIdentityV2 {
            provider_id: "xai".into(),
            api: "responses".into(),
            endpoint_fingerprint: "endpoint".into(),
            model: "grok-test".into(),
            auth_principal_fingerprint: "principal".into(),
            contract_version: RESPONSES_COMPACTION_CONTRACT_V2.into(),
            prompt_envelope_fingerprint: "envelope-fp".into(),
            base_instructions_sha256: "base-hash".into(),
            prior_checkpoint_id: prior.map(str::to_owned),
            cache_route_fingerprint: None,
        }
    }

    fn portable_fixture() -> Vec<ConversationItem> {
        vec![
            ConversationItem::base_instructions("base prompt"),
            ConversationItem::user("first"),
            ConversationItem::assistant("answer"),
        ]
    }

    fn wrapper_fixture(
        portable: &[ConversationItem],
        prior: Option<&str>,
    ) -> ServerResponsesCheckpointV2 {
        let digest = portable_history_digest(portable).unwrap();
        ServerResponsesCheckpointV2 {
            schema_version: RESPONSES_CHECKPOINT_SCHEMA_V2,
            checkpoint_id: "cp-v2".into(),
            operation_id: "op-v2".into(),
            prompt_index: 5,
            created_at: chrono::Utc::now(),
            auto_continue: false,
            mode: ResponsesCompactionModeV1 {
                name: "default".into(),
                detail: None,
            },
            branch_id: "branch-1".into(),
            identity: identity_fixture(prior),
            output: vec![serde_json::json!({
                "type": "compaction",
                "encrypted_content": "opaque"
            })],
            portable_history_path: "compaction_checkpoints/cp-v2.json".into(),
            portable_history_sha256: digest,
            portable_history_bytes: 128,
            checkpoint_token_seed: 42,
            token_seed_source: xai_grok_sampling_types::TokenSeedSource::UsageOutputTokens,
            server_output_item_count: 1,
            prior_checkpoint_id: prior.map(str::to_owned),
            memory_revision: Some(3),
        }
    }

    fn material_fixture(
        wrapper: &ServerResponsesCheckpointV2,
        portable: &[ConversationItem],
    ) -> CheckpointReplayMaterialV2 {
        CheckpointReplayMaterialV2::try_new(wrapper, envelope_fixture(), portable).unwrap()
    }

    const RELATIVE_PATH: &str = "compaction_checkpoints/cp-v2.json";

    fn write_fixture_sidecar(
        session_dir: &Path,
    ) -> (CompactionCheckpointFileV3, ServerResponsesCheckpointV2) {
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let material = material_fixture(&wrapper, &portable);
        let sidecar = CompactionCheckpointFileV3::new(
            wrapper.clone(),
            material,
            portable,
            Some("original user".into()),
            vec!["rel/readme.md".into()],
        )
        .unwrap();
        write_checkpoint_v3_durable(session_dir, RELATIVE_PATH, &sidecar).unwrap();
        (sidecar, wrapper)
    }

    fn rewrite_sidecar_on_disk(session_dir: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
        let path = session_dir.join(RELATIVE_PATH);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).expect("sidecar parses as json");
        mutate(&mut value);
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    }

    #[test]
    fn v3_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let (sidecar, wrapper) = write_fixture_sidecar(dir.path());
        assert_eq!(sidecar.schema_version, 3);
        assert_eq!(sidecar.kind, "responses_server_v2");
        assert_eq!(sidecar.original_user_info.as_deref(), Some("original user"));
        assert_eq!(sidecar.reread_file_paths, vec!["rel/readme.md"]);

        let read = read_checkpoint_v3(
            dir.path(),
            RELATIVE_PATH,
            "cp-v2",
            5,
            &wrapper.portable_history_sha256,
        )
        .unwrap();
        assert_eq!(read.checkpoint_id, "cp-v2");
        assert_eq!(read.operation_id, "op-v2");
        assert_eq!(read.prompt_index_at_compaction, 5);
        assert_eq!(
            read.portable_history_sha256,
            wrapper.portable_history_sha256
        );
        assert_eq!(
            read.replay_material.wrapper_digest(),
            wrapper.wrapper_digest()
        );
        assert_eq!(read.wrapper.wrapper_digest(), wrapper.wrapper_digest());

        let bound = read_checkpoint_for_wrapper_v2(dir.path(), &wrapper).unwrap();
        assert_eq!(bound.checkpoint_id, wrapper.checkpoint_id);
    }

    #[test]
    fn v3_rejects_wrong_expected_digest() {
        let dir = tempfile::tempdir().unwrap();
        let (_, wrapper) = write_fixture_sidecar(dir.path());
        let error =
            read_checkpoint_v3(dir.path(), RELATIVE_PATH, "cp-v2", 5, "deadbeef").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let _ = wrapper;
    }

    #[test]
    fn v3_new_rejects_mismatched_replay_material() {
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let mut other = wrapper.clone();
        other.operation_id = "op-forged".into();
        let forged =
            CheckpointReplayMaterialV2::try_new(&other, envelope_fixture(), &portable).unwrap();
        assert!(CompactionCheckpointFileV3::new(wrapper, forged, portable, None, vec![],).is_err());
    }

    #[test]
    fn v3_tampered_portable_history_fails() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_sidecar(dir.path());
        rewrite_sidecar_on_disk(dir.path(), |value| {
            value["portable_history"] = serde_json::json!([
                {"type": "user", "content": [{"type": "text", "text": "tampered"}]}
            ]);
        });
        let error = read_checkpoint_v3(dir.path(), RELATIVE_PATH, "cp-v2", 5, "x").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn v3_tampered_wrapper_digest_fails() {
        let dir = tempfile::tempdir().unwrap();
        let (_, wrapper) = write_fixture_sidecar(dir.path());
        rewrite_sidecar_on_disk(dir.path(), |value| {
            value["replay_material"]["wrapper_digest"] = serde_json::json!("deadbeef");
        });
        let error = read_checkpoint_v3(
            dir.path(),
            RELATIVE_PATH,
            "cp-v2",
            5,
            &wrapper.portable_history_sha256,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn v3_rotation_tolerant_read() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_sidecar(dir.path());
        // Rewind/fork rotated only the live branch.
        let mut rotated = wrapper_fixture(&portable_fixture(), None);
        rotated.branch_id = "branch-rotated".into();
        let bound = read_checkpoint_for_wrapper_v2(dir.path(), &rotated).unwrap();
        assert_eq!(bound.wrapper.branch_id, "branch-1");

        // Any other live-field difference still fails closed.
        let mut tampered = rotated;
        tampered.checkpoint_id = "cp-forged".into();
        let error = read_checkpoint_for_wrapper_v2(dir.path(), &tampered).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn marker_v3_validates_stable_fields_only() {
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, Some("cp-prior"));
        let marker = marker_for_wrapper_v3(&wrapper);
        assert_eq!(marker.schema_version, 3);
        assert_eq!(
            marker.wrapper_digest.as_deref(),
            Some(wrapper.wrapper_digest().as_str())
        );
        assert_eq!(marker.prior_checkpoint_id.as_deref(), Some("cp-prior"));
        validate_marker_for_wrapper_v3(&marker, &wrapper).unwrap();

        // Branch rotation and repair timestamps are deliberately ignored.
        let mut rotated_marker = marker.clone();
        rotated_marker.branch_id = Some("branch-rotated".into());
        rotated_marker.created_at = "2000-01-01T00:00:00Z".into();
        validate_marker_for_wrapper_v3(&rotated_marker, &wrapper).unwrap();
    }

    #[test]
    fn marker_v3_rejects_mismatches() {
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let marker = marker_for_wrapper_v3(&wrapper);

        let mut wrong_digest = marker.clone();
        wrong_digest.wrapper_digest = Some("deadbeef".into());
        assert!(validate_marker_for_wrapper_v3(&wrong_digest, &wrapper).is_err());

        let mut wrong_schema = marker.clone();
        wrong_schema.schema_version = 2;
        assert!(validate_marker_for_wrapper_v3(&wrong_schema, &wrapper).is_err());

        let mut wrong_operation = marker.clone();
        wrong_operation.operation_id = Some("op-forged".into());
        assert!(validate_marker_for_wrapper_v3(&wrong_operation, &wrapper).is_err());

        let mut wrong_prior = marker.clone();
        wrong_prior.prior_checkpoint_id = Some("cp-forged".into());
        assert!(validate_marker_for_wrapper_v3(&wrong_prior, &wrapper).is_err());
    }

    fn staging_fixture(
        wrapper: &ServerResponsesCheckpointV2,
    ) -> ResponsesCompactionSegmentStagingV2 {
        ResponsesCompactionSegmentStagingV2::new(
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
    fn staging_v2_strong_binding() {
        let dir = tempfile::tempdir().unwrap();
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let staging = staging_fixture(&wrapper);
        stage_compaction_segment_v2_durable(dir.path(), &staging).unwrap();

        let read = read_segment_staging_v2_for_wrapper(dir.path(), &wrapper)
            .unwrap()
            .expect("staging binds to the live wrapper");
        assert_eq!(read.items.len(), 1);
        let read_plain = read_segment_staging_v2(
            dir.path(),
            &wrapper.checkpoint_id,
            &wrapper.operation_id,
            &wrapper.wrapper_digest(),
        )
        .unwrap()
        .expect("staging binds by strong keys");
        assert_eq!(read_plain.checkpoint_id, wrapper.checkpoint_id);

        // Idempotent restage of the identical payload is allowed.
        stage_compaction_segment_v2_durable(dir.path(), &staging).unwrap();
    }

    #[test]
    fn staging_v2_rotation_tolerance() {
        let dir = tempfile::tempdir().unwrap();
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let staging = staging_fixture(&wrapper);
        stage_compaction_segment_v2_durable(dir.path(), &staging).unwrap();

        let mut rotated = wrapper.clone();
        rotated.branch_id = "branch-rotated".into();
        assert_ne!(rotated.wrapper_digest(), staging.wrapper_digest);
        let read = read_segment_staging_v2_for_wrapper(dir.path(), &rotated)
            .unwrap()
            .expect("branch-only rotation is tolerated");
        assert_eq!(read.branch_id, "branch-1");
    }

    #[test]
    fn staging_v2_rejects_wrong_binding_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        stage_compaction_segment_v2_durable(dir.path(), &staging_fixture(&wrapper)).unwrap();

        // Wrong operation id is rejected (strong binding).
        let mut wrong_operation = wrapper.clone();
        wrong_operation.operation_id = "op-forged".into();
        let error = read_segment_staging_v2_for_wrapper(dir.path(), &wrong_operation).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        // Wrong digest via the key-based reader is rejected.
        let error = read_segment_staging_v2(
            dir.path(),
            &wrapper.checkpoint_id,
            &wrapper.operation_id,
            "deadbeef",
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        // No staging file for an unknown checkpoint is Ok(None).
        let mut unknown = wrapper_fixture(&portable_fixture(), None);
        unknown.checkpoint_id = "cp-unknown".into();
        unknown.portable_history_path = "compaction_checkpoints/cp-unknown.json".into();
        let missing = read_segment_staging_v2_for_wrapper(dir.path(), &unknown).unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn staging_v2_rejects_empty_items() {
        let wrapper = wrapper_fixture(&portable_fixture(), None);
        assert!(
            ResponsesCompactionSegmentStagingV2::new(
                wrapper.checkpoint_id.clone(),
                wrapper.operation_id.clone(),
                wrapper.branch_id.clone(),
                wrapper.wrapper_digest(),
                vec![],
                "summary",
                xai_chat_state::CompactionDetail::Balanced,
                "2026-01-01T00:00:00Z",
            )
            .is_err()
        );
    }

    #[test]
    fn staging_v2_sidecar_schema_mismatch_rejected() {
        // A V1 staging payload must never satisfy the V2 strong-binding
        // reader even when it shares the checkpoint id.
        let dir = tempfile::tempdir().unwrap();
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let v1 = ResponsesCompactionSegmentStagingV1::new(
            wrapper.checkpoint_id.clone(),
            vec![ConversationItem::user("segmented turn")],
            "summary",
            xai_chat_state::CompactionDetail::Balanced,
            "2026-01-01T00:00:00Z",
        )
        .unwrap();
        stage_compaction_segment_durable(dir.path(), &v1).unwrap();
        let error = read_segment_staging_v2_for_wrapper(dir.path(), &wrapper).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn v3_sidecar_fills_portable_history_bytes() {
        // The V3 sidecar construction keeps the wrapper's byte count
        // truthful (mirror of the V2 sidecar), feeding checkpoint-bytes
        // telemetry.
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let material = material_fixture(&wrapper, &portable);
        let sidecar = CompactionCheckpointFileV3::new(
            wrapper.clone(),
            material,
            portable,
            None,
            Vec::new(),
        )
        .unwrap();
        let expected = portable_history_bytes(&portable_fixture()).unwrap().len() as u64;
        assert_eq!(sidecar.wrapper.portable_history_bytes, expected);
        // An unsupported wrapper schema is still rejected by the sidecar
        // constructor (material `try_new` only binds the portable digest,
        // so this exercises the V3 schema guard).
        let mut tampered = wrapper_fixture(&portable_fixture(), None);
        tampered.schema_version = 99;
        let material = material_fixture(&tampered, &portable_fixture());
        assert!(CompactionCheckpointFileV3::new(
            tampered,
            material,
            portable_fixture(),
            None,
            Vec::new(),
        )
        .is_err());
    }

    #[test]
    fn publish_v2_segment_consumes_staging_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let staging = staging_fixture(&wrapper);
        stage_compaction_segment_v2_durable(dir.path(), &staging).unwrap();

        let published = publish_staged_compaction_segment_durable_v2(
            dir.path(),
            &wrapper.checkpoint_id,
            &wrapper.operation_id,
            &wrapper.wrapper_digest(),
        )
        .unwrap();
        assert!(published.newly_published);
        let compaction_dir = dir
            .path()
            .join(xai_chat_state::compaction_transcript::COMPACTION_DIR);
        let segment = compaction_dir
            .join(xai_chat_state::compaction_transcript::segment_filename(published.index));
        let markdown = std::fs::read_to_string(&segment).unwrap();
        assert!(
            markdown.contains(&format!("<!-- {RESPONSES_SEGMENT_MARKER_PREFIX}{}", wrapper.checkpoint_id)),
            "published segment carries the checkpoint marker"
        );
        // Staging is consumed and the publish is idempotent.
        assert!(!segment_staging_path(dir.path(), &wrapper.checkpoint_id)
            .unwrap()
            .exists());
        let again = publish_staged_compaction_segment_durable_v2(
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
    fn publish_v2_rejects_operation_or_digest_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let portable = portable_fixture();
        let wrapper = wrapper_fixture(&portable, None);
        let staging = staging_fixture(&wrapper);
        stage_compaction_segment_v2_durable(dir.path(), &staging).unwrap();

        let wrong_operation = publish_staged_compaction_segment_durable_v2(
            dir.path(),
            &wrapper.checkpoint_id,
            "op-forged",
            &wrapper.wrapper_digest(),
        )
        .unwrap_err();
        assert_eq!(wrong_operation.kind(), io::ErrorKind::InvalidData);
        let wrong_digest = publish_staged_compaction_segment_durable_v2(
            dir.path(),
            &wrapper.checkpoint_id,
            &wrapper.operation_id,
            "deadbeef",
        )
        .unwrap_err();
        assert_eq!(wrong_digest.kind(), io::ErrorKind::InvalidData);
        // Staging survived both rejected publishes.
        assert!(segment_staging_path(dir.path(), &wrapper.checkpoint_id)
            .unwrap()
            .exists());
    }
}

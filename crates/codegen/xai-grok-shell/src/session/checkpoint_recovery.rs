//! V1 Responses checkpoint recovery scanner and status persistence.
//!
//! Stage D0 of the Responses compaction continuity plan: classify how a live
//! V1 [`ServerResponsesCheckpointV1`] could be recovered into safe local
//! continuity **without** changing any request behaviour:
//!
//! * [`V1RecoveryOutcome::Lossless`] — the full portable history is recovered
//!   item-for-item from a verified source (sidecar V2, or segment staging V1
//!   anchored on the live wrapper) and the post-compact typed tail is
//!   recovered losslessly from the prepared/committed journal.
//! * [`V1RecoveryOutcome::LossySalvageAvailable`] — only the lossy legacy
//!   `updates.jsonl` text rebuild is available for the pre-compact section.
//!   The assembled history gets a **new** `salvage_digest`; the old checkpoint
//!   output, identity and portable digest are never reused.
//! * [`V1RecoveryOutcome::Unrecoverable`] — fail closed.
//!
//! The legacy `updates.jsonl` text rebuild never counts as lossless, and the
//! journal typed tail is never downgraded to plain text.
//!
//! Recovery status is persisted to `updates.jsonl` as
//! [`CheckpointRecoveryRecord`] (keyed by `checkpoint_id` +
//! `checkpoint_operation_id`) so a checkpoint in a terminal state is not
//! re-scanned every user turn; a scan is only repeated when the recovery
//! source fingerprint changes or an explicit repair requests it.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use xai_grok_sampling_types::{ConversationItem, ServerResponsesCheckpointV1};

use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
use crate::session::storage::responses_compaction::{
    TailJournalRecord, portable_history_digest, read_checkpoint_for_wrapper,
    read_segment_staging_for_recovery,
};
use crate::session::storage::{self, SessionUpdate};

pub const CHECKPOINT_RECOVERY_RECORD_SCHEMA: u32 = 1;

// ============================================================================
// Outcome types
// ============================================================================

/// Source that produced a lossless pre-compact portable history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LosslessRecoverySource {
    /// `CompactionCheckpointFileV2` sidecar, validated against the live
    /// wrapper with rotation-aware branch binding (checkpoint ID, operation
    /// ID, prompt index, portable digest and all immutable replay fields).
    SidecarV2,
    /// `ResponsesCompactionSegmentStagingV1` items, anchored on the live
    /// wrapper's checkpoint ID and verified against the live wrapper's
    /// portable digest. The V1 staging schema has no operation ID, so this
    /// source never claims operation-ID verification.
    SegmentStagingV1,
}

impl LosslessRecoverySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SidecarV2 => "sidecar_v2",
            Self::SegmentStagingV1 => "segment_staging_v1",
        }
    }
}

/// Categories of content that cannot be recovered losslessly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SalvageOmissionCategory {
    /// Pre-compact tool calls exist only as ACP `ToolCall`/`ToolCallUpdate`
    /// updates; their typed items are lost.
    PreCompactToolCalls,
    /// Pre-compact reasoning (`AgentThoughtChunk`) is lost.
    PreCompactReasoning,
    /// Pre-compact image content blocks are lost (text-only rebuild).
    PreCompactImages,
    /// Typed tail items dropped because the journal did not close
    /// (uncommitted prepared records, sequence gaps or integrity conflicts).
    TailItems,
}

/// Count of unrecoverable items in one category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SalvageOmission {
    pub category: SalvageOmissionCategory,
    pub count: u32,
}

/// Stable machine-diagnosable reason a checkpoint cannot be recovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryReasonCode {
    /// Wrapper schema is not V1; this scanner only handles V1.
    UnsupportedWrapperSchema,
    /// Neither sidecar nor staging could restore the portable history, and
    /// no trusted base instructions were supplied for salvage.
    TrustedBaseUnavailable,
    /// The checkpoint marker is missing or bound to a different operation,
    /// so the pre-compact update boundary is ambiguous and no lossless
    /// source exists.
    MarkerMissing,
    /// No lossless source exists and the salvage assembly is structurally
    /// unusable (no user turns survived).
    SalvageStructureInvalid,
    /// No recovery source exists at all (no sidecar, no staging, no marker,
    /// no journal records).
    NoRecoverySource,
}

impl RecoveryReasonCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedWrapperSchema => "unsupported_wrapper_schema",
            Self::TrustedBaseUnavailable => "trusted_base_unavailable",
            Self::MarkerMissing => "marker_missing",
            Self::SalvageStructureInvalid => "salvage_structure_invalid",
            Self::NoRecoverySource => "no_recovery_source",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryError {
    pub reason_code: RecoveryReasonCode,
    pub detail: String,
}

impl RecoveryError {
    fn new(reason_code: RecoveryReasonCode, detail: impl Into<String>) -> Self {
        Self {
            reason_code,
            detail: detail.into(),
        }
    }
}

/// How completely the post-compact typed tail could be recovered from the
/// `ConversationAppendPreparedV2`/`ConversationAppendCommittedV2` journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalTailRecovery {
    /// No journal records exist for this checkpoint branch.
    Empty,
    /// Every journaled item was recovered; nothing omitted.
    Lossless { items: usize },
    /// A contiguous committed prefix was recovered; the rest is omitted.
    Partial { recovered: usize, omitted: u32 },
    /// The journal contradicts itself; no tail item is trustworthy.
    Unusable { omitted: u32 },
}

impl JournalTailRecovery {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Lossless { .. } => "lossless",
            Self::Partial { .. } => "partial",
            Self::Unusable { .. } => "unusable",
        }
    }
}

/// Graded recovery result for one live V1 checkpoint wrapper.
#[derive(Debug, Clone)]
pub enum V1RecoveryOutcome {
    /// Full portable history + typed tail recovered with the original digest
    /// verified against the recovery source.
    Lossless {
        history: Vec<ConversationItem>,
        source: LosslessRecoverySource,
        original_digest: String,
        /// `history[..portable_len]` is the verified pre-compact portable
        /// history; the remainder is the journal-recovered typed tail.
        /// Migration replaces the live wrapper with the portable prefix plus
        /// the **live** tail (which the journal proved durable).
        portable_len: usize,
    },
    /// Pre-compact section (and possibly part of the tail) is lossy. The
    /// history is freshly assembled: trusted base instructions + lossy
    /// pre-compact User/Assistant text + lossless journal tail. A brand new
    /// `salvage_digest` covers the assembled history; the old checkpoint
    /// output/identity/portable digest are never reused.
    LossySalvageAvailable {
        history: Vec<ConversationItem>,
        salvage_digest: String,
        omissions: Vec<SalvageOmission>,
    },
    /// No safe recovery exists. Callers must fail closed.
    Unrecoverable(RecoveryError),
}

impl V1RecoveryOutcome {
    pub fn class_str(&self) -> &'static str {
        match self {
            Self::Lossless { .. } => "lossless",
            Self::LossySalvageAvailable { .. } => "lossy_salvage_available",
            Self::Unrecoverable(_) => "unrecoverable",
        }
    }
}

/// Full scan result: the outcome plus journal/fingerprint metadata used for
/// retry suppression and telemetry.
#[derive(Debug, Clone)]
pub struct V1RecoveryScan {
    pub outcome: V1RecoveryOutcome,
    pub journal_tail: JournalTailRecovery,
    /// Fingerprint of every recovery source. A status record carrying this
    /// fingerprint suppresses re-scans until a source actually changes.
    pub source_fingerprint: String,
}

// ============================================================================
// Persisted recovery status
// ============================================================================

/// Recovery state machine persisted to `updates.jsonl`. Migration and
/// salvage operation IDs are idempotent: the same checkpoint in the same
/// source state is only attempted once per `source_fingerprint`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum CheckpointRecoveryStatus {
    /// Lossless recovery is available; eligible for automatic migration.
    Pending,
    /// A migration attempt with this idempotent operation ID is in flight.
    Migrating { operation_id: String },
    /// Migration committed; the V1 wrapper no longer exists.
    Migrated,
    /// Only lossy salvage is available. Never auto-migrated; requires an
    /// explicit repair action or an explicit deployment policy.
    SalvageRequired {
        salvage_digest: String,
        omissions: Vec<SalvageOmission>,
    },
    /// A salvage repair with this idempotent operation ID completed.
    Salvaged { operation_id: String },
    /// Fail-closed terminal state with a stable reason code.
    Unrecoverable { reason_code: String },
}

/// Persist-only bookkeeping record for one checkpoint's recovery state,
/// appended to `updates.jsonl` as
/// [`XaiSessionUpdate::CheckpointRecovery`](crate::extensions::notification::SessionUpdate::CheckpointRecovery).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRecoveryRecord {
    pub schema_version: u32,
    pub checkpoint_id: String,
    /// The checkpoint's own operation ID (part of the dedup key).
    pub checkpoint_operation_id: String,
    /// Idempotent recovery/migration operation ID, when one has started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_operation_id: Option<String>,
    pub status: CheckpointRecoveryStatus,
    pub source_fingerprint: String,
    /// RFC3339 timestamp.
    pub updated_at: String,
}

impl CheckpointRecoveryRecord {
    pub fn new(
        wrapper: &ServerResponsesCheckpointV1,
        status: CheckpointRecoveryStatus,
        source_fingerprint: String,
    ) -> Self {
        Self {
            schema_version: CHECKPOINT_RECOVERY_RECORD_SCHEMA,
            checkpoint_id: wrapper.checkpoint_id.clone(),
            checkpoint_operation_id: wrapper.operation_id.clone(),
            recovery_operation_id: None,
            status,
            source_fingerprint,
            updated_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Latest persisted recovery record for (`checkpoint_id`, `operation_id`),
/// if any. The pair is the dedup key: a record left by an older operation
/// must never suppress recovery of a newer checkpoint operation.
pub fn latest_recovery_record(
    updates: &[SessionUpdate],
    checkpoint_id: &str,
    operation_id: &str,
) -> Option<CheckpointRecoveryRecord> {
    updates.iter().rev().find_map(|update| {
        let SessionUpdate::Xai(notification) = update else {
            return None;
        };
        let XaiSessionUpdate::CheckpointRecovery(record) = &notification.update else {
            return None;
        };
        (record.checkpoint_id == checkpoint_id && record.checkpoint_operation_id == operation_id)
            .then(|| (**record).clone())
    })
}

/// Operation ID a migration attempt should use: the persisted one when a
/// previous attempt already reached `Migrating` for the same source
/// fingerprint, so retries stay idempotent instead of fragmenting the
/// journal with fresh IDs.
pub fn migration_operation_id(
    updates: &[SessionUpdate],
    checkpoint_id: &str,
    operation_id: &str,
    source_fingerprint: &str,
) -> Option<String> {
    let record = latest_recovery_record(updates, checkpoint_id, operation_id)?;
    match &record.status {
        CheckpointRecoveryStatus::Migrating { operation_id }
            if record.source_fingerprint == source_fingerprint =>
        {
            Some(operation_id.clone())
        }
        _ => None,
    }
}

/// Map a scan outcome to the status that should be persisted for it.
pub fn status_for_outcome(outcome: &V1RecoveryOutcome) -> CheckpointRecoveryStatus {
    match outcome {
        V1RecoveryOutcome::Lossless { .. } => CheckpointRecoveryStatus::Pending,
        V1RecoveryOutcome::LossySalvageAvailable {
            salvage_digest,
            omissions,
            ..
        } => CheckpointRecoveryStatus::SalvageRequired {
            salvage_digest: salvage_digest.clone(),
            omissions: omissions.clone(),
        },
        V1RecoveryOutcome::Unrecoverable(error) => CheckpointRecoveryStatus::Unrecoverable {
            reason_code: error.reason_code.as_str().to_string(),
        },
    }
}

/// Whether an automatic migration attempt may start given the latest
/// persisted record. `Pending` and `Migrating` (a crashed attempt is safe
/// to resume: the commit is an idempotent CAS) allow an attempt; every
/// terminal or externally-owned state blocks it, so a checkpoint is never
/// auto-migrated twice and never retried in a loop.
pub fn migration_eligible(record: Option<&CheckpointRecoveryRecord>) -> bool {
    match &record {
        None => true,
        Some(record) => matches!(
            record.status,
            CheckpointRecoveryStatus::Pending | CheckpointRecoveryStatus::Migrating { .. }
        ),
    }
}

// ============================================================================
// Scanner
// ============================================================================

/// Load and rewind-filter `updates.jsonl` for recovery scanning.
///
/// Strictly fail-closed: any malformed line aborts the scan. A recovery
/// classification built on a partially-parsed journal could call a corrupted
/// tail "lossless", so parse errors are never swallowed here.
pub fn load_updates_for_recovery(session_dir: &Path) -> io::Result<Vec<SessionUpdate>> {
    let updates_path = session_dir.join(storage::UPDATES_FILE);
    let Some(iter) = storage::UpdatesIterator::open(&updates_path)? else {
        return Ok(Vec::new());
    };
    let mut updates = Vec::new();
    for (index, update) in iter.enumerate() {
        updates.push(update.map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed updates.jsonl line {}: {error}", index + 1),
            )
        })?);
    }
    Ok(storage::filter_rewind_updates(updates))
}

/// Scan every recovery source for `wrapper` and classify the outcome.
///
/// `updates` must be rewind-filtered (see [`load_updates_for_recovery`]).
/// `trusted_base` is the session's trusted base instructions (normally the
/// persisted `system_prompt.txt` content); without it, lossy salvage is
/// impossible and the scan fails closed.
///
/// This function never uses the checkpoint's `output`, never reuses the old
/// portable digest for salvage, and never counts the legacy text rebuild as
/// lossless.
pub fn scan_v1_checkpoint_recovery(
    session_dir: &Path,
    updates: &[SessionUpdate],
    wrapper: &ServerResponsesCheckpointV1,
    trusted_base: Option<&str>,
) -> V1RecoveryScan {
    if wrapper.schema_version != 1 {
        return V1RecoveryScan {
            outcome: V1RecoveryOutcome::Unrecoverable(RecoveryError::new(
                RecoveryReasonCode::UnsupportedWrapperSchema,
                format!("wrapper schema_version {}", wrapper.schema_version),
            )),
            journal_tail: JournalTailRecovery::Empty,
            source_fingerprint: fingerprint_sources(wrapper, None, None, &[], None),
        };
    }

    let marker = updates.iter().rev().find_map(|update| {
        let SessionUpdate::Xai(notification) = update else {
            return None;
        };
        let XaiSessionUpdate::CompactionCheckpoint(marker) = &notification.update else {
            return None;
        };
        // The boundary marker must bind to this exact checkpoint operation:
        // checkpoint ID, operation ID, prompt index and portable digest.
        // Branch is deliberately excluded — rewind/fork rotates only the
        // branch, and a rotated live wrapper keeps its original marker.
        (marker.checkpoint_id == wrapper.checkpoint_id
            && marker.operation_id.as_deref() == Some(wrapper.operation_id.as_str())
            && marker.prompt_index_at_compaction == wrapper.prompt_index
            && marker.portable_history_sha256.as_deref()
                == Some(wrapper.portable_history_sha256.as_str()))
        .then(|| {
            (
                marker.checkpoint_id.clone(),
                marker.prompt_index_at_compaction,
            )
        })
    });
    let marker_present = marker.is_some();

    // The journal records self-bind to checkpoint ID + branch ID, so the
    // typed tail can be recovered even when the marker is missing (crash
    // between the history CAS and the marker write).
    let records = journal_records_for(updates, wrapper);
    let (tail, journal_tail) = recover_journal_tail(wrapper, &records);

    // Lossless source A: sidecar V2 (rotation-aware binding, digest verified).
    let sidecar = read_checkpoint_for_wrapper(session_dir, wrapper).ok();
    // Lossless source B: V1 segment staging, anchored on the live wrapper's
    // checkpoint ID and verified against the live wrapper's portable digest.
    // The staging schema has no operation ID, so none is claimed. Orphan
    // staging (no live wrapper / checkpoint-ID mismatch) is ignored.
    let staging_verified = read_segment_staging_for_recovery(session_dir, &wrapper.checkpoint_id)
        .ok()
        .flatten()
        .and_then(|staging| {
            let digest = portable_history_digest(&staging.items).ok()?;
            (digest == wrapper.portable_history_sha256).then_some((staging.items, digest))
        });
    let staging_digest = staging_verified.as_ref().map(|(_, digest)| digest.clone());

    let verified_portable = if let Some(file) = &sidecar {
        Some((
            file.portable_history.clone(),
            file.portable_history_sha256.clone(),
            LosslessRecoverySource::SidecarV2,
        ))
    } else {
        staging_verified
            .map(|(items, digest)| (items, digest, LosslessRecoverySource::SegmentStagingV1))
    };

    let fingerprint = fingerprint_sources(
        wrapper,
        sidecar_probe(&sidecar),
        staging_digest.as_deref(),
        &records,
        marker.as_ref(),
    );

    let tail_omissions = match journal_tail {
        JournalTailRecovery::Partial { omitted, .. }
        | JournalTailRecovery::Unusable { omitted } => omitted,
        _ => 0,
    };

    if let Some((portable, original_digest, source)) = verified_portable {
        if !portable_structure_ok(&portable, &tail) {
            // A structurally invalid portable history cannot anchor recovery;
            // fall through to the text-salvage path below.
        } else if tail_omissions == 0 {
            let mut history = Vec::with_capacity(portable.len() + tail.len());
            let portable_len = portable.len();
            history.extend_from_slice(&portable);
            history.extend_from_slice(&tail);
            return V1RecoveryScan {
                outcome: V1RecoveryOutcome::Lossless {
                    history,
                    source,
                    original_digest,
                    portable_len,
                },
                journal_tail,
                source_fingerprint: fingerprint,
            };
        } else {
            // Portable history is lossless but the tail journal did not close:
            // salvage keeps the verified portable base and only the closed
            // tail prefix, recording the dropped tail items as omissions.
            let mut history = portable;
            history.extend_from_slice(&tail);
            let salvage_digest = portable_history_digest(&history).unwrap_or_default();
            return V1RecoveryScan {
                outcome: V1RecoveryOutcome::LossySalvageAvailable {
                    history,
                    salvage_digest,
                    omissions: vec![SalvageOmission {
                        category: SalvageOmissionCategory::TailItems,
                        count: tail_omissions,
                    }],
                },
                journal_tail,
                source_fingerprint: fingerprint,
            };
        }
    }

    // Lossy path: needs the marker to bound the pre-compact update section.
    if !marker_present {
        return V1RecoveryScan {
            outcome: V1RecoveryOutcome::Unrecoverable(RecoveryError::new(
                if updates.is_empty() {
                    RecoveryReasonCode::NoRecoverySource
                } else {
                    RecoveryReasonCode::MarkerMissing
                },
                "no lossless source and no matching checkpoint marker",
            )),
            journal_tail,
            source_fingerprint: fingerprint,
        };
    }

    let Some(trusted_base) = trusted_base.filter(|base| !base.trim().is_empty()) else {
        return V1RecoveryScan {
            outcome: V1RecoveryOutcome::Unrecoverable(RecoveryError::new(
                RecoveryReasonCode::TrustedBaseUnavailable,
                "no lossless source and no trusted base instructions",
            )),
            journal_tail,
            source_fingerprint: fingerprint,
        };
    };

    let pre_compact = rebuild_pre_compact_text(updates, wrapper);
    let mut omissions = pre_compact.omissions();
    if tail_omissions > 0 {
        omissions.push(SalvageOmission {
            category: SalvageOmissionCategory::TailItems,
            count: tail_omissions,
        });
    }

    let mut history = Vec::with_capacity(1 + pre_compact.items.len() + tail.len());
    history.push(ConversationItem::system(trusted_base.to_string()));
    history.extend(pre_compact.items);
    history.extend(tail);
    // A salvage that cannot feed builtin compaction (e.g. every pre-compact
    // user turn was lost and the tail is empty) is not a recovery at all.
    if !history
        .iter()
        .any(|item| matches!(item, ConversationItem::User(_)))
    {
        return V1RecoveryScan {
            outcome: V1RecoveryOutcome::Unrecoverable(RecoveryError::new(
                RecoveryReasonCode::SalvageStructureInvalid,
                "salvage assembled no user turns",
            )),
            journal_tail,
            source_fingerprint: fingerprint,
        };
    }
    let salvage_digest = portable_history_digest(&history).unwrap_or_default();

    V1RecoveryScan {
        outcome: V1RecoveryOutcome::LossySalvageAvailable {
            history,
            salvage_digest,
            omissions,
        },
        journal_tail,
        source_fingerprint: fingerprint,
    }
}

/// Structural checks a lossless portable history must pass before it can
/// anchor a recovered conversation:
///
/// * a leading/base System item must exist;
/// * the typed tail must not introduce another checkpoint wrapper;
/// * the assembled history must contain at least one User item so builtin
///   compaction accepts it as input.
fn portable_structure_ok(portable: &[ConversationItem], tail: &[ConversationItem]) -> bool {
    if !matches!(portable.first(), Some(ConversationItem::System(_))) {
        return false;
    }
    if tail
        .iter()
        .any(|item| matches!(item, ConversationItem::ResponsesCompactionCheckpoint(_)))
    {
        return false;
    }
    portable
        .iter()
        .chain(tail)
        .any(|item| matches!(item, ConversationItem::User(_)))
}

/// Collect the prepared/committed journal records bound to this wrapper's
/// checkpoint ID + branch ID.
fn journal_records_for(
    updates: &[SessionUpdate],
    wrapper: &ServerResponsesCheckpointV1,
) -> Vec<TailJournalRecord> {
    updates
        .iter()
        .filter_map(|update| {
            let SessionUpdate::Xai(notification) = update else {
                return None;
            };
            match &notification.update {
                XaiSessionUpdate::ConversationAppendPreparedV2(prepared)
                    if prepared.checkpoint_id == wrapper.checkpoint_id
                        && prepared.branch_id == wrapper.branch_id =>
                {
                    Some(TailJournalRecord::Prepared((**prepared).clone()))
                }
                XaiSessionUpdate::ConversationAppendCommittedV2(committed)
                    if committed.checkpoint_id == wrapper.checkpoint_id
                        && committed.branch_id == wrapper.branch_id =>
                {
                    Some(TailJournalRecord::Committed(committed.clone()))
                }
                _ => None,
            }
        })
        .collect()
}

/// Recover the typed tail from journal records with the same strictness as
/// `rebuild_updates_only_v2`, except violations degrade into omissions
/// instead of hard errors:
///
/// * conflicting duplicate prepared records make the whole tail unusable;
/// * duplicate committed sequences make the whole tail unusable;
/// * orphaned committed records (no matching prepared) count as omissions;
/// * only the contiguous committed prefix starting at sequence 1 is
///   recovered; everything past a gap counts as omissions.
fn recover_journal_tail(
    wrapper: &ServerResponsesCheckpointV1,
    records: &[TailJournalRecord],
) -> (Vec<ConversationItem>, JournalTailRecovery) {
    type Key = (String, String, String, u64, usize);
    let key_of = |operation_id: &str, sequence: u64, prompt_index: usize| -> Key {
        (
            operation_id.to_string(),
            wrapper.checkpoint_id.clone(),
            wrapper.branch_id.clone(),
            sequence,
            prompt_index,
        )
    };
    let mut prepared: BTreeMap<Key, ConversationItem> = BTreeMap::new();
    let mut committed: BTreeSet<Key> = BTreeSet::new();
    let mut total_records = 0_u32;
    for record in records {
        total_records += 1;
        match record {
            TailJournalRecord::Prepared(record) => {
                let key = key_of(&record.operation_id, record.sequence, record.prompt_index);
                if let Some(existing) = prepared.get(&key) {
                    if serde_json::to_value(existing).ok()
                        != serde_json::to_value(&record.item).ok()
                    {
                        return (
                            Vec::new(),
                            JournalTailRecovery::Unusable {
                                omitted: total_records.max(1),
                            },
                        );
                    }
                    // Identical re-prepare: idempotent, ignore.
                    continue;
                }
                prepared.insert(key, record.item.clone());
            }
            TailJournalRecord::Committed(record) => {
                committed.insert(key_of(
                    &record.operation_id,
                    record.sequence,
                    record.prompt_index,
                ));
            }
        }
    }
    if prepared.is_empty() && committed.is_empty() {
        return (Vec::new(), JournalTailRecovery::Empty);
    }

    let mut by_sequence: BTreeMap<u64, ConversationItem> = BTreeMap::new();
    let mut unclosed = 0_u32;
    for (key, item) in &prepared {
        if committed.contains(key) {
            if by_sequence.insert(key.3, item.clone()).is_some() {
                // Two distinct committed keys claim the same sequence: the
                // journal contradicts itself, no tail item is trustworthy.
                return (
                    Vec::new(),
                    JournalTailRecovery::Unusable {
                        omitted: total_records.max(1),
                    },
                );
            }
        } else {
            unclosed += 1;
        }
    }
    // Orphaned committed records reference prepared items we do not have;
    // they cannot be recovered and must be counted, not silently dropped.
    let orphan_committed = committed
        .iter()
        .filter(|key| !prepared.contains_key(*key))
        .count() as u32;
    let mut tail = Vec::new();
    let mut expected = 1_u64;
    let mut gap_omitted = 0_u32;
    for (sequence, item) in by_sequence {
        if sequence != expected {
            gap_omitted += 1;
            continue;
        }
        tail.push(item);
        expected += 1;
    }
    let omitted = unclosed + orphan_committed + gap_omitted;
    let recovery = if omitted == 0 {
        JournalTailRecovery::Lossless { items: tail.len() }
    } else {
        JournalTailRecovery::Partial {
            recovered: tail.len(),
            omitted,
        }
    };
    (tail, recovery)
}

/// Lossy pre-compact User/Assistant text rebuild from the ACP chunk updates
/// that precede the checkpoint marker. Mirrors the legacy replay semantics:
/// user runs are split on prompt-index transitions, trailing partial agent
/// text is flushed, and host-turn chunks only act as flush boundaries.
struct PreCompactText {
    items: Vec<ConversationItem>,
    tool_updates: u32,
    thought_chunks: u32,
    image_blocks: u32,
}

impl PreCompactText {
    fn omissions(&self) -> Vec<SalvageOmission> {
        let mut omissions = Vec::new();
        if self.tool_updates > 0 {
            omissions.push(SalvageOmission {
                category: SalvageOmissionCategory::PreCompactToolCalls,
                count: self.tool_updates,
            });
        }
        if self.thought_chunks > 0 {
            omissions.push(SalvageOmission {
                category: SalvageOmissionCategory::PreCompactReasoning,
                count: self.thought_chunks,
            });
        }
        if self.image_blocks > 0 {
            omissions.push(SalvageOmission {
                category: SalvageOmissionCategory::PreCompactImages,
                count: self.image_blocks,
            });
        }
        omissions
    }
}

fn rebuild_pre_compact_text(
    updates: &[SessionUpdate],
    wrapper: &ServerResponsesCheckpointV1,
) -> PreCompactText {
    let mut items = Vec::new();
    let mut tool_updates = 0_u32;
    let mut thought_chunks = 0_u32;
    let mut image_blocks = 0_u32;

    let mut in_user_message = false;
    let mut current_user_text = String::new();
    let mut current_user_prompt_index: Option<usize> = None;
    let mut seen_prompt_index_marker = false;
    let mut current_agent_text = String::new();
    let mut has_pending_agent = false;

    macro_rules! flush_user {
        () => {
            if !current_user_text.is_empty() {
                let text = std::mem::take(&mut current_user_text);
                match current_user_prompt_index.take() {
                    Some(index) => {
                        let mut item = ConversationItem::user(text);
                        item.set_prompt_index(index);
                        items.push(item);
                    }
                    None => items.push(ConversationItem::user(text)),
                }
            }
        };
    }
    macro_rules! flush_agent {
        () => {
            if has_pending_agent {
                items.push(ConversationItem::assistant(std::mem::take(
                    &mut current_agent_text,
                )));
                has_pending_agent = false;
            }
        };
    }

    for update in updates {
        // Stop at the checkpoint marker: everything after it belongs to the
        // typed tail, which is recovered from the journal instead.
        if let SessionUpdate::Xai(notification) = update
            && let XaiSessionUpdate::CompactionCheckpoint(marker) = &notification.update
            && marker.checkpoint_id == wrapper.checkpoint_id
            && marker.operation_id.as_deref() == Some(wrapper.operation_id.as_str())
        {
            break;
        }
        let SessionUpdate::Acp(notification) = update else {
            continue;
        };
        use agent_client_protocol::SessionUpdate as AcpUpdate;
        match &notification.update {
            AcpUpdate::UserMessageChunk(chunk) => {
                let chunk_prompt_index = chunk
                    .meta
                    .as_ref()
                    .and_then(|meta| meta.get("promptIndex"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|value| value as usize);
                if chunk_prompt_index.is_some() {
                    seen_prompt_index_marker = true;
                }
                if !in_user_message {
                    flush_agent!();
                    in_user_message = true;
                    current_user_text.clear();
                    current_user_prompt_index = chunk_prompt_index;
                } else if chunk_prompt_index != current_user_prompt_index
                    && (chunk_prompt_index.is_some() || current_user_prompt_index.is_some())
                {
                    flush_user!();
                    current_user_text.clear();
                    current_user_prompt_index = chunk_prompt_index;
                } else if current_user_prompt_index.is_none() {
                    current_user_prompt_index = chunk_prompt_index;
                }
                match &chunk.content {
                    agent_client_protocol::ContentBlock::Text(text) => {
                        current_user_text.push_str(&text.text);
                    }
                    agent_client_protocol::ContentBlock::Image(_) => {
                        image_blocks += 1;
                    }
                    _ => {}
                }
            }
            AcpUpdate::AgentMessageChunk(chunk) => {
                if storage::is_host_turn_chunk(chunk) {
                    flush_user!();
                    in_user_message = false;
                    flush_agent!();
                    continue;
                }
                if in_user_message {
                    flush_user!();
                    in_user_message = false;
                }
                match &chunk.content {
                    agent_client_protocol::ContentBlock::Text(text) => {
                        current_agent_text.push_str(&text.text);
                        has_pending_agent = true;
                    }
                    agent_client_protocol::ContentBlock::Image(_) => {
                        image_blocks += 1;
                    }
                    _ => {}
                }
            }
            AcpUpdate::AgentThoughtChunk(_) => {
                thought_chunks += 1;
            }
            AcpUpdate::ToolCall(_) | AcpUpdate::ToolCallUpdate(_) => {
                tool_updates += 1;
            }
            _ => {}
        }
    }
    flush_user!();
    flush_agent!();
    let _ = seen_prompt_index_marker;
    let _ = has_pending_agent;
    PreCompactText {
        items,
        tool_updates,
        thought_chunks,
        image_blocks,
    }
}

/// Fingerprint every recovery source so a persisted status record only
/// suppresses retries while the sources are unchanged. Covers: validated
/// sidecar/staging digests, the sidecar byte count (so corrupt or replaced
/// files change the fingerprint too), the full journal record key set, and
/// the matched marker identity.
fn fingerprint_sources(
    wrapper: &ServerResponsesCheckpointV1,
    sidecar_probe: Option<(String, u64)>,
    staging_digest: Option<&str>,
    records: &[TailJournalRecord],
    marker: Option<&(String, usize)>,
) -> String {
    #[derive(Serialize)]
    struct FingerprintMaterial<'a> {
        checkpoint_id: &'a str,
        operation_id: &'a str,
        branch_id: &'a str,
        sidecar_probe: Option<(String, u64)>,
        staging_digest: Option<&'a str>,
        journal_keys: String,
        marker: Option<(String, usize)>,
    }
    // Hash the full journal key set (not just count/last) so any record
    // mutation — even one that keeps the last key identical — retries.
    let journal_keys = {
        let mut hasher = sha2::Sha256::new();
        for record in records {
            match record {
                TailJournalRecord::Prepared(record) => {
                    hasher.update(b"p\0");
                    hasher.update(record.operation_id.as_bytes());
                    hasher.update(record.sequence.to_le_bytes());
                    hasher.update(record.prompt_index.to_le_bytes());
                }
                TailJournalRecord::Committed(record) => {
                    hasher.update(b"c\0");
                    hasher.update(record.operation_id.as_bytes());
                    hasher.update(record.sequence.to_le_bytes());
                    hasher.update(record.prompt_index.to_le_bytes());
                }
            }
        }
        format!("{:x}", hasher.finalize())
    };
    let material = FingerprintMaterial {
        checkpoint_id: &wrapper.checkpoint_id,
        operation_id: &wrapper.operation_id,
        branch_id: &wrapper.branch_id,
        sidecar_probe,
        staging_digest,
        journal_keys,
        marker: marker.cloned(),
    };
    let bytes = serde_json::to_vec(&material).unwrap_or_default();
    format!("{:x}", sha2::Sha256::digest(&bytes))
}

/// Probe of the on-disk sidecar for fingerprinting: the validated portable
/// digest plus the recorded byte count, so corrupt or replaced files still
/// change the fingerprint.
fn sidecar_probe(
    sidecar: &Option<crate::session::storage::responses_compaction::CompactionCheckpointFileV2>,
) -> Option<(String, u64)> {
    sidecar.as_ref().map(|file| {
        (
            file.portable_history_sha256.clone(),
            file.wrapper.portable_history_bytes,
        )
    })
}

#[cfg(test)]
mod tests;

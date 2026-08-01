//! D4: per-session GC and retention quota for Responses server-compaction
//! artifacts (sidecars under `compaction_checkpoints/` and segment staging
//! under `compaction/staging/`).
//!
//! Retention rules (hard, see the D4 plan section):
//! * ALWAYS retain sidecars/staging reachable from the live wrapper (the
//!   first `chat_history.jsonl` item when it is a checkpoint), from ANY
//!   compaction marker in `updates.jsonl`, from pending/migrating/
//!   salvage-required checkpoint recovery records, from prepared/committed
//!   journal tail records, and from published `compaction/segment_*.md`
//!   marker headers (retained rewind/fork branches keep their markers, so
//!   they are covered by the marker scan).
//! * Pre-CAS orphan sidecar/staging (referenced by NOTHING after the full
//!   scan) may be deleted only once its file mtime is older than
//!   [`GcOptions::grace_period`].
//! * The scan fails closed: if `updates.jsonl` or `chat_history.jsonl`
//!   exists but cannot be parsed, the whole GC aborts with no deletions.
//! * Quota pressure never deletes active recovery data: exceeding the
//!   per-session quota only stops NEW remote checkpoints (see
//!   [`quota_exceeded`]); the caller falls back to builtin compaction.

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
use crate::session::checkpoint_recovery::CheckpointRecoveryStatus;
use crate::session::storage::{self, SessionUpdate};

/// Default per-session cap on total compaction checkpoint bytes (sidecars +
/// segment staging). Exceeding it stops NEW remote checkpoints and falls
/// back to builtin compaction; it never deletes active recovery data.
pub(crate) const DEFAULT_SESSION_CHECKPOINT_QUOTA_BYTES: u64 = 256 * 1024 * 1024;

/// Env override for the per-session checkpoint quota, in MiB. A missing,
/// empty, zero or unparseable value falls back to
/// [`DEFAULT_SESSION_CHECKPOINT_QUOTA_BYTES`].
const QUOTA_ENV_OVERRIDE_MB: &str = "GROK_SESSION_CHECKPOINT_QUOTA_MB";

/// Directory holding published V2/V3 sidecar files,
/// `{session_dir}/compaction_checkpoints/{checkpoint_id}.json`.
const CHECKPOINT_DIR: &str = "compaction_checkpoints";

/// Directory holding staged (unpublished) compaction segments,
/// `{session_dir}/compaction/staging/{checkpoint_id}.json`.
const STAGING_SUBDIR: &str = "compaction/staging";

/// First-line marker prefix of published segments; mirrors
/// `responses_compaction::RESPONSES_SEGMENT_MARKER_PREFIX`. A published
/// segment referencing a checkpoint id retains that id's sidecar.
const SEGMENT_MARKER_HEADER: &str = "<!-- responses-compaction-checkpoint:";

/// Resolved per-session checkpoint quota in bytes, read from the
/// `GROK_SESSION_CHECKPOINT_QUOTA_MB` env override (default 256 MiB).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GrokCompactionQuota {
    bytes: u64,
}

impl GrokCompactionQuota {
    pub(crate) fn from_env() -> Self {
        Self {
            bytes: session_checkpoint_quota_bytes(),
        }
    }

    pub(crate) fn bytes(self) -> u64 {
        self.bytes
    }
}

impl Default for GrokCompactionQuota {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Resolve the per-session checkpoint quota (bytes) from the
/// `GROK_SESSION_CHECKPOINT_QUOTA_MB` env override, defaulting to
/// [`DEFAULT_SESSION_CHECKPOINT_QUOTA_BYTES`] (256 MiB).
pub(crate) fn session_checkpoint_quota_bytes() -> u64 {
    quota_from_mb_override(std::env::var(QUOTA_ENV_OVERRIDE_MB).ok().as_deref())
}

/// Pure quota resolution from the raw env override value (MiB), so the
/// defaulting rules are testable without mutating process env. Missing,
/// empty, zero or unparseable values fall back to
/// [`DEFAULT_SESSION_CHECKPOINT_QUOTA_BYTES`].
fn quota_from_mb_override(value: Option<&str>) -> u64 {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|mb| *mb > 0)
        .map(|mb| mb.saturating_mul(1024 * 1024))
        .unwrap_or(DEFAULT_SESSION_CHECKPOINT_QUOTA_BYTES)
}

/// Total on-disk bytes of this session's compaction artifacts: every regular
/// file under `compaction_checkpoints/` and `compaction/staging/` (symlinks
/// and special files are not counted). Missing directories count as zero.
pub(crate) fn session_checkpoint_bytes(session_dir: &Path) -> io::Result<u64> {
    let mut total = 0_u64;
    for dir in [session_dir.join(CHECKPOINT_DIR), session_dir.join(STAGING_SUBDIR)] {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if !metadata.file_type().is_symlink() && metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

/// Whether this session exceeds `quota` bytes of compaction artifacts. This
/// only gates NEW remote checkpoints; active recovery data is never deleted
/// to make room.
pub(crate) fn quota_exceeded(session_dir: &Path, quota: u64) -> io::Result<bool> {
    Ok(session_checkpoint_bytes(session_dir)? > quota)
}

/// Options for one GC pass.
#[derive(Debug, Clone)]
pub(crate) struct GcOptions {
    /// Orphans younger than this are retained. Defaults to 24h.
    pub grace_period: Duration,
    /// When set, nothing is deleted; the report still classifies orphans and
    /// grace-period retention.
    pub dry_run: bool,
}

impl Default for GcOptions {
    fn default() -> Self {
        Self {
            grace_period: Duration::from_secs(24 * 60 * 60),
            dry_run: false,
        }
    }
}

/// Outcome of one GC pass over one session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SessionGcReport {
    /// Total bytes of every scanned candidate file (sidecars + staging).
    pub scanned_bytes: u64,
    /// Bytes of candidates reachable from the live wrapper / markers /
    /// recovery records / journal / published segments (always retained).
    pub referenced_bytes: u64,
    /// Bytes of candidates referenced by nothing (pre-CAS orphans).
    pub orphan_bytes: u64,
    /// Bytes actually deleted this pass (dry-run and grace-period orphans
    /// never count here).
    pub deleted_bytes: u64,
    /// Bytes of orphans retained because their mtime is inside the grace
    /// period (or their mtime could not be verified).
    pub retained_orphan_bytes: u64,
    /// Number of candidate files scanned.
    pub files_scanned: u32,
    /// Number of files deleted.
    pub files_deleted: u32,
    /// Human-readable reasons for skipped/unparseable files (they are
    /// retained, never deleted).
    pub skipped_due_to: Vec<String>,
}

/// GC the session's compaction artifacts.
///
/// Fail-closed full scan: `updates.jsonl` and `chat_history.jsonl` are fully
/// parsed BEFORE anything is deleted; a missing file is treated as empty,
/// but a file that exists and cannot be parsed aborts the whole GC with
/// [`io::ErrorKind::InvalidData`] and no deletions happen.
pub(crate) async fn gc_session_compaction_artifacts(
    session_dir: &Path,
    opts: GcOptions,
) -> io::Result<SessionGcReport> {
    let (referenced_ids, referenced_files) = scan_references(session_dir)?;
    let (candidates, mut skipped) = collect_candidates(session_dir)?;

    let now = std::time::SystemTime::now();
    let mut report = SessionGcReport {
        files_scanned: candidates.len() as u32,
        ..SessionGcReport::default()
    };
    for candidate in candidates {
        report.scanned_bytes = report.scanned_bytes.saturating_add(candidate.bytes);
        if referenced_ids.contains(&candidate.checkpoint_id)
            || referenced_files.contains(&candidate.file_name)
        {
            report.referenced_bytes = report.referenced_bytes.saturating_add(candidate.bytes);
            continue;
        }
        report.orphan_bytes = report.orphan_bytes.saturating_add(candidate.bytes);
        let in_grace = match candidate.mtime_older_than(now, opts.grace_period) {
            Ok(true) => false,
            Ok(false) => true,
            Err(error) => {
                // mtime unverifiable: retain (fail closed on age) and note.
                skipped.push(format!(
                    "{}: cannot verify mtime ({error}); retained",
                    candidate.path.display()
                ));
                true
            }
        };
        if in_grace {
            report.retained_orphan_bytes =
                report.retained_orphan_bytes.saturating_add(candidate.bytes);
            continue;
        }
        if opts.dry_run {
            continue;
        }
        match std::fs::remove_file(&candidate.path) {
            Ok(()) => {
                report.deleted_bytes = report.deleted_bytes.saturating_add(candidate.bytes);
                report.files_deleted += 1;
                if candidate.staging {
                    remove_staging_dir_if_empty(session_dir);
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // Raced with the writer; nothing to delete.
            }
            Err(error) => {
                skipped.push(format!(
                    "{}: delete failed ({error}); retained",
                    candidate.path.display()
                ));
            }
        }
    }
    report.skipped_due_to = skipped;
    Ok(report)
}

/// One deletable candidate (sidecar or staging file).
struct Candidate {
    path: PathBuf,
    file_name: String,
    checkpoint_id: String,
    bytes: u64,
    mtime: std::time::SystemTime,
    staging: bool,
}

impl Candidate {
    /// Whether the file's mtime is strictly older than `grace_period` (i.e.
    /// it is eligible for deletion). Future mtimes count as young.
    fn mtime_older_than(&self, now: std::time::SystemTime, grace: Duration) -> io::Result<bool> {
        let age = now
            .duration_since(self.mtime)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "mtime in the future"))?;
        Ok(age > grace)
    }
}

/// Full reachability scan: live wrapper + typed tail from `chat_history.jsonl`,
/// and markers + recovery records + journal from `updates.jsonl`, plus
/// published segment marker headers. Fail-closed on any parse error.
///
/// Returns the set of referenced checkpoint ids and the set of referenced
/// file names (basenames), so candidates can be matched either way.
fn scan_references(session_dir: &Path) -> io::Result<(BTreeSet<String>, BTreeSet<String>)> {
    let mut ids = BTreeSet::new();
    let mut files = BTreeSet::new();

    // updates.jsonl — full scan; every line must parse.
    let updates_path = session_dir.join(storage::UPDATES_FILE);
    if let Some(iter) = storage::UpdatesIterator::open(&updates_path)? {
        for (index, update) in iter.enumerate() {
            let update = update.map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("malformed updates.jsonl line {}: {error}", index + 1),
                )
            })?;
            let SessionUpdate::Xai(notification) = update else {
                continue;
            };
            match &notification.update {
                XaiSessionUpdate::CompactionCheckpoint(marker) => {
                    ids.insert(marker.checkpoint_id.clone());
                    if let Some(prior) = &marker.prior_checkpoint_id {
                        // Recompact chain link / retained rewind-fork lineage:
                        // the prior sidecar stays reachable.
                        ids.insert(prior.clone());
                    }
                }
                XaiSessionUpdate::CheckpointRecovery(record) => {
                    // Pending/migrating/salvage-required records still need
                    // their recovery source; terminal states no longer do.
                    if matches!(
                        record.status,
                        CheckpointRecoveryStatus::Pending
                            | CheckpointRecoveryStatus::Migrating { .. }
                            | CheckpointRecoveryStatus::SalvageRequired { .. }
                    ) {
                        ids.insert(record.checkpoint_id.clone());
                    }
                }
                XaiSessionUpdate::ConversationAppendPreparedV2(prepared) => {
                    ids.insert(prepared.checkpoint_id.clone());
                }
                XaiSessionUpdate::ConversationAppendCommittedV2(committed) => {
                    ids.insert(committed.checkpoint_id.clone());
                }
                _ => {}
            }
        }
    }

    // chat_history.jsonl — full scan; a missing file means no wrapper yet,
    // an unparseable file aborts the GC.
    let chat_path = session_dir.join(storage::CHAT_HISTORY_FILE);
    if chat_path.exists() {
        for entry in storage::read_persisted_chat_entries(&chat_path)? {
            match entry {
                storage::responses_compaction::PersistedChatEntry::Legacy(item) => {
                    if let Some(reference) = item.as_responses_checkpoint() {
                        ids.insert(reference.checkpoint_id().to_string());
                        if let Some(file_name) = Path::new(reference.portable_history_path())
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                        {
                            files.insert(file_name);
                        }
                    }
                }
                storage::responses_compaction::PersistedChatEntry::TailV2(tail) => {
                    ids.insert(tail.checkpoint_id.clone());
                }
            }
        }
    }

    // Published segments: a segment marker header referencing a checkpoint
    // id retains that id's sidecar (never deleted while a marker may still
    // reference it). A recognized segment file that cannot be read fails the
    // scan closed — the reference set would be incomplete.
    let compaction_dir = session_dir.join(xai_chat_state::compaction_transcript::COMPACTION_DIR);
    let entries = match std::fs::read_dir(&compaction_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok((ids, files)),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if xai_chat_state::compaction_transcript::parse_segment_index(&name).is_none() {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            continue;
        }
        let first_line = first_line(&entry.path())?;
        if let Some(id) = first_line
            .strip_prefix(SEGMENT_MARKER_HEADER)
            .and_then(|rest| rest.strip_suffix(" -->"))
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            ids.insert(id.to_string());
        }
    }
    Ok((ids, files))
}

fn first_line(path: &Path) -> io::Result<String> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(line.trim_end().to_string())
}

/// Collect every deletable candidate (regular `*.json` file) from
/// `compaction_checkpoints/` and `compaction/staging/`. Files that are
/// symlinks, non-regular, not named `{checkpoint_id}.json` with a safe
/// checkpoint id, or whose age cannot be trusted are skipped and reported —
/// they are always retained.
fn collect_candidates(session_dir: &Path) -> io::Result<(Vec<Candidate>, Vec<String>)> {
    let mut candidates = Vec::new();
    let mut skipped = Vec::new();
    for (dir, staging) in [
        (session_dir.join(CHECKPOINT_DIR), false),
        (session_dir.join(STAGING_SUBDIR), true),
    ] {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let metadata = match std::fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                skipped.push(format!(
                    "{file_name}: not a regular file; retained"
                ));
                continue;
            }
            let Some(stem) = file_name.strip_suffix(".json") else {
                skipped.push(format!("{file_name}: not a checkpoint json file; retained"));
                continue;
            };
            if !is_safe_checkpoint_component(stem) {
                skipped.push(format!(
                    "{file_name}: unrecognized checkpoint file name; retained"
                ));
                continue;
            }
            let checkpoint_id = stem.to_string();
            candidates.push(Candidate {
                path,
                file_name,
                checkpoint_id,
                bytes: metadata.len(),
                mtime: metadata
                    .modified()
                    .map_err(|error| {
                        io::Error::new(
                            error.kind(),
                            format!("cannot read mtime of {}: {error}", entry.path().display()),
                        )
                    })?,
                staging,
            });
        }
    }
    Ok((candidates, skipped))
}

/// Mirrors `responses_compaction::validate_checkpoint_component`: only
/// alphanumeric, `-` and `_` ids are safe to derive a file name from.
fn is_safe_checkpoint_component(component: &str) -> bool {
    !component.is_empty()
        && component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

/// Remove the staging directory when it is empty after a deletion (best
/// effort — errors are ignored).
fn remove_staging_dir_if_empty(session_dir: &Path) {
    let staging_dir = session_dir.join(STAGING_SUBDIR);
    if let Ok(mut entries) = std::fs::read_dir(&staging_dir) {
        if entries.next().is_none() {
            let _ = std::fs::remove_dir(staging_dir);
        }
    }
}

#[cfg(test)]
mod tests;

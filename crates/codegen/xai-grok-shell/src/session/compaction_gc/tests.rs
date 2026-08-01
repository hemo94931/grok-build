use super::*;

use agent_client_protocol as acp;
use tempfile::TempDir;
use xai_grok_sampling_types::{
    CheckpointIdentityV1, ConversationItem, ResponsesCompactionModeV1, ServerResponsesCheckpointV1,
    TokenSeedSource,
};

use crate::extensions::notification::{
    SessionNotification as XaiNotification, SessionUpdate as XaiSessionUpdate,
};
use crate::session::checkpoint_recovery::{
    CheckpointRecoveryRecord, CheckpointRecoveryStatus,
};
use crate::session::storage::responses_compaction::{
    CompactionCheckpointFileV2, ConversationAppendCommittedV2, ConversationAppendPreparedV2,
    PersistedChatEntry, marker_for_wrapper, portable_history_digest, write_checkpoint_v2_durable,
    write_history_v2_durable,
};

const SESSION_ID: &str = "gc-test";

fn portable_fixture() -> Vec<ConversationItem> {
    vec![
        ConversationItem::system("base instructions"),
        ConversationItem::user("first prompt"),
        ConversationItem::assistant("first answer"),
    ]
}

fn identity_fixture() -> CheckpointIdentityV1 {
    CheckpointIdentityV1 {
        provider_id: "xai".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: "responses-compact-codex-v1".into(),
        prompt_envelope_fingerprint: "envelope".into(),
        canonical_prompt_projection: None,
    }
}

fn wrapper_fixture(checkpoint_id: &str) -> ServerResponsesCheckpointV1 {
    let portable = portable_fixture();
    let digest = portable_history_digest(&portable).unwrap();
    let bytes =
        crate::session::storage::responses_compaction::portable_history_bytes(&portable).unwrap();
    ServerResponsesCheckpointV1 {
        schema_version: 1,
        checkpoint_id: checkpoint_id.into(),
        operation_id: format!("op-{checkpoint_id}"),
        prompt_index: 1,
        created_at: chrono::Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionModeV1 {
            name: "default".into(),
            detail: None,
        },
        branch_id: "branch-a".into(),
        identity: identity_fixture(),
        output: vec![serde_json::json!({ "type": "compaction", "encrypted_content": "opaque" })],
        portable_history_path: format!("compaction_checkpoints/{checkpoint_id}.json"),
        portable_history_sha256: digest,
        portable_history_bytes: bytes.len() as u64,
        checkpoint_token_seed: 10,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 1,
    }
}

/// Write a real V2 sidecar file for `checkpoint_id`.
fn write_sidecar(session_dir: &Path, checkpoint_id: &str) {
    let wrapper = wrapper_fixture(checkpoint_id);
    let file = CompactionCheckpointFileV2::new(
        wrapper.clone(),
        portable_fixture(),
        None,
        Vec::new(),
    )
    .unwrap();
    write_checkpoint_v2_durable(session_dir, &wrapper.portable_history_path, &file).unwrap();
}

/// Write an orphan artifact (valid sidecar-shaped or arbitrary content) that
/// no reference points at, with an old mtime so it is past the grace period.
fn write_orphan(session_dir: &Path, checkpoint_id: &str, content: &[u8]) {
    let dir = session_dir.join(CHECKPOINT_DIR);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{checkpoint_id}.json"));
    std::fs::write(&path, content).unwrap();
    set_mtime_old(&path);
}

fn set_mtime_old(path: &Path) {
    let old = std::time::SystemTime::now() - Duration::from_secs(48 * 60 * 60);
    filetime::set_file_mtime(path, filetime::FileTime::from_system_time(old)).unwrap();
}

fn write_orphan_staging(session_dir: &Path, checkpoint_id: &str, content: &[u8]) {
    let dir = session_dir.join(STAGING_SUBDIR);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{checkpoint_id}.json"));
    std::fs::write(&path, content).unwrap();
    set_mtime_old(&path);
}

fn write_updates(session_dir: &Path, updates: &[SessionUpdate]) {
    let mut bytes = Vec::new();
    for update in updates {
        serde_json::to_writer(&mut bytes, update).unwrap();
        bytes.push(b'\n');
    }
    std::fs::write(session_dir.join(storage::UPDATES_FILE), bytes).unwrap();
}

fn xai(update: XaiSessionUpdate) -> SessionUpdate {
    SessionUpdate::Xai(Box::new(XaiNotification {
        session_id: acp::SessionId::new(SESSION_ID),
        update,
        meta: None,
    }))
}

fn marker_update(checkpoint_id: &str) -> SessionUpdate {
    xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
        marker_for_wrapper(&wrapper_fixture(checkpoint_id)),
    )))
}

fn recovery_update(checkpoint_id: &str, status: CheckpointRecoveryStatus) -> SessionUpdate {
    let record =
        CheckpointRecoveryRecord::new(&wrapper_fixture(checkpoint_id), status, "fingerprint".into());
    xai(XaiSessionUpdate::CheckpointRecovery(Box::new(record)))
}

fn journal_updates(checkpoint_id: &str) -> Vec<SessionUpdate> {
    let wrapper = wrapper_fixture(checkpoint_id);
    let prepared = ConversationAppendPreparedV2 {
        operation_id: format!("{}-tail-1", wrapper.operation_id),
        checkpoint_id: wrapper.checkpoint_id.clone(),
        branch_id: wrapper.branch_id.clone(),
        sequence: 1,
        prompt_index: wrapper.prompt_index,
        item: ConversationItem::user("tail user"),
    };
    vec![
        xai(XaiSessionUpdate::ConversationAppendPreparedV2(Box::new(
            prepared.clone(),
        ))),
        xai(XaiSessionUpdate::ConversationAppendCommittedV2(
            ConversationAppendCommittedV2::from(&prepared),
        )),
    ]
}

fn chat_history_with_wrapper(session_dir: &Path, checkpoint_id: &str) {
    let wrapper = wrapper_fixture(checkpoint_id);
    let entries = vec![PersistedChatEntry::Legacy(
        ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper)),
    )];
    write_history_v2_durable(&session_dir.join(storage::CHAT_HISTORY_FILE), &entries).unwrap();
}

async fn gc(session_dir: &Path, grace: Duration) -> SessionGcReport {
    gc_session_compaction_artifacts(
        session_dir,
        GcOptions {
            grace_period: grace,
            dry_run: false,
        },
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn sidecar_referenced_by_live_wrapper_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-wrapper");
    write_orphan(tmp.path(), "cp-orphan", b"{}");
    chat_history_with_wrapper(tmp.path(), "cp-wrapper");

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-wrapper.json").exists());
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-orphan.json").exists());
    assert!(report.referenced_bytes > 0);
    assert_eq!(report.deleted_bytes, report.orphan_bytes);
    assert_eq!(report.retained_orphan_bytes, 0);
}

#[tokio::test]
async fn sidecar_referenced_by_marker_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-marker");
    write_orphan(tmp.path(), "cp-orphan", b"{}");
    write_updates(tmp.path(), &[marker_update("cp-marker")]);

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-marker.json").exists());
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-orphan.json").exists());
    assert_eq!(report.referenced_bytes, report.scanned_bytes - report.orphan_bytes);
}

#[tokio::test]
async fn marker_prior_checkpoint_chain_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-latest", b"{}");
    write_orphan(tmp.path(), "cp-prior", b"{}");
    // Marker for cp-latest links back to cp-prior (recompact chain).
    let mut marker = marker_for_wrapper(&wrapper_fixture("cp-latest"));
    marker.prior_checkpoint_id = Some("cp-prior".into());
    write_updates(tmp.path(), &[xai(XaiSessionUpdate::CompactionCheckpoint(
        Box::new(marker),
    ))]);

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-latest.json").exists());
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-prior.json").exists());
    assert_eq!(report.deleted_bytes, 0);
}

#[tokio::test]
async fn sidecar_referenced_by_pending_recovery_record_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-recovery");
    write_orphan(tmp.path(), "cp-orphan", b"{}");
    write_updates(
        tmp.path(),
        &[recovery_update("cp-recovery", CheckpointRecoveryStatus::Pending)],
    );

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-recovery.json").exists());
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-orphan.json").exists());
    assert_eq!(report.deleted_bytes, report.orphan_bytes);
}

#[tokio::test]
async fn terminal_recovery_record_does_not_retain() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-recovered", b"{}");
    write_updates(
        tmp.path(),
        &[recovery_update(
            "cp-recovered",
            CheckpointRecoveryStatus::Unrecoverable {
                reason_code: "no_recovery_source".into(),
            },
        )],
    );
    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-recovered.json").exists());
    assert_eq!(report.deleted_bytes, report.orphan_bytes);
}

#[tokio::test]
async fn sidecar_referenced_by_journal_tail_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-journal");
    write_orphan(tmp.path(), "cp-orphan", b"{}");
    write_updates(tmp.path(), &journal_updates("cp-journal"));

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-journal.json").exists());
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-orphan.json").exists());
    assert_eq!(report.deleted_bytes, report.orphan_bytes);
}

#[tokio::test]
async fn orphan_staging_deleted_but_referenced_staging_retained() {
    let tmp = TempDir::new().unwrap();
    write_orphan_staging(tmp.path(), "cp-staged", b"{}");
    write_orphan_staging(tmp.path(), "cp-orphan", b"{}");
    write_updates(tmp.path(), &[marker_update("cp-staged")]);

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(STAGING_SUBDIR).join("cp-staged.json").exists());
    assert!(!tmp.path().join(STAGING_SUBDIR).join("cp-orphan.json").exists());
    assert_eq!(report.files_deleted, 1);
}

#[tokio::test]
async fn orphan_younger_than_grace_is_retained() {
    let tmp = TempDir::new().unwrap();
    // Fresh mtime (written just now), no references, default 24h grace.
    let dir = tmp.path().join(CHECKPOINT_DIR);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cp-fresh.json");
    std::fs::write(&path, b"{}").unwrap();

    let report = gc_session_compaction_artifacts(tmp.path(), GcOptions::default())
        .await
        .unwrap();
    assert!(path.exists());
    assert_eq!(report.orphan_bytes, 2);
    assert_eq!(report.retained_orphan_bytes, 2);
    assert_eq!(report.deleted_bytes, 0);
    assert_eq!(report.files_deleted, 0);
}

#[tokio::test]
async fn orphan_older_than_grace_is_deleted() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-old", b"{}");
    let report = gc(tmp.path(), Duration::from_secs(24 * 60 * 60)).await;
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-old.json").exists());
    assert_eq!(report.deleted_bytes, 2);
    assert_eq!(report.files_deleted, 1);
    assert_eq!(report.retained_orphan_bytes, 0);
}

#[tokio::test]
async fn dry_run_deletes_nothing_but_reports_orphans() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-old", b"{}");
    let report = gc_session_compaction_artifacts(
        tmp.path(),
        GcOptions {
            grace_period: Duration::ZERO,
            dry_run: true,
        },
    )
    .await
    .unwrap();
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-old.json").exists());
    assert_eq!(report.orphan_bytes, 2);
    assert_eq!(report.deleted_bytes, 0);
    assert_eq!(report.files_deleted, 0);
}

#[tokio::test]
async fn corrupt_updates_jsonl_aborts_with_no_deletions() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-old", b"{}");
    std::fs::write(
        tmp.path().join(storage::UPDATES_FILE),
        b"{ not valid json\n",
    )
    .unwrap();

    let error = gc_session_compaction_artifacts(
        tmp.path(),
        GcOptions {
            grace_period: Duration::ZERO,
            dry_run: false,
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-old.json").exists());
}

#[tokio::test]
async fn corrupt_chat_history_aborts_with_no_deletions() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-old", b"{}");
    std::fs::write(
        tmp.path().join(storage::CHAT_HISTORY_FILE),
        b"{ not valid json\n",
    )
    .unwrap();

    let error = gc_session_compaction_artifacts(
        tmp.path(),
        GcOptions {
            grace_period: Duration::ZERO,
            dry_run: false,
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-old.json").exists());
}

#[tokio::test]
async fn unknown_and_non_json_files_are_skipped_and_retained() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join(CHECKPOINT_DIR);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("README.txt"), b"keep me").unwrap();
    std::fs::write(dir.join("bad name.json"), b"{}").unwrap();

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(dir.join("README.txt").exists());
    assert!(dir.join("bad name.json").exists());
    assert_eq!(report.orphan_bytes, 0);
    assert_eq!(report.files_deleted, 0);
    assert_eq!(report.skipped_due_to.len(), 2);
}

#[tokio::test]
async fn published_segment_marker_retains_its_checkpoint() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-published", b"{}");
    write_orphan(tmp.path(), "cp-orphan", b"{}");
    let compaction_dir = tmp.path().join("compaction");
    std::fs::create_dir_all(&compaction_dir).unwrap();
    std::fs::write(
        compaction_dir.join("segment_001.md"),
        format!("<!-- responses-compaction-checkpoint:cp-published -->\n# Segment\n"),
    )
    .unwrap();

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-published.json").exists());
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-orphan.json").exists());
    assert_eq!(report.files_deleted, 1);
}

#[tokio::test]
async fn missing_chat_history_is_empty_not_an_error() {
    let tmp = TempDir::new().unwrap();
    write_orphan(tmp.path(), "cp-old", b"{}");
    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(!tmp.path().join(CHECKPOINT_DIR).join("cp-old.json").exists());
    assert_eq!(report.files_deleted, 1);
}

#[test]
fn quota_defaults_and_override() {
    assert_eq!(DEFAULT_SESSION_CHECKPOINT_QUOTA_BYTES, 256 * 1024 * 1024);
    assert_eq!(quota_from_mb_override(None), 256 * 1024 * 1024);
    assert_eq!(quota_from_mb_override(Some("")), 256 * 1024 * 1024);
    assert_eq!(quota_from_mb_override(Some("0")), 256 * 1024 * 1024);
    assert_eq!(quota_from_mb_override(Some("junk")), 256 * 1024 * 1024);
    assert_eq!(quota_from_mb_override(Some("512")), 512 * 1024 * 1024);
    assert_eq!(quota_from_mb_override(Some(" 128 ")), 128 * 1024 * 1024);
    assert_eq!(GrokCompactionQuota::default().bytes(), session_checkpoint_quota_bytes());
}

#[test]
fn quota_computation_and_exceeded() {
    let tmp = TempDir::new().unwrap();
    // Nothing on disk yet.
    assert_eq!(session_checkpoint_bytes(tmp.path()).unwrap(), 0);
    assert!(!quota_exceeded(tmp.path(), 1024).unwrap());

    let checkpoint_dir = tmp.path().join(CHECKPOINT_DIR);
    std::fs::create_dir_all(&checkpoint_dir).unwrap();
    std::fs::write(checkpoint_dir.join("cp-a.json"), [0_u8; 10]).unwrap();
    std::fs::write(checkpoint_dir.join("cp-b.json"), [0_u8; 100]).unwrap();
    assert_eq!(session_checkpoint_bytes(tmp.path()).unwrap(), 110);

    let staging_dir = tmp.path().join(STAGING_SUBDIR);
    std::fs::create_dir_all(&staging_dir).unwrap();
    std::fs::write(staging_dir.join("cp-c.json"), [0_u8; 50]).unwrap();
    assert_eq!(session_checkpoint_bytes(tmp.path()).unwrap(), 160);

    assert!(quota_exceeded(tmp.path(), 100).unwrap());
    assert!(!quota_exceeded(tmp.path(), 200).unwrap());
}

#[test]
fn missing_dirs_count_as_zero_bytes() {
    let tmp = TempDir::new().unwrap();
    assert_eq!(session_checkpoint_bytes(tmp.path()).unwrap(), 0);
    assert!(!quota_exceeded(tmp.path(), DEFAULT_SESSION_CHECKPOINT_QUOTA_BYTES).unwrap());
}

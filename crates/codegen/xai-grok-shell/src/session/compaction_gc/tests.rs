use super::*;

use agent_client_protocol as acp;
use tempfile::TempDir;
use xai_grok_sampling_types::{
    CheckpointIdentity, CheckpointReplayMaterial, ConversationItem, RESPONSES_COMPACTION_CONTRACT,
    ResponsesCompactionMode, ServerResponsesCheckpoint, TokenSeedSource, TrustedPromptEnvelope,
};

use crate::extensions::notification::{
    CompactionCheckpointKind, SessionNotification as XaiNotification,
    SessionUpdate as XaiSessionUpdate,
};
use crate::session::storage::responses_compaction::{
    CompactionCheckpointFile, ConversationAppendCommitted, ConversationAppendPrepared,
    PersistedChatEntry, ResponsesCompactionSegmentStaging, marker_for_wrapper,
    portable_history_bytes, portable_history_digest, stage_compaction_segment_durable,
    write_checkpoint_durable, write_history_durable,
};

const SESSION_ID: &str = "gc-test";

fn portable_fixture() -> Vec<ConversationItem> {
    vec![
        ConversationItem::base_instructions("base instructions"),
        ConversationItem::user("first prompt"),
        ConversationItem::assistant("first answer"),
    ]
}

fn envelope_fixture() -> TrustedPromptEnvelope {
    TrustedPromptEnvelope {
        base_instructions_sha256: "base-hash".into(),
        memory_revision: None,
        envelope_fingerprint: "envelope-fp".into(),
        wire_prompt_sha256: "wire-hash".into(),
    }
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

fn wrapper_fixture(checkpoint_id: &str, prior: Option<&str>) -> ServerResponsesCheckpoint {
    let portable = portable_fixture();
    let digest = portable_history_digest(&portable).unwrap();
    let bytes = portable_history_bytes(&portable).unwrap();
    ServerResponsesCheckpoint {
        checkpoint_id: checkpoint_id.into(),
        operation_id: format!("op-{checkpoint_id}"),
        prompt_index: 3,
        created_at: chrono::Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionMode {
            name: "default".into(),
            detail: None,
        },
        branch_id: "branch-1".into(),
        identity: identity_fixture(prior),
        retained_prefix: vec![ConversationItem::user("first")],
        compaction_item: serde_json::json!({
            "type": "compaction",
            "encrypted_content": "opaque"
        }),
        portable_history_path: format!("compaction_checkpoints/{checkpoint_id}.json"),
        portable_history_sha256: digest,
        portable_history_bytes: bytes.len() as u64,
        checkpoint_token_seed: 42,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        prior_checkpoint_id: prior.map(str::to_owned),
        memory_revision: None,
    }
}

/// Write a fully valid current sidecar, optionally linked to a prior
/// checkpoint, and return its live wrapper.
fn write_sidecar(
    session_dir: &Path,
    checkpoint_id: &str,
    prior: Option<&str>,
) -> ServerResponsesCheckpoint {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(checkpoint_id, prior);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    let sidecar =
        CompactionCheckpointFile::new(wrapper.clone(), material, portable, None, Vec::new())
            .unwrap();
    write_checkpoint_durable(session_dir, &wrapper.portable_history_path, &sidecar).unwrap();
    wrapper
}

/// Write a valid unreferenced current sidecar with an old mtime.
fn write_orphan_sidecar(session_dir: &Path, checkpoint_id: &str) {
    write_sidecar(session_dir, checkpoint_id, None);
    set_mtime_old(
        &session_dir
            .join(CHECKPOINT_DIR)
            .join(format!("{checkpoint_id}.json")),
    );
}

fn set_mtime_old(path: &Path) {
    let old = std::time::SystemTime::now() - Duration::from_secs(48 * 60 * 60);
    filetime::set_file_mtime(path, filetime::FileTime::from_system_time(old)).unwrap();
}

fn write_orphan_staging(session_dir: &Path, checkpoint_id: &str) {
    let wrapper = wrapper_fixture(checkpoint_id, None);
    let staging = ResponsesCompactionSegmentStaging::new(
        checkpoint_id,
        &wrapper.operation_id,
        &wrapper.branch_id,
        wrapper.wrapper_digest(),
        vec![ConversationItem::user("staged turn")],
        "summary",
        xai_chat_state::CompactionDetail::Balanced,
        "2026-01-01T00:00:00Z",
    )
    .unwrap();
    stage_compaction_segment_durable(session_dir, &staging).unwrap();
    set_mtime_old(
        &session_dir
            .join(STAGING_SUBDIR)
            .join(format!("{checkpoint_id}.json")),
    );
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
        marker_for_wrapper(&wrapper_fixture(checkpoint_id, None)),
    )))
}

fn journal_updates(checkpoint_id: &str) -> Vec<SessionUpdate> {
    let wrapper = wrapper_fixture(checkpoint_id, None);
    let prepared = ConversationAppendPrepared {
        operation_id: format!("{}-tail-1", wrapper.operation_id),
        checkpoint_id: wrapper.checkpoint_id.clone(),
        branch_id: wrapper.branch_id.clone(),
        sequence: 1,
        prompt_index: wrapper.prompt_index,
        item: ConversationItem::user("tail user"),
    };
    vec![
        xai(XaiSessionUpdate::ConversationAppendPrepared(Box::new(
            prepared.clone(),
        ))),
        xai(XaiSessionUpdate::ConversationAppendCommitted(
            ConversationAppendCommitted::from(&prepared),
        )),
    ]
}

fn chat_history_with_wrapper(session_dir: &Path, checkpoint_id: &str) {
    let wrapper = wrapper_fixture(checkpoint_id, None);
    let entries = vec![PersistedChatEntry::Item(
        ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper)),
    )];
    write_history_durable(&session_dir.join(storage::CHAT_HISTORY_FILE), &entries).unwrap();
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
    write_sidecar(tmp.path(), "cp-wrapper", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    chat_history_with_wrapper(tmp.path(), "cp-wrapper");

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-wrapper.json")
            .exists()
    );
    assert!(
        !tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists()
    );
    assert!(report.referenced_bytes > 0);
    assert_eq!(report.deleted_bytes, report.orphan_bytes);
    assert_eq!(report.retained_orphan_bytes, 0);
}

#[tokio::test]
async fn sidecar_referenced_by_marker_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-marker", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    write_updates(tmp.path(), &[marker_update("cp-marker")]);

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-marker.json")
            .exists()
    );
    assert!(
        !tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists()
    );
    assert_eq!(
        report.referenced_bytes,
        report.scanned_bytes - report.orphan_bytes
    );
}

#[tokio::test]
async fn marker_prior_checkpoint_chain_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-prior", None);
    let latest = write_sidecar(tmp.path(), "cp-latest", Some("cp-prior"));
    write_updates(
        tmp.path(),
        &[xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
            marker_for_wrapper(&latest),
        )))],
    );

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-latest.json")
            .exists()
    );
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-prior.json")
            .exists()
    );
    assert_eq!(report.deleted_bytes, 0);
}

#[tokio::test]
async fn prior_chain_transitive_closure_retains_grandparent() {
    let tmp = TempDir::new().unwrap();
    let head = write_sidecar(tmp.path(), "cp-a", Some("cp-b"));
    write_sidecar(tmp.path(), "cp-b", Some("cp-c"));
    write_sidecar(tmp.path(), "cp-c", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    write_updates(
        tmp.path(),
        &[xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
            marker_for_wrapper(&head),
        )))],
    );

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-a.json").exists());
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-b.json").exists());
    assert!(
        tmp.path().join(CHECKPOINT_DIR).join("cp-c.json").exists(),
        "grandparent sidecar must survive via the transitive prior chain"
    );
    assert!(
        !tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists()
    );
    assert_eq!(report.files_deleted, 1);
}

#[tokio::test]
async fn corrupt_current_sidecar_aborts_gc() {
    let tmp = TempDir::new().unwrap();
    let head = write_sidecar(tmp.path(), "cp-a", Some("cp-b"));
    let path = tmp.path().join(CHECKPOINT_DIR).join("cp-a.json");
    std::fs::write(&path, b"{not json").unwrap();
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    write_updates(
        tmp.path(),
        &[xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
            marker_for_wrapper(&head),
        )))],
    );

    let result = gc_session_compaction_artifacts(
        tmp.path(),
        GcOptions {
            grace_period: Duration::ZERO,
            dry_run: false,
        },
    )
    .await;
    assert!(result.is_err(), "corrupt chain sidecar must abort the GC");
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists(),
        "an aborted GC deletes nothing"
    );
}

#[tokio::test]
async fn deserializable_chain_tampering_aborts_before_deletion() {
    let tmp = TempDir::new().unwrap();
    let head = write_sidecar(tmp.path(), "cp-a", Some("cp-b"));
    write_sidecar(tmp.path(), "cp-b", Some("cp-c"));
    write_sidecar(tmp.path(), "cp-c", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");

    let middle_path = tmp.path().join(CHECKPOINT_DIR).join("cp-b.json");
    let mut middle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&middle_path).unwrap()).unwrap();
    // Keep the payload valid JSON and structurally deserializable while
    // corrupting one strong prior-chain binding.
    middle["wrapper"]["prior_checkpoint_id"] = serde_json::Value::Null;
    std::fs::write(&middle_path, serde_json::to_vec(&middle).unwrap()).unwrap();
    write_updates(
        tmp.path(),
        &[xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
            marker_for_wrapper(&head),
        )))],
    );

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
    assert!(tmp.path().join(CHECKPOINT_DIR).join("cp-c.json").exists());
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists(),
        "failed-closed chain validation must delete nothing"
    );
}

#[tokio::test]
async fn unknown_marker_kind_aborts_gc_before_deletion() {
    let tmp = TempDir::new().unwrap();
    let wrapper = write_sidecar(tmp.path(), "cp-live", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    let mut marker = marker_for_wrapper(&wrapper);
    marker.kind = CompactionCheckpointKind::Unknown;
    write_updates(
        tmp.path(),
        &[xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
            marker,
        )))],
    );

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
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists()
    );
}

#[tokio::test]
async fn unreadable_reachable_sidecar_aborts_gc_instead_of_cutting_the_chain() {
    let tmp = TempDir::new().unwrap();
    let head = write_sidecar(tmp.path(), "cp-a", Some("cp-b"));
    write_sidecar(tmp.path(), "cp-b", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    let path = tmp.path().join(CHECKPOINT_DIR).join("cp-a.json");
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    write_updates(
        tmp.path(),
        &[xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
            marker_for_wrapper(&head),
        )))],
    );

    let result = gc_session_compaction_artifacts(
        tmp.path(),
        GcOptions {
            grace_period: Duration::ZERO,
            dry_run: false,
        },
    )
    .await;
    assert!(
        result.is_err(),
        "non-NotFound read errors must abort the GC"
    );
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists(),
        "a failed-closed scan deletes no unrelated orphan"
    );
}

#[tokio::test]
async fn versioned_sidecar_fails_closed_without_deletions() {
    let tmp = TempDir::new().unwrap();
    let wrapper = write_sidecar(tmp.path(), "cp-versioned", None);
    let dir = tmp.path().join(CHECKPOINT_DIR);
    let path = dir.join("cp-versioned.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["schema_version"] = serde_json::json!(2);
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    write_updates(
        tmp.path(),
        &[xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
            marker_for_wrapper(&wrapper),
        )))],
    );

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
    assert!(dir.join("cp-orphan.json").exists());
}

#[tokio::test]
async fn sidecar_referenced_by_journal_tail_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-journal", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    write_updates(tmp.path(), &journal_updates("cp-journal"));

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-journal.json")
            .exists()
    );
    assert!(
        !tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists()
    );
    assert_eq!(report.deleted_bytes, report.orphan_bytes);
}

#[tokio::test]
async fn orphan_staging_deleted_but_referenced_staging_retained() {
    let tmp = TempDir::new().unwrap();
    write_orphan_staging(tmp.path(), "cp-staged");
    write_orphan_staging(tmp.path(), "cp-orphan");
    write_updates(tmp.path(), &[marker_update("cp-staged")]);

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(
        tmp.path()
            .join(STAGING_SUBDIR)
            .join("cp-staged.json")
            .exists()
    );
    assert!(
        !tmp.path()
            .join(STAGING_SUBDIR)
            .join("cp-orphan.json")
            .exists()
    );
    assert_eq!(report.files_deleted, 1);
}

#[tokio::test]
async fn orphan_younger_than_grace_is_retained() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-fresh", None);
    let path = tmp.path().join(CHECKPOINT_DIR).join("cp-fresh.json");
    let bytes = std::fs::metadata(&path).unwrap().len();

    let report = gc_session_compaction_artifacts(tmp.path(), GcOptions::default())
        .await
        .unwrap();
    assert!(path.exists());
    assert_eq!(report.orphan_bytes, bytes);
    assert_eq!(report.retained_orphan_bytes, bytes);
    assert_eq!(report.deleted_bytes, 0);
    assert_eq!(report.files_deleted, 0);
}

#[tokio::test]
async fn orphan_older_than_grace_is_deleted() {
    let tmp = TempDir::new().unwrap();
    write_orphan_sidecar(tmp.path(), "cp-old");
    let path = tmp.path().join(CHECKPOINT_DIR).join("cp-old.json");
    let bytes = std::fs::metadata(&path).unwrap().len();
    let report = gc(tmp.path(), Duration::from_secs(24 * 60 * 60)).await;
    assert!(!path.exists());
    assert_eq!(report.deleted_bytes, bytes);
    assert_eq!(report.files_deleted, 1);
    assert_eq!(report.retained_orphan_bytes, 0);
}

#[tokio::test]
async fn dry_run_deletes_nothing_but_reports_orphans() {
    let tmp = TempDir::new().unwrap();
    write_orphan_sidecar(tmp.path(), "cp-old");
    let path = tmp.path().join(CHECKPOINT_DIR).join("cp-old.json");
    let bytes = std::fs::metadata(&path).unwrap().len();
    let report = gc_session_compaction_artifacts(
        tmp.path(),
        GcOptions {
            grace_period: Duration::ZERO,
            dry_run: true,
        },
    )
    .await
    .unwrap();
    assert!(path.exists());
    assert_eq!(report.orphan_bytes, bytes);
    assert_eq!(report.deleted_bytes, 0);
    assert_eq!(report.files_deleted, 0);
}

#[tokio::test]
async fn corrupt_updates_jsonl_aborts_with_no_deletions() {
    let tmp = TempDir::new().unwrap();
    write_orphan_sidecar(tmp.path(), "cp-old");
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
    write_orphan_sidecar(tmp.path(), "cp-old");
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
async fn builtin_sidecar_is_retained() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join(CHECKPOINT_DIR);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("builtin-checkpoint.json");
    let builtin = crate::extensions::notification::CompactionCheckpointFile {
        kind: CompactionCheckpointKind::Builtin,
        checkpoint_id: "builtin-checkpoint".into(),
        prompt_index_at_compaction: 1,
        compacted_history: vec![ConversationItem::user("builtin summary")],
        created_at: "2026-01-01T00:00:00Z".into(),
        original_user_info: None,
        reread_file_paths: Vec::new(),
    };
    std::fs::write(&path, serde_json::to_vec(&builtin).unwrap()).unwrap();

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(path.exists());
    assert_eq!(report.files_deleted, 0);
    assert_eq!(report.orphan_bytes, 0);
}

#[tokio::test]
async fn published_segment_marker_retains_its_checkpoint() {
    let tmp = TempDir::new().unwrap();
    write_sidecar(tmp.path(), "cp-published", None);
    write_orphan_sidecar(tmp.path(), "cp-orphan");
    let compaction_dir = tmp.path().join("compaction");
    std::fs::create_dir_all(&compaction_dir).unwrap();
    std::fs::write(
        compaction_dir.join("segment_001.md"),
        "<!-- responses-compaction-checkpoint:cp-published -->\n# Segment\n",
    )
    .unwrap();

    let report = gc(tmp.path(), Duration::ZERO).await;
    assert!(
        tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-published.json")
            .exists()
    );
    assert!(
        !tmp.path()
            .join(CHECKPOINT_DIR)
            .join("cp-orphan.json")
            .exists()
    );
    assert_eq!(report.files_deleted, 1);
}

#[tokio::test]
async fn missing_chat_history_is_empty_not_an_error() {
    let tmp = TempDir::new().unwrap();
    write_orphan_sidecar(tmp.path(), "cp-old");
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
    assert_eq!(
        GrokCompactionQuota::default().bytes(),
        session_checkpoint_quota_bytes()
    );
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

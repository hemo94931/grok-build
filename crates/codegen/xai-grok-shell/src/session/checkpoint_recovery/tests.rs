use super::*;

use agent_client_protocol as acp;
use tempfile::TempDir;
use xai_grok_sampling_types::{CheckpointIdentityV1, ResponsesCompactionModeV1, TokenSeedSource};

use crate::extensions::notification::{
    CompactionCheckpointInfo, SessionNotification as XaiNotification,
};
use crate::session::storage::responses_compaction::{
    CompactionCheckpointFileV2, ConversationAppendCommittedV2, ConversationAppendPreparedV2,
    ResponsesCompactionSegmentStagingV1, marker_for_wrapper, portable_history_digest,
    stage_compaction_segment_durable, write_checkpoint_v2_durable,
};

const SESSION_ID: &str = "recovery-test";

fn portable_fixture() -> Vec<ConversationItem> {
    vec![
        ConversationItem::system("base instructions"),
        ConversationItem::user("first prompt"),
        ConversationItem::assistant("first answer"),
        ConversationItem::user("second prompt"),
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

fn wrapper_fixture(
    checkpoint_id: &str,
    operation_id: &str,
    branch_id: &str,
    portable: &[ConversationItem],
) -> ServerResponsesCheckpointV1 {
    let digest = portable_history_digest(portable).unwrap();
    let bytes =
        crate::session::storage::responses_compaction::portable_history_bytes(portable).unwrap();
    ServerResponsesCheckpointV1 {
        schema_version: 1,
        checkpoint_id: checkpoint_id.into(),
        operation_id: operation_id.into(),
        prompt_index: 1,
        created_at: chrono::Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionModeV1 {
            name: "default".into(),
            detail: None,
        },
        branch_id: branch_id.into(),
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

/// Write a V2 sidecar for `wrapper` (which must carry the default branch) and
/// return the file.
fn write_sidecar(
    session_dir: &Path,
    wrapper: &ServerResponsesCheckpointV1,
    portable: &[ConversationItem],
) -> CompactionCheckpointFileV2 {
    let file =
        CompactionCheckpointFileV2::new(wrapper.clone(), portable.to_vec(), None, Vec::new())
            .unwrap();
    write_checkpoint_v2_durable(session_dir, &wrapper.portable_history_path, &file).unwrap();
    file
}

fn write_staging(session_dir: &Path, checkpoint_id: &str, items: Vec<ConversationItem>) {
    let staging = ResponsesCompactionSegmentStagingV1::new(
        checkpoint_id,
        items,
        "summary",
        xai_chat_state::CompactionDetail::Balanced,
        "2024-01-01T00:00:00Z",
    )
    .unwrap();
    stage_compaction_segment_durable(session_dir, &staging).unwrap();
}

fn xai(update: XaiSessionUpdate) -> SessionUpdate {
    SessionUpdate::Xai(Box::new(XaiNotification {
        session_id: acp::SessionId::new(SESSION_ID),
        update,
        meta: None,
    }))
}

fn marker_update(wrapper: &ServerResponsesCheckpointV1) -> SessionUpdate {
    xai(XaiSessionUpdate::CompactionCheckpoint(Box::new(
        marker_for_wrapper(wrapper),
    )))
}

fn user_chunk(text: &str, prompt_index: usize) -> SessionUpdate {
    SessionUpdate::Acp(Box::new(acp::SessionNotification::new(
        acp::SessionId::new(SESSION_ID),
        acp::SessionUpdate::UserMessageChunk(
            acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(
                text.to_string(),
            )))
            .meta(
                serde_json::json!({ "promptIndex": prompt_index })
                    .as_object()
                    .cloned(),
            ),
        ),
    )))
}

fn agent_chunk(text: &str) -> SessionUpdate {
    SessionUpdate::Acp(Box::new(acp::SessionNotification::new(
        acp::SessionId::new(SESSION_ID),
        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
            acp::TextContent::new(text.to_string()),
        ))),
    )))
}

fn tool_call_update(id: &str) -> SessionUpdate {
    SessionUpdate::Acp(Box::new(acp::SessionNotification::new(
        acp::SessionId::new(SESSION_ID),
        acp::SessionUpdate::ToolCall(acp::ToolCall::new(id.to_string(), "tool title")),
    )))
}

fn thought_chunk() -> SessionUpdate {
    SessionUpdate::Acp(Box::new(acp::SessionNotification::new(
        acp::SessionId::new(SESSION_ID),
        acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
            acp::TextContent::new("thinking".to_string()),
        ))),
    )))
}

fn prepared(
    wrapper: &ServerResponsesCheckpointV1,
    sequence: u64,
    prompt_index: usize,
    item: ConversationItem,
) -> ConversationAppendPreparedV2 {
    ConversationAppendPreparedV2 {
        operation_id: format!("{}-tail-{sequence}", wrapper.operation_id),
        checkpoint_id: wrapper.checkpoint_id.clone(),
        branch_id: wrapper.branch_id.clone(),
        sequence,
        prompt_index,
        item,
    }
}

fn prepared_update(prepared: ConversationAppendPreparedV2) -> SessionUpdate {
    xai(XaiSessionUpdate::ConversationAppendPreparedV2(Box::new(
        prepared,
    )))
}

fn committed_update(prepared: &ConversationAppendPreparedV2) -> SessionUpdate {
    xai(XaiSessionUpdate::ConversationAppendCommittedV2(
        ConversationAppendCommittedV2::from(prepared),
    ))
}

fn scan(
    session_dir: &Path,
    updates: &[SessionUpdate],
    wrapper: &ServerResponsesCheckpointV1,
    trusted_base: Option<&str>,
) -> V1RecoveryScan {
    scan_v1_checkpoint_recovery(session_dir, updates, wrapper, trusted_base)
}

#[test]
fn lossless_via_sidecar_with_rotated_live_branch() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let stored = wrapper_fixture("cp1", "op1", "branch-a", &portable);
    let file = write_sidecar(tmp.path(), &stored, &portable);
    // Rewind/fork rotated only the live branch; every other immutable replay
    // field is identical.
    let mut live = file.wrapper.clone();
    live.branch_id = "branch-b".into();

    let result = scan(tmp.path(), &[], &live, None);
    match result.outcome {
        V1RecoveryOutcome::Lossless {
            history,
            source,
            original_digest,
            ..
        } => {
            assert_eq!(source, LosslessRecoverySource::SidecarV2);
            assert_eq!(history.len(), portable.len());
            assert_eq!(original_digest, portable_history_digest(&portable).unwrap());
        }
        other => panic!("expected lossless, got {}", other.class_str()),
    }
    assert_eq!(result.journal_tail, JournalTailRecovery::Empty);
}

#[test]
fn lossless_via_sidecar_with_committed_journal_tail() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp2", "op2", "branch-a", &portable);
    write_sidecar(tmp.path(), &wrapper, &portable);

    let tail_user = prepared(&wrapper, 1, 1, ConversationItem::user("tail user"));
    let tail_assistant = prepared(
        &wrapper,
        2,
        1,
        ConversationItem::assistant("tail assistant"),
    );
    let updates = vec![
        marker_update(&wrapper),
        prepared_update(tail_user.clone()),
        committed_update(&tail_user),
        prepared_update(tail_assistant.clone()),
        committed_update(&tail_assistant),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, None);
    match result.outcome {
        V1RecoveryOutcome::Lossless { history, .. } => {
            assert_eq!(history.len(), portable.len() + 2);
            assert_eq!(history[portable.len()].text_content(), "tail user");
            assert_eq!(history[portable.len() + 1].text_content(), "tail assistant");
        }
        other => panic!("expected lossless, got {}", other.class_str()),
    }
    assert_eq!(
        result.journal_tail,
        JournalTailRecovery::Lossless { items: 2 }
    );
}

#[test]
fn lossless_via_staging_anchored_on_live_wrapper() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp3", "op3", "branch-a", &portable);
    // No sidecar on disk: only the staging payload remains.
    write_staging(tmp.path(), "cp3", portable.clone());
    let updates = vec![marker_update(&wrapper)];

    let result = scan(tmp.path(), &updates, &wrapper, None);
    match result.outcome {
        V1RecoveryOutcome::Lossless {
            history,
            source,
            original_digest,
            ..
        } => {
            assert_eq!(source, LosslessRecoverySource::SegmentStagingV1);
            assert_eq!(history.len(), portable.len());
            assert_eq!(original_digest, wrapper.portable_history_sha256);
        }
        other => panic!("expected lossless, got {}", other.class_str()),
    }
}

#[test]
fn orphan_staging_with_mismatched_digest_is_ignored() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp4", "op4", "branch-a", &portable);
    // Staging from a superseded operation: items do not match the live
    // wrapper's portable digest and must never be used.
    write_staging(
        tmp.path(),
        "cp4",
        vec![
            ConversationItem::system("other base"),
            ConversationItem::user("other"),
        ],
    );
    let updates = vec![
        user_chunk("hello", 0),
        agent_chunk("hi"),
        marker_update(&wrapper),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, Some("base instructions"));
    assert!(matches!(
        result.outcome,
        V1RecoveryOutcome::LossySalvageAvailable { .. }
    ));
}

#[test]
fn salvage_records_omissions_and_fresh_digest() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp5", "op5", "branch-a", &portable);
    let tail_item = prepared(&wrapper, 1, 1, ConversationItem::user("tail user"));
    let updates = vec![
        user_chunk("pre-compact question", 0),
        thought_chunk(),
        tool_call_update("call-1"),
        agent_chunk("pre-compact answer"),
        marker_update(&wrapper),
        prepared_update(tail_item.clone()),
        committed_update(&tail_item),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, Some("trusted base"));
    match result.outcome {
        V1RecoveryOutcome::LossySalvageAvailable {
            history,
            salvage_digest,
            omissions,
        } => {
            // trusted base + lossy text + lossless typed tail
            assert_eq!(history[0].text_content(), "trusted base");
            assert!(matches!(history[0], ConversationItem::System(_)));
            assert_eq!(history[1].text_content(), "pre-compact question");
            assert_eq!(history[2].text_content(), "pre-compact answer");
            assert_eq!(history[3].text_content(), "tail user");
            // Brand-new digest; never the old portable digest.
            assert_eq!(salvage_digest, portable_history_digest(&history).unwrap());
            assert_ne!(salvage_digest, wrapper.portable_history_sha256);
            assert!(omissions.contains(&SalvageOmission {
                category: SalvageOmissionCategory::PreCompactToolCalls,
                count: 1,
            }));
            assert!(omissions.contains(&SalvageOmission {
                category: SalvageOmissionCategory::PreCompactReasoning,
                count: 1,
            }));
        }
        other => panic!("expected salvage, got {}", other.class_str()),
    }
    assert_eq!(
        result.journal_tail,
        JournalTailRecovery::Lossless { items: 1 }
    );
}

#[test]
fn uncommitted_prepared_tail_becomes_omission_on_verified_portable() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp6", "op6", "branch-a", &portable);
    write_sidecar(tmp.path(), &wrapper, &portable);

    let closed = prepared(&wrapper, 1, 1, ConversationItem::user("closed tail"));
    let open = prepared(
        &wrapper,
        2,
        1,
        ConversationItem::assistant("uncommitted tail"),
    );
    let updates = vec![
        marker_update(&wrapper),
        prepared_update(closed.clone()),
        committed_update(&closed),
        prepared_update(open),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, None);
    match result.outcome {
        V1RecoveryOutcome::LossySalvageAvailable {
            history,
            omissions,
            salvage_digest,
        } => {
            // Verified portable base + closed tail prefix only.
            assert_eq!(history.len(), portable.len() + 1);
            assert_eq!(history[portable.len()].text_content(), "closed tail");
            assert_eq!(
                omissions,
                vec![SalvageOmission {
                    category: SalvageOmissionCategory::TailItems,
                    count: 1,
                }]
            );
            assert_eq!(salvage_digest, portable_history_digest(&history).unwrap());
        }
        other => panic!("expected salvage, got {}", other.class_str()),
    }
    assert_eq!(
        result.journal_tail,
        JournalTailRecovery::Partial {
            recovered: 1,
            omitted: 1,
        }
    );
}

#[test]
fn conflicting_prepared_records_make_tail_unusable() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp7", "op7", "branch-a", &portable);
    write_sidecar(tmp.path(), &wrapper, &portable);

    let first = prepared(&wrapper, 1, 1, ConversationItem::user("version one"));
    let mut conflicting = first.clone();
    conflicting.item = ConversationItem::user("version two");
    let updates = vec![
        marker_update(&wrapper),
        prepared_update(first.clone()),
        committed_update(&first),
        prepared_update(conflicting),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, None);
    assert!(matches!(
        result.journal_tail,
        JournalTailRecovery::Unusable { .. }
    ));
    // Tail is dropped entirely; the verified portable base is preserved with
    // the dropped records reported as omissions.
    match result.outcome {
        V1RecoveryOutcome::LossySalvageAvailable {
            history, omissions, ..
        } => {
            assert_eq!(history.len(), portable.len());
            assert!(omissions.iter().any(|omission| {
                omission.category == SalvageOmissionCategory::TailItems && omission.count >= 1
            }));
        }
        other => panic!("expected salvage, got {}", other.class_str()),
    }
}

#[test]
fn unrecoverable_when_no_source_exists() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp8", "op8", "branch-a", &portable);

    let result = scan(tmp.path(), &[], &wrapper, Some("trusted base"));
    match result.outcome {
        V1RecoveryOutcome::Unrecoverable(error) => {
            assert_eq!(error.reason_code, RecoveryReasonCode::NoRecoverySource);
        }
        other => panic!("expected unrecoverable, got {}", other.class_str()),
    }
}

#[test]
fn unrecoverable_without_trusted_base() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp9", "op9", "branch-a", &portable);
    let updates = vec![
        user_chunk("question", 0),
        agent_chunk("answer"),
        marker_update(&wrapper),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, None);
    match result.outcome {
        V1RecoveryOutcome::Unrecoverable(error) => {
            assert_eq!(
                error.reason_code,
                RecoveryReasonCode::TrustedBaseUnavailable
            );
        }
        other => panic!("expected unrecoverable, got {}", other.class_str()),
    }
}

#[test]
fn missing_marker_with_journal_records_fails_closed() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp10", "op10", "branch-a", &portable);
    let tail_item = prepared(&wrapper, 1, 1, ConversationItem::user("tail user"));
    // Journal records exist (crash between CAS and marker write) but no
    // sidecar/staging: the pre-compact boundary is ambiguous.
    let updates = vec![
        user_chunk("question", 0),
        prepared_update(tail_item.clone()),
        committed_update(&tail_item),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, Some("trusted base"));
    match result.outcome {
        V1RecoveryOutcome::Unrecoverable(error) => {
            assert_eq!(error.reason_code, RecoveryReasonCode::MarkerMissing);
        }
        other => panic!("expected unrecoverable, got {}", other.class_str()),
    }
}

#[test]
fn fingerprint_changes_with_recovery_sources() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp11", "op11", "branch-a", &portable);
    let updates = vec![marker_update(&wrapper)];

    let before = scan(tmp.path(), &updates, &wrapper, None);
    write_sidecar(tmp.path(), &wrapper, &portable);
    let after = scan(tmp.path(), &updates, &wrapper, None);
    assert_ne!(before.source_fingerprint, after.source_fingerprint);

    // A second scan with unchanged sources reproduces the fingerprint, which
    // is what suppresses per-turn rescans.
    let again = scan(tmp.path(), &updates, &wrapper, None);
    assert_eq!(after.source_fingerprint, again.source_fingerprint);
}

#[test]
fn orphaned_committed_records_become_omissions() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp-orphan", "op", "branch-a", &portable);
    write_sidecar(tmp.path(), &wrapper, &portable);

    let closed = prepared(&wrapper, 1, 1, ConversationItem::user("closed"));
    let orphan = prepared(&wrapper, 2, 1, ConversationItem::user("orphan"));
    // Only the committed half of the second record survives (the prepared
    // line was lost): the tail item is unrecoverable and must be counted.
    let updates = vec![
        marker_update(&wrapper),
        prepared_update(closed.clone()),
        committed_update(&closed),
        committed_update(&orphan),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, None);
    assert_eq!(
        result.journal_tail,
        JournalTailRecovery::Partial {
            recovered: 1,
            omitted: 1,
        }
    );
    match result.outcome {
        V1RecoveryOutcome::LossySalvageAvailable {
            history, omissions, ..
        } => {
            assert_eq!(history.len(), portable.len() + 1);
            assert!(omissions.iter().any(|omission| {
                omission.category == SalvageOmissionCategory::TailItems && omission.count == 1
            }));
        }
        other => panic!("expected salvage, got {}", other.class_str()),
    }
}

#[test]
fn duplicate_committed_sequence_makes_tail_unusable() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp-dup", "op", "branch-a", &portable);
    write_sidecar(tmp.path(), &wrapper, &portable);

    // Two distinct operation keys both claim sequence 1.
    let first = prepared(&wrapper, 1, 1, ConversationItem::user("first"));
    let mut second = prepared(&wrapper, 1, 1, ConversationItem::assistant("second"));
    second.operation_id = format!("{}-other", wrapper.operation_id);
    let updates = vec![
        marker_update(&wrapper),
        prepared_update(first.clone()),
        committed_update(&first),
        prepared_update(second.clone()),
        committed_update(&second),
    ];

    let result = scan(tmp.path(), &updates, &wrapper, None);
    assert!(matches!(
        result.journal_tail,
        JournalTailRecovery::Unusable { .. }
    ));
}

#[test]
fn marker_with_mismatched_prompt_index_is_no_boundary() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp-marker", "op", "branch-a", &portable);
    // Marker exists but binds to a different prompt index: the pre-compact
    // update boundary is ambiguous and cannot anchor salvage.
    let mut marker = marker_update(&wrapper);
    if let SessionUpdate::Xai(notification) = &mut marker
        && let XaiSessionUpdate::CompactionCheckpoint(info) = &mut notification.update
    {
        info.prompt_index_at_compaction = wrapper.prompt_index + 99;
    }
    let updates = vec![user_chunk("question", 0), agent_chunk("answer"), marker];

    let result = scan(tmp.path(), &updates, &wrapper, Some("trusted base"));
    match result.outcome {
        V1RecoveryOutcome::Unrecoverable(error) => {
            assert_eq!(error.reason_code, RecoveryReasonCode::MarkerMissing);
        }
        other => panic!("expected unrecoverable, got {}", other.class_str()),
    }
}

#[test]
fn salvage_without_any_user_turn_is_unrecoverable() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp-empty", "op", "branch-a", &portable);
    // Only the marker: no chunks survived, no tail. The assembled salvage
    // would be a bare System item, which builtin compaction cannot accept.
    let updates = vec![marker_update(&wrapper)];

    let result = scan(tmp.path(), &updates, &wrapper, Some("trusted base"));
    match result.outcome {
        V1RecoveryOutcome::Unrecoverable(error) => {
            assert_eq!(
                error.reason_code,
                RecoveryReasonCode::SalvageStructureInvalid
            );
        }
        other => panic!("expected unrecoverable, got {}", other.class_str()),
    }
}

#[test]
fn fingerprint_changes_with_journal_growth_and_marker() {
    let tmp = TempDir::new().unwrap();
    let portable = portable_fixture();
    let wrapper = wrapper_fixture("cp-fp", "op", "branch-a", &portable);
    write_sidecar(tmp.path(), &wrapper, &portable);

    let no_marker = scan(tmp.path(), &[], &wrapper, None);
    let with_marker = scan(tmp.path(), &[marker_update(&wrapper)], &wrapper, None);
    assert_ne!(
        no_marker.source_fingerprint, with_marker.source_fingerprint,
        "marker appearance must change the fingerprint"
    );

    let tail_item = prepared(&wrapper, 1, 1, ConversationItem::user("tail"));
    let with_tail = scan(
        tmp.path(),
        &[
            marker_update(&wrapper),
            prepared_update(tail_item.clone()),
            committed_update(&tail_item),
        ],
        &wrapper,
        None,
    );
    assert_ne!(
        with_marker.source_fingerprint, with_tail.source_fingerprint,
        "journal growth must change the fingerprint"
    );
}

#[test]
fn status_mapping_and_record_readback() {
    let wrapper = wrapper_fixture("cp12", "op12", "branch-a", &portable_fixture());
    let record = CheckpointRecoveryRecord::new(
        &wrapper,
        CheckpointRecoveryStatus::SalvageRequired {
            salvage_digest: "digest".into(),
            omissions: vec![SalvageOmission {
                category: SalvageOmissionCategory::PreCompactToolCalls,
                count: 3,
            }],
        },
        "fingerprint".into(),
    );
    let updates = vec![
        xai(XaiSessionUpdate::CheckpointRecovery(Box::new(
            record.clone(),
        ))),
        xai(XaiSessionUpdate::CheckpointRecovery(Box::new(
            CheckpointRecoveryRecord {
                status: CheckpointRecoveryStatus::Pending,
                ..record.clone()
            },
        ))),
    ];
    let latest = latest_recovery_record(&updates, "cp12", "op12").unwrap();
    assert_eq!(latest.status, CheckpointRecoveryStatus::Pending);
    assert!(latest_recovery_record(&updates, "other", "op12").is_none());
    // The dedup key is checkpoint ID + operation ID: a record from an older
    // operation must not match the newer one.
    assert!(latest_recovery_record(&updates, "cp12", "op-old").is_none());
}

#[test]
fn status_for_outcome_maps_all_classes() {
    let lossless = V1RecoveryOutcome::Lossless {
        history: portable_fixture(),
        source: LosslessRecoverySource::SidecarV2,
        original_digest: "d".into(),
        portable_len: 2,
    };
    assert_eq!(
        status_for_outcome(&lossless),
        CheckpointRecoveryStatus::Pending
    );
    let salvage = V1RecoveryOutcome::LossySalvageAvailable {
        history: portable_fixture(),
        salvage_digest: "s".into(),
        omissions: vec![],
    };
    assert!(matches!(
        status_for_outcome(&salvage),
        CheckpointRecoveryStatus::SalvageRequired { .. }
    ));
    let unrecoverable = V1RecoveryOutcome::Unrecoverable(RecoveryError::new(
        RecoveryReasonCode::NoRecoverySource,
        "detail",
    ));
    assert_eq!(
        status_for_outcome(&unrecoverable),
        CheckpointRecoveryStatus::Unrecoverable {
            reason_code: "no_recovery_source".into(),
        }
    );
}

#[test]
fn migration_eligibility_by_status() {
    let wrapper = wrapper_fixture("cp-elig", "op", "branch-a", &portable_fixture());
    let record = |status| {
        CheckpointRecoveryRecord::new(&wrapper, status, "fingerprint".into())
    };
    assert!(migration_eligible(None));
    assert!(migration_eligible(Some(&record(
        CheckpointRecoveryStatus::Pending
    ))));
    assert!(migration_eligible(Some(&record(
        CheckpointRecoveryStatus::Migrating {
            operation_id: "op-1".into(),
        }
    ))));
    assert!(!migration_eligible(Some(&record(
        CheckpointRecoveryStatus::Migrated
    ))));
    assert!(!migration_eligible(Some(&record(
        CheckpointRecoveryStatus::SalvageRequired {
            salvage_digest: "d".into(),
            omissions: vec![],
        }
    ))));
    assert!(!migration_eligible(Some(&record(
        CheckpointRecoveryStatus::Salvaged {
            operation_id: "op-2".into(),
        }
    ))));
    assert!(!migration_eligible(Some(&record(
        CheckpointRecoveryStatus::Unrecoverable {
            reason_code: "migration_commit_failed".into(),
        }
    ))));
}

#[test]
fn migration_operation_id_reused_only_for_same_fingerprint() {
    let wrapper = wrapper_fixture("cp-mop", "op", "branch-a", &portable_fixture());
    let record = |status, fingerprint: &str| {
        CheckpointRecoveryRecord::new(&wrapper, status, fingerprint.into())
    };
    let migrating = |fingerprint: &str| {
        record(
            CheckpointRecoveryStatus::Migrating {
                operation_id: "v1-migrate-stable".into(),
            },
            fingerprint,
        )
    };
    let updates_same = vec![xai(XaiSessionUpdate::CheckpointRecovery(Box::new(
        migrating("fp-a"),
    )))];
    // Same fingerprint: the stable operation ID is reused.
    assert_eq!(
        migration_operation_id(&updates_same, "cp-mop", "op", "fp-a").as_deref(),
        Some("v1-migrate-stable")
    );
    // Changed source fingerprint: a fresh operation ID must be minted.
    assert_eq!(
        migration_operation_id(&updates_same, "cp-mop", "op", "fp-b"),
        None
    );
    // Non-Migrating latest record: fresh ID.
    let updates_pending = vec![xai(XaiSessionUpdate::CheckpointRecovery(Box::new(
        record(CheckpointRecoveryStatus::Pending, "fp-a"),
    )))];
    assert_eq!(
        migration_operation_id(&updates_pending, "cp-mop", "op", "fp-a"),
        None
    );
    // No record at all: fresh ID.
    assert_eq!(migration_operation_id(&[], "cp-mop", "op", "fp-a"), None);
    // Unrelated checkpoint records do not interfere.
    let other = wrapper_fixture("cp-other", "op", "branch-a", &portable_fixture());
    let updates_other = vec![xai(XaiSessionUpdate::CheckpointRecovery(Box::new(
        CheckpointRecoveryRecord::new(
            &other,
            CheckpointRecoveryStatus::Migrating {
                operation_id: "v1-migrate-other".into(),
            },
            "fp-a".into(),
        ),
    )))];
    assert_eq!(
        migration_operation_id(&updates_other, "cp-mop", "op", "fp-a"),
        None
    );
}

#[test]
fn recovery_record_serde_roundtrip() {
    let wrapper = wrapper_fixture("cp13", "op13", "branch-a", &portable_fixture());
    let record = CheckpointRecoveryRecord::new(
        &wrapper,
        CheckpointRecoveryStatus::Migrating {
            operation_id: "migration-op".into(),
        },
        "fingerprint".into(),
    );
    let update = xai(XaiSessionUpdate::CheckpointRecovery(Box::new(
        record.clone(),
    )));
    let envelope = crate::session::storage::SessionUpdateEnvelope::from_update(&update).unwrap();
    let line = serde_json::to_string(&envelope).unwrap();
    let parsed = crate::session::storage::SessionUpdateEnvelope::from_str(&line).unwrap();
    match parsed {
        SessionUpdate::Xai(notification) => match &notification.update {
            XaiSessionUpdate::CheckpointRecovery(parsed_record) => {
                assert_eq!(**parsed_record, record);
            }
            other => panic!("unexpected update: {other:?}"),
        },
        SessionUpdate::Acp(_) => panic!("unexpected acp update"),
    }
}

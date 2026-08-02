use std::sync::Arc;

use serde_json::json;
use tempfile::tempdir;
use xai_grok_sampling_types::{
    AssistantItem, CheckpointIdentity, CheckpointReplayMaterial, ContentPart, ConversationItem,
    RESPONSES_COMPACTION_CONTRACT, ResponsesCompactionMode, ServerResponsesCheckpoint,
    TokenSeedSource, ToolCall, ToolResultItem, TrustedPromptEnvelope, rs,
};
use xai_grok_shell::extensions::notification::CompactionCheckpointKind;
use xai_grok_shell::session::storage::responses_compaction::{
    CompactionCheckpointFile, ConversationAppendCommitted, ConversationAppendPrepared,
    PersistedChatEntry, PersistedTail, ResponsesCompactionSegmentStaging, TailJournalRecord,
    append_history_tail_durable, marker_for_wrapper, publish_staged_compaction_segment_durable,
    read_checkpoint, read_checkpoint_for_wrapper, read_history, rebuild_updates_only,
    rewind_history_entries, stage_compaction_segment_durable, validate_marker_for_wrapper,
    write_checkpoint_durable, write_history_durable,
};

const CHECKPOINT_ID: &str = "checkpoint-current";
const OPERATION_ID: &str = "operation-current";
const BRANCH_ID: &str = "branch-current";
const RELATIVE_PATH: &str = "compaction_checkpoints/checkpoint-current.json";

fn portable_history() -> Vec<ConversationItem> {
    let mut prompt_0 = ConversationItem::user("prompt 0");
    prompt_0.set_prompt_index(0);
    let mut prompt_1 = ConversationItem::user("prompt 1");
    prompt_1.set_prompt_index(1);
    let mut prompt_2 = ConversationItem::user("prompt 2");
    prompt_2.set_prompt_index(2);
    vec![
        ConversationItem::base_instructions("base instructions"),
        prompt_0,
        ConversationItem::assistant("answer 0"),
        prompt_1,
        ConversationItem::assistant("answer 1"),
        prompt_2,
        ConversationItem::assistant("answer 2"),
    ]
}

fn identity() -> CheckpointIdentity {
    CheckpointIdentity {
        provider_id: "xai".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
        prompt_envelope_fingerprint: "envelope-fingerprint".into(),
        base_instructions_sha256: "base-hash".into(),
        prior_checkpoint_id: None,
        cache_route_fingerprint: Some("route-fingerprint".into()),
    }
}

fn trusted_envelope() -> TrustedPromptEnvelope {
    TrustedPromptEnvelope {
        base_instructions_sha256: "base-hash".into(),
        memory_revision: Some(7),
        envelope_fingerprint: "envelope-fingerprint".into(),
        wire_prompt_sha256: "wire-hash".into(),
    }
}

fn wrapper(portable: &[ConversationItem]) -> ServerResponsesCheckpoint {
    let digest = xai_grok_sampling_types::portable_history_digest(portable).unwrap();
    let bytes = xai_grok_sampling_types::portable_history_bytes(portable).unwrap();
    ServerResponsesCheckpoint {
        checkpoint_id: CHECKPOINT_ID.into(),
        operation_id: OPERATION_ID.into(),
        prompt_index: 3,
        created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
        auto_continue: true,
        mode: ResponsesCompactionMode {
            name: "transcript".into(),
            detail: Some(json!({"location": "transcript.md"})),
        },
        branch_id: BRANCH_ID.into(),
        identity: identity(),
        output: vec![
            json!({"type": "future_item", "provider": {"z": 1}}),
            json!({"type": "compaction", "encrypted_content": "opaque"}),
        ],
        portable_history_path: RELATIVE_PATH.into(),
        portable_history_sha256: digest,
        portable_history_bytes: bytes.len() as u64,
        checkpoint_token_seed: 40,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 2,
        prior_checkpoint_id: None,
        memory_revision: Some(7),
    }
}

fn checkpoint_file(portable: Vec<ConversationItem>) -> CompactionCheckpointFile {
    let wrapper = wrapper(&portable);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, trusted_envelope(), &portable).unwrap();
    CompactionCheckpointFile::new(
        wrapper,
        material,
        portable,
        Some("original user".into()),
        vec!["src/lib.rs".into()],
    )
    .unwrap()
}

fn typed_tail() -> Vec<ConversationItem> {
    vec![
        ConversationItem::Reasoning(rs::ReasoningItem {
            id: "reasoning-1".into(),
            summary: vec![rs::SummaryPart::SummaryText(rs::SummaryTextContent {
                text: "reasoning".into(),
            })],
            content: None,
            encrypted_content: Some("encrypted".into()),
            status: None,
        }),
        ConversationItem::Assistant(AssistantItem {
            content: Arc::from("assistant"),
            tool_calls: vec![ToolCall {
                id: Arc::from("call-1"),
                name: "read".into(),
                arguments: Arc::from("{\"path\":\"a.png\"}"),
            }],
            model_id: Some("grok-test".into()),
            model_fingerprint: None,
            reasoning_effort: None,
        }),
        ConversationItem::ToolResult(ToolResultItem {
            tool_call_id: "call-1".into(),
            content: Arc::from("result"),
            images: vec![ContentPart::Image {
                url: Arc::from("data:image/png;base64,AA=="),
            }],
        }),
    ]
}

fn persisted_tail(
    checkpoint: &ServerResponsesCheckpoint,
    sequence: u64,
    item: ConversationItem,
) -> PersistedTail {
    PersistedTail {
        operation_id: format!("tail-operation-{sequence}"),
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        branch_id: checkpoint.branch_id.clone(),
        sequence,
        prompt_index: checkpoint.prompt_index + 1,
        item,
    }
}

#[test]
fn current_checkpoint_round_trip_binds_marker_and_detects_tampering() {
    let dir = tempdir().unwrap();
    let file = checkpoint_file(portable_history());
    write_checkpoint_durable(dir.path(), RELATIVE_PATH, &file).unwrap();

    let loaded = read_checkpoint(
        dir.path(),
        RELATIVE_PATH,
        CHECKPOINT_ID,
        3,
        &file.portable_history_sha256,
    )
    .unwrap();
    assert_eq!(loaded.kind, CompactionCheckpointKind::ResponsesServer);
    assert_eq!(
        loaded.replay_material.wrapper_digest(),
        loaded.wrapper.wrapper_digest()
    );

    let marker = marker_for_wrapper(&loaded.wrapper);
    assert_eq!(marker.kind, CompactionCheckpointKind::ResponsesServer);
    validate_marker_for_wrapper(&marker, &loaded.wrapper).unwrap();
    read_checkpoint_for_wrapper(dir.path(), &loaded.wrapper).unwrap();

    let mut rotated = loaded.wrapper.clone();
    rotated.branch_id = "rewound-branch".into();
    read_checkpoint_for_wrapper(dir.path(), &rotated).unwrap();
    let repaired_marker = marker_for_wrapper(&rotated);
    validate_marker_for_wrapper(&repaired_marker, &loaded.wrapper).unwrap();
    rotated.operation_id = "forged-operation".into();
    assert!(read_checkpoint_for_wrapper(dir.path(), &rotated).is_err());

    let path = dir.path().join(RELATIVE_PATH);
    let mut tampered: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    tampered["portable_history"][1] =
        serde_json::to_value(ConversationItem::user("tampered")).unwrap();
    std::fs::write(&path, serde_json::to_vec(&tampered).unwrap()).unwrap();
    assert!(
        read_checkpoint(
            dir.path(),
            RELATIVE_PATH,
            CHECKPOINT_ID,
            3,
            &file.portable_history_sha256,
        )
        .is_err()
    );

    for unsafe_path in ["../checkpoint.json", "/absolute/checkpoint.json"] {
        assert!(write_checkpoint_durable(dir.path(), unsafe_path, &file).is_err());
    }
}

#[test]
fn versioned_sidecar_is_rejected_fail_closed() {
    let dir = tempdir().unwrap();
    let file = checkpoint_file(portable_history());
    write_checkpoint_durable(dir.path(), RELATIVE_PATH, &file).unwrap();
    let path = dir.path().join(RELATIVE_PATH);
    let mut versioned: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(versioned.get("schema_version").is_none());
    versioned["schema_version"] = json!(2);
    std::fs::write(&path, serde_json::to_vec(&versioned).unwrap()).unwrap();

    let error = read_checkpoint(
        dir.path(),
        RELATIVE_PATH,
        CHECKPOINT_ID,
        3,
        &file.portable_history_sha256,
    )
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn persisted_history_preserves_typed_tail_and_rejects_boundary_gaps() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chat_history.jsonl");
    let checkpoint = wrapper(&portable_history());
    let mut entries = vec![PersistedChatEntry::Item(
        ConversationItem::ResponsesCompactionCheckpoint(Box::new(checkpoint.clone())),
    )];
    entries.extend(typed_tail().into_iter().enumerate().map(|(index, item)| {
        PersistedChatEntry::Tail(persisted_tail(&checkpoint, (index + 1) as u64, item))
    }));
    write_history_durable(&path, &entries).unwrap();

    let serialized = std::fs::read_to_string(&path).unwrap();
    let first_tail: serde_json::Value =
        serde_json::from_str(serialized.lines().nth(1).unwrap()).unwrap();
    assert_eq!(first_tail["persisted_entry"], "tail");

    let recovered = read_history(&path).unwrap();
    assert_eq!(recovered.conversation.len(), 4);
    assert!(matches!(
        recovered.conversation[1],
        ConversationItem::Reasoning(_)
    ));
    assert!(matches!(
        recovered.conversation[2],
        ConversationItem::Assistant(_)
    ));
    assert!(matches!(
        recovered.conversation[3],
        ConversationItem::ToolResult(_)
    ));
    assert_eq!(recovered.prepared_repairs.len(), 3);
    assert_eq!(recovered.committed_repairs.len(), 3);

    let mut broken = entries;
    let PersistedChatEntry::Tail(tail) = &mut broken[2] else {
        unreachable!();
    };
    tail.sequence = 9;
    write_history_durable(&path, &broken).unwrap();
    assert!(read_history(&path).is_err(), "sequence gaps fail closed");
}

#[test]
fn durable_tail_append_is_contiguous_idempotent_and_conflict_safe() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chat_history.jsonl");
    let checkpoint = wrapper(&portable_history());
    write_history_durable(
        &path,
        &[PersistedChatEntry::Item(
            ConversationItem::ResponsesCompactionCheckpoint(Box::new(checkpoint.clone())),
        )],
    )
    .unwrap();
    let tail = persisted_tail(&checkpoint, 1, ConversationItem::user("next"));

    assert!(append_history_tail_durable(&path, &tail).unwrap());
    assert!(!append_history_tail_durable(&path, &tail).unwrap());
    assert_eq!(read_history(&path).unwrap().conversation.len(), 2);

    let conflicting = PersistedTail {
        operation_id: "different-operation".into(),
        item: ConversationItem::user("different"),
        ..tail.clone()
    };
    assert!(append_history_tail_durable(&path, &conflicting).is_err());
    let skipped = PersistedTail {
        operation_id: "tail-operation-3".into(),
        sequence: 3,
        ..tail
    };
    assert!(append_history_tail_durable(&path, &skipped).is_err());
}

#[test]
fn crash_rebuild_uses_only_fully_committed_active_branch_tail() {
    let checkpoint = wrapper(&portable_history());
    let items = typed_tail();
    let prepared_1 = ConversationAppendPrepared {
        operation_id: "tail-1".into(),
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        branch_id: checkpoint.branch_id.clone(),
        sequence: 1,
        prompt_index: 4,
        item: items[0].clone(),
    };
    let prepared_2 = ConversationAppendPrepared {
        operation_id: "tail-2".into(),
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        branch_id: checkpoint.branch_id.clone(),
        sequence: 2,
        prompt_index: 4,
        item: items[1].clone(),
    };
    let records = vec![
        TailJournalRecord::Prepared(prepared_1.clone()),
        TailJournalRecord::Committed(ConversationAppendCommitted::from(&prepared_1)),
        TailJournalRecord::Prepared(prepared_2), // crash before commit: ignored
        TailJournalRecord::Committed(ConversationAppendCommitted {
            operation_id: "abandoned-tail".into(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: "abandoned-branch".into(),
            sequence: 2,
            prompt_index: 4,
        }),
    ];

    let rebuilt = rebuild_updates_only(checkpoint.clone(), &records).unwrap();
    assert_eq!(rebuilt.len(), 2);
    assert!(matches!(
        rebuilt[0],
        ConversationItem::ResponsesCompactionCheckpoint(_)
    ));
    assert!(matches!(rebuilt[1], ConversationItem::Reasoning(_)));

    let gap_prepared = ConversationAppendPrepared {
        sequence: 2,
        ..prepared_1
    };
    let gap = vec![
        TailJournalRecord::Prepared(gap_prepared.clone()),
        TailJournalRecord::Committed(ConversationAppendCommitted::from(&gap_prepared)),
    ];
    assert!(rebuild_updates_only(checkpoint, &gap).is_err());
}

#[test]
fn rewind_rotates_typed_branch_or_restores_portable_history() {
    let file = checkpoint_file(portable_history());
    let checkpoint = file.wrapper.clone();
    let active = vec![
        PersistedChatEntry::Item(ConversationItem::ResponsesCompactionCheckpoint(Box::new(
            checkpoint.clone(),
        ))),
        PersistedChatEntry::Tail(PersistedTail {
            prompt_index: 3,
            ..persisted_tail(&checkpoint, 1, ConversationItem::system_reminder("mode"))
        }),
        PersistedChatEntry::Tail(PersistedTail {
            prompt_index: 4,
            ..persisted_tail(&checkpoint, 2, ConversationItem::user("prompt 3"))
        }),
        PersistedChatEntry::Tail(PersistedTail {
            prompt_index: 4,
            ..persisted_tail(&checkpoint, 3, ConversationItem::assistant("answer 3"))
        }),
        PersistedChatEntry::Tail(PersistedTail {
            prompt_index: 5,
            ..persisted_tail(&checkpoint, 4, ConversationItem::user("prompt 4"))
        }),
    ];

    let after = rewind_history_entries(&file, active.clone(), 4, "rewound-branch").unwrap();
    assert_eq!(after.len(), 4);
    let PersistedChatEntry::Item(ConversationItem::ResponsesCompactionCheckpoint(wrapper)) =
        &after[0]
    else {
        panic!("expected checkpoint wrapper");
    };
    assert_eq!(wrapper.branch_id, "rewound-branch");
    assert!(after.iter().skip(1).all(|entry| matches!(
        entry,
        PersistedChatEntry::Tail(tail) if tail.branch_id == "rewound-branch"
    )));

    let before = rewind_history_entries(&file, active, 2, "portable-branch").unwrap();
    assert!(
        before
            .iter()
            .all(|entry| matches!(entry, PersistedChatEntry::Item(_)))
    );
    assert!(!before.iter().any(|entry| matches!(
        entry,
        PersistedChatEntry::Item(ConversationItem::ResponsesCompactionCheckpoint(_))
    )));
}

#[test]
fn staged_segment_is_invisible_until_strongly_bound_publish() {
    let dir = tempdir().unwrap();
    let checkpoint = wrapper(&portable_history());
    let staging = ResponsesCompactionSegmentStaging::new(
        checkpoint.checkpoint_id.clone(),
        checkpoint.operation_id.clone(),
        checkpoint.branch_id.clone(),
        checkpoint.wrapper_digest(),
        vec![ConversationItem::user("portable turn")],
        "Server Responses checkpoint",
        xai_chat_state::CompactionDetail::Verbose,
        "2026-01-01T00:00:00Z",
    )
    .unwrap();

    stage_compaction_segment_durable(dir.path(), &staging).unwrap();
    stage_compaction_segment_durable(dir.path(), &staging).unwrap();
    let compaction_dir = dir.path().join("compaction");
    assert!(!compaction_dir.join("INDEX.md").exists());
    assert!(!compaction_dir.join("segment_000.md").exists());
    assert!(
        publish_staged_compaction_segment_durable(
            dir.path(),
            &checkpoint.checkpoint_id,
            "forged-operation",
            &checkpoint.wrapper_digest(),
        )
        .is_err()
    );
    assert!(
        compaction_dir
            .join("staging/checkpoint-current.json")
            .exists()
    );

    let published = publish_staged_compaction_segment_durable(
        dir.path(),
        &checkpoint.checkpoint_id,
        &checkpoint.operation_id,
        &checkpoint.wrapper_digest(),
    )
    .unwrap();
    assert_eq!(published.index, 0);
    assert!(published.newly_published);
    let segment = std::fs::read_to_string(compaction_dir.join("segment_000.md")).unwrap();
    assert!(segment.contains("responses-compaction-checkpoint:checkpoint-current"));
    assert!(segment.contains("portable turn"));

    let repeated = publish_staged_compaction_segment_durable(
        dir.path(),
        &checkpoint.checkpoint_id,
        &checkpoint.operation_id,
        &checkpoint.wrapper_digest(),
    )
    .unwrap();
    assert_eq!(repeated.index, 0);
    assert!(!repeated.newly_published);
    assert!(!compaction_dir.join("segment_001.md").exists());
}

#[test]
fn orphan_staging_from_crash_does_not_consume_segment_index() {
    let dir = tempdir().unwrap();
    for checkpoint_id in ["orphan", "committed"] {
        let portable = portable_history();
        let mut checkpoint = wrapper(&portable);
        checkpoint.checkpoint_id = checkpoint_id.into();
        checkpoint.operation_id = format!("operation-{checkpoint_id}");
        checkpoint.portable_history_path = format!("compaction_checkpoints/{checkpoint_id}.json");
        let staging = ResponsesCompactionSegmentStaging::new(
            checkpoint_id,
            &checkpoint.operation_id,
            &checkpoint.branch_id,
            checkpoint.wrapper_digest(),
            vec![ConversationItem::user(checkpoint_id)],
            "Server Responses checkpoint",
            xai_chat_state::CompactionDetail::Minimal,
            "2026-01-01T00:00:00Z",
        )
        .unwrap();
        stage_compaction_segment_durable(dir.path(), &staging).unwrap();
    }

    let portable = portable_history();
    let mut committed = wrapper(&portable);
    committed.checkpoint_id = "committed".into();
    committed.operation_id = "operation-committed".into();
    committed.portable_history_path = "compaction_checkpoints/committed.json".into();
    let published = publish_staged_compaction_segment_durable(
        dir.path(),
        "committed",
        &committed.operation_id,
        &committed.wrapper_digest(),
    )
    .unwrap();
    assert_eq!(published.index, 0);
    assert!(!dir.path().join("compaction/segment_001.md").exists());
    assert!(dir.path().join("compaction/staging/orphan.json").exists());
}

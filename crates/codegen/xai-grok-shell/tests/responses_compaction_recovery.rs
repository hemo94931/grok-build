use std::sync::Arc;

use serde_json::json;
use tempfile::tempdir;
use xai_grok_sampling_types::{
    AssistantItem, CheckpointIdentityV1, ContentPart, ConversationItem, ResponsesCompactionModeV1,
    ServerResponsesCheckpointV1, TokenSeedSource, ToolCall, ToolResultItem, rs,
};
use xai_grok_shell::session::storage::responses_compaction::{
    CompactionCheckpointFileV2, ConversationAppendCommittedV2, ConversationAppendPreparedV2,
    PersistedChatEntry, ResponsesCompactionSegmentStagingV1, TailJournalRecord, TailV2,
    append_history_tail_v2_durable, marker_for_wrapper, publish_staged_compaction_segment_durable,
    read_checkpoint_for_wrapper, read_checkpoint_v2, read_history_v2, rebuild_updates_only_v2,
    rewind_history_entries_v2, stage_compaction_segment_durable, validate_marker_for_wrapper,
    write_checkpoint_v2_durable, write_history_v2_durable,
};

fn identity() -> CheckpointIdentityV1 {
    CheckpointIdentityV1 {
        provider_id: "provider".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: "responses-compact-codex-v1".into(),
        prompt_envelope_fingerprint: "prompt".into(),
        canonical_prompt_projection: Some(json!({"system": "system"})),
    }
}

fn wrapper() -> ServerResponsesCheckpointV1 {
    ServerResponsesCheckpointV1 {
        schema_version: 1,
        checkpoint_id: "checkpoint-1".into(),
        operation_id: "compact-operation".into(),
        prompt_index: 3,
        created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
        auto_continue: true,
        mode: ResponsesCompactionModeV1 {
            name: "transcript".into(),
            detail: Some(json!({"location": "transcript.md"})),
        },
        branch_id: "branch-1".into(),
        identity: identity(),
        output: vec![
            json!({"type": "future_item", "provider": {"z": 1}}),
            json!({"type": "compaction", "encrypted_content": "opaque"}),
        ],
        portable_history_path: "compaction_checkpoints/checkpoint-1.json".into(),
        portable_history_sha256: String::new(),
        portable_history_bytes: 0,
        checkpoint_token_seed: 40,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 2,
    }
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

#[test]
fn checkpoint_v2_digest_round_trip_and_tamper_detection() {
    let dir = tempdir().unwrap();
    let portable = vec![
        ConversationItem::system("system"),
        ConversationItem::user("original prompt"),
    ];
    let file = CompactionCheckpointFileV2::new(
        wrapper(),
        portable.clone(),
        Some("user info".into()),
        vec!["src/lib.rs".into()],
    )
    .unwrap();
    let relative = "compaction_checkpoints/checkpoint-1.json";
    write_checkpoint_v2_durable(dir.path(), relative, &file).unwrap();

    let loaded = read_checkpoint_v2(
        dir.path(),
        relative,
        "checkpoint-1",
        3,
        &file.portable_history_sha256,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&loaded.portable_history).unwrap(),
        serde_json::to_value(&portable).unwrap()
    );
    assert_eq!(loaded.schema_version, 2);
    assert_eq!(loaded.kind, "responses_server");
    let marker = marker_for_wrapper(&loaded.wrapper);
    validate_marker_for_wrapper(&marker, &loaded.wrapper).unwrap();
    read_checkpoint_for_wrapper(dir.path(), &loaded.wrapper).unwrap();
    let mut rewound_wrapper = loaded.wrapper.clone();
    rewound_wrapper.branch_id = "new-rewind-branch".into();
    read_checkpoint_for_wrapper(dir.path(), &rewound_wrapper).unwrap();
    let mut future_wrapper = loaded.wrapper.clone();
    future_wrapper.schema_version = 2;
    assert!(read_checkpoint_for_wrapper(dir.path(), &future_wrapper).is_err());
    let mut bad_marker = marker;
    bad_marker.prompt_index_at_compaction += 1;
    assert!(validate_marker_for_wrapper(&bad_marker, &loaded.wrapper).is_err());

    let path = dir.path().join(relative);
    let mut tampered: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    tampered["portable_history"][1] =
        serde_json::to_value(ConversationItem::user("tampered")).unwrap();
    std::fs::write(&path, serde_json::to_vec(&tampered).unwrap()).unwrap();
    assert!(
        read_checkpoint_v2(
            dir.path(),
            relative,
            "checkpoint-1",
            3,
            &file.portable_history_sha256,
        )
        .is_err()
    );

    for unsafe_path in ["../checkpoint.json", "/absolute/checkpoint.json"] {
        assert!(write_checkpoint_v2_durable(dir.path(), unsafe_path, &file).is_err());
    }

    std::fs::remove_file(path).unwrap();
    assert!(read_checkpoint_for_wrapper(dir.path(), &loaded.wrapper).is_err());
}

#[cfg(unix)]
#[test]
fn checkpoint_reader_rejects_symlink_components() {
    use std::os::unix::fs::symlink;

    let dir = tempdir().unwrap();
    let outside = tempdir().unwrap();
    symlink(outside.path(), dir.path().join("compaction_checkpoints")).unwrap();
    let file =
        CompactionCheckpointFileV2::new(wrapper(), vec![ConversationItem::user("p")], None, vec![])
            .unwrap();
    assert!(
        write_checkpoint_v2_durable(
            dir.path(),
            "compaction_checkpoints/checkpoint-1.json",
            &file,
        )
        .is_err()
    );
}

#[test]
fn persisted_history_v2_preserves_typed_tail_and_authoritative_boundary() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chat_history.jsonl");
    let checkpoint = wrapper();
    let mut entries = vec![PersistedChatEntry::Legacy(
        ConversationItem::ResponsesCompactionCheckpoint(Box::new(checkpoint.clone())),
    )];
    entries.extend(typed_tail().into_iter().enumerate().map(|(index, item)| {
        PersistedChatEntry::TailV2(TailV2 {
            operation_id: format!("tail-operation-{}", index + 1),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: checkpoint.branch_id.clone(),
            sequence: (index + 1) as u64,
            prompt_index: checkpoint.prompt_index,
            item,
        })
    }));
    write_history_v2_durable(&path, &entries).unwrap();

    let recovered = read_history_v2(&path).unwrap();
    assert_eq!(recovered.conversation.len(), 4);
    assert!(matches!(
        recovered.conversation[0],
        ConversationItem::ResponsesCompactionCheckpoint(_)
    ));
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
    assert_eq!(recovered.committed_repairs.len(), 3);

    let mut broken = entries;
    if let PersistedChatEntry::TailV2(tail) = &mut broken[2] {
        tail.sequence = 9;
    }
    write_history_v2_durable(&path, &broken).unwrap();
    assert!(read_history_v2(&path).is_err(), "sequence gaps fail closed");
}

#[test]
fn durable_tail_append_is_contiguous_and_idempotent() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chat_history.jsonl");
    let checkpoint = wrapper();
    write_history_v2_durable(
        &path,
        &[PersistedChatEntry::Legacy(
            ConversationItem::ResponsesCompactionCheckpoint(Box::new(checkpoint.clone())),
        )],
    )
    .unwrap();
    let tail = TailV2 {
        operation_id: "tail-1".into(),
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        branch_id: checkpoint.branch_id.clone(),
        sequence: 1,
        prompt_index: 4,
        item: ConversationItem::user("next"),
    };

    assert!(append_history_tail_v2_durable(&path, &tail).unwrap());
    assert!(!append_history_tail_v2_durable(&path, &tail).unwrap());
    assert_eq!(read_history_v2(&path).unwrap().conversation.len(), 2);

    let conflicting = TailV2 {
        operation_id: "different-operation".into(),
        item: ConversationItem::user("different"),
        ..tail.clone()
    };
    assert!(append_history_tail_v2_durable(&path, &conflicting).is_err());
    let skipped = TailV2 {
        operation_id: "tail-3".into(),
        sequence: 3,
        ..tail
    };
    assert!(append_history_tail_v2_durable(&path, &skipped).is_err());
}

#[test]
fn updates_only_rebuild_requires_prepared_and_committed_on_active_branch() {
    let checkpoint = wrapper();
    let items = typed_tail();
    let prepared_1 = ConversationAppendPreparedV2 {
        operation_id: "tail-1".into(),
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        branch_id: checkpoint.branch_id.clone(),
        sequence: 1,
        prompt_index: 4,
        item: items[0].clone(),
    };
    let prepared_2 = ConversationAppendPreparedV2 {
        operation_id: "tail-2".into(),
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        branch_id: checkpoint.branch_id.clone(),
        sequence: 2,
        prompt_index: 4,
        item: items[1].clone(),
    };
    let records = vec![
        TailJournalRecord::Prepared(prepared_1.clone()),
        TailJournalRecord::Committed(ConversationAppendCommittedV2::from(&prepared_1)),
        TailJournalRecord::Prepared(prepared_2), // orphan: never committed
        TailJournalRecord::Committed(ConversationAppendCommittedV2 {
            operation_id: "old-branch".into(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: "old-branch".into(),
            sequence: 2,
            prompt_index: 4,
        }),
    ];

    let rebuilt = rebuild_updates_only_v2(checkpoint.clone(), &records).unwrap();
    assert_eq!(rebuilt.len(), 2);
    assert!(matches!(rebuilt[1], ConversationItem::Reasoning(_)));

    let gap = vec![
        TailJournalRecord::Prepared(ConversationAppendPreparedV2 {
            sequence: 2,
            ..prepared_1.clone()
        }),
        TailJournalRecord::Committed(ConversationAppendCommittedV2 {
            sequence: 2,
            ..ConversationAppendCommittedV2::from(&prepared_1)
        }),
    ];
    assert!(rebuild_updates_only_v2(checkpoint, &gap).is_err());
}

#[test]
fn rewind_rotates_branch_and_restores_portable_history_before_checkpoint() {
    let portable = vec![
        ConversationItem::system("system"),
        ConversationItem::user("user info"),
        ConversationItem::user("prompt 0"),
        ConversationItem::assistant("answer 0"),
        ConversationItem::user("prompt 1"),
        ConversationItem::assistant("answer 1"),
        ConversationItem::user("prompt 2"),
        ConversationItem::assistant("answer 2"),
    ];
    let file = CompactionCheckpointFileV2::new(wrapper(), portable, None, vec![]).unwrap();
    let checkpoint = file.wrapper.clone();
    let active = vec![
        PersistedChatEntry::Legacy(ConversationItem::ResponsesCompactionCheckpoint(Box::new(
            checkpoint.clone(),
        ))),
        PersistedChatEntry::TailV2(TailV2 {
            operation_id: "mode".into(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: checkpoint.branch_id.clone(),
            sequence: 1,
            prompt_index: 3,
            item: ConversationItem::system("mode reminder"),
        }),
        PersistedChatEntry::TailV2(TailV2 {
            operation_id: "prompt-3".into(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: checkpoint.branch_id.clone(),
            sequence: 2,
            prompt_index: 4,
            item: ConversationItem::user("prompt 3"),
        }),
        PersistedChatEntry::TailV2(TailV2 {
            operation_id: "answer-3".into(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: checkpoint.branch_id.clone(),
            sequence: 3,
            prompt_index: 4,
            item: ConversationItem::assistant("answer 3"),
        }),
        PersistedChatEntry::TailV2(TailV2 {
            operation_id: "prompt-4".into(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_id: checkpoint.branch_id.clone(),
            sequence: 4,
            prompt_index: 5,
            item: ConversationItem::user("prompt 4"),
        }),
    ];

    let after = rewind_history_entries_v2(&file, active.clone(), 4, "branch-2").unwrap();
    assert_eq!(after.len(), 4);
    let PersistedChatEntry::Legacy(ConversationItem::ResponsesCompactionCheckpoint(wrapper)) =
        &after[0]
    else {
        panic!("expected wrapper");
    };
    assert_eq!(wrapper.branch_id, "branch-2");
    assert!(after.iter().skip(1).all(|entry| matches!(
        entry,
        PersistedChatEntry::TailV2(tail) if tail.branch_id == "branch-2"
    )));

    let before = rewind_history_entries_v2(&file, active, 2, "branch-3").unwrap();
    assert!(
        before
            .iter()
            .all(|entry| matches!(entry, PersistedChatEntry::Legacy(_)))
    );
    assert!(!before.iter().any(|entry| matches!(
        entry,
        PersistedChatEntry::Legacy(ConversationItem::ResponsesCompactionCheckpoint(_))
    )));
}

#[test]
fn segment_staging_is_invisible_until_idempotent_publish() {
    let dir = tempdir().unwrap();
    let staged = ResponsesCompactionSegmentStagingV1::new(
        "checkpoint-1",
        vec![ConversationItem::user("portable turn")],
        "Server Responses checkpoint",
        xai_chat_state::CompactionDetail::Verbose,
        "2026-01-01T00:00:00Z",
    )
    .unwrap();

    stage_compaction_segment_durable(dir.path(), &staged).unwrap();
    stage_compaction_segment_durable(dir.path(), &staged).unwrap();
    let compaction_dir = dir.path().join("compaction");
    assert!(!compaction_dir.join("INDEX.md").exists());
    assert!(!compaction_dir.join("segment_000.md").exists());

    let published = publish_staged_compaction_segment_durable(dir.path(), "checkpoint-1").unwrap();
    assert_eq!(published.index, 0);
    assert!(published.newly_published);
    let segment = std::fs::read_to_string(compaction_dir.join("segment_000.md")).unwrap();
    assert!(segment.contains("responses-compaction-checkpoint:checkpoint-1"));
    assert!(segment.contains("portable turn"));
    assert!(
        std::fs::read_to_string(compaction_dir.join("INDEX.md"))
            .unwrap()
            .contains("segment_000.md")
    );

    let repeated = publish_staged_compaction_segment_durable(dir.path(), "checkpoint-1").unwrap();
    assert_eq!(repeated.index, 0);
    assert!(!repeated.newly_published);
    assert!(!compaction_dir.join("segment_001.md").exists());
}

#[test]
fn orphan_segment_staging_never_consumes_a_formal_index() {
    let dir = tempdir().unwrap();
    for checkpoint_id in ["orphan", "committed"] {
        stage_compaction_segment_durable(
            dir.path(),
            &ResponsesCompactionSegmentStagingV1::new(
                checkpoint_id,
                vec![ConversationItem::user(checkpoint_id)],
                "Server Responses checkpoint",
                xai_chat_state::CompactionDetail::Minimal,
                "2026-01-01T00:00:00Z",
            )
            .unwrap(),
        )
        .unwrap();
    }

    let published = publish_staged_compaction_segment_durable(dir.path(), "committed").unwrap();
    assert_eq!(published.index, 0);
    assert!(!dir.path().join("compaction/segment_001.md").exists());
    assert!(dir.path().join("compaction/staging/orphan.json").exists());
}

#[test]
fn future_checkpoint_schema_is_rejected() {
    let dir = tempdir().unwrap();
    let file =
        CompactionCheckpointFileV2::new(wrapper(), vec![ConversationItem::user("p")], None, vec![])
            .unwrap();
    let relative = "compaction_checkpoints/checkpoint-1.json";
    write_checkpoint_v2_durable(dir.path(), relative, &file).unwrap();
    let path = dir.path().join(relative);
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["schema_version"] = json!(3);
    std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(
        read_checkpoint_v2(
            dir.path(),
            relative,
            "checkpoint-1",
            3,
            &file.portable_history_sha256,
        )
        .is_err()
    );
}

use serde_json::json;
use std::num::NonZeroU64;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use xai_chat_state::{
    ChatCompactionSnapshot, ChatStateActor, CheckpointReplayStatus, CommitCompaction,
    CommitCompactionResult, HistoryReplaceError, MockChatPersistence, ReplaceSystemHeadResult,
    RequestIdentityBindResult, RequestIdentityBinding, estimate_conversation_tokens,
};
use xai_grok_sampling_types::{
    ApiBackend, CheckpointIdentity, ContentPart, ConversationItem, RESPONSES_COMPACTION_CONTRACT,
    ResponsesCompactionMode, SamplingConfig, ServerResponsesCheckpoint, TokenSeedSource,
    base_instructions_sha256,
};

fn sampling_config(model: &str) -> SamplingConfig {
    SamplingConfig {
        base_url: "https://api.example.test/v1".into(),
        model: model.into(),
        max_completion_tokens: Some(4096),
        temperature: None,
        top_p: None,
        api_backend: ApiBackend::Responses,
        extra_headers: Default::default(),
        query_params: Default::default(),
        env_http_headers: Default::default(),
        context_window: NonZeroU64::new(128_000).unwrap(),
        reasoning_effort: None,
        stream_tool_calls: None,
    }
}

fn identity(model: &str, prompt: &str) -> CheckpointIdentity {
    CheckpointIdentity {
        provider_id: "provider".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: model.into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
        prompt_envelope_fingerprint: format!("prompt-{prompt}"),
        base_instructions_sha256: base_instructions_sha256(prompt),
        prior_checkpoint_id: Some("checkpoint-0".into()),
        cache_route_fingerprint: Some("cache-route".into()),
    }
}

fn wrapper(seed: u64, identity: CheckpointIdentity) -> ConversationItem {
    let prior_checkpoint_id = identity.prior_checkpoint_id.clone();
    ConversationItem::ResponsesCompactionCheckpoint(Box::new(ServerResponsesCheckpoint {
        checkpoint_id: "checkpoint-1".into(),
        operation_id: "operation-1".into(),
        prompt_index: 1,
        created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
        auto_continue: false,
        mode: ResponsesCompactionMode {
            name: "summary".into(),
            detail: None,
        },
        branch_id: "branch-1".into(),
        identity,
        output: vec![json!({"type": "compaction", "encrypted_content": "opaque"})],
        portable_history_path: "compaction_checkpoints/checkpoint-1.json".into(),
        portable_history_sha256: "digest".into(),
        portable_history_bytes: 10,
        checkpoint_token_seed: seed,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 1,
        prior_checkpoint_id,
        memory_revision: Some(7),
    }))
}

fn spawn(
    initial: Vec<ConversationItem>,
    persistence: MockChatPersistence,
) -> xai_chat_state::ChatStateHandle {
    let (events, _event_rx) = mpsc::unbounded_channel();
    ChatStateActor::spawn(
        initial,
        sampling_config("grok-test"),
        Box::new(persistence),
        events,
        CancellationToken::new(),
    )
}

async fn bind(
    handle: &xai_chat_state::ChatStateHandle,
    identity: CheckpointIdentity,
) -> RequestIdentityBinding {
    handle.bind_request_identity(identity).await.unwrap()
}

async fn snapshot(handle: &xai_chat_state::ChatStateHandle) -> ChatCompactionSnapshot {
    handle.get_compaction_snapshot().await.unwrap()
}

#[test]
fn tool_result_image_tokens_are_counted_once() {
    let item = ConversationItem::tool_result_with_images(
        "call",
        "result",
        vec![ContentPart::Image {
            url: "data:image/png;base64,large-payload".into(),
        }],
    );
    assert_eq!(
        estimate_conversation_tokens(&[item]),
        xai_token_estimation::estimate_tokens("result")
            + xai_token_estimation::IMAGE_TOKEN_ESTIMATE
    );
}

#[tokio::test]
async fn request_revision_binding_is_atomic_and_rejects_stale_history() {
    let (persistence, _records) = MockChatPersistence::new();
    let handle = spawn(
        vec![
            ConversationItem::system("system"),
            ConversationItem::user("first"),
        ],
        persistence,
    );
    let request = handle
        .build_request(vec![], None, false, None, "conv".into(), "req".into())
        .await
        .unwrap();
    let request_revision = request.history_revision.expect("actor request revision");
    let before = snapshot(&handle).await;

    handle
        .push_user_message_and_ack(ConversationItem::user("concurrent"))
        .await
        .unwrap();
    let stale = handle
        .bind_request_identity_at_revision(identity("grok-test", "system"), request_revision)
        .await
        .unwrap();
    assert!(matches!(
        stale,
        RequestIdentityBindResult::StaleHistory { current_revision }
            if current_revision > request_revision
    ));
    let after_stale = snapshot(&handle).await;
    assert_eq!(
        after_stale.total_tokens,
        before.total_tokens + estimate_conversation_tokens(&[ConversationItem::user("concurrent")]),
        "the compaction baseline includes user/tool deltas since the last model usage"
    );
    assert_eq!(
        after_stale.request_identity_generation, before.request_identity_generation,
        "a stale request must not bind its identity"
    );
    assert!(after_stale.bound_request_identity.is_none());

    let rebuilt = handle
        .build_request(vec![], None, false, None, "conv".into(), "req-2".into())
        .await
        .unwrap();
    let rebuilt_revision = rebuilt.history_revision.expect("rebuilt request revision");
    let bound = handle
        .bind_request_identity_at_revision(identity("grok-test", "system"), rebuilt_revision)
        .await
        .unwrap();
    let RequestIdentityBindResult::Bound {
        binding,
        compaction_snapshot,
    } = bound
    else {
        panic!("rebuilt request should bind");
    };
    let compaction_snapshot = *compaction_snapshot;
    assert_eq!(binding.request_identity_generation, 1);
    assert_eq!(compaction_snapshot.history_revision, rebuilt_revision);
    assert_eq!(
        serde_json::to_value(compaction_snapshot.conversation).unwrap(),
        serde_json::to_value(rebuilt.items).unwrap()
    );
}

#[tokio::test]
async fn dual_generation_cas_commits_once_and_reseeds_exactly() {
    let (persistence, mut records) = MockChatPersistence::new();
    let handle = spawn(
        vec![
            ConversationItem::system("system"),
            ConversationItem::user("first"),
        ],
        persistence,
    );
    let bound = bind(&handle, identity("grok-test", "system")).await;
    assert_eq!(
        bound.checkpoint_status,
        CheckpointReplayStatus::NoCheckpoint
    );
    assert_eq!(bound.request_identity_generation, 1);

    let stale = snapshot(&handle).await;
    handle
        .push_user_message_and_ack(ConversationItem::user("concurrent"))
        .await
        .unwrap();
    let stale_result = handle
        .commit_compaction(CommitCompaction {
            operation_id: "stale-operation".into(),
            expected_history_revision: stale.history_revision,
            expected_request_identity_generation: stale.request_identity_generation,
            replacement: vec![wrapper(50, identity("grok-test", "system"))],
            committed_total_tokens: 50,
        })
        .await
        .unwrap();
    assert!(matches!(
        stale_result,
        CommitCompactionResult::Superseded { .. }
    ));
    assert!(!records.drain().iter().any(|record| matches!(
        record,
        xai_chat_state::PersistenceRecord::AcknowledgedReplaceHistory { .. }
    )));

    let current = snapshot(&handle).await;
    let tail = ConversationItem::user("12345678901234567890"); // 5 tokens
    let replacement = vec![wrapper(50, identity("grok-test", "system")), tail];
    assert_eq!(estimate_conversation_tokens(&replacement), 55);
    let committed = handle
        .commit_compaction(CommitCompaction {
            operation_id: "operation-1".into(),
            expected_history_revision: current.history_revision,
            expected_request_identity_generation: current.request_identity_generation,
            replacement,
            committed_total_tokens: 55,
        })
        .await
        .unwrap();
    assert!(matches!(
        committed,
        CommitCompactionResult::Committed { .. }
    ));
    assert_eq!(handle.get_total_tokens().await, 55);
    assert_eq!(handle.get_conversation().await.len(), 2);

    let after = snapshot(&handle).await;
    assert_eq!(after.history_revision, current.history_revision + 1);
    assert_eq!(
        after.request_identity_generation,
        current.request_identity_generation
    );

    handle.record_token_usage(99);
    assert_eq!(
        handle.get_total_tokens().await,
        99,
        "provider usage becomes authoritative"
    );
}

#[tokio::test]
async fn generic_replace_cannot_overwrite_an_active_checkpoint() {
    let (persistence, _records) = MockChatPersistence::new();
    let handle = spawn(vec![ConversationItem::system("system")], persistence);
    bind(&handle, identity("grok-test", "system")).await;
    let before = snapshot(&handle).await;
    assert!(matches!(
        handle
            .commit_compaction(CommitCompaction {
                operation_id: "install-wrapper".into(),
                expected_history_revision: before.history_revision,
                expected_request_identity_generation: before.request_identity_generation,
                replacement: vec![wrapper(10, identity("grok-test", "system"))],
                committed_total_tokens: 10,
            })
            .await
            .unwrap(),
        CommitCompactionResult::Committed { .. }
    ));
    let installed = snapshot(&handle).await;

    handle.replace_conversation(vec![ConversationItem::system("must not win")]);
    let after = snapshot(&handle).await;
    assert!(matches!(
        after.conversation.first(),
        Some(ConversationItem::ResponsesCompactionCheckpoint(_))
    ));
    assert_eq!(after.history_revision, installed.history_revision);
    assert_eq!(
        after.request_identity_generation,
        installed.request_identity_generation + 1
    );
}

#[tokio::test]
async fn every_checkpoint_identity_component_mismatch_requires_migration() {
    let expected = identity("grok-test", "system");
    let (persistence, _records) = MockChatPersistence::new();
    let handle = spawn(vec![wrapper(40, expected.clone())], persistence);
    assert_eq!(
        bind(&handle, expected.clone()).await.checkpoint_status,
        CheckpointReplayStatus::Replayable
    );

    let mut mismatches = Vec::new();
    let mut changed = expected.clone();
    changed.provider_id = "other-provider".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.api = "other-api".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.endpoint_fingerprint = "other-endpoint".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.model = "other-model".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.auth_principal_fingerprint = "other-principal".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.contract_version = "future-contract".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.prompt_envelope_fingerprint = "other-prompt".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.base_instructions_sha256 = "other-base-instructions".into();
    mismatches.push(changed);
    let mut changed = expected.clone();
    changed.prior_checkpoint_id = None;
    mismatches.push(changed);
    let mut changed = expected;
    changed.cache_route_fingerprint = None;
    mismatches.push(changed);

    for mismatch in mismatches {
        assert_eq!(
            bind(&handle, mismatch).await.checkpoint_status,
            CheckpointReplayStatus::MigrationRequired
        );
    }
}

#[tokio::test]
async fn invalid_checkpoint_layout_and_output_are_rejected() {
    let id = identity("grok-test", "system");
    let valid = wrapper(40, id.clone());
    let misplaced = vec![ConversationItem::user("before checkpoint"), valid.clone()];
    let duplicate = vec![valid.clone(), valid.clone()];

    for conversation in [misplaced, duplicate] {
        let (persistence, _records) = MockChatPersistence::new();
        let handle = spawn(conversation, persistence);
        assert_eq!(
            bind(&handle, id.clone()).await.checkpoint_status,
            CheckpointReplayStatus::InvalidCheckpoint
        );
    }

    let mut empty_output = valid.clone();
    let ConversationItem::ResponsesCompactionCheckpoint(checkpoint) = &mut empty_output else {
        unreachable!("fixture is a checkpoint");
    };
    checkpoint.output.clear();

    let (persistence, mut records) = MockChatPersistence::new();
    let handle = spawn(vec![ConversationItem::system("system")], persistence);
    bind(&handle, id).await;
    let before = snapshot(&handle).await;
    for (operation_id, replacement) in [
        (
            "invalid-layout",
            vec![ConversationItem::user("before checkpoint"), valid],
        ),
        ("empty-output", vec![empty_output]),
    ] {
        let result = handle
            .commit_compaction(CommitCompaction {
                operation_id: operation_id.into(),
                expected_history_revision: before.history_revision,
                expected_request_identity_generation: before.request_identity_generation,
                replacement,
                committed_total_tokens: 40,
            })
            .await
            .unwrap();
        assert!(matches!(
            result,
            CommitCompactionResult::PersistenceFailed(_)
        ));
    }
    assert!(matches!(
        handle.get_conversation().await[0],
        ConversationItem::System(_)
    ));
    assert!(!records.drain().iter().any(|record| matches!(
        record,
        xai_chat_state::PersistenceRecord::AcknowledgedReplaceHistory { .. }
    )));
}

#[tokio::test]
async fn identity_change_after_snapshot_supersedes_without_persistence() {
    let (persistence, mut records) = MockChatPersistence::new();
    let handle = spawn(vec![ConversationItem::system("system")], persistence);
    bind(&handle, identity("grok-test", "system")).await;
    let before = snapshot(&handle).await;

    handle.update_sampling_config(sampling_config("grok-next"));
    let after_identity_change = snapshot(&handle).await;
    assert_eq!(
        after_identity_change.request_identity_generation,
        before.request_identity_generation + 1
    );

    let result = handle
        .commit_compaction(CommitCompaction {
            operation_id: "operation-stale-identity".into(),
            expected_history_revision: before.history_revision,
            expected_request_identity_generation: before.request_identity_generation,
            replacement: vec![wrapper(10, identity("grok-test", "system"))],
            committed_total_tokens: 10,
        })
        .await
        .unwrap();
    assert!(matches!(result, CommitCompactionResult::Superseded { .. }));
    assert!(!records.drain().iter().any(|record| matches!(
        record,
        xai_chat_state::PersistenceRecord::AcknowledgedReplaceHistory { .. }
    )));
}

#[tokio::test]
async fn persistence_not_committed_keeps_memory_but_lost_ack_committed_converges() {
    let (persistence, mut records) = MockChatPersistence::new_with_manual_history_replace_ack();
    let handle = spawn(vec![ConversationItem::system("system")], persistence);
    bind(&handle, identity("grok-test", "system")).await;

    let before = snapshot(&handle).await;
    let task = tokio::spawn({
        let handle = handle.clone();
        async move {
            handle
                .commit_compaction(CommitCompaction {
                    operation_id: "not-committed".into(),
                    expected_history_revision: before.history_revision,
                    expected_request_identity_generation: before.request_identity_generation,
                    replacement: vec![wrapper(10, identity("grok-test", "system"))],
                    committed_total_tokens: 10,
                })
                .await
                .unwrap()
        }
    });
    records
        .next_history_replace_ack()
        .await
        .unwrap()
        .send(Err(HistoryReplaceError::NotCommitted(
            std::io::Error::other("disk full"),
        )))
        .unwrap();
    assert!(matches!(
        task.await.unwrap(),
        CommitCompactionResult::PersistenceFailed(_)
    ));
    assert!(matches!(
        handle.get_conversation().await[0],
        ConversationItem::System(_)
    ));

    let current = snapshot(&handle).await;
    let task = tokio::spawn({
        let handle = handle.clone();
        async move {
            handle
                .commit_compaction(CommitCompaction {
                    operation_id: "lost-ack".into(),
                    expected_history_revision: current.history_revision,
                    expected_request_identity_generation: current.request_identity_generation,
                    replacement: vec![wrapper(10, identity("grok-test", "system"))],
                    committed_total_tokens: 10,
                })
                .await
                .unwrap()
        }
    });
    records
        .next_history_replace_ack()
        .await
        .unwrap()
        .send(Err(HistoryReplaceError::Committed(std::io::Error::other(
            "ack channel lost",
        ))))
        .unwrap();
    assert!(matches!(
        task.await.unwrap(),
        CommitCompactionResult::Committed { .. }
    ));
    assert!(matches!(
        handle.get_conversation().await[0],
        ConversationItem::ResponsesCompactionCheckpoint(_)
    ));
}

#[tokio::test]
async fn snapshot_serde_restore_preserves_wrapper_tokens_and_monotonic_generations() {
    let id = identity("grok-test", "system");
    let (persistence, _records) = MockChatPersistence::new();
    let handle = spawn(vec![ConversationItem::system("system")], persistence);
    bind(&handle, id.clone()).await;
    let before = snapshot(&handle).await;
    let replacement = vec![
        wrapper(40, id.clone()),
        ConversationItem::user("12345678901234567890"),
    ];
    assert!(matches!(
        handle
            .commit_compaction(CommitCompaction {
                operation_id: "snapshot-wrapper".into(),
                expected_history_revision: before.history_revision,
                expected_request_identity_generation: before.request_identity_generation,
                replacement: replacement.clone(),
                committed_total_tokens: 45,
            })
            .await
            .unwrap(),
        CommitCompactionResult::Committed { .. }
    ));

    let serialized = serde_json::to_vec(&handle.snapshot().await.unwrap()).unwrap();
    let saved: xai_chat_state::ChatStateSnapshot = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(saved.total_tokens, 45);
    assert_eq!(saved.bound_request_identity.as_ref(), Some(&id));
    assert_eq!(estimate_conversation_tokens(&saved.conversation), 45);

    handle
        .push_user_message_and_ack(ConversationItem::user("abandoned"))
        .await
        .unwrap();
    handle.update_sampling_config(sampling_config("grok-next"));
    let mutated = snapshot(&handle).await;
    handle.restore_snapshot(saved.clone());
    let restored = snapshot(&handle).await;

    assert_eq!(restored.total_tokens, 45);
    assert_eq!(restored.conversation.len(), 2);
    assert!(matches!(
        restored.conversation[0],
        ConversationItem::ResponsesCompactionCheckpoint(_)
    ));
    assert_eq!(restored.bound_request_identity.as_ref(), Some(&id));
    assert!(restored.history_revision > mutated.history_revision);
    assert!(restored.request_identity_generation > mutated.request_identity_generation);
    assert_eq!(estimate_conversation_tokens(&restored.conversation), 45);
}

#[tokio::test]
async fn wrapper_tail_append_waits_for_typed_persistence_ack() {
    let checkpoint = wrapper(40, identity("grok-test", "system"));
    let (persistence, mut records) = MockChatPersistence::new_with_manual_tail_append_ack();
    let handle = spawn(vec![checkpoint], persistence);

    handle.push_assistant_response(ConversationItem::assistant("durable tail"));
    let acknowledgement = records.next_tail_append_ack().await.unwrap();
    let record = records
        .drain()
        .into_iter()
        .find_map(|record| match record {
            xai_chat_state::PersistenceRecord::AcknowledgedTail(tail) => Some(tail),
            _ => None,
        })
        .unwrap();
    assert_eq!(record.checkpoint_id, "checkpoint-1");
    assert_eq!(record.branch_id, "branch-1");
    assert_eq!(record.sequence, 1);
    acknowledgement.send(Ok(())).unwrap();

    let conversation = handle.get_conversation().await;
    assert_eq!(conversation.len(), 2);
    assert!(matches!(conversation[1], ConversationItem::Assistant(_)));
}

#[tokio::test]
async fn wrapper_memory_and_system_head_are_checkpoint_aware() {
    let id = identity("grok-test", "system");
    let (persistence, _records) = MockChatPersistence::new();
    let handle = spawn(
        vec![wrapper(40, id.clone()), ConversationItem::user("tail")],
        persistence,
    );
    bind(&handle, id).await;

    let request = handle
        .build_request(
            vec![],
            Some("<memory-context>remember</memory-context>".into()),
            true,
            None,
            "conv".into(),
            "req".into(),
        )
        .await
        .unwrap();
    assert!(matches!(
        request.items[0],
        ConversationItem::ResponsesCompactionCheckpoint(_)
    ));
    assert!(
        request
            .items
            .iter()
            .skip(1)
            .any(|item| matches!(item, ConversationItem::System(_)))
    );
    let first_len = request.items.len();

    let second = handle
        .build_request(
            vec![],
            Some("<memory-context>remember</memory-context>".into()),
            true,
            None,
            "conv".into(),
            "req".into(),
        )
        .await
        .unwrap();
    assert_eq!(
        second.items.len(),
        first_len,
        "memory reminder is an idempotent tail upsert"
    );

    assert_eq!(
        handle.replace_system_head("system").await,
        Some(ReplaceSystemHeadResult::Unchanged)
    );
    let before = snapshot(&handle).await;
    assert_eq!(
        handle.replace_system_head("different").await,
        Some(ReplaceSystemHeadResult::MigrationRequired)
    );
    let after = snapshot(&handle).await;
    assert_eq!(after.history_revision, before.history_revision);
    assert_eq!(
        after.request_identity_generation,
        before.request_identity_generation + 1
    );
    assert!(matches!(
        handle.get_conversation().await[0],
        ConversationItem::ResponsesCompactionCheckpoint(_)
    ));
}

use super::{
    fingerprint_prefix, prefire_layout_allows, prefire_lead_percent,
    verified_remote_shrink_for_prefire_discard,
};
use xai_grok_sampler::ResponsesCompactResponse;
use xai_grok_sampling_types::{
    CheckpointIdentity, ConversationItem, RESPONSES_COMPACTION_CONTRACT, ResponsesCompactionMode,
    ServerResponsesCheckpoint, TokenSeedSource,
};

fn checkpoint_item() -> ConversationItem {
    ConversationItem::ResponsesCompactionCheckpoint(Box::new(ServerResponsesCheckpoint {
        checkpoint_id: "checkpoint-current".into(),
        operation_id: "operation-current".into(),
        prompt_index: 2,
        created_at: chrono::Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionMode {
            name: "default".into(),
            detail: None,
        },
        branch_id: "branch-current".into(),
        identity: CheckpointIdentity {
            provider_id: "xai".into(),
            api: "responses".into(),
            endpoint_fingerprint: "endpoint".into(),
            model: "grok-test".into(),
            auth_principal_fingerprint: "principal".into(),
            contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
            prompt_envelope_fingerprint: "envelope".into(),
            base_instructions_sha256: "base".into(),
            prior_checkpoint_id: None,
            cache_route_fingerprint: None,
        },
        output: vec![serde_json::json!({
            "type": "compaction",
            "encrypted_content": "opaque"
        })],
        portable_history_path: "compaction_checkpoints/checkpoint-current.json".into(),
        portable_history_sha256: "portable-digest".into(),
        portable_history_bytes: 10,
        checkpoint_token_seed: 25,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 1,
        prior_checkpoint_id: None,
        memory_revision: None,
    }))
}

#[test]
fn fingerprint_stable_for_same_prefix() {
    let items = vec![
        ConversationItem::system("sys"),
        ConversationItem::user("hello"),
        ConversationItem::assistant("hi"),
    ];
    assert_eq!(fingerprint_prefix(&items), fingerprint_prefix(&items));
}

#[test]
fn fingerprint_changes_when_prefix_content_changes() {
    let base = vec![
        ConversationItem::system("sys"),
        ConversationItem::user("hello"),
    ];
    let edited = vec![
        ConversationItem::system("sys"),
        ConversationItem::user("HELLO there"), // a real edit/rewind of the prefix
    ];
    assert_ne!(
        fingerprint_prefix(&base),
        fingerprint_prefix(&edited),
        "a changed prefix must invalidate the cached NOTE1 fingerprint"
    );
}

#[test]
fn fingerprint_changes_with_length() {
    let short = vec![ConversationItem::user("a")];
    let long = vec![
        ConversationItem::user("a"),
        ConversationItem::assistant("b"),
    ];
    assert_ne!(fingerprint_prefix(&short), fingerprint_prefix(&long));
}

#[test]
fn prefire_layout_accepts_normal_and_unique_leading_checkpoint_only() {
    assert!(prefire_layout_allows(&[
        ConversationItem::user("normal"),
        ConversationItem::assistant("history"),
    ]));

    let checkpoint = checkpoint_item();
    assert!(prefire_layout_allows(&[
        checkpoint.clone(),
        ConversationItem::user("typed tail"),
    ]));
    assert!(!prefire_layout_allows(&[
        ConversationItem::user("misplaced"),
        checkpoint.clone(),
    ]));
    assert!(!prefire_layout_allows(&[checkpoint.clone(), checkpoint,]));
}

#[test]
fn prefire_discard_requires_seed_and_committed_total_to_shrink() {
    let response = ResponsesCompactResponse {
        output: vec![serde_json::json!({
            "type": "compaction",
            "encrypted_content": "opaque"
        })],
        usage_output_tokens: Some(95),
        usage_total_tokens: None,
        response_bytes: 32,
        attempts: 1,
    };
    let rejected_seed = crate::session::responses_server_compaction::server_checkpoint_token_seed(
        &response, 5, 100,
    )
    .ok()
    .map(|(seed, _)| seed);
    assert_eq!(rejected_seed, None, "DidNotShrink cannot discard prefire");
    assert_eq!(
        verified_remote_shrink_for_prefire_discard(rejected_seed, 0, 100),
        None
    );
    assert_eq!(
        verified_remote_shrink_for_prefire_discard(Some(80), 20, 100),
        None,
        "a retained tail that erases the shrink must keep prefire"
    );
    assert_eq!(
        verified_remote_shrink_for_prefire_discard(Some(80), 10, 100),
        Some(90)
    );
}

#[test]
fn prefire_lead_percent_defaults_to_10() {
    // SAFETY: single-threaded test mutation of our own env var.
    unsafe { std::env::remove_var("GROK_PREFIRE_LEAD_PERCENT") };
    assert_eq!(prefire_lead_percent(), 10);
}

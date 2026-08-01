use std::sync::Arc;

use chrono::Utc;
use serde_json::{Value, json};
use xai_grok_sampling_types::{
    ApiBackend, CheckpointIdentityV1, ConversationItem, ConversationRequest, FinalResponsesRequest,
    ResponsesCompactionModeV1, ServerResponsesCheckpointV1, SystemItem, TokenSeedSource, UserItem,
    canonical_json_bytes, rs,
};

fn checkpoint(output: Vec<Value>) -> ConversationItem {
    ConversationItem::ResponsesCompactionCheckpoint(Box::new(ServerResponsesCheckpointV1 {
        schema_version: 1,
        checkpoint_id: "checkpoint-1".into(),
        operation_id: "operation-1".into(),
        prompt_index: 7,
        created_at: Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionModeV1 {
            name: "summary".into(),
            detail: None,
        },
        branch_id: "branch-1".into(),
        identity: CheckpointIdentityV1 {
            provider_id: "provider".into(),
            api: "responses".into(),
            endpoint_fingerprint: "endpoint".into(),
            model: "grok-test".into(),
            auth_principal_fingerprint: "principal".into(),
            contract_version: "responses-compact-codex-v1".into(),
            prompt_envelope_fingerprint: "prompt".into(),
            canonical_prompt_projection: Some(json!({"system": "system"})),
        },
        output,
        portable_history_path: "compaction_checkpoints/checkpoint-1.json".into(),
        portable_history_sha256: "digest".into(),
        portable_history_bytes: 123,
        checkpoint_token_seed: 42,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 1,
    }))
}

#[test]
fn wrapper_v1_replay_projection_preserves_raw_output_prefix() {
    let raw_reasoning = json!({
        "type": "reasoning",
        "id": "raw-r1",
        "content": [{"text": "canonical-prefix"}],
        "provider_extension": {"keep": [3, 2, 1]}
    });
    let typed_reasoning = ConversationItem::Reasoning(rs::ReasoningItem {
        id: "tail-r1".into(),
        summary: vec![],
        content: Some(vec![rs::ReasoningTextContent {
            text: "typed-tail".into(),
        }]),
        encrypted_content: None,
        status: None,
    });
    let request = ConversationRequest {
        model: Some("grok-test".into()),
        items: vec![checkpoint(vec![raw_reasoning.clone()]), typed_reasoning],
        ..Default::default()
    };

    // The public typed conversion refuses to flatten checkpoints: replay
    // bodies only exist behind validated construction.
    assert!(FinalResponsesRequest::try_from(&request).is_err());

    // The gate's projection helper still exposes the flattened body shape
    // for identity computation.
    let body = FinalResponsesRequest::replay_projection_body(&request).unwrap();
    let input = body.get("input").and_then(Value::as_array).unwrap();

    assert_eq!(
        input[0], raw_reasoning,
        "canonical prefix must be byte-shape stable"
    );
    assert_eq!(input[1]["id"], "tail-r1");
    assert_eq!(input[1]["content"][0]["type"], "reasoning_text");
}

#[test]
fn wrapper_layout_and_non_responses_backends_fail_closed() {
    let wrapper = checkpoint(vec![json!({
        "type": "compaction",
        "encrypted_content": "opaque"
    })]);
    let request = ConversationRequest {
        items: vec![
            ConversationItem::User(UserItem {
                content: vec![],
                ..Default::default()
            }),
            wrapper.clone(),
        ],
        ..Default::default()
    };
    assert!(FinalResponsesRequest::try_from(&request).is_err());

    let duplicate = ConversationRequest {
        items: vec![wrapper.clone(), wrapper],
        ..Default::default()
    };
    assert!(FinalResponsesRequest::try_from(&duplicate).is_err());

    let valid = ConversationRequest {
        items: vec![checkpoint(vec![json!({
            "type": "compaction",
            "encrypted_content": "opaque"
        })])],
        ..Default::default()
    };
    assert!(valid.validate_for_backend(&ApiBackend::Responses).is_ok());
    // Even a structurally valid lone wrapper cannot be flattened through
    // the public typed conversion.
    assert!(FinalResponsesRequest::try_from(&valid).is_err());
    assert!(
        valid
            .validate_for_backend(&ApiBackend::ChatCompletions)
            .is_err()
    );
    assert!(valid.validate_for_backend(&ApiBackend::Messages).is_err());
}

#[test]
fn normal_responses_fields_are_serialized() {
    let request = ConversationRequest {
        items: vec![ConversationItem::System(SystemItem {
            content: Arc::from("system"),
            source: Default::default(),
        })],
        model: Some("grok-test".into()),
        instructions: Some("instructions".into()),
        prompt_cache_key: Some("cache-key".into()),
        prompt_cache_options: Some(json!({"scope": "session"})),
        prompt_cache_retention: Some("24h".into()),
        service_tier: Some("priority".into()),
        ..Default::default()
    };

    let body = FinalResponsesRequest::try_from(&request)
        .unwrap()
        .into_body();
    assert_eq!(body["instructions"], "instructions");
    assert_eq!(body["prompt_cache_key"], "cache-key");
    assert_eq!(body["prompt_cache_options"], json!({"scope": "session"}));
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["service_tier"], "priority");
}

#[test]
fn canonical_json_sorts_nested_object_keys() {
    let a = json!({"z": 1, "a": {"y": 2, "b": 3}, "m": [{"d": 4, "c": 5}]});
    let b = json!({"m": [{"c": 5, "d": 4}], "a": {"b": 3, "y": 2}, "z": 1});

    assert_eq!(
        canonical_json_bytes(&a).unwrap(),
        canonical_json_bytes(&b).unwrap()
    );
    assert_eq!(
        String::from_utf8(canonical_json_bytes(&a).unwrap()).unwrap(),
        r#"{"a":{"b":3,"y":2},"m":[{"c":5,"d":4}],"z":1}"#
    );
}

use std::sync::Arc;

use chrono::Utc;
use serde_json::{Value, json};
use xai_grok_sampling_types::{
    ApiBackend, CheckpointIdentity, CheckpointReplayMaterial, ConversationItem,
    ConversationRequest, FinalResponsesRequest, RESPONSES_COMPACTION_CONTRACT,
    ResolvedResponsesRequest, ResponsesCompactionMode, ServerResponsesCheckpoint, SystemItem,
    TokenSeedSource, TrustedPromptEnvelope, ValidatedResponsesReplay, canonical_json_bytes,
    compose_instructions, portable_history_digest, rs,
};

fn envelope() -> TrustedPromptEnvelope {
    TrustedPromptEnvelope {
        base_instructions_sha256: "base-hash".into(),
        memory_revision: None,
        envelope_fingerprint: "prompt".into(),
        wire_prompt_sha256: "wire-hash".into(),
    }
}

fn checkpoint(portable_history: &[ConversationItem], output: Vec<Value>) -> ConversationItem {
    ConversationItem::ResponsesCompactionCheckpoint(Box::new(ServerResponsesCheckpoint {
        checkpoint_id: "checkpoint-1".into(),
        operation_id: "operation-1".into(),
        prompt_index: 7,
        created_at: Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionMode {
            name: "summary".into(),
            detail: None,
        },
        branch_id: "branch-1".into(),
        identity: CheckpointIdentity {
            provider_id: "provider".into(),
            api: "responses".into(),
            endpoint_fingerprint: "endpoint".into(),
            model: "grok-test".into(),
            auth_principal_fingerprint: "principal".into(),
            contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
            prompt_envelope_fingerprint: "prompt".into(),
            base_instructions_sha256: "base-hash".into(),
            prior_checkpoint_id: None,
            cache_route_fingerprint: None,
        },
        output,
        portable_history_path: "compaction_checkpoints/checkpoint-1.json".into(),
        portable_history_sha256: portable_history_digest(portable_history).unwrap(),
        portable_history_bytes: 123,
        checkpoint_token_seed: 42,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 1,
        prior_checkpoint_id: None,
        memory_revision: None,
    }))
}

#[test]
fn validated_replay_preserves_raw_output_prefix() {
    let portable_history = vec![
        ConversationItem::base_instructions("base"),
        ConversationItem::user("compacted user"),
    ];
    let raw_reasoning = json!({
        "type": "reasoning",
        "id": "raw-r1",
        "content": [{"text": "canonical-prefix"}],
        "provider_extension": {"keep": [3, 2, 1]}
    });
    let checkpoint = checkpoint(&portable_history, vec![raw_reasoning.clone()]);
    let wrapper = checkpoint.as_responses_checkpoint().unwrap();
    let material =
        CheckpointReplayMaterial::try_new(wrapper, envelope(), &portable_history).unwrap();
    let typed_tail = vec![ConversationItem::Reasoning(rs::ReasoningItem {
        id: "tail-r1".into(),
        summary: vec![],
        content: Some(vec![rs::ReasoningTextContent {
            text: "typed-tail".into(),
        }]),
        encrypted_content: None,
        status: None,
    })];
    let replay = ValidatedResponsesReplay::verify(
        wrapper,
        &material,
        &portable_history,
        &envelope(),
        &typed_tail,
        7,
        1,
    )
    .unwrap();
    let request = ConversationRequest {
        model: Some("grok-test".into()),
        instructions: compose_instructions(&portable_history),
        items: std::iter::once(checkpoint).chain(typed_tail).collect(),
        ..Default::default()
    };

    assert!(FinalResponsesRequest::try_from(&request).is_err());
    let resolved = ResolvedResponsesRequest::from_validated_replay(&replay, &request).unwrap();
    let input = resolved
        .body()
        .get("input")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(input[0], raw_reasoning);
    assert_eq!(input[1]["id"], "tail-r1");
    assert_eq!(input[1]["content"][0]["type"], "reasoning_text");
}

#[test]
fn wrapper_layout_and_non_responses_backends_fail_closed() {
    let portable = vec![ConversationItem::user("old")];
    let wrapper = checkpoint(
        &portable,
        vec![json!({
            "type": "compaction",
            "encrypted_content": "opaque"
        })],
    );
    let request = ConversationRequest {
        items: vec![ConversationItem::user("before"), wrapper.clone()],
        ..Default::default()
    };
    assert!(
        request
            .validate_for_backend(&ApiBackend::Responses)
            .is_err()
    );

    let duplicate = ConversationRequest {
        items: vec![wrapper.clone(), wrapper.clone()],
        ..Default::default()
    };
    assert!(
        duplicate
            .validate_for_backend(&ApiBackend::Responses)
            .is_err()
    );

    let valid = ConversationRequest {
        items: vec![wrapper],
        ..Default::default()
    };
    assert!(valid.validate_for_backend(&ApiBackend::Responses).is_ok());
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

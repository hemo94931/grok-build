use std::time::Duration;

use xai_grok_sampler::{ResponsesCompactFailure, ResponsesCompactResponse};
use xai_grok_sampling_types::{
    CheckpointIdentity, ConversationItem, RESPONSES_COMPACTION_CONTRACT, ResponsesCompactionMode,
    TokenSeedSource, is_valid_compaction_item,
};
use xai_grok_shell::session::responses_server_compaction::{
    CapabilityKey, NegativeCapabilityCache, ServerCheckpointSeedError,
    ServerCompactionFailureReason, build_server_successor, classify_compact_failure,
    current_identity_for_recompact_binding, prompt_envelope_token_estimate,
    resolve_compact_model_layers, resolve_server_compaction_layers, resolved_prompt_envelope,
    server_checkpoint_token_seed, should_send_inline_compaction_headers,
};

fn identity(prior: Option<&str>) -> CheckpointIdentity {
    CheckpointIdentity {
        provider_id: "xai".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
        prompt_envelope_fingerprint: "envelope-fingerprint".into(),
        base_instructions_sha256: "base-hash".into(),
        prior_checkpoint_id: prior.map(str::to_owned),
        cache_route_fingerprint: Some("route-fingerprint".into()),
    }
}

fn compaction_blob() -> serde_json::Value {
    serde_json::json!({
        "type": "compaction",
        "encrypted_content": "opaque",
        "future": {"z": 1, "a": 2}
    })
}

#[test]
fn remote_setting_and_default_enable_the_single_writer_without_an_extra_gate() {
    assert!(resolve_server_compaction_layers(None, None, None));
    assert!(
        resolve_server_compaction_layers(None, None, Some(true)),
        "the remote setting alone must enable the current writer"
    );
    assert!(!resolve_server_compaction_layers(None, None, Some(false)));
    assert!(resolve_server_compaction_layers(
        None,
        Some(true),
        Some(false)
    ));
    assert!(!resolve_server_compaction_layers(
        Some(false),
        Some(true),
        Some(true)
    ));

    assert!(!should_send_inline_compaction_headers(
        &xai_grok_sampling_types::ApiBackend::Responses
    ));
    assert!(should_send_inline_compaction_headers(
        &xai_grok_sampling_types::ApiBackend::ChatCompletions
    ));
}

#[test]
fn compact_model_layers_keep_valid_precedence_and_safe_fallback() {
    let known = ["current", "compact", "remote"];
    assert_eq!(
        resolve_compact_model_layers(
            Some("compact"),
            Some("ignored"),
            Some("remote"),
            "current",
            |model| known.contains(&model),
        ),
        "compact"
    );
    assert_eq!(
        resolve_compact_model_layers(None, Some("unknown"), Some("remote"), "current", |model| {
            known.contains(&model)
        }),
        "current",
        "an invalid configured slug falls back to the current model"
    );
    assert_eq!(
        resolve_compact_model_layers(None, None, Some("remote"), "current", |model| {
            known.contains(&model)
        }),
        "remote"
    );
}

#[test]
fn generic_successor_uses_retained_prefix_plus_blob() {
    assert_eq!(RESPONSES_COMPACTION_CONTRACT, "responses-compact-grok");
    let retained = vec![
        ConversationItem::user("kept-user"),
        ConversationItem::system("kept-system"),
    ];
    let blob = compaction_blob();
    let tail = vec![ConversationItem::system_reminder("transcript pointer")];
    let successor = build_server_successor(
        "checkpoint-current",
        "operation-current",
        7,
        true,
        ResponsesCompactionMode {
            name: "segments".into(),
            detail: Some(serde_json::Value::String("balanced".into())),
        },
        "branch-current",
        identity(Some("checkpoint-prior")),
        retained.clone(),
        blob.clone(),
        "compaction_checkpoints/checkpoint-current.json",
        "portable-digest".into(),
        25,
        TokenSeedSource::UsageOutputTokens,
        Some("checkpoint-prior".into()),
        Some(3),
        tail.clone(),
    );

    let ConversationItem::ResponsesCompactionCheckpoint(wrapper) = &successor[0] else {
        panic!("checkpoint wrapper must be index zero");
    };
    assert_eq!(
        wrapper.identity.contract_version,
        RESPONSES_COMPACTION_CONTRACT
    );
    assert_eq!(wrapper.retained_prefix.len(), retained.len());
    assert_eq!(wrapper.compaction_item, blob);
    assert!(is_valid_compaction_item(&wrapper.compaction_item));
    assert_eq!(wrapper.checkpoint_token_seed, 25);
    assert_eq!(
        wrapper.prior_checkpoint_id.as_deref(),
        Some("checkpoint-prior")
    );
    assert_eq!(successor[1..].len(), tail.len());
    assert_eq!(successor[1].text_content(), tail[0].text_content());
    assert_eq!(
        wrapper.wrapper_digest(),
        wrapper.wrapper_digest_for_branch(&wrapper.branch_id)
    );

    let serialized = serde_json::to_value(&successor[0]).unwrap();
    assert_eq!(
        serialized.get("type").and_then(serde_json::Value::as_str),
        Some("responses_compaction_checkpoint")
    );
    // Old unary shape must not appear on the wire of the live wrapper.
    assert!(serialized.get("output").is_none() || serialized["output"].is_null());
    // serde tag nests fields under the variant; check wrapper fields via typed access.
    assert!(serialized.get("server_output_item_count").is_none());

    let mut successor_identity = wrapper.identity.clone();
    successor_identity.prior_checkpoint_id = Some(wrapper.checkpoint_id.clone());
    let current = current_identity_for_recompact_binding(&successor_identity, wrapper);
    assert_eq!(current, wrapper.identity);
    assert_eq!(
        successor_identity.prior_checkpoint_id.as_deref(),
        Some(wrapper.checkpoint_id.as_str())
    );
}

#[test]
fn checkpoint_seed_prefers_usage_and_reports_did_not_shrink() {
    let blob = serde_json::json!({
        "type": "compaction",
        "encrypted_content": "opaque"
    });
    let with_usage = ResponsesCompactResponse {
        compaction_item: blob.clone(),
        usage_output_tokens: Some(40),
        usage_total_tokens: Some(999),
        response_bytes: 100,
        attempts: 1,
    };
    assert_eq!(
        server_checkpoint_token_seed(&with_usage, 5, 100).unwrap(),
        (45, TokenSeedSource::UsageOutputTokens)
    );
    assert!(matches!(
        server_checkpoint_token_seed(&with_usage, 60, 100),
        Err(ServerCheckpointSeedError::DidNotShrink)
    ));

    let estimated = ResponsesCompactResponse {
        compaction_item: blob,
        usage_output_tokens: None,
        usage_total_tokens: None,
        response_bytes: 100,
        attempts: 1,
    };
    let (seed, source) = server_checkpoint_token_seed(&estimated, 5, 100).unwrap();
    assert!((6..100).contains(&seed));
    assert_eq!(source, TokenSeedSource::EstimatedCanonicalOutput);

    // Estimate falls back to the single blob; large encrypted content still
    // produces a non-zero seed without treating image bytes specially inside
    // the opaque provider payload.
    let large_blob = ResponsesCompactResponse {
        compaction_item: serde_json::json!({
            "type": "compaction",
            "encrypted_content": "a".repeat(10_000)
        }),
        usage_output_tokens: None,
        usage_total_tokens: None,
        response_bytes: 100_000,
        attempts: 1,
    };
    let (large_seed, source) = server_checkpoint_token_seed(&large_blob, 0, 10_000).unwrap();
    assert_eq!(source, TokenSeedSource::EstimatedCanonicalOutput);
    assert!(large_seed >= 1);
    assert!(large_seed < 10_000);
}

#[test]
fn semantic_envelope_excludes_transcript_and_transport_fields() {
    let first = serde_json::json!({
        "input": [{"role": "user", "content": "first transcript"}],
        "instructions": "stable",
        "tools": [{"type": "function", "name": "read"}],
        "tool_choice": "auto",
        "reasoning": {"effort": "medium"},
        "text": {"format": {"type": "text"}},
        "parallel_tool_calls": true,
        "prompt_cache_key": "cache-a",
        "service_tier": "flex"
    });
    let mut transport_changed = first.clone();
    transport_changed["input"] = serde_json::json!([{"role": "user", "content": "different"}]);
    transport_changed["prompt_cache_key"] = serde_json::json!("cache-b");
    transport_changed["service_tier"] = serde_json::json!("priority");

    let envelope = resolved_prompt_envelope(&first).unwrap();
    assert_eq!(
        envelope,
        resolved_prompt_envelope(&transport_changed).unwrap()
    );
    assert!(envelope.get("input").is_none());
    assert!(envelope.get("prompt_cache_key").is_none());
    assert_eq!(
        prompt_envelope_token_estimate(&first).unwrap(),
        prompt_envelope_token_estimate(&transport_changed).unwrap()
    );

    transport_changed["instructions"] = serde_json::json!("changed");
    assert_ne!(
        envelope,
        resolved_prompt_envelope(&transport_changed).unwrap()
    );
}

#[test]
fn failure_mapping_is_fixed_and_cancel_never_falls_back() {
    use ServerCompactionFailureReason as Reason;

    assert_eq!(
        classify_compact_failure(ResponsesCompactFailure::Cancelled, None, None),
        None
    );
    for (failure, status, code, expected) in [
        (
            ResponsesCompactFailure::HttpStatus,
            Some(404),
            None,
            Reason::Unsupported,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(400),
            None,
            Reason::Unsupported,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(422),
            None,
            Reason::Unsupported,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(405),
            None,
            Reason::Unsupported,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(501),
            None,
            Reason::Unsupported,
        ),
        (
            ResponsesCompactFailure::CompletedWithoutCompaction,
            None,
            None,
            Reason::Unsupported,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(401),
            None,
            Reason::Auth,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(429),
            None,
            Reason::RateLimited,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(400),
            Some("insufficient_quota"),
            Reason::Quota,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(400),
            Some("context_length_exceeded"),
            Reason::ContextOverflow,
        ),
        (
            ResponsesCompactFailure::Timeout,
            None,
            None,
            Reason::Timeout,
        ),
        (
            ResponsesCompactFailure::Transport,
            None,
            None,
            Reason::Transport,
        ),
        (
            ResponsesCompactFailure::MissingCredential,
            None,
            None,
            Reason::Auth,
        ),
        (
            ResponsesCompactFailure::ResponseTooLarge,
            None,
            None,
            Reason::InvalidResponse,
        ),
        (
            ResponsesCompactFailure::HttpStatus,
            Some(503),
            None,
            Reason::Server,
        ),
        (
            ResponsesCompactFailure::RequestTooLarge,
            None,
            None,
            Reason::RequestTooLarge,
        ),
    ] {
        assert_eq!(
            classify_compact_failure(failure, status, code),
            Some(expected)
        );
    }
}

#[test]
fn unsupported_negative_cache_is_keyed_and_expires_after_one_hour() {
    let key = CapabilityKey {
        endpoint_fingerprint: "endpoint-a".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
    };
    let distinct = CapabilityKey {
        endpoint_fingerprint: "endpoint-b".into(),
        ..key.clone()
    };
    let mut cache = NegativeCapabilityCache::new(Duration::from_secs(3600));

    assert!(!cache.is_unsupported(&key, Duration::ZERO));
    cache.record_unsupported(key.clone(), Duration::from_secs(10));
    assert!(cache.is_unsupported(&key, Duration::from_secs(3609)));
    assert!(!cache.is_unsupported(&distinct, Duration::from_secs(20)));
    assert!(!cache.is_unsupported(&key, Duration::from_secs(3610)));

    assert!(NegativeCapabilityCache::status_is_unsupported(400));
    assert!(NegativeCapabilityCache::status_is_unsupported(404));
    assert!(NegativeCapabilityCache::status_is_unsupported(405));
    assert!(NegativeCapabilityCache::status_is_unsupported(422));
    assert!(NegativeCapabilityCache::status_is_unsupported(501));
    assert!(!NegativeCapabilityCache::status_is_unsupported(429));
    assert!(!NegativeCapabilityCache::status_is_unsupported(500));
}

/// D6 recording policy: CompletedWithoutCompaction and HTTP 400/404/405/422/501
/// are unsupported (and thus negative-cache candidates); transport/timeout are not.
#[test]
fn d6_unsupported_classification_for_negative_cache() {
    assert_eq!(
        classify_compact_failure(
            ResponsesCompactFailure::CompletedWithoutCompaction,
            None,
            None
        ),
        Some(ServerCompactionFailureReason::Unsupported)
    );
    for status in [400, 404, 405, 422, 501] {
        assert_eq!(
            classify_compact_failure(ResponsesCompactFailure::HttpStatus, Some(status), None),
            Some(ServerCompactionFailureReason::Unsupported),
            "status {status} must be unsupported"
        );
        assert!(NegativeCapabilityCache::status_is_unsupported(status));
    }
    // Transport / timeout fall back without recording.
    assert_eq!(
        classify_compact_failure(ResponsesCompactFailure::Transport, None, None),
        Some(ServerCompactionFailureReason::Transport)
    );
    assert_eq!(
        classify_compact_failure(ResponsesCompactFailure::Timeout, None, None),
        Some(ServerCompactionFailureReason::Timeout)
    );
    assert!(!NegativeCapabilityCache::status_is_unsupported(408));
    assert!(!NegativeCapabilityCache::status_is_unsupported(500));
}

#[test]
fn old_output_shape_sidecar_fails_closed_on_deserialize() {
    // D1: no compat reader for the unary `/responses/compact` checkpoint.
    let old_shape = serde_json::json!({
        "checkpoint_id": "c",
        "operation_id": "o",
        "prompt_index": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "auto_continue": false,
        "mode": {"name": "default"},
        "branch_id": "b",
        "identity": {
            "provider_id": "xai",
            "api": "responses",
            "endpoint_fingerprint": "e",
            "model": "m",
            "auth_principal_fingerprint": "p",
            "contract_version": RESPONSES_COMPACTION_CONTRACT,
            "prompt_envelope_fingerprint": "env",
            "base_instructions_sha256": "base"
        },
        "output": [{"type": "compaction", "encrypted_content": "opaque"}],
        "portable_history_path": "compaction_checkpoints/c.json",
        "portable_history_sha256": "digest",
        "portable_history_bytes": 1,
        "checkpoint_token_seed": 1,
        "token_seed_source": "usage_output_tokens",
        "server_output_item_count": 1
    });
    assert!(
        serde_json::from_value::<xai_grok_sampling_types::ServerResponsesCheckpoint>(old_shape)
            .is_err(),
        "old unary output shape must fail closed"
    );
}

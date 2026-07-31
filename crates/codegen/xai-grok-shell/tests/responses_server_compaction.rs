use std::time::Duration;

use xai_grok_sampler::{ResponsesCompactFailure, ResponsesCompactResponse};
use xai_grok_sampling_types::{
    ConversationItem, ConversationRequest, FinalResponsesRequest, ResponsesCompactionModeV1,
    TokenSeedSource,
};
use xai_grok_shell::session::responses_server_compaction::{
    CapabilityKey, NegativeCapabilityCache, ServerCompactionFailureReason,
    build_checkpoint_identity, build_server_successor, classify_compact_failure,
    resolve_compact_model_layers, resolve_server_compaction_layers, server_checkpoint_token_seed,
    should_send_inline_compaction_headers,
};

#[test]
fn config_precedence_and_model_fallback_are_frozen() {
    assert!(resolve_server_compaction_layers(None, None, None));
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
    assert!(!should_send_inline_compaction_headers(
        true,
        &xai_grok_sampling_types::ApiBackend::Responses
    ));
    assert!(should_send_inline_compaction_headers(
        false,
        &xai_grok_sampling_types::ApiBackend::Responses
    ));
    assert!(should_send_inline_compaction_headers(
        true,
        &xai_grok_sampling_types::ApiBackend::ChatCompletions
    ));
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
            Some(403),
            None,
            Reason::InvalidResponse,
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
        (
            ResponsesCompactFailure::InvalidResponse,
            None,
            Some("untrusted free-form text"),
            Reason::InvalidResponse,
        ),
    ] {
        assert_eq!(
            classify_compact_failure(failure, status, code),
            Some(expected)
        );
    }
}

#[test]
fn prompt_identity_is_semantic_and_excludes_transcript_cache_and_tier() {
    fn final_request(
        system: &str,
        user: &str,
        cache: &str,
        instructions: &str,
    ) -> FinalResponsesRequest {
        FinalResponsesRequest::try_from(&ConversationRequest {
            items: vec![
                ConversationItem::system(system),
                ConversationItem::user(user),
            ],
            model: Some("grok".into()),
            instructions: Some(instructions.into()),
            prompt_cache_key: Some(cache.into()),
            service_tier: Some("flex".into()),
            ..Default::default()
        })
        .unwrap()
    }

    let first = build_checkpoint_identity(
        "provider",
        "endpoint",
        "principal",
        &final_request("system", "first transcript", "cache-a", "stable"),
    )
    .unwrap();
    let same_semantics = build_checkpoint_identity(
        "provider",
        "endpoint",
        "principal",
        &final_request("system", "different transcript", "cache-b", "stable"),
    )
    .unwrap();
    let changed_system = build_checkpoint_identity(
        "provider",
        "endpoint",
        "principal",
        &final_request(
            "changed system",
            "different transcript",
            "cache-b",
            "stable",
        ),
    )
    .unwrap();
    let changed_instructions = build_checkpoint_identity(
        "provider",
        "endpoint",
        "principal",
        &final_request("system", "different transcript", "cache-b", "changed"),
    )
    .unwrap();

    assert_eq!(
        first.prompt_envelope_fingerprint,
        same_semantics.prompt_envelope_fingerprint
    );
    assert_ne!(
        first.prompt_envelope_fingerprint,
        changed_system.prompt_envelope_fingerprint
    );
    assert_ne!(
        first.prompt_envelope_fingerprint,
        changed_instructions.prompt_envelope_fingerprint
    );
    assert_eq!(
        first.canonical_prompt_projection.unwrap()[0]["role"],
        "system"
    );
}

#[test]
fn checkpoint_seed_prefers_usage_and_rejects_non_shrinking_output() {
    let output = vec![serde_json::json!({
        "type": "compaction",
        "encrypted_content": "opaque"
    })];
    let with_usage = ResponsesCompactResponse {
        output: output.clone(),
        usage_output_tokens: Some(40),
        usage_total_tokens: Some(999),
        response_bytes: 100,
        attempts: 1,
    };
    assert_eq!(
        server_checkpoint_token_seed(&with_usage, 5, 100).unwrap(),
        (45, TokenSeedSource::UsageOutputTokens)
    );

    let estimated = ResponsesCompactResponse {
        output,
        usage_output_tokens: None,
        usage_total_tokens: None,
        response_bytes: 100,
        attempts: 1,
    };
    let (seed, source) = server_checkpoint_token_seed(&estimated, 5, 100).unwrap();
    assert!((6..100).contains(&seed));
    assert_eq!(source, TokenSeedSource::EstimatedCanonicalOutput);
    assert!(server_checkpoint_token_seed(&with_usage, 60, 100).is_err());

    let image_response = ResponsesCompactResponse {
        output: vec![
            serde_json::json!({
                "type": "compaction",
                "encrypted_content": "opaque"
            }),
            serde_json::json!({
                "type": "message",
                "content": [{
                    "type": "input_image",
                    "image_url": format!("data:image/png;base64,{}", "a".repeat(100_000))
                }]
            }),
        ],
        usage_output_tokens: None,
        usage_total_tokens: None,
        response_bytes: 100_000,
        attempts: 1,
    };
    let (image_seed, source) = server_checkpoint_token_seed(&image_response, 0, 10_000).unwrap();
    assert_eq!(source, TokenSeedSource::EstimatedCanonicalOutput);
    assert!(image_seed >= xai_token_estimation::IMAGE_TOKEN_ESTIMATE);
    assert!(
        image_seed < 10_000,
        "image bytes must use the image estimate"
    );
}

#[test]
fn server_successor_preserves_opaque_output_and_counts_tail_once() {
    let output = vec![serde_json::json!({
        "type": "compaction",
        "encrypted_content": "opaque",
        "future": {"z": 1, "a": 2}
    })];
    let tail = vec![ConversationItem::system_reminder("transcript pointer")];
    let successor = build_server_successor(
        "checkpoint",
        "operation",
        7,
        false,
        ResponsesCompactionModeV1 {
            name: "transcript".into(),
            detail: None,
        },
        "branch",
        build_checkpoint_identity(
            "provider",
            "endpoint",
            "principal",
            &FinalResponsesRequest::try_from(&ConversationRequest {
                items: vec![
                    ConversationItem::system("system"),
                    ConversationItem::user("u"),
                ],
                model: Some("grok".into()),
                ..Default::default()
            })
            .unwrap(),
        )
        .unwrap(),
        output.clone(),
        "compaction_checkpoints/checkpoint.json",
        25,
        TokenSeedSource::UsageOutputTokens,
        tail.clone(),
    );

    let ConversationItem::ResponsesCompactionCheckpoint(wrapper) = &successor[0] else {
        panic!("wrapper must be index zero");
    };
    assert_eq!(wrapper.output, output);
    assert_eq!(successor[1..].len(), tail.len());
    assert_eq!(wrapper.checkpoint_token_seed, 25);
}

#[test]
fn unsupported_negative_cache_is_keyed_and_expires_after_one_hour() {
    let key = CapabilityKey {
        endpoint_fingerprint: "endpoint-a".into(),
        model: "grok".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: "responses-compact-codex-v1".into(),
    };
    let distinct_keys = [
        CapabilityKey {
            endpoint_fingerprint: "endpoint-b".into(),
            ..key.clone()
        },
        CapabilityKey {
            model: "other".into(),
            ..key.clone()
        },
        CapabilityKey {
            auth_principal_fingerprint: "other-principal".into(),
            ..key.clone()
        },
        CapabilityKey {
            contract_version: "future-contract".into(),
            ..key.clone()
        },
    ];
    let mut cache = NegativeCapabilityCache::new(Duration::from_secs(3600));

    assert!(!cache.is_unsupported(&key, Duration::ZERO));
    cache.record_unsupported(key.clone(), Duration::from_secs(10));
    assert!(cache.is_unsupported(&key, Duration::from_secs(3609)));
    for distinct in &distinct_keys {
        assert!(!cache.is_unsupported(distinct, Duration::from_secs(20)));
    }
    assert!(!cache.is_unsupported(&key, Duration::from_secs(3610)));

    assert!(NegativeCapabilityCache::status_is_unsupported(404));
    assert!(NegativeCapabilityCache::status_is_unsupported(405));
    assert!(NegativeCapabilityCache::status_is_unsupported(501));
    assert!(!NegativeCapabilityCache::status_is_unsupported(429));
    assert!(!NegativeCapabilityCache::status_is_unsupported(500));
}

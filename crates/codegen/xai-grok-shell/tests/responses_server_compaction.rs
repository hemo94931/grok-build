use std::time::Duration;

use xai_grok_sampler::{ResponsesCompactFailure, ResponsesCompactResponse};
use xai_grok_sampling_types::{
    ConversationItem, ConversationRequest, FinalResponsesRequest, RESPONSES_CHECKPOINT_SCHEMA_V2,
    RESPONSES_COMPACTION_CONTRACT_V2, ResponsesCompactionModeV1, TokenSeedSource,
};
use xai_grok_shell::session::responses_server_compaction::{
    CapabilityKey, NegativeCapabilityCache, ServerCompactionFailureReason,
    build_checkpoint_identity, build_server_successor, build_server_successor_v2,
    classify_compact_failure, current_v2_identity_for_recompact_binding,
    resolve_compact_model_layers, resolve_server_compaction_layers, server_checkpoint_token_seed,
    should_send_inline_compaction_headers,
    v1_migration_cohort_percent, v2_server_compaction_writers_enabled, v2_writer_cohort_allows,
    v2_writer_cohort_percent,
};

/// Serializes env-mutating tests: the harness runs tests in one process,
/// and `GROK_V2_WRITER_PERCENT` / `GROK_RESPONSES_V2_SERVER_COMPACTION` are
/// process-global.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn v2_writer_flags_default_off_and_cohort_is_deterministic() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    // Defaults: V2 writers off, 0% cohort.
    unsafe {
        std::env::remove_var("GROK_RESPONSES_V2_SERVER_COMPACTION");
        std::env::remove_var("GROK_V2_WRITER_PERCENT");
    }
    assert!(!v2_server_compaction_writers_enabled());
    assert_eq!(v2_writer_cohort_percent(), 0);
    assert!(!v2_writer_cohort_allows("any-session"));
    assert!(v1_migration_cohort_percent("any-session", 100));

    // Flag accepts the same truthy spellings as the V1 kill switch.
    for value in ["1", "true", "yes", "on", " 1 "] {
        unsafe { std::env::set_var("GROK_RESPONSES_V2_SERVER_COMPACTION", value) };
        assert!(v2_server_compaction_writers_enabled());
    }
    for value in ["0", "false", "no", "off", "", "garbage"] {
        unsafe { std::env::set_var("GROK_RESPONSES_V2_SERVER_COMPACTION", value) };
        assert!(!v2_server_compaction_writers_enabled());
    }
    unsafe { std::env::remove_var("GROK_RESPONSES_V2_SERVER_COMPACTION") };

    // Percent is clamped to [0, 100] and 0% admits nobody, 100% everybody.
    unsafe { std::env::set_var("GROK_V2_WRITER_PERCENT", "0") };
    assert_eq!(v2_writer_cohort_percent(), 0);
    assert!(!v2_writer_cohort_allows("session-a"));
    unsafe { std::env::set_var("GROK_V2_WRITER_PERCENT", "100") };
    assert_eq!(v2_writer_cohort_percent(), 100);
    assert!(v2_writer_cohort_allows("session-a"));
    unsafe { std::env::set_var("GROK_V2_WRITER_PERCENT", "250") };
    assert_eq!(v2_writer_cohort_percent(), 100);
    unsafe { std::env::set_var("GROK_V2_WRITER_PERCENT", "-5") };
    assert_eq!(v2_writer_cohort_percent(), 0);
    unsafe { std::env::set_var("GROK_V2_WRITER_PERCENT", "not-a-number") };
    assert_eq!(v2_writer_cohort_percent(), 0);
    unsafe { std::env::remove_var("GROK_V2_WRITER_PERCENT") };

    // Deterministic + monotonic: same session lands in the same bucket and
    // a session admitted at p% stays admitted at every q >= p.
    unsafe { std::env::set_var("GROK_V2_WRITER_PERCENT", "50") };
    let first = v2_writer_cohort_allows("session-b");
    for _ in 0..8 {
        assert_eq!(first, v2_writer_cohort_allows("session-b"));
    }
    for session in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"] {
        let mut admitted_at = None;
        for percent in 1..=100u32 {
            if v1_migration_cohort_percent(session, percent) {
                admitted_at = Some(percent);
                break;
            }
        }
        let admitted_at = admitted_at.expect("every session admits at 100%");
        for percent in admitted_at..=100u32 {
            assert!(
                v1_migration_cohort_percent(session, percent),
                "session {session} admitted at {admitted_at} must stay admitted at {percent}"
            );
        }
    }
    unsafe { std::env::remove_var("GROK_V2_WRITER_PERCENT") };
}

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
        final_request("system", "first transcript", "cache-a", "stable").body(),
    )
    .unwrap();
    let same_semantics = build_checkpoint_identity(
        "provider",
        "endpoint",
        "principal",
        final_request("system", "different transcript", "cache-b", "stable").body(),
    )
    .unwrap();
    let changed_system = build_checkpoint_identity(
        "provider",
        "endpoint",
        "principal",
        final_request(
            "changed system",
            "different transcript",
            "cache-b",
            "stable",
        )
        .body(),
    )
    .unwrap();
    let changed_instructions = build_checkpoint_identity(
        "provider",
        "endpoint",
        "principal",
        final_request("system", "different transcript", "cache-b", "changed").body(),
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
            FinalResponsesRequest::try_from(&ConversationRequest {
                items: vec![
                    ConversationItem::system("system"),
                    ConversationItem::user("u"),
                ],
                model: Some("grok".into()),
                ..Default::default()
            })
            .unwrap()
            .body(),
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
fn server_successor_v2_shape_and_digest_binding() {
    use xai_grok_sampling_types::{wrapper_digest_v2, CheckpointIdentityV2};

    let output = vec![serde_json::json!({
        "type": "compaction",
        "encrypted_content": "opaque",
        "future": {"z": 1, "a": 2}
    })];
    let portable = vec![
        ConversationItem::base_instructions("base prompt"),
        ConversationItem::user("first"),
        ConversationItem::assistant("answer"),
    ];
    let digest = xai_grok_shell::session::storage::responses_compaction::portable_history_digest(
        &portable,
    )
    .unwrap();
    let identity = CheckpointIdentityV2 {
        provider_id: "xai".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: RESPONSES_COMPACTION_CONTRACT_V2.into(),
        prompt_envelope_fingerprint: "envelope-fp".into(),
        base_instructions_sha256: "base-hash".into(),
        prior_checkpoint_id: Some("cp-prior".into()),
        cache_route_fingerprint: Some("route-fp".into()),
    };
    let tail = vec![ConversationItem::system_reminder("transcript pointer")];
    let successor = build_server_successor_v2(
        "checkpoint",
        "operation",
        7,
        true,
        ResponsesCompactionModeV1 {
            name: "segments".into(),
            detail: Some(serde_json::Value::String("balanced".into())),
        },
        "branch",
        identity.clone(),
        output.clone(),
        "compaction_checkpoints/checkpoint.json",
        digest.clone(),
        25,
        TokenSeedSource::UsageOutputTokens,
        Some("cp-prior".into()),
        tail.clone(),
    );

    let ConversationItem::ResponsesCompactionCheckpointV2(wrapper) = &successor[0] else {
        panic!("V2 wrapper must be index zero");
    };
    assert_eq!(wrapper.schema_version, RESPONSES_CHECKPOINT_SCHEMA_V2);
    assert_eq!(wrapper.identity, identity);
    assert_eq!(wrapper.output, output);
    assert_eq!(wrapper.checkpoint_token_seed, 25);
    assert_eq!(wrapper.prior_checkpoint_id.as_deref(), Some("cp-prior"));
    assert_eq!(wrapper.portable_history_sha256, digest);
    assert_eq!(successor[1..].len(), tail.len());
    assert_eq!(wrapper.server_output_item_count, output.len());
    assert!(wrapper.auto_continue);

    // Recompact computes a successor identity whose prior link is the live
    // checkpoint, but must bind the current wrapper with its existing prior.
    let mut successor_identity = wrapper.identity.clone();
    successor_identity.prior_checkpoint_id = Some(wrapper.checkpoint_id.clone());
    let binding_identity =
        current_v2_identity_for_recompact_binding(&successor_identity, wrapper);
    assert_eq!(binding_identity, wrapper.identity);
    assert_eq!(
        successor_identity.prior_checkpoint_id.as_deref(),
        Some(wrapper.checkpoint_id.as_str())
    );

    // The wrapper digest binds the immutable fields the V3 sidecar and V2
    // staging re-verify at read time.
    assert_eq!(
        wrapper.wrapper_digest(),
        wrapper_digest_v2(
            &wrapper.checkpoint_id,
            &wrapper.operation_id,
            wrapper.prompt_index,
            &wrapper.branch_id,
            &wrapper.identity,
            &wrapper.portable_history_sha256,
            wrapper.prior_checkpoint_id.as_deref(),
        )
    );
    // A different portable digest must produce a different wrapper digest
    // (the sidecar constructor rejects mismatches).
    let mut other = successor.clone();
    let ConversationItem::ResponsesCompactionCheckpointV2(other_wrapper) = &mut other[0] else {
        unreachable!()
    };
    other_wrapper.portable_history_sha256 = "deadbeef".into();
    assert_ne!(other_wrapper.wrapper_digest(), wrapper.wrapper_digest());
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

#[test]
fn v1_migration_cohort_is_stable_and_monotonic() {
    use xai_grok_shell::session::responses_server_compaction::v1_migration_cohort_percent;

    // Extremes are unconditional.
    assert!(v1_migration_cohort_percent("session-a", 100));
    assert!(!v1_migration_cohort_percent("session-a", 0));

    // The same session always lands in the same bucket.
    let first = v1_migration_cohort_percent("session-b", 50);
    for _ in 0..8 {
        assert_eq!(first, v1_migration_cohort_percent("session-b", 50));
    }

    // Monotonic ramp: a session admitted at p% is admitted at every q >= p.
    for session in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"] {
        let mut admitted_at = None;
        for percent in 1..=100u32 {
            if v1_migration_cohort_percent(session, percent) {
                admitted_at = Some(percent);
                break;
            }
        }
        let admitted_at = admitted_at.expect("every session admits at 100%");
        for percent in admitted_at..=100u32 {
            assert!(
                v1_migration_cohort_percent(session, percent),
                "session {session} admitted at {admitted_at} must stay admitted at {percent}"
            );
        }
    }

    // Distinct sessions spread across buckets (not all in the same cohort).
    let admitted: usize = (0..100)
        .filter(|i| v1_migration_cohort_percent(&format!("session-{i}"), 10))
        .count();
    assert!(
        admitted > 0 && admitted < 40,
        "10% cohort should admit a small nonzero share, got {admitted}/100"
    );
}

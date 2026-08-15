//! Responses compaction replay contract tests: material binding, verification
//! and wire rules.

use super::*;

fn envelope_fixture() -> TrustedPromptEnvelope {
    TrustedPromptEnvelope {
        base_instructions_sha256: "base-hash".into(),
        memory_revision: Some(3),
        envelope_fingerprint: "envelope-fp".into(),
        wire_prompt_sha256: "wire-hash".into(),
    }
}

fn identity_fixture(prior: Option<&str>) -> CheckpointIdentity {
    CheckpointIdentity {
        provider_id: "xai".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
        prompt_envelope_fingerprint: "envelope-fp".into(),
        base_instructions_sha256: "base-hash".into(),
        prior_checkpoint_id: prior.map(str::to_owned),
        cache_route_fingerprint: None,
    }
}

fn portable_fixture() -> Vec<ConversationItem> {
    vec![
        ConversationItem::base_instructions("base prompt"),
        ConversationItem::user("first"),
        ConversationItem::assistant("answer"),
    ]
}

fn compaction_blob() -> serde_json::Value {
    serde_json::json!({
        "type": "compaction",
        "encrypted_content": "opaque"
    })
}

fn wrapper_fixture(
    portable: &[ConversationItem],
    prior: Option<&str>,
) -> ServerResponsesCheckpoint {
    let digest = portable_history_digest(portable).unwrap();
    ServerResponsesCheckpoint {
        checkpoint_id: "checkpoint-current".into(),
        operation_id: "operation-current".into(),
        prompt_index: 5,
        created_at: chrono::Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionMode {
            name: "default".into(),
            detail: None,
        },
        branch_id: "branch-1".into(),
        identity: identity_fixture(prior),
        retained_prefix: vec![ConversationItem::user("first")],
        compaction_item: compaction_blob(),
        portable_history_path: "compaction_checkpoints/checkpoint-current.json".into(),
        portable_history_sha256: digest,
        portable_history_bytes: 128,
        checkpoint_token_seed: 42,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        prior_checkpoint_id: prior.map(str::to_owned),
        memory_revision: Some(3),
    }
}

fn tail_fixture() -> Vec<ConversationItem> {
    vec![
        ConversationItem::user("after compaction"),
        ConversationItem::assistant("tail answer"),
    ]
}

fn verify_ok() -> ValidatedResponsesReplay {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    ValidatedResponsesReplay::verify(
        &wrapper,
        &material,
        &portable,
        &envelope_fixture(),
        &tail_fixture(),
        7,
        2,
    )
    .unwrap()
}

#[test]
fn checkpoint_uses_the_single_unversioned_contract() {
    assert_eq!(RESPONSES_COMPACTION_CONTRACT, "responses-compact-grok");

    let wrapper = wrapper_fixture(&portable_fixture(), None);
    let mut value = serde_json::to_value(&wrapper).unwrap();
    assert!(value.get("schema_version").is_none());
    assert!(value.get("output").is_none());
    assert!(value.get("server_output_item_count").is_none());
    assert!(value.get("retained_prefix").is_some());
    assert!(value.get("compaction_item").is_some());

    value
        .as_object_mut()
        .unwrap()
        .insert("schema_version".into(), serde_json::json!(2));
    assert!(serde_json::from_value::<ServerResponsesCheckpoint>(value).is_err());
}

#[test]
fn old_output_shape_fails_closed() {
    let mut value = serde_json::to_value(wrapper_fixture(&portable_fixture(), None)).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("retained_prefix");
    object.remove("compaction_item");
    object.insert(
        "output".into(),
        serde_json::json!([{"type": "compaction", "encrypted_content": "opaque"}]),
    );
    object.insert("server_output_item_count".into(), serde_json::json!(1));
    assert!(serde_json::from_value::<ServerResponsesCheckpoint>(value).is_err());
}

#[test]
fn material_binds_wrapper_and_portable_digests() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, Some("checkpoint-prior"));
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    assert_eq!(
        material.portable_history_sha256(),
        wrapper.portable_history_sha256
    );
    assert_eq!(material.wrapper_digest(), wrapper.wrapper_digest());
    assert_eq!(material.prior_checkpoint_id(), Some("checkpoint-prior"));
    assert_eq!(material.branch_id(), "branch-1");
    assert_eq!(material.memory_revision(), Some(3));
    assert_eq!(material.contract_version(), RESPONSES_COMPACTION_CONTRACT);
}

#[test]
fn material_rejects_wrong_portable_history() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let wrong = vec![ConversationItem::user("other")];
    assert!(matches!(
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &wrong),
        Err(ReplayMaterialError::PortableDigestMismatch)
    ));
}

#[test]
fn verify_accepts_and_freezes_replay() {
    let replay = verify_ok();
    assert_eq!(replay.checkpoint_id(), "checkpoint-current");
    assert_eq!(replay.operation_id(), "operation-current");
    assert_eq!(replay.branch_id(), "branch-1");
    assert_eq!(replay.history_revision(), 7);
    assert_eq!(replay.request_identity_generation(), 2);
    assert_eq!(replay.memory_revision(), Some(3));
    assert_eq!(replay.retained_prefix().len(), 1);
    assert_eq!(
        replay
            .compaction_item()
            .get("encrypted_content")
            .and_then(|v| v.as_str()),
        Some("opaque")
    );
    assert_eq!(replay.typed_tail().len(), 2);
}

#[test]
fn verify_accepts_branch_rotation() {
    let portable = portable_fixture();
    let mut wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    wrapper.branch_id = "branch-rotated".into();
    assert_ne!(wrapper.wrapper_digest(), material.wrapper_digest());
    let replay = ValidatedResponsesReplay::verify(
        &wrapper,
        &material,
        &portable,
        &envelope_fixture(),
        &tail_fixture(),
        7,
        2,
    )
    .unwrap();
    assert_eq!(replay.branch_id(), "branch-rotated");
}

#[test]
fn verify_rejects_tampered_wrapper() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();

    let mut forged_operation = wrapper.clone();
    forged_operation.operation_id = "operation-forged".into();
    let mut forged_blob = wrapper.clone();
    forged_blob.compaction_item = serde_json::json!({
        "type": "compaction",
        "encrypted_content": "attacker-controlled"
    });
    let mut forged_seed = wrapper.clone();
    forged_seed.checkpoint_token_seed += 1;
    let mut forged_source = wrapper.clone();
    forged_source.token_seed_source = TokenSeedSource::EstimatedCanonicalOutput;
    let mut forged_prefix = wrapper.clone();
    forged_prefix.retained_prefix = vec![ConversationItem::user("forged")];

    for forged in [
        forged_operation,
        forged_blob,
        forged_seed,
        forged_source,
        forged_prefix,
    ] {
        assert!(matches!(
            ValidatedResponsesReplay::verify(
                &forged,
                &material,
                &portable,
                &envelope_fixture(),
                &tail_fixture(),
                7,
                2,
            ),
            Err(ReplayVerificationError::WrapperDigestMismatch)
        ));
    }
}

#[test]
fn verify_rejects_envelope_mismatch() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    let mut other = envelope_fixture();
    other.base_instructions_sha256 = "different-base".into();
    assert!(matches!(
        ValidatedResponsesReplay::verify(
            &wrapper,
            &material,
            &portable,
            &other,
            &tail_fixture(),
            7,
            2,
        ),
        Err(ReplayVerificationError::EnvelopeMismatch(
            "base_instructions"
        ))
    ));
}

#[test]
fn verify_ignores_memory_content_drift() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    let mut current = envelope_fixture();
    current.memory_revision = Some(9);
    current.wire_prompt_sha256 = "wire-changed".into();
    assert!(
        ValidatedResponsesReplay::verify(
            &wrapper,
            &material,
            &portable,
            &current,
            &tail_fixture(),
            7,
            2,
        )
        .is_ok()
    );
}

#[test]
fn verify_rejects_checkpoint_in_tail() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterial::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    let mut tail = tail_fixture();
    tail.push(ConversationItem::ResponsesCompactionCheckpoint(Box::new(
        wrapper_fixture(&portable, None),
    )));
    assert!(matches!(
        ValidatedResponsesReplay::verify(
            &wrapper,
            &material,
            &portable,
            &envelope_fixture(),
            &tail,
            7,
            2,
        ),
        Err(ReplayVerificationError::CheckpointInTail)
    ));
}

#[test]
fn compose_instructions_lifts_only_marked_sources() {
    let items = vec![
        ConversationItem::base_instructions("base"),
        ConversationItem::memory_context("memory"),
        ConversationItem::system("legacy unclassified"),
        ConversationItem::runtime_system("runtime notice"),
        ConversationItem::user("hi"),
    ];
    let instructions = compose_instructions(&items).unwrap();
    assert_eq!(instructions, "base\n\nmemory");
    let tail = replay_input_tail(&items);
    assert_eq!(tail.len(), 3);
    assert_eq!(tail[0].text_content(), "legacy unclassified");
    assert_eq!(tail[1].text_content(), "runtime notice");
    assert!(matches!(tail[2], ConversationItem::User(_)));
}

#[test]
fn replay_request_body_is_retained_prefix_plus_blob_plus_tail() {
    let replay = verify_ok();
    let request = ConversationRequest {
        items: std::iter::once(ConversationItem::ResponsesCompactionCheckpoint(Box::new(
            wrapper_fixture(&portable_fixture(), None),
        )))
        .chain(tail_fixture())
        .collect(),
        model: Some("grok-test".into()),
        instructions: compose_instructions(&[ConversationItem::base_instructions("base")]),
        prompt_cache_key: Some("key".into()),
        ..Default::default()
    };
    let resolved = ResolvedResponsesRequest::from_validated_replay(&replay, &request).unwrap();
    let input = resolved
        .body()
        .get("input")
        .and_then(|value| value.as_array())
        .unwrap();
    // retained user + compaction blob + two tail items
    assert!(input.len() >= 3);
    assert_eq!(
        input
            .iter()
            .find(|item| item.get("type").and_then(|v| v.as_str()) == Some("compaction"))
            .and_then(|item| item.get("encrypted_content"))
            .and_then(|v| v.as_str()),
        Some("opaque")
    );
    let binding = resolved.checkpoint_binding().unwrap();
    assert_eq!(binding.checkpoint_id, "checkpoint-current");
    assert_eq!(binding.contract_version, RESPONSES_COMPACTION_CONTRACT);
    assert_eq!(resolved.request_identity_generation(), Some(2));
}

#[test]
fn compact_try_normal_rejects_checkpoint_and_lifts_instructions() {
    let items = vec![
        ConversationItem::base_instructions("base"),
        ConversationItem::memory_context("memory"),
        ConversationItem::user("hello"),
    ];
    let request = ConversationRequest {
        items: items.clone(),
        model: Some("grok-test".into()),
        instructions: compose_instructions(&items),
        parallel_tool_calls: Some(true),
        ..Default::default()
    };
    let resolved = ResolvedCompactRequest::try_normal(&request, Some("focus on rust")).unwrap();
    let body = resolved.body();
    let instructions = body
        .get("instructions")
        .and_then(|value| value.as_str())
        .unwrap();
    assert!(instructions.starts_with("base\n\nmemory"));
    assert!(instructions.contains(USER_CONTEXT_DELIMITER));
    assert!(instructions.ends_with("focus on rust"));
    let input = body
        .get("input")
        .and_then(|value| value.as_array())
        .unwrap();
    // user + trigger
    assert_eq!(input.len(), 2);
    assert_eq!(
        input
            .last()
            .and_then(|item| item.get("type"))
            .and_then(|v| v.as_str()),
        Some("compaction_trigger")
    );
    assert_eq!(body.get("stream").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(body.get("store").and_then(|v| v.as_bool()), Some(false));
    assert_eq!(
        body.get("tool_choice").and_then(|v| v.as_str()),
        Some("auto")
    );
    let include = body.get("include").and_then(|v| v.as_array()).unwrap();
    assert!(
        include
            .iter()
            .any(|v| v.as_str() == Some("reasoning.encrypted_content"))
    );
    assert!(resolved.checkpoint_binding().is_none());

    let mut checkpointed = request.clone();
    checkpointed.items.insert(
        0,
        ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper_fixture(
            &portable_fixture(),
            None,
        ))),
    );
    assert!(matches!(
        ResolvedCompactRequest::try_normal(&checkpointed, None),
        Err(ResolvedCompactError::CheckpointInNormalRequest)
    ));
}

#[test]
fn recompact_uses_retained_prefix_blob_and_trigger() {
    let replay = verify_ok();
    let request = ConversationRequest {
        items: std::iter::once(ConversationItem::ResponsesCompactionCheckpoint(Box::new(
            wrapper_fixture(&portable_fixture(), None),
        )))
        .chain(tail_fixture())
        .collect(),
        model: Some("grok-test".into()),
        instructions: Some("base".into()),
        prompt_cache_key: Some("grok:stable-main-session-key".into()),
        parallel_tool_calls: Some(true),
        ..Default::default()
    };
    let resolved =
        ResolvedCompactRequest::from_validated_recompact(&replay, &request, None).unwrap();
    let input = resolved
        .body()
        .get("input")
        .and_then(|value| value.as_array())
        .unwrap();
    assert_eq!(
        input
            .last()
            .and_then(|item| item.get("type"))
            .and_then(|v| v.as_str()),
        Some("compaction_trigger")
    );
    assert!(
        input
            .iter()
            .any(|item| item.get("type").and_then(|v| v.as_str()) == Some("compaction"))
    );
    assert_eq!(
        resolved
            .body()
            .get("prompt_cache_key")
            .and_then(|value| value.as_str()),
        Some("grok:stable-main-session-key")
    );
    assert_eq!(
        resolved.checkpoint_binding().unwrap().checkpoint_id,
        "checkpoint-current"
    );
    assert_eq!(resolved.request_identity_generation(), Some(2));
    assert_eq!(
        resolved.body().get("stream").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn checkpoint_serde_uses_only_the_unversioned_variant() {
    let item = ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper_fixture(
        &portable_fixture(),
        Some("checkpoint-prior"),
    )));
    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(
        json.get("type").and_then(|value| value.as_str()),
        Some("responses_compaction_checkpoint")
    );
    let parsed: ConversationItem = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(
        canonical_json_bytes(&serde_json::to_value(&item).unwrap()).unwrap(),
        canonical_json_bytes(&serde_json::to_value(&parsed).unwrap()).unwrap()
    );

    let mut old_versioned = json.clone();
    old_versioned["type"] = serde_json::Value::String("responses_compaction_checkpoint_v2".into());
    assert!(serde_json::from_value::<ConversationItem>(old_versioned).is_err());

    let mut old_shape = json;
    old_shape
        .get_mut("identity")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap()
        .remove("base_instructions_sha256");
    assert!(serde_json::from_value::<ConversationItem>(old_shape).is_err());
}

#[test]
fn envelope_fingerprint_ignores_items_and_cache_key() {
    let base_request = ConversationRequest {
        items: vec![ConversationItem::user("hello")],
        model: Some("grok-test".into()),
        prompt_cache_key: Some("key-a".into()),
        service_tier: Some("priority".into()),
        ..Default::default()
    };
    let fingerprint = canonical_envelope_fingerprint(&base_request);
    let mut other_items = base_request.clone();
    other_items.items = vec![
        ConversationItem::system("sys"),
        ConversationItem::assistant("answer"),
    ];
    assert_eq!(canonical_envelope_fingerprint(&other_items), fingerprint);
    let mut other_key = base_request.clone();
    other_key.prompt_cache_key = Some("key-b".into());
    assert_eq!(canonical_envelope_fingerprint(&other_key), fingerprint);
    let mut other_tier = base_request;
    other_tier.service_tier = Some("flex".into());
    assert_ne!(canonical_envelope_fingerprint(&other_tier), fingerprint);
}

#[test]
fn instruction_hash_helpers_are_stable() {
    assert_eq!(
        base_instructions_sha256("base"),
        base_instructions_sha256("base")
    );
    assert_ne!(base_instructions_sha256("a"), base_instructions_sha256("b"));
    assert_ne!(wire_prompt_sha256("a"), wire_prompt_sha256("b"));
}

#[test]
fn checkpoint_helpers_return_the_single_wrapper_type() {
    let item = ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper_fixture(
        &portable_fixture(),
        None,
    )));
    assert!(item.is_responses_checkpoint());
    let checkpoint: &ServerResponsesCheckpoint = item.as_responses_checkpoint().unwrap();
    assert_eq!(checkpoint.checkpoint_id, "checkpoint-current");
    assert!(!ConversationItem::user("x").is_responses_checkpoint());
}

#[test]
fn backend_validation_requires_one_leading_current_checkpoint() {
    let wrapper = wrapper_fixture(&portable_fixture(), None);
    let valid = ConversationRequest {
        items: vec![
            ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper.clone())),
            ConversationItem::user("tail"),
        ],
        ..Default::default()
    };
    assert!(
        valid
            .validate_for_backend(&crate::ApiBackend::Responses)
            .is_ok()
    );
    assert!(
        valid
            .validate_for_backend(&crate::ApiBackend::ChatCompletions)
            .is_err()
    );

    let misplaced = ConversationRequest {
        items: vec![
            ConversationItem::user("head"),
            ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper.clone())),
        ],
        ..Default::default()
    };
    assert!(matches!(
        misplaced.validate_for_backend(&crate::ApiBackend::Responses),
        Err(ConversationValidationError::InvalidCheckpointLayout)
    ));

    let duplicate = ConversationRequest {
        items: vec![
            ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper.clone())),
            ConversationItem::ResponsesCompactionCheckpoint(Box::new(wrapper.clone())),
        ],
        ..Default::default()
    };
    assert!(matches!(
        duplicate.validate_for_backend(&crate::ApiBackend::Responses),
        Err(ConversationValidationError::InvalidCheckpointLayout)
    ));

    let mut empty = wrapper;
    empty.compaction_item = serde_json::json!({"type": "compaction", "encrypted_content": ""});
    let empty = ConversationRequest {
        items: vec![ConversationItem::ResponsesCompactionCheckpoint(Box::new(
            empty,
        ))],
        ..Default::default()
    };
    assert!(matches!(
        empty.validate_for_backend(&crate::ApiBackend::Responses),
        Err(ConversationValidationError::EmptyCheckpointOutput)
    ));
}

#[test]
fn retained_prefix_keeps_user_system_and_budget_truncates_newest_first() {
    let history = vec![
        ConversationItem::base_instructions("lifted base"),
        ConversationItem::user("oldest"),
        ConversationItem::assistant("agent short"),
        ConversationItem::tool_result("c1", "tool out"),
        ConversationItem::user("middle"),
        ConversationItem::user("newest"),
    ];
    let retained = build_retained_prefix(&history);
    assert!(
        retained
            .iter()
            .all(|item| !matches!(item, ConversationItem::ToolResult(_)))
    );
    assert!(
        retained
            .iter()
            .all(|item| !matches!(item, ConversationItem::System(_)))
    );
    // lifted base instructions must not appear
    assert!(
        !retained
            .iter()
            .any(|item| item.text_content() == "lifted base")
    );

    // Tiny budget keeps only the newest eligible item(s).
    let truncated = truncate_retained_newest_first(
        vec![
            ConversationItem::user("aaaa"), // 1 token
            ConversationItem::user("bbbb"), // 1 token
            ConversationItem::user("cccc"), // 1 token
        ],
        2,
    );
    assert_eq!(truncated.len(), 2);
    assert_eq!(truncated[0].text_content(), "bbbb");
    assert_eq!(truncated[1].text_content(), "cccc");
}

#[test]
fn retained_prefix_drops_oversized_non_final_agent_messages() {
    let huge = "x".repeat((MAX_RETAINED_AGENT_MESSAGE_TOKENS as usize + 1) * 4);
    let history = vec![
        ConversationItem::user("keep"),
        ConversationItem::assistant(huge),
        ConversationItem::assistant("Message Type: FINAL_ANSWER\nbody"),
    ];
    let retained = build_retained_prefix(&history);
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].text_content(), "keep");
}

#[test]
fn compaction_trigger_wire_item_is_bare_type_object() {
    let trigger = compaction_trigger_wire_item();
    assert_eq!(trigger, serde_json::json!({ "type": "compaction_trigger" }));
    assert!(is_valid_compaction_item(&compaction_blob()));
    assert!(is_valid_compaction_item(&serde_json::json!({
        "type": "compaction_summary",
        "encrypted_content": "x"
    })));
    assert!(!is_valid_compaction_item(&serde_json::json!({
        "type": "compaction",
        "encrypted_content": ""
    })));
    assert!(!is_valid_compaction_item(&compaction_trigger_wire_item()));
}

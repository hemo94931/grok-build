//! V2 replay contract tests: material binding, verification, wire rules.

use super::v2::{
    CheckpointIdentityV2, RESPONSES_CHECKPOINT_SCHEMA_V2, RESPONSES_COMPACTION_CONTRACT_V2,
    ServerResponsesCheckpointV2, TrustedPromptEnvelopeV2,
};
use super::*;
use crate::TraceContext;

fn envelope_fixture() -> TrustedPromptEnvelopeV2 {
    TrustedPromptEnvelopeV2 {
        base_instructions_sha256: "base-hash".into(),
        memory_revision: Some(3),
        envelope_fingerprint: "envelope-fp".into(),
        wire_prompt_sha256: "wire-hash".into(),
    }
}

fn identity_fixture(prior: Option<&str>) -> CheckpointIdentityV2 {
    CheckpointIdentityV2 {
        provider_id: "xai".into(),
        api: "responses".into(),
        endpoint_fingerprint: "endpoint".into(),
        model: "grok-test".into(),
        auth_principal_fingerprint: "principal".into(),
        contract_version: RESPONSES_COMPACTION_CONTRACT_V2.into(),
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

fn wrapper_fixture(
    portable: &[ConversationItem],
    prior: Option<&str>,
) -> ServerResponsesCheckpointV2 {
    let digest = portable_history_digest(portable).unwrap();
    ServerResponsesCheckpointV2 {
        schema_version: RESPONSES_CHECKPOINT_SCHEMA_V2,
        checkpoint_id: "cp-v2".into(),
        operation_id: "op-v2".into(),
        prompt_index: 5,
        created_at: chrono::Utc::now(),
        auto_continue: false,
        mode: ResponsesCompactionModeV1 {
            name: "default".into(),
            detail: None,
        },
        branch_id: "branch-1".into(),
        identity: identity_fixture(prior),
        output: vec![serde_json::json!({
            "type": "compaction",
            "encrypted_content": "opaque"
        })],
        portable_history_path: "compaction_checkpoints/cp-v2.json".into(),
        portable_history_sha256: digest,
        portable_history_bytes: 128,
        checkpoint_token_seed: 42,
        token_seed_source: TokenSeedSource::UsageOutputTokens,
        server_output_item_count: 1,
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

fn verify_ok() -> ValidatedResponsesReplayV2 {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    ValidatedResponsesReplayV2::verify(
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
fn material_binds_wrapper_and_portable_digests() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, Some("cp-prior"));
    let material =
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    assert_eq!(
        material.portable_history_sha256(),
        wrapper.portable_history_sha256
    );
    assert_eq!(material.wrapper_digest(), wrapper.wrapper_digest());
    assert_eq!(material.prior_checkpoint_id(), Some("cp-prior"));
    assert_eq!(material.branch_id(), "branch-1");
    assert_eq!(material.memory_revision(), Some(3));
    assert_eq!(material.contract_version(), RESPONSES_COMPACTION_CONTRACT_V2);
}

#[test]
fn material_rejects_wrong_portable_history() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let wrong = vec![ConversationItem::user("other")];
    assert!(matches!(
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &wrong),
        Err(ReplayMaterialError::PortableDigestMismatch)
    ));
}

#[test]
fn verify_accepts_and_freezes_replay() {
    let replay = verify_ok();
    assert_eq!(replay.checkpoint_id(), "cp-v2");
    assert_eq!(replay.operation_id(), "op-v2");
    assert_eq!(replay.branch_id(), "branch-1");
    assert_eq!(replay.history_revision(), 7);
    assert_eq!(replay.request_identity_generation(), 2);
    assert_eq!(replay.memory_revision(), Some(3));
    assert_eq!(replay.output().len(), 1);
    assert_eq!(replay.typed_tail().len(), 2);
}

#[test]
fn verify_accepts_branch_rotation() {
    let portable = portable_fixture();
    let mut wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    // Rewind rotated the live branch; everything else is untouched.
    wrapper.branch_id = "branch-rotated".into();
    assert!(wrapper.wrapper_digest() != material.wrapper_digest());
    let replay = ValidatedResponsesReplayV2::verify(
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
    let mut wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    wrapper.operation_id = "op-forged".into();
    assert!(matches!(
        ValidatedResponsesReplayV2::verify(
            &wrapper,
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

#[test]
fn verify_rejects_envelope_mismatch() {
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    let mut other = envelope_fixture();
    other.base_instructions_sha256 = "different-base".into();
    assert!(matches!(
        ValidatedResponsesReplayV2::verify(
            &wrapper,
            &material,
            &portable,
            &other,
            &tail_fixture(),
            7,
            2,
        ),
        Err(ReplayVerificationError::EnvelopeMismatch("base_instructions"))
    ));
}

#[test]
fn verify_ignores_memory_content_drift() {
    // A different current memory revision/wire hash must not invalidate the
    // checkpoint: memory is not part of compatibility identity.
    let portable = portable_fixture();
    let wrapper = wrapper_fixture(&portable, None);
    let material =
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    let mut current = envelope_fixture();
    current.memory_revision = Some(9);
    current.wire_prompt_sha256 = "wire-changed".into();
    assert!(
        ValidatedResponsesReplayV2::verify(
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
        CheckpointReplayMaterialV2::try_new(&wrapper, envelope_fixture(), &portable).unwrap();
    let mut tail = tail_fixture();
    tail.push(ConversationItem::ResponsesCompactionCheckpointV2(Box::new(
        wrapper_fixture(&portable, None),
    )));
    assert!(matches!(
        ValidatedResponsesReplayV2::verify(
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
    let instructions = compose_instructions_v2(&items).unwrap();
    assert_eq!(instructions, "base\n\nmemory");
    // Input tail keeps unclassified/runtime systems in position, drops the
    // lifted ones, and never touches non-system items.
    let tail = replay_input_tail_v2(&items);
    assert_eq!(tail.len(), 3);
    assert_eq!(tail[0].text_content(), "legacy unclassified");
    assert_eq!(tail[1].text_content(), "runtime notice");
    assert!(matches!(tail[2], ConversationItem::User(_)));
}

#[test]
fn replay_request_body_is_output_prefix_plus_tail() {
    let replay = verify_ok();
    let request = ConversationRequest {
        items: std::iter::once(ConversationItem::ResponsesCompactionCheckpointV2(Box::new(
            wrapper_fixture(&portable_fixture(), None),
        )))
        .chain(tail_fixture())
        .collect(),
        model: Some("grok-test".into()),
        instructions: compose_instructions_v2(&[ConversationItem::base_instructions("base")]),
        prompt_cache_key: Some("key".into()),
        ..Default::default()
    };
    let resolved = ResolvedResponsesRequest::from_validated_replay(&replay, &request).unwrap();
    let input = resolved
        .body()
        .get("input")
        .and_then(|v| v.as_array())
        .unwrap();
    assert_eq!(
        input[0].get("type").and_then(|v| v.as_str()),
        Some("compaction")
    );
    assert_eq!(input.len(), 3);
    let binding = resolved.checkpoint_binding().unwrap();
    assert_eq!(binding.checkpoint_id, "cp-v2");
    assert_eq!(binding.contract_version, RESPONSES_COMPACTION_CONTRACT_V2);
    assert_eq!(resolved.request_identity_generation(), Some(2));
}

#[test]
fn compact_try_normal_rejects_checkpoints_and_lifts_instructions() {
    let items = vec![
        ConversationItem::base_instructions("base"),
        ConversationItem::memory_context("memory"),
        ConversationItem::user("hello"),
    ];
    let request = ConversationRequest {
        items: items.clone(),
        model: Some("grok-test".into()),
        instructions: compose_instructions_v2(&items),
        ..Default::default()
    };
    let resolved = ResolvedCompactRequest::try_normal(&request, Some("focus on rust")).unwrap();
    let body = resolved.body();
    let instructions = body
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap();
    assert!(instructions.starts_with("base\n\nmemory"));
    assert!(instructions.contains(USER_CONTEXT_DELIMITER));
    assert!(instructions.ends_with("focus on rust"));
    // Base/memory items are lifted out of the compactable input.
    let input = body.get("input").and_then(|v| v.as_array()).unwrap();
    assert_eq!(input.len(), 1);
    assert!(resolved.checkpoint_binding().is_none());

    let mut checkpointed = request.clone();
    checkpointed.items.insert(
        0,
        ConversationItem::ResponsesCompactionCheckpointV2(Box::new(wrapper_fixture(
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
fn recompact_uses_prior_output_prefix() {
    let replay = verify_ok();
    let request = ConversationRequest {
        items: std::iter::once(ConversationItem::ResponsesCompactionCheckpointV2(Box::new(
            wrapper_fixture(&portable_fixture(), None),
        )))
        .chain(tail_fixture())
        .collect(),
        model: Some("grok-test".into()),
        instructions: Some("base".into()),
        ..Default::default()
    };
    let resolved =
        ResolvedCompactRequest::from_validated_recompact(&replay, &request, None).unwrap();
    let input = resolved
        .body()
        .get("input")
        .and_then(|v| v.as_array())
        .unwrap();
    assert_eq!(
        input[0].get("type").and_then(|v| v.as_str()),
        Some("compaction"),
        "prior opaque output stays the input prefix"
    );
    assert_eq!(input.len(), 3);
    let binding = resolved.checkpoint_binding().unwrap();
    assert_eq!(binding.checkpoint_id, "cp-v2");
    assert_eq!(resolved.request_identity_generation(), Some(2));
}

#[test]
fn wrapper_v2_serde_roundtrip_preserves_bytes() {
    let wrapper = wrapper_fixture(&portable_fixture(), Some("cp-prior"));
    let item = ConversationItem::ResponsesCompactionCheckpointV2(Box::new(wrapper));
    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(
        json.get("type").and_then(|v| v.as_str()),
        Some("responses_compaction_checkpoint_v2")
    );
    let parsed: ConversationItem = serde_json::from_value(json.clone()).unwrap();
    let canonical = |value: &ConversationItem| {
        canonical_json_bytes(&serde_json::to_value(value).unwrap()).unwrap()
    };
    assert_eq!(canonical(&item), canonical(&parsed));
}

#[test]
fn checkpoint_identity_untagged_serde_direction() {
    // Old bare-V1 JSON parses as V1 (rewind snapshots stay readable).
    let v1 = CheckpointIdentityV1 {
        provider_id: "xai".into(),
        api: "responses".into(),
        endpoint_fingerprint: "ep".into(),
        model: "m".into(),
        auth_principal_fingerprint: "p".into(),
        contract_version: "responses-compact-codex-v1".into(),
        prompt_envelope_fingerprint: "e".into(),
        canonical_prompt_projection: None,
    };
    let v1_json = serde_json::to_value(&v1).unwrap();
    let parsed: CheckpointIdentity = serde_json::from_value(v1_json).unwrap();
    assert!(matches!(parsed, CheckpointIdentity::V1(_)));

    // V2 JSON parses as V2 (V2-required fields prevent collapse into V1).
    let v2 = identity_fixture(Some("cp-prior"));
    let v2_json = serde_json::to_value(&v2).unwrap();
    let parsed: CheckpointIdentity = serde_json::from_value(v2_json.clone()).unwrap();
    match &parsed {
        CheckpointIdentity::V2(identity) => assert_eq!(identity, &v2),
        CheckpointIdentity::V1(_) => panic!("V2 identity must not parse as V1"),
    }

    // Roundtrip through the enum preserves both.
    for identity in [
        CheckpointIdentity::V1(v1),
        CheckpointIdentity::V2(identity_fixture(None)),
    ] {
        let json = serde_json::to_value(&identity).unwrap();
        let back: CheckpointIdentity = serde_json::from_value(json).unwrap();
        assert_eq!(back, identity);
    }
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
    let fp_a = canonical_envelope_fingerprint_v2(&base_request);
    // Items never affect the envelope.
    let mut other_items = base_request.clone();
    other_items.items = vec![
        ConversationItem::system("sys"),
        ConversationItem::assistant("answer"),
    ];
    assert_eq!(canonical_envelope_fingerprint_v2(&other_items), fp_a);
    // prompt_cache_key is routing, not compatibility.
    let mut other_key = base_request.clone();
    other_key.prompt_cache_key = Some("key-b".into());
    assert_eq!(canonical_envelope_fingerprint_v2(&other_key), fp_a);
    // service_tier is compatibility-relevant.
    let mut other_tier = base_request.clone();
    other_tier.service_tier = Some("flex".into());
    assert_ne!(canonical_envelope_fingerprint_v2(&other_tier), fp_a);
}

#[test]
fn base_and_wire_hashes_are_stable() {
    assert_eq!(
        base_instructions_sha256_v2("base"),
        base_instructions_sha256_v2("base")
    );
    assert_ne!(
        base_instructions_sha256_v2("a"),
        base_instructions_sha256_v2("b")
    );
    assert_ne!(wire_prompt_sha256_v2("a"), wire_prompt_sha256_v2("b"));
}

#[test]
fn checkpoint_helpers_cover_both_variants() {
    let v1 = ConversationItem::ResponsesCompactionCheckpoint(Box::new(
        ServerResponsesCheckpointV1 {
            schema_version: 1,
            checkpoint_id: "cp1".into(),
            operation_id: "op1".into(),
            prompt_index: 1,
            created_at: chrono::Utc::now(),
            auto_continue: false,
            mode: ResponsesCompactionModeV1 {
                name: "default".into(),
                detail: None,
            },
            branch_id: "b1".into(),
            identity: CheckpointIdentityV1 {
                provider_id: "xai".into(),
                api: "responses".into(),
                endpoint_fingerprint: "ep".into(),
                model: "m".into(),
                auth_principal_fingerprint: "p".into(),
                contract_version: "responses-compact-codex-v1".into(),
                prompt_envelope_fingerprint: "e".into(),
                canonical_prompt_projection: None,
            },
            output: vec![serde_json::json!({"type": "compaction"})],
            portable_history_path: "p".into(),
            portable_history_sha256: "d".into(),
            portable_history_bytes: 1,
            checkpoint_token_seed: 1,
            token_seed_source: TokenSeedSource::UsageOutputTokens,
            server_output_item_count: 1,
        },
    ));
    let v2 = ConversationItem::ResponsesCompactionCheckpointV2(Box::new(wrapper_fixture(
        &portable_fixture(),
        None,
    )));
    assert!(v1.is_responses_checkpoint());
    assert!(v2.is_responses_checkpoint());
    assert_eq!(v1.as_responses_checkpoint().unwrap().checkpoint_id(), "cp1");
    assert_eq!(
        v2.as_responses_checkpoint().unwrap().checkpoint_id(),
        "cp-v2"
    );
    assert!(!ConversationItem::user("x").is_responses_checkpoint());
}

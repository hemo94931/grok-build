//! Opaque resolved Responses request types.
//!
//! Security boundary: the real Responses POST points
//! (`SamplingClient::create_response`, `create_response_stream` and
//! `compact_responses`) only accept bodies produced by the validating
//! constructors in this module or by [`FinalResponsesRequest`]'s typed
//! conversion. Callers can never inject arbitrary raw JSON checkpoint
//! bodies.
//!
//! The resolved/validated types deliberately do **not** implement
//! `Deserialize`: deserialization can never imply validation.

use super::responses::{FinalResponsesRequest, ResponsesRequestBuildError};
use super::{ConversationItem, ConversationRequest, ConversationValidationError};
use crate::TraceContext;

/// Correlation headers frozen into a resolved request so retries reuse the
/// exact same metadata instead of re-reading session state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponsesCorrelation {
    pub x_grok_conv_id: Option<String>,
    pub x_grok_req_id: Option<String>,
    pub x_grok_session_id: Option<String>,
    pub x_grok_turn_idx: Option<String>,
    pub x_grok_agent_id: Option<String>,
    pub x_grok_deployment_id: Option<String>,
    pub x_grok_user_id: Option<String>,
}

impl ResponsesCorrelation {
    pub fn from_request(request: &ConversationRequest) -> Self {
        Self {
            x_grok_conv_id: request.x_grok_conv_id.clone(),
            x_grok_req_id: request.x_grok_req_id.clone(),
            x_grok_session_id: request.x_grok_session_id.clone(),
            x_grok_turn_idx: request.x_grok_turn_idx.clone(),
            x_grok_agent_id: request.x_grok_agent_id.clone(),
            x_grok_deployment_id: request.x_grok_deployment_id.clone(),
            x_grok_user_id: request.x_grok_user_id.clone(),
        }
    }
}

/// Checkpoint binding frozen into a validated replay/legacy permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCheckpointBinding {
    pub checkpoint_id: String,
    pub operation_id: String,
    pub branch_id: String,
    pub contract_version: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ResolvedRequestError {
    #[error("typed normal requests reject every checkpoint variant")]
    CheckpointInNormalRequest,
    #[error(transparent)]
    Build(#[from] ResponsesRequestBuildError),
}

/// Opaque Responses request: an immutable wire body plus frozen
/// correlation/continuity metadata.
///
/// Obtained only through validating constructors:
///
/// * [`ResolvedResponsesRequest::try_normal`] — typed normal request that
///   structurally cannot contain a checkpoint;
/// * replay constructors (added with the V2 checkpoint contract) that carry
///   a [`ResolvedCheckpointBinding`].
#[derive(Debug, Clone)]
pub struct ResolvedResponsesRequest {
    body: FinalResponsesRequest,
    model: String,
    correlation: ResponsesCorrelation,
    history_revision: Option<u64>,
    request_identity_generation: Option<u64>,
    checkpoint_binding: Option<ResolvedCheckpointBinding>,
    trace: Option<Box<dyn TraceContext>>,
}

impl ResolvedResponsesRequest {
    /// Build a typed normal resolved request. Rejects every checkpoint
    /// variant; checkpoints must go through validated replay construction.
    pub fn try_normal(request: &ConversationRequest) -> Result<Self, ResolvedRequestError> {
        if request
            .items
            .iter()
            .any(|item| matches!(item, ConversationItem::ResponsesCompactionCheckpoint(_)))
        {
            return Err(ResolvedRequestError::CheckpointInNormalRequest);
        }
        Ok(Self {
            body: FinalResponsesRequest::try_from(request)?,
            model: request.model.clone().unwrap_or_default(),
            correlation: ResponsesCorrelation::from_request(request),
            history_revision: request.history_revision,
            request_identity_generation: None,
            checkpoint_binding: None,
            trace: request.trace.clone(),
        })
    }

    pub fn body(&self) -> &serde_json::Value {
        self.body.body()
    }

    pub(crate) fn into_body(self) -> serde_json::Value {
        self.body.into_body()
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn correlation(&self) -> &ResponsesCorrelation {
        &self.correlation
    }

    pub fn history_revision(&self) -> Option<u64> {
        self.history_revision
    }

    pub fn request_identity_generation(&self) -> Option<u64> {
        self.request_identity_generation
    }

    pub fn checkpoint_binding(&self) -> Option<&ResolvedCheckpointBinding> {
        self.checkpoint_binding.as_ref()
    }

    pub fn take_trace(&mut self) -> Option<Box<dyn TraceContext>> {
        self.trace.take()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LegacyReplayPermitError {
    #[error("legacy replay requires exactly one leading V1 responses checkpoint")]
    MissingCheckpoint,
    #[error("portable history digest does not match the live checkpoint wrapper")]
    DigestMismatch,
    #[error(transparent)]
    Validation(#[from] ConversationValidationError),
    #[error(transparent)]
    Build(#[from] ResponsesRequestBuildError),
}

/// Temporary V1 replay permit for the migration gray window.
///
/// Only the shell's checkpoint gate issues this permit, after
/// `ensure_checkpoint_replayable_for_request` returned
/// `CheckpointReplayStatus::Replayable` and the live wrapper + sidecar
/// validated successfully. The body is produced exclusively by the typed
/// [`FinalResponsesRequest`] conversion; the frozen body is reused verbatim
/// across sampler retries (plan-once, body-frozen retry).
///
/// This type never creates new V1 checkpoints and is deleted once V1
/// migration reaches 100%.
#[derive(Debug, Clone)]
pub struct ValidatedLegacyReplayV1 {
    body: FinalResponsesRequest,
    model: String,
    checkpoint_id: String,
    operation_id: String,
    branch_id: String,
    history_revision: Option<u64>,
    request_identity_generation: u64,
    correlation: ResponsesCorrelation,
    trace: Option<Box<dyn TraceContext>>,
}

impl ValidatedLegacyReplayV1 {
    /// Validate a V1 replay request and freeze its flattened body.
    ///
    /// The constructor consumes the full material proof:
    ///
    /// * `portable_history` must hash to the live wrapper's
    ///   `portable_history_sha256` — only the shell gate reads the sidecar,
    ///   so ungated callers cannot satisfy this;
    /// * `request_identity_generation` must come from the
    ///   `bind_request_identity_at_revision` call that proved replayability.
    ///
    /// Continuity between the checkpoint output and the live tail is
    /// structural: the wrapper sits at index 0 of the live chat-state
    /// history and the tail is whatever the actor appended after the
    /// compaction commit, so the flattened body is coherent by
    /// construction.
    pub fn try_new(
        request: &ConversationRequest,
        portable_history: &[ConversationItem],
        request_identity_generation: u64,
    ) -> Result<Self, LegacyReplayPermitError> {
        let Some(ConversationItem::ResponsesCompactionCheckpoint(wrapper)) = request.items.first()
        else {
            return Err(LegacyReplayPermitError::MissingCheckpoint);
        };
        // Exactly one schema-1 checkpoint at index 0 with non-empty output.
        request.validate_for_backend(&crate::ApiBackend::Responses)?;
        let digest = super::responses::portable_history_digest(portable_history)
            .map_err(ResponsesRequestBuildError::Serialization)?;
        if digest != wrapper.portable_history_sha256 {
            return Err(LegacyReplayPermitError::DigestMismatch);
        }
        Ok(Self {
            body: FinalResponsesRequest::from_legacy_replay(request, wrapper.output.clone())?,
            model: request.model.clone().unwrap_or_default(),
            checkpoint_id: wrapper.checkpoint_id.clone(),
            operation_id: wrapper.operation_id.clone(),
            branch_id: wrapper.branch_id.clone(),
            history_revision: request.history_revision,
            request_identity_generation,
            correlation: ResponsesCorrelation::from_request(request),
            trace: request.trace.clone(),
        })
    }

    pub fn body(&self) -> &serde_json::Value {
        self.body.body()
    }

    pub(crate) fn into_body(self) -> serde_json::Value {
        self.body.into_body()
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }

    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    pub fn branch_id(&self) -> &str {
        &self.branch_id
    }

    pub fn history_revision(&self) -> Option<u64> {
        self.history_revision
    }

    pub fn request_identity_generation(&self) -> u64 {
        self.request_identity_generation
    }

    pub fn correlation(&self) -> &ResponsesCorrelation {
        &self.correlation
    }

    pub fn take_trace(&mut self) -> Option<Box<dyn TraceContext>> {
        self.trace.take()
    }
}

/// Sealed body of a [`crate::CreateResponseWrapper`]. The `Typed` variant
/// serializes `wrapper.inner`; every other variant carries a body that can
/// only have come from a validating constructor above.
#[derive(Debug, Clone, Default)]
pub(crate) enum SealedResponsesBody {
    #[default]
    Typed,
    Resolved(Box<ResolvedResponsesRequest>),
    LegacyV1(Box<ValidatedLegacyReplayV1>),
}

impl SealedResponsesBody {
    /// Borrow the pre-built body, if this wrapper carries one.
    pub(crate) fn override_value(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Typed => None,
            Self::Resolved(resolved) => Some(resolved.body()),
            Self::LegacyV1(permit) => Some(permit.body()),
        }
    }

    /// Consume the wrapper body, returning the pre-built body if present.
    pub(crate) fn into_override_value(self) -> Option<serde_json::Value> {
        match self {
            Self::Typed => None,
            Self::Resolved(resolved) => Some(resolved.into_body()),
            Self::LegacyV1(permit) => Some(permit.into_body()),
        }
    }
}

// ============================================================================
// V2 replay contract (reader-first)
// ============================================================================

use serde::{Deserialize, Serialize};

use super::responses::portable_history_digest;
use super::v2::{
    RESPONSES_CHECKPOINT_SCHEMA_V2, RESPONSES_COMPACTION_CONTRACT_V2, ServerResponsesCheckpointV2,
    TrustedPromptEnvelopeV2, wrapper_digest_v2,
};

/// Separator between base instructions and memory context inside the
/// composed top-level `instructions` field (V2 wire rule).
pub const INSTRUCTIONS_MEMORY_SEPARATOR: &str = "\n\n";

/// Delimiter for compact-only user context appended to the compact
/// request's instructions. Wire contract constant; the sampler re-exports
/// it for its legacy compact request type.
pub const USER_CONTEXT_DELIMITER: &str = "\n\n--- user-provided compaction context ---\n";

/// Compose the V2 top-level instructions from conversation items: every
/// `BaseInstructions` item in order, then every `MemoryContext` item,
/// joined by the fixed separator. Only sources that
/// [`SystemSource::lifts_into_instructions`] participate; every other
/// System stays in `input` at its original position.
pub fn compose_instructions_v2(items: &[ConversationItem]) -> Option<String> {
    let mut parts = Vec::new();
    for item in items {
        let ConversationItem::System(system) = item else {
            continue;
        };
        if system.source.lifts_into_instructions() {
            let content = system.content.trim();
            if !content.is_empty() {
                parts.push(content.to_string());
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join(INSTRUCTIONS_MEMORY_SEPARATOR))
}

/// Whether an item is lifted into `instructions` and therefore must not
/// also appear in `input`. Removal is by `SystemSource`, never by text.
fn lifted_into_instructions(item: &ConversationItem) -> bool {
    matches!(
        item,
        ConversationItem::System(system) if system.source.lifts_into_instructions()
    )
}

/// Replay/recompact input items: typed items with instruction-lifted
/// systems removed *by source*.
pub fn replay_input_tail_v2(typed_items: &[ConversationItem]) -> Vec<ConversationItem> {
    typed_items
        .iter()
        .filter(|item| !lifted_into_instructions(item))
        .cloned()
        .collect()
}

/// Persistable V2 replay material. `Deserialize` yields an **unvalidated**
/// value: after reading the V3 sidecar, `ValidatedResponsesReplayV2::verify`
/// must recompute every digest before the material may influence a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointReplayMaterialV2 {
    trusted_envelope: TrustedPromptEnvelopeV2,
    portable_history_sha256: String,
    wrapper_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    memory_revision: Option<u64>,
    branch_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prior_checkpoint_id: Option<String>,
    contract_version: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayMaterialError {
    #[error("portable history digest does not match the checkpoint wrapper")]
    PortableDigestMismatch,
    #[error("failed to serialize portable history: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl CheckpointReplayMaterialV2 {
    /// Construct material bound to a wrapper and its portable history,
    /// computing both digests from the actual data.
    pub fn try_new(
        checkpoint: &ServerResponsesCheckpointV2,
        trusted_envelope: TrustedPromptEnvelopeV2,
        portable_history: &[ConversationItem],
    ) -> Result<Self, ReplayMaterialError> {
        let portable_history_sha256 = portable_history_digest(portable_history)?;
        if portable_history_sha256 != checkpoint.portable_history_sha256 {
            return Err(ReplayMaterialError::PortableDigestMismatch);
        }
        Ok(Self {
            trusted_envelope,
            portable_history_sha256,
            wrapper_digest: checkpoint.wrapper_digest(),
            memory_revision: checkpoint.memory_revision,
            branch_id: checkpoint.branch_id.clone(),
            prior_checkpoint_id: checkpoint.prior_checkpoint_id.clone(),
            contract_version: RESPONSES_COMPACTION_CONTRACT_V2.into(),
        })
    }

    pub fn trusted_envelope(&self) -> &TrustedPromptEnvelopeV2 {
        &self.trusted_envelope
    }

    pub fn portable_history_sha256(&self) -> &str {
        &self.portable_history_sha256
    }

    pub fn wrapper_digest(&self) -> &str {
        &self.wrapper_digest
    }

    pub fn memory_revision(&self) -> Option<u64> {
        self.memory_revision
    }

    pub fn branch_id(&self) -> &str {
        &self.branch_id
    }

    pub fn prior_checkpoint_id(&self) -> Option<&str> {
        self.prior_checkpoint_id.as_deref()
    }

    pub fn contract_version(&self) -> &str {
        &self.contract_version
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayVerificationError {
    #[error("unsupported V2 checkpoint schema {0}")]
    UnsupportedSchema(u8),
    #[error("contract version mismatch")]
    ContractMismatch,
    #[error("portable history digest mismatch")]
    PortableDigestMismatch,
    #[error("wrapper digest mismatch")]
    WrapperDigestMismatch,
    #[error("checkpoint identity mismatch: {0}")]
    IdentityMismatch(&'static str),
    #[error("compatibility envelope mismatch: {0}")]
    EnvelopeMismatch(&'static str),
    #[error("typed tail must not contain a checkpoint")]
    CheckpointInTail,
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// A V2 replay that passed full material verification. Pure data proof —
/// the shell additionally re-checks revision/identity freshness adjacent
/// to dispatch. Never implements `Deserialize`.
#[derive(Debug, Clone)]
pub struct ValidatedResponsesReplayV2 {
    checkpoint_id: String,
    operation_id: String,
    branch_id: String,
    prior_checkpoint_id: Option<String>,
    contract_version: String,
    output: Vec<serde_json::Value>,
    typed_tail: Vec<ConversationItem>,
    portable_history: Vec<ConversationItem>,
    memory_revision: Option<u64>,
    history_revision: u64,
    request_identity_generation: u64,
}

impl ValidatedResponsesReplayV2 {
    /// Verify a V2 checkpoint against its replay material, the portable
    /// history and the *current* compatibility envelope.
    ///
    /// Pure data validation (no I/O): recomputes the portable and wrapper
    /// digests, checks identity/compatibility fields and the typed tail.
    /// Memory content is intentionally not compared — a memory update never
    /// invalidates a checkpoint (`memory_revision` records the expected
    /// staleness for telemetry).
    ///
    /// Branch rotation tolerance: a rewind that only rotated the live
    /// branch keeps replaying when every other immutable wrapper field
    /// matches (the digest is recomputed with the compaction-time branch).
    pub fn verify(
        checkpoint: &ServerResponsesCheckpointV2,
        material: &CheckpointReplayMaterialV2,
        portable_history: &[ConversationItem],
        current_envelope: &TrustedPromptEnvelopeV2,
        typed_tail: &[ConversationItem],
        history_revision: u64,
        request_identity_generation: u64,
    ) -> Result<Self, ReplayVerificationError> {
        if checkpoint.schema_version != RESPONSES_CHECKPOINT_SCHEMA_V2 {
            return Err(ReplayVerificationError::UnsupportedSchema(
                checkpoint.schema_version,
            ));
        }
        if material.contract_version != RESPONSES_COMPACTION_CONTRACT_V2
            || checkpoint.identity.contract_version != RESPONSES_COMPACTION_CONTRACT_V2
        {
            return Err(ReplayVerificationError::ContractMismatch);
        }
        let digest = portable_history_digest(portable_history)?;
        if digest != checkpoint.portable_history_sha256
            || digest != material.portable_history_sha256
        {
            return Err(ReplayVerificationError::PortableDigestMismatch);
        }
        let strict = checkpoint.wrapper_digest() == material.wrapper_digest;
        let rotation_ok = !strict
            && checkpoint.branch_id != material.branch_id
            && wrapper_digest_v2(
                &checkpoint.checkpoint_id,
                &checkpoint.operation_id,
                checkpoint.prompt_index,
                &material.branch_id,
                &checkpoint.identity,
                &checkpoint.portable_history_sha256,
                checkpoint.prior_checkpoint_id.as_deref(),
            ) == material.wrapper_digest;
        if !strict && !rotation_ok {
            return Err(ReplayVerificationError::WrapperDigestMismatch);
        }
        if material.prior_checkpoint_id() != checkpoint.prior_checkpoint_id.as_deref()
            || checkpoint.identity.prior_checkpoint_id != checkpoint.prior_checkpoint_id
        {
            return Err(ReplayVerificationError::IdentityMismatch(
                "prior_checkpoint_id",
            ));
        }
        if material.memory_revision() != checkpoint.memory_revision {
            return Err(ReplayVerificationError::IdentityMismatch(
                "memory_revision",
            ));
        }
        if checkpoint.output.is_empty() {
            return Err(ReplayVerificationError::IdentityMismatch("empty_output"));
        }
        // Compatibility envelope: base instructions and the canonical
        // envelope fingerprint must match; memory content is excluded.
        if current_envelope.base_instructions_sha256
            != checkpoint.identity.base_instructions_sha256
            || current_envelope.base_instructions_sha256
                != material.trusted_envelope().base_instructions_sha256
        {
            return Err(ReplayVerificationError::EnvelopeMismatch(
                "base_instructions",
            ));
        }
        if current_envelope.envelope_fingerprint != checkpoint.identity.prompt_envelope_fingerprint
            || current_envelope.envelope_fingerprint
                != material.trusted_envelope().envelope_fingerprint
        {
            return Err(ReplayVerificationError::EnvelopeMismatch(
                "prompt_envelope",
            ));
        }
        if typed_tail.iter().any(|item| item.is_responses_checkpoint()) {
            return Err(ReplayVerificationError::CheckpointInTail);
        }
        Ok(Self {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            operation_id: checkpoint.operation_id.clone(),
            branch_id: checkpoint.branch_id.clone(),
            prior_checkpoint_id: checkpoint.prior_checkpoint_id.clone(),
            contract_version: material.contract_version().to_string(),
            output: checkpoint.output.clone(),
            typed_tail: typed_tail.to_vec(),
            portable_history: portable_history.to_vec(),
            memory_revision: checkpoint.memory_revision,
            history_revision,
            request_identity_generation,
        })
    }

    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }

    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    pub fn branch_id(&self) -> &str {
        &self.branch_id
    }

    pub fn prior_checkpoint_id(&self) -> Option<&str> {
        self.prior_checkpoint_id.as_deref()
    }

    pub fn output(&self) -> &[serde_json::Value] {
        &self.output
    }

    pub fn typed_tail(&self) -> &[ConversationItem] {
        &self.typed_tail
    }

    pub fn portable_history(&self) -> &[ConversationItem] {
        &self.portable_history
    }

    pub fn memory_revision(&self) -> Option<u64> {
        self.memory_revision
    }

    pub fn history_revision(&self) -> u64 {
        self.history_revision
    }

    pub fn request_identity_generation(&self) -> u64 {
        self.request_identity_generation
    }

    fn checkpoint_binding(&self) -> ResolvedCheckpointBinding {
        ResolvedCheckpointBinding {
            checkpoint_id: self.checkpoint_id.clone(),
            operation_id: self.operation_id.clone(),
            branch_id: self.branch_id.clone(),
            contract_version: self.contract_version.clone(),
        }
    }
}

impl ResolvedResponsesRequest {
    /// Build a frozen replay request from a verified V2 replay.
    ///
    /// `request` supplies the envelope context (model, tools, cache fields,
    /// correlation) with `instructions` pre-composed via
    /// [`compose_instructions_v2`]; the body is
    /// `opaque output ++ serialized typed tail` with instruction-lifted
    /// systems removed by source.
    pub fn from_validated_replay(
        replay: &ValidatedResponsesReplayV2,
        request: &ConversationRequest,
    ) -> Result<Self, ResolvedRequestError> {
        let tail = replay_input_tail_v2(replay.typed_tail());
        let body =
            FinalResponsesRequest::from_replay_parts(request, replay.output().to_vec(), &tail)?;
        Ok(Self {
            body,
            model: request.model.clone().unwrap_or_default(),
            correlation: ResponsesCorrelation::from_request(request),
            history_revision: Some(replay.history_revision()),
            request_identity_generation: Some(replay.request_identity_generation()),
            checkpoint_binding: Some(replay.checkpoint_binding()),
            trace: request.trace.clone(),
        })
    }
}

/// Opaque frozen `/responses/compact` request. Obtained only through
/// [`ResolvedCompactRequest::try_normal`] (first compact on a
/// checkpoint-free conversation) or
/// [`ResolvedCompactRequest::from_validated_recompact`] (continuous compact
/// on a verified V2 checkpoint).
#[derive(Debug, Clone)]
pub struct ResolvedCompactRequest {
    body: serde_json::Value,
    model: String,
    correlation: ResponsesCorrelation,
    checkpoint_binding: Option<ResolvedCheckpointBinding>,
    history_revision: Option<u64>,
    request_identity_generation: Option<u64>,
    trace: Option<Box<dyn TraceContext>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ResolvedCompactError {
    #[error("compact request requires a non-empty model")]
    MissingModel,
    #[error("compact request requires a non-empty input")]
    MissingInput,
    #[error("normal compact rejects every checkpoint variant")]
    CheckpointInNormalRequest,
    #[error(transparent)]
    Build(#[from] ResponsesRequestBuildError),
}

/// Project a flattened full-request body onto the compact endpoint's field
/// allowlist, applying the V2 instructions semantics (composed
/// instructions already present in the body) and the compact-only user
/// context suffix.
fn compact_body_from_final(
    final_body: serde_json::Value,
    user_context: Option<&str>,
) -> Result<serde_json::Value, ResolvedCompactError> {
    let model = final_body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .filter(|model| !model.is_empty())
        .ok_or(ResolvedCompactError::MissingModel)?
        .to_string();
    let input = final_body
        .get("input")
        .and_then(serde_json::Value::as_array)
        .filter(|input| !input.is_empty())
        .cloned()
        .ok_or(ResolvedCompactError::MissingInput)?;
    let string_field = |name: &str| {
        final_body
            .get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    let optional_value = |name: &str| {
        final_body
            .get(name)
            .filter(|value| !value.is_null())
            .filter(|value| !value.as_array().is_some_and(Vec::is_empty))
            .cloned()
    };
    let mut instructions = string_field("instructions");
    if let Some(context) = user_context.filter(|context| !context.is_empty()) {
        let value = instructions.get_or_insert_with(String::new);
        value.push_str(USER_CONTEXT_DELIMITER);
        value.push_str(context);
    }
    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::Value::String(model));
    body.insert("input".into(), serde_json::Value::Array(input));
    // `parallel_tool_calls` comes from the canonical context explicitly —
    // never an invented default.
    if let Some(value) = final_body.get("parallel_tool_calls") {
        body.insert("parallel_tool_calls".into(), value.clone());
    }
    if let Some(value) = instructions {
        body.insert("instructions".into(), serde_json::Value::String(value));
    }
    for (name, value) in [
        ("tools", optional_value("tools")),
        ("reasoning", optional_value("reasoning")),
        ("text", optional_value("text")),
        (
            "service_tier",
            string_field("service_tier").map(serde_json::Value::String),
        ),
        (
            "prompt_cache_key",
            string_field("prompt_cache_key").map(serde_json::Value::String),
        ),
        ("prompt_cache_options", optional_value("prompt_cache_options")),
        (
            "prompt_cache_retention",
            string_field("prompt_cache_retention").map(serde_json::Value::String),
        ),
    ] {
        if let Some(value) = value {
            body.insert(name.into(), value);
        }
    }
    Ok(serde_json::Value::Object(body))
}

impl ResolvedCompactRequest {
    fn from_final_body(
        final_body: serde_json::Value,
        request: &ConversationRequest,
        user_context: Option<&str>,
        checkpoint_binding: Option<ResolvedCheckpointBinding>,
        history_revision: Option<u64>,
        request_identity_generation: Option<u64>,
    ) -> Result<Self, ResolvedCompactError> {
        let body = compact_body_from_final(final_body, user_context)?;
        Ok(Self {
            model: request.model.clone().unwrap_or_default(),
            body,
            correlation: ResponsesCorrelation::from_request(request),
            checkpoint_binding,
            history_revision,
            request_identity_generation,
            trace: request.trace.clone(),
        })
    }

    /// First compact on a checkpoint-free conversation. Rejects every
    /// checkpoint variant.
    pub fn try_normal(
        request: &ConversationRequest,
        user_context: Option<&str>,
    ) -> Result<Self, ResolvedCompactError> {
        if request
            .items
            .iter()
            .any(|item| item.is_responses_checkpoint())
        {
            return Err(ResolvedCompactError::CheckpointInNormalRequest);
        }
        // Input excludes instruction-lifted systems by source.
        let input_items = replay_input_tail_v2(&request.items);
        let final_body =
            FinalResponsesRequest::from_replay_parts(request, Vec::new(), &input_items)?
                .into_body();
        Self::from_final_body(
            final_body,
            request,
            user_context,
            None,
            request.history_revision,
            None,
        )
    }

    /// Continuous compact on a verified V2 checkpoint:
    /// `input = prior opaque output ++ serialized typed tail`, with
    /// instruction-lifted systems removed by source.
    pub fn from_validated_recompact(
        replay: &ValidatedResponsesReplayV2,
        request: &ConversationRequest,
        user_context: Option<&str>,
    ) -> Result<Self, ResolvedCompactError> {
        let tail = replay_input_tail_v2(replay.typed_tail());
        let final_body =
            FinalResponsesRequest::from_replay_parts(request, replay.output().to_vec(), &tail)?
                .into_body();
        Self::from_final_body(
            final_body,
            request,
            user_context,
            Some(replay.checkpoint_binding()),
            Some(replay.history_revision()),
            Some(replay.request_identity_generation()),
        )
    }

    pub fn body(&self) -> &serde_json::Value {
        &self.body
    }

    pub(crate) fn into_body(self) -> serde_json::Value {
        self.body
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn correlation(&self) -> &ResponsesCorrelation {
        &self.correlation
    }

    pub fn checkpoint_binding(&self) -> Option<&ResolvedCheckpointBinding> {
        self.checkpoint_binding.as_ref()
    }

    pub fn history_revision(&self) -> Option<u64> {
        self.history_revision
    }

    pub fn request_identity_generation(&self) -> Option<u64> {
        self.request_identity_generation
    }

    pub fn take_trace(&mut self) -> Option<Box<dyn TraceContext>> {
        self.trace.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_request() -> ConversationRequest {
        ConversationRequest {
            items: vec![
                ConversationItem::system("base"),
                ConversationItem::user("hello"),
            ],
            model: Some("grok-test".into()),
            x_grok_conv_id: Some("conv".into()),
            ..Default::default()
        }
    }

    fn checkpointed_request() -> ConversationRequest {
        let wrapper = Box::new(super::super::ServerResponsesCheckpointV1 {
            schema_version: 1,
            checkpoint_id: "cp".into(),
            operation_id: "op".into(),
            prompt_index: 0,
            created_at: chrono::Utc::now(),
            auto_continue: false,
            mode: super::super::ResponsesCompactionModeV1 {
                name: "default".into(),
                detail: None,
            },
            branch_id: "branch".into(),
            identity: super::super::CheckpointIdentityV1 {
                provider_id: "xai".into(),
                api: "responses".into(),
                endpoint_fingerprint: "ep".into(),
                model: "grok-test".into(),
                auth_principal_fingerprint: "principal".into(),
                contract_version: "responses-compact-codex-v1".into(),
                prompt_envelope_fingerprint: "envelope".into(),
                canonical_prompt_projection: None,
            },
            output: vec![serde_json::json!({
                "type": "compaction",
                "encrypted_content": "opaque"
            })],
            portable_history_path: "compaction_checkpoints/cp.json".into(),
            portable_history_sha256: "digest".into(),
            portable_history_bytes: 1,
            checkpoint_token_seed: 1,
            token_seed_source: super::super::TokenSeedSource::UsageOutputTokens,
            server_output_item_count: 1,
        });
        ConversationRequest {
            items: vec![
                ConversationItem::ResponsesCompactionCheckpoint(wrapper),
                ConversationItem::user("tail user"),
            ],
            model: Some("grok-test".into()),
            history_revision: Some(7),
            ..Default::default()
        }
    }

    #[test]
    fn try_normal_rejects_checkpoints() {
        let request = checkpointed_request();
        assert!(matches!(
            ResolvedResponsesRequest::try_normal(&request),
            Err(ResolvedRequestError::CheckpointInNormalRequest)
        ));
    }

    #[test]
    fn try_normal_freezes_body_and_correlation() {
        let request = normal_request();
        let resolved = ResolvedResponsesRequest::try_normal(&request).unwrap();
        assert_eq!(resolved.model(), "grok-test");
        assert_eq!(
            resolved.correlation().x_grok_conv_id.as_deref(),
            Some("conv")
        );
        assert_eq!(
            resolved.body().get("model").and_then(|v| v.as_str()),
            Some("grok-test")
        );
        assert!(resolved.checkpoint_binding().is_none());
    }

    #[test]
    fn legacy_permit_requires_checkpoint() {
        let request = normal_request();
        assert!(matches!(
            ValidatedLegacyReplayV1::try_new(&request, &[], 1),
            Err(LegacyReplayPermitError::MissingCheckpoint)
        ));
    }

    /// Portable history whose digest matches `checkpointed_request()`'s
    /// wrapper. Computed once via `portable_history_digest`.
    fn matching_portable_history() -> Vec<ConversationItem> {
        vec![
            ConversationItem::system("base"),
            ConversationItem::user("compacted away"),
        ]
    }

    fn checkpointed_request_with_digest() -> ConversationRequest {
        let mut request = checkpointed_request();
        let digest = super::super::responses::portable_history_digest(
            &matching_portable_history(),
        )
        .unwrap();
        let Some(ConversationItem::ResponsesCompactionCheckpoint(wrapper)) =
            request.items.first_mut()
        else {
            unreachable!();
        };
        wrapper.portable_history_sha256 = digest;
        request
    }

    #[test]
    fn legacy_permit_freezes_wrapper_binding() {
        let request = checkpointed_request_with_digest();
        let permit =
            ValidatedLegacyReplayV1::try_new(&request, &matching_portable_history(), 3).unwrap();
        assert_eq!(permit.checkpoint_id(), "cp");
        assert_eq!(permit.operation_id(), "op");
        assert_eq!(permit.branch_id(), "branch");
        assert_eq!(permit.history_revision(), Some(7));
        assert_eq!(permit.request_identity_generation(), 3);
        // The flattened body starts with the checkpoint output prefix.
        let input = permit
            .body()
            .get("input")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(
            input[0].get("type").and_then(|v| v.as_str()),
            Some("compaction")
        );
    }

    #[test]
    fn legacy_permit_rejects_digest_mismatch() {
        let request = checkpointed_request();
        // The wrapper still carries the placeholder digest "digest": no
        // caller can build a permit without the real sidecar material.
        assert!(matches!(
            ValidatedLegacyReplayV1::try_new(&request, &matching_portable_history(), 1),
            Err(LegacyReplayPermitError::DigestMismatch)
        ));
    }

    #[test]
    fn legacy_permit_rejects_empty_output_checkpoint() {
        let mut request = checkpointed_request_with_digest();
        let Some(ConversationItem::ResponsesCompactionCheckpoint(wrapper)) =
            request.items.first_mut()
        else {
            unreachable!();
        };
        wrapper.output.clear();
        assert!(matches!(
            ValidatedLegacyReplayV1::try_new(&request, &matching_portable_history(), 1),
            Err(LegacyReplayPermitError::Validation(_))
        ));
    }
}

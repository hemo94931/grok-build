//! Responses compaction checkpoint V2 contract types.
//!
//! V2 fixes the structural flaws of V1: the prompt envelope is captured
//! explicitly (never re-derived by scanning `role == "system"` items),
//! memory is modeled by [`SystemSource`](super::SystemSource) instead of
//! string tags, and the replay material is a separately persisted,
//! re-verifiable record.
//!
//! Reader-first rollout: these types are fully readable/validatable before
//! any writer is enabled. Old binaries cannot deserialize the new
//! [`super::ConversationItem::ResponsesCompactionCheckpointV2`] variant, so
//! rollback targets must keep V2/V3 read capability with writers disabled.

use serde::{Deserialize, Serialize};

use super::responses::canonical_json_bytes;
use super::{ConversationItem, ResponsesCompactionModeV1, TokenSeedSource};

/// Frozen V2 `/responses/compact` contract version.
pub const RESPONSES_COMPACTION_CONTRACT_V2: &str = "responses-compact-grok-v2";

/// Schema version stored inside [`ServerResponsesCheckpointV2`].
pub const RESPONSES_CHECKPOINT_SCHEMA_V2: u8 = 2;

/// Trusted prompt envelope captured at V2 compaction time.
///
/// Identity semantics (plan 阶段 3):
///
/// * `base_instructions_sha256` enters checkpoint **compatibility
///   identity** — a checkpoint may not replay under different base
///   instructions;
/// * `memory_revision` is metadata/telemetry only — a memory update never
///   invalidates an existing checkpoint;
/// * `wire_prompt_sha256` covers the *complete* rendered instructions
///   (base + separator + memory at compaction time) for exact wire
///   accounting; it is diagnostic, not a compatibility gate;
/// * `envelope_fingerprint` covers the canonical non-transcript request
///   semantics (tools, tool_choice, reasoning, text, parallel_tool_calls,
///   cache-route fields) and enters compatibility identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedPromptEnvelopeV2 {
    pub base_instructions_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_revision: Option<u64>,
    pub envelope_fingerprint: String,
    pub wire_prompt_sha256: String,
}

/// Identity that must match before replaying a V2 server compaction
/// checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointIdentityV2 {
    pub provider_id: String,
    pub api: String,
    pub endpoint_fingerprint: String,
    pub model: String,
    pub auth_principal_fingerprint: String,
    pub contract_version: String,
    /// Fingerprint over the compatibility-relevant canonical envelope
    /// (includes base instructions; excludes memory content).
    pub prompt_envelope_fingerprint: String,
    /// Base instructions hash at compaction; part of compatibility
    /// identity, duplicated from the trusted envelope for cheap comparison.
    pub base_instructions_sha256: String,
    /// Recompact chain link: the checkpoint this one was built on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_checkpoint_id: Option<String>,
    /// Normalized cache route fingerprint (provider + deployment + model
    /// family + logical namespace). Diagnostic in D2; routing lands in D3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_route_fingerprint: Option<String>,
}

/// Local wrapper for the canonical output of `POST /responses/compact`
/// under the V2 contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerResponsesCheckpointV2 {
    pub schema_version: u8,
    pub checkpoint_id: String,
    pub operation_id: String,
    pub prompt_index: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub auto_continue: bool,
    pub mode: ResponsesCompactionModeV1,
    pub branch_id: String,
    pub identity: CheckpointIdentityV2,
    /// Opaque provider compact output. Never scanned, never re-ordered,
    /// never filtered by role.
    pub output: Vec<serde_json::Value>,
    /// V3 sidecar path (relative to the session dir).
    pub portable_history_path: String,
    pub portable_history_sha256: String,
    pub portable_history_bytes: u64,
    pub checkpoint_token_seed: u64,
    pub token_seed_source: TokenSeedSource,
    pub server_output_item_count: usize,
    /// Recompact chain link (mirrors identity.prior_checkpoint_id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_checkpoint_id: Option<String>,
    /// Memory revision at compaction (metadata/telemetry only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_revision: Option<u64>,
}

impl ServerResponsesCheckpointV2 {
    /// Digest over the wrapper's immutable identity fields. The replay
    /// material, the V3 sidecar and V2 segment staging all carry this
    /// digest to bind themselves to *this* wrapper — replacing V1's fragile
    /// whole-JSON equality comparisons.
    pub fn wrapper_digest(&self) -> String {
        wrapper_digest_v2(
            &self.checkpoint_id,
            &self.operation_id,
            self.prompt_index,
            &self.branch_id,
            &self.identity,
            &self.portable_history_sha256,
            self.prior_checkpoint_id.as_deref(),
        )
    }
}

/// Compute the V2 wrapper digest from its immutable binding fields.
pub fn wrapper_digest_v2(
    checkpoint_id: &str,
    operation_id: &str,
    prompt_index: usize,
    branch_id: &str,
    identity: &CheckpointIdentityV2,
    portable_history_sha256: &str,
    prior_checkpoint_id: Option<&str>,
) -> String {
    use sha2::Digest as _;
    let value = serde_json::json!({
        "checkpoint_id": checkpoint_id,
        "operation_id": operation_id,
        "prompt_index": prompt_index,
        "branch_id": branch_id,
        "identity": identity,
        "portable_history_sha256": portable_history_sha256,
        "prior_checkpoint_id": prior_checkpoint_id,
    });
    let bytes = canonical_json_bytes(&value)
        .expect("wrapper digest payload is always serializable");
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// sha256 hex of the base-instructions string that enters checkpoint
/// compatibility identity. The same function feeds the V2 writer and the
/// reader's compatibility check; semantics must never change without a
/// contract bump.
pub fn base_instructions_sha256_v2(base_instructions: &str) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(base_instructions.as_bytes()))
}

/// sha256 hex of the complete rendered instructions (base + separator +
/// memory) — the exact wire prompt hash. Diagnostic; never a
/// compatibility gate.
pub fn wire_prompt_sha256_v2(rendered_instructions: &str) -> String {
    use sha2::Digest as _;
    format!(
        "{:x}",
        sha2::Sha256::digest(rendered_instructions.as_bytes())
    )
}

/// Canonical envelope fingerprint for the V2 contract: sha256 over the
/// canonical JSON of the non-transcript request semantics (tools incl.
/// hosted extra entries, tool_choice, reasoning, text,
/// parallel_tool_calls, service tier, prompt cache options/retention).
///
/// `prompt_cache_key` is deliberately excluded (it is a routing concern,
/// not compatibility); model enters identity separately. The same function
/// feeds the V2 writer (D4) and the reader's compatibility check, so its
/// semantics must never change without a contract bump.
pub fn canonical_envelope_fingerprint_v2(request: &super::ConversationRequest) -> String {
    use sha2::Digest as _;
    // Serialize the envelope exactly as the wire body would carry it: a
    // typed conversion over an item-less request yields the same
    // tools/reasoning/text/parallel_tool_calls/cache fields.
    let mut envelope_request = request.clone();
    envelope_request.items = Vec::new();
    let body = super::responses::FinalResponsesRequest::try_from(&envelope_request)
        .map(|final_request| final_request.into_body())
        .unwrap_or_else(|_| serde_json::json!({}));
    let mut envelope = serde_json::Map::new();
    for field in [
        "tools",
        "tool_choice",
        "reasoning",
        "text",
        "parallel_tool_calls",
        "service_tier",
        "prompt_cache_options",
        "prompt_cache_retention",
    ] {
        if let Some(value) = body.get(field).filter(|value| !value.is_null()) {
            envelope.insert(field.into(), value.clone());
        }
    }
    let bytes = canonical_json_bytes(&serde_json::Value::Object(envelope))
        .expect("envelope fingerprint payload is always serializable");
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Borrowed view over either checkpoint wrapper variant, for the variant
/// fan-out audit: call sites that only need identity/binding fields must
/// go through this instead of matching a single variant.
#[derive(Debug, Clone, Copy)]
pub enum ResponsesCheckpointRef<'a> {
    V1(&'a super::ServerResponsesCheckpointV1),
    V2(&'a ServerResponsesCheckpointV2),
}

impl ResponsesCheckpointRef<'_> {
    pub fn checkpoint_id(&self) -> &str {
        match self {
            Self::V1(wrapper) => &wrapper.checkpoint_id,
            Self::V2(wrapper) => &wrapper.checkpoint_id,
        }
    }

    pub fn operation_id(&self) -> &str {
        match self {
            Self::V1(wrapper) => &wrapper.operation_id,
            Self::V2(wrapper) => &wrapper.operation_id,
        }
    }

    pub fn branch_id(&self) -> &str {
        match self {
            Self::V1(wrapper) => &wrapper.branch_id,
            Self::V2(wrapper) => &wrapper.branch_id,
        }
    }

    pub fn prompt_index(&self) -> usize {
        match self {
            Self::V1(wrapper) => wrapper.prompt_index,
            Self::V2(wrapper) => wrapper.prompt_index,
        }
    }

    pub fn portable_history_path(&self) -> &str {
        match self {
            Self::V1(wrapper) => &wrapper.portable_history_path,
            Self::V2(wrapper) => &wrapper.portable_history_path,
        }
    }

    pub fn portable_history_sha256(&self) -> &str {
        match self {
            Self::V1(wrapper) => &wrapper.portable_history_sha256,
            Self::V2(wrapper) => &wrapper.portable_history_sha256,
        }
    }

    /// Recompact chain link (V2 only; V1 has no chain concept).
    pub fn prior_checkpoint_id(&self) -> Option<&str> {
        match self {
            Self::V1(_) => None,
            Self::V2(wrapper) => wrapper.prior_checkpoint_id.as_deref(),
        }
    }

    pub fn schema_version(&self) -> u8 {
        match self {
            Self::V1(wrapper) => wrapper.schema_version,
            Self::V2(wrapper) => wrapper.schema_version,
        }
    }

    pub fn checkpoint_token_seed(&self) -> u64 {
        match self {
            Self::V1(wrapper) => wrapper.checkpoint_token_seed,
            Self::V2(wrapper) => wrapper.checkpoint_token_seed,
        }
    }

    pub fn token_seed_source(&self) -> TokenSeedSource {
        match self {
            Self::V1(wrapper) => wrapper.token_seed_source,
            Self::V2(wrapper) => wrapper.token_seed_source,
        }
    }

    pub fn output(&self) -> &[serde_json::Value] {
        match self {
            Self::V1(wrapper) => &wrapper.output,
            Self::V2(wrapper) => &wrapper.output,
        }
    }

    pub fn mode(&self) -> &ResponsesCompactionModeV1 {
        match self {
            Self::V1(wrapper) => &wrapper.mode,
            Self::V2(wrapper) => &wrapper.mode,
        }
    }

    pub fn auto_continue(&self) -> bool {
        match self {
            Self::V1(wrapper) => wrapper.auto_continue,
            Self::V2(wrapper) => wrapper.auto_continue,
        }
    }

    pub fn created_at(&self) -> chrono::DateTime<chrono::Utc> {
        match self {
            Self::V1(wrapper) => wrapper.created_at,
            Self::V2(wrapper) => wrapper.created_at,
        }
    }
}

impl ConversationItem {
    /// Borrow either checkpoint wrapper variant.
    pub fn as_responses_checkpoint(&self) -> Option<ResponsesCheckpointRef<'_>> {
        match self {
            Self::ResponsesCompactionCheckpoint(wrapper) => {
                Some(ResponsesCheckpointRef::V1(wrapper))
            }
            Self::ResponsesCompactionCheckpointV2(wrapper) => {
                Some(ResponsesCheckpointRef::V2(wrapper))
            }
            _ => None,
        }
    }

    /// Whether this item is any Responses compaction checkpoint variant.
    pub fn is_responses_checkpoint(&self) -> bool {
        self.as_responses_checkpoint().is_some()
    }
}

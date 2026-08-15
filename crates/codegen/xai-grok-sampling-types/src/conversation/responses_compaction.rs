//! Responses compaction checkpoint contract types.
//!
//! The prompt envelope is captured explicitly (never re-derived by scanning
//! `role == "system"` items), memory is modeled by
//! [`SystemSource`](super::SystemSource) instead of string tags, and replay
//! material is a separately persisted, re-verifiable record.
//!
//! Checkpoint shape (compaction_trigger wire contract): retained typed prefix
//! + a single opaque compaction blob. The trigger control item is request-only
//! and is never persisted here.

use serde::{Deserialize, Serialize};

use super::responses::canonical_json_bytes;
use super::{ConversationItem, ResponsesCompactionMode, TokenSeedSource};

/// Frozen Responses remote-compaction contract identifier.
///
/// Unchanged across the unary→trigger migration (D7).
pub const RESPONSES_COMPACTION_CONTRACT: &str = "responses-compact-grok";

/// Mirror of upstream `RETAINED_MESSAGE_TOKEN_BUDGET` (64k tokens).
pub const RETAINED_MESSAGE_TOKEN_BUDGET: u64 = 64_000;

/// Non-final agent messages larger than this are dropped from the retained
/// prefix (upstream `MAX_RETAINED_AGENT_MESSAGE_TOKENS`).
pub const MAX_RETAINED_AGENT_MESSAGE_TOKENS: u64 = 10_000;

/// Bytes-per-token heuristic used for retained-prefix budgeting (matches
/// `xai_token_estimation` and upstream's 4-bytes/token estimate).
const BYTES_PER_TOKEN: u64 = 4;

/// Wire control item appended last on every remote-compaction request.
/// Never enters history and is never persisted.
pub fn compaction_trigger_wire_item() -> serde_json::Value {
    serde_json::json!({ "type": "compaction_trigger" })
}

/// Whether a wire value is a valid opaque compaction blob: type
/// `compaction` or alias `compaction_summary`, with non-empty
/// `encrypted_content`.
pub fn is_valid_compaction_item(value: &serde_json::Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let Some(item_type) = object.get("type").and_then(serde_json::Value::as_str) else {
        return false;
    };
    if !matches!(item_type, "compaction" | "compaction_summary") {
        return false;
    }
    object
        .get("encrypted_content")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|content| !content.is_empty())
}

/// Token estimate for one conversation item under the bytes/4 heuristic.
pub fn estimate_item_tokens(item: &ConversationItem) -> u64 {
    let text = item.text_content();
    (text.len() as u64) / BYTES_PER_TOKEN
}

/// Whether an item is eligible for the retained typed prefix (D8).
///
/// Retains user messages, non-lifted system messages (developer/system
/// roles on the wire), and non-final agent messages whose estimate is at
/// most [`MAX_RETAINED_AGENT_MESSAGE_TOKENS`]. Reasoning, tool results,
/// backend tool calls, checkpoints, and instruction-lifted systems are
/// excluded — those live only inside the opaque blob after compaction.
pub fn is_retained_for_compaction(item: &ConversationItem) -> bool {
    match item {
        ConversationItem::User(_) => true,
        ConversationItem::System(system) => !system.source.lifts_into_instructions(),
        ConversationItem::Assistant(assistant) => {
            if assistant
                .content
                .as_ref()
                .starts_with("Message Type: FINAL_ANSWER\n")
            {
                return false;
            }
            estimate_item_tokens(item) <= MAX_RETAINED_AGENT_MESSAGE_TOKENS
        }
        ConversationItem::ToolResult(_)
        | ConversationItem::BackendToolCall(_)
        | ConversationItem::Reasoning(_)
        | ConversationItem::ResponsesCompactionCheckpoint(_) => false,
    }
}

/// Build the retained typed prefix from pre-compaction history: keep
/// eligible items, then truncate newest-first to
/// [`RETAINED_MESSAGE_TOKEN_BUDGET`].
pub fn build_retained_prefix(history: &[ConversationItem]) -> Vec<ConversationItem> {
    let candidates: Vec<ConversationItem> = history
        .iter()
        .filter(|item| is_retained_for_compaction(item))
        .cloned()
        .collect();
    truncate_retained_newest_first(candidates, RETAINED_MESSAGE_TOKEN_BUDGET)
}

/// Newest-first truncation to a token budget. Returns items in original
/// chronological order.
pub fn truncate_retained_newest_first(
    items: Vec<ConversationItem>,
    max_tokens: u64,
) -> Vec<ConversationItem> {
    let mut remaining = max_tokens;
    let mut kept_reversed = Vec::with_capacity(items.len());
    for item in items.into_iter().rev() {
        if remaining == 0 {
            break;
        }
        let tokens = estimate_item_tokens(&item).max(1);
        if tokens > remaining {
            continue;
        }
        remaining = remaining.saturating_sub(tokens);
        kept_reversed.push(item);
    }
    kept_reversed.reverse();
    kept_reversed
}

/// Trusted prompt envelope captured at compaction time.
///
/// Identity semantics:
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
pub struct TrustedPromptEnvelope {
    pub base_instructions_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_revision: Option<u64>,
    pub envelope_fingerprint: String,
    pub wire_prompt_sha256: String,
}

/// Identity that must match before replaying a server compaction checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointIdentity {
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
    /// family + logical namespace).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_route_fingerprint: Option<String>,
}

/// Local wrapper for a remote-compaction checkpoint.
///
/// Replacement history = [`retained_prefix`] + [`compaction_item`] (blob
/// last). Old unary-compact shapes (`output: [...]`) fail closed on
/// deserialize (D1 — no compat reader).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerResponsesCheckpoint {
    pub checkpoint_id: String,
    pub operation_id: String,
    pub prompt_index: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub auto_continue: bool,
    pub mode: ResponsesCompactionMode,
    pub branch_id: String,
    pub identity: CheckpointIdentity,
    /// Retained typed prefix (user/developer/system, budget-truncated).
    /// Never contains checkpoints or the request-only trigger item.
    pub retained_prefix: Vec<ConversationItem>,
    /// Single opaque provider compaction item (`compaction` /
    /// `compaction_summary` + non-empty `encrypted_content`). Never
    /// scanned, never re-ordered.
    pub compaction_item: serde_json::Value,
    /// Portable-history sidecar path (relative to the session directory).
    pub portable_history_path: String,
    pub portable_history_sha256: String,
    pub portable_history_bytes: u64,
    pub checkpoint_token_seed: u64,
    pub token_seed_source: TokenSeedSource,
    /// Recompact chain link (mirrors identity.prior_checkpoint_id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_checkpoint_id: Option<String>,
    /// Memory revision at compaction (metadata/telemetry only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_revision: Option<u64>,
}

impl ServerResponsesCheckpoint {
    /// Digest over every replay- and accounting-relevant immutable field.
    /// Replay material, markers, and segment staging carry this digest to bind
    /// themselves to the exact opaque provider output and token seed.
    ///
    /// `portable_history_bytes` is deliberately excluded: the sidecar fills
    /// that derived byte count after replay material is created and validates
    /// it independently against the canonical portable history.
    pub fn wrapper_digest(&self) -> String {
        self.wrapper_digest_for_branch(&self.branch_id)
    }

    /// Recompute the digest with a different active branch while keeping all
    /// other fields frozen. This is the only tolerated rewind/fork mutation.
    pub fn wrapper_digest_for_branch(&self, branch_id: &str) -> String {
        wrapper_digest_for_branch(self, branch_id)
    }
}

/// Compute the full wrapper digest with an explicit active branch.
pub fn wrapper_digest_for_branch(
    checkpoint: &ServerResponsesCheckpoint,
    branch_id: &str,
) -> String {
    use sha2::Digest as _;
    let value = serde_json::json!({
        "checkpoint_id": checkpoint.checkpoint_id,
        "operation_id": checkpoint.operation_id,
        "prompt_index": checkpoint.prompt_index,
        "created_at": checkpoint.created_at,
        "auto_continue": checkpoint.auto_continue,
        "mode": checkpoint.mode,
        "branch_id": branch_id,
        "identity": checkpoint.identity,
        "retained_prefix": checkpoint.retained_prefix,
        "compaction_item": checkpoint.compaction_item,
        "portable_history_path": checkpoint.portable_history_path,
        "portable_history_sha256": checkpoint.portable_history_sha256,
        "checkpoint_token_seed": checkpoint.checkpoint_token_seed,
        "token_seed_source": checkpoint.token_seed_source,
        "prior_checkpoint_id": checkpoint.prior_checkpoint_id,
        "memory_revision": checkpoint.memory_revision,
    });
    let bytes =
        canonical_json_bytes(&value).expect("wrapper digest payload is always serializable");
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// SHA-256 hex of the base-instructions string that enters checkpoint
/// compatibility identity. The same function feeds the writer and the
/// reader's compatibility check; semantics must not change without a
/// contract bump.
pub fn base_instructions_sha256(base_instructions: &str) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(base_instructions.as_bytes()))
}

/// SHA-256 hex of the complete rendered instructions (base + separator +
/// memory), the exact wire prompt hash. Diagnostic; never a compatibility
/// gate.
pub fn wire_prompt_sha256(rendered_instructions: &str) -> String {
    use sha2::Digest as _;
    format!(
        "{:x}",
        sha2::Sha256::digest(rendered_instructions.as_bytes())
    )
}

/// Canonical envelope fingerprint: SHA-256 over the canonical JSON of the
/// non-transcript request semantics (tools including hosted extra entries,
/// tool_choice, reasoning, text, parallel_tool_calls, service tier, prompt
/// cache options and retention).
///
/// `prompt_cache_key` is deliberately excluded because it is a routing
/// concern, not compatibility; model enters identity separately. The same
/// function feeds the writer and reader compatibility check, so its semantics
/// must not change without a contract bump.
pub fn canonical_envelope_fingerprint(request: &super::ConversationRequest) -> String {
    use sha2::Digest as _;
    // Serialize the envelope exactly as the wire body would carry it: a typed
    // conversion over an item-less request yields the same fields.
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

impl ConversationItem {
    /// Borrow the Responses compaction checkpoint wrapper, if present.
    pub fn as_responses_checkpoint(&self) -> Option<&ServerResponsesCheckpoint> {
        match self {
            Self::ResponsesCompactionCheckpoint(wrapper) => Some(wrapper.as_ref()),
            _ => None,
        }
    }

    /// Whether this item is a Responses compaction checkpoint.
    pub fn is_responses_checkpoint(&self) -> bool {
        self.as_responses_checkpoint().is_some()
    }
}

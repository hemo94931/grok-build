use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use xai_grok_sampler::{ResponsesCompactFailure, ResponsesCompactResponse};
use xai_grok_sampling_types::{
    CheckpointIdentity, ConversationItem, ResponsesCompactionMode, ServerResponsesCheckpoint,
    TokenSeedSource, TrustedPromptEnvelope,
};

pub const CAPABILITY_CACHE_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Clone)]
pub struct ResponsesRequestSnapshot {
    pub chat_revision: u64,
    pub request_identity_generation: u64,
    pub prompt_index: usize,
    pub pre_compaction_tokens: u64,
    /// Flattened provider-visible body at snapshot time. Raw JSON used for
    /// diagnostics and token accounting, never as a normal POST gate.
    pub final_request: serde_json::Value,
    pub credential: xai_grok_sampler::RequestCredentialSnapshot,
    pub model: String,
    pub input: Vec<serde_json::Value>,
    pub portable_history: Vec<ConversationItem>,
    pub instructions: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_options: Option<serde_json::Value>,
    pub prompt_cache_retention: Option<String>,
    pub service_tier: Option<String>,
    pub semantic_envelope: serde_json::Value,
    pub semantic_envelope_tokens: u64,
    pub identity: CheckpointIdentity,
    /// Trusted prompt envelope resolved from the same request that fed the
    /// gate, so writer identity matches the next turn's gate identity.
    pub trusted_envelope: TrustedPromptEnvelope,
    pub request_bytes: Vec<u8>,
    pub trigger: xai_grok_telemetry::events::CompactionTrigger,
    pub mode: ResponsesCompactionMode,
    pub user_context: Option<String>,
    pub cancellation: tokio_util::sync::CancellationToken,
}

impl std::fmt::Debug for ResponsesRequestSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesRequestSnapshot")
            .field("chat_revision", &self.chat_revision)
            .field(
                "request_identity_generation",
                &self.request_identity_generation,
            )
            .field("prompt_index", &self.prompt_index)
            .field("pre_compaction_tokens", &self.pre_compaction_tokens)
            .field("model", &self.model)
            .field("input_items", &self.input.len())
            .field("portable_history_items", &self.portable_history.len())
            .field("request_bytes", &self.request_bytes.len())
            .field("trigger", &self.trigger)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub enum PreparedCompaction {
    Server {
        wrapper: Box<ServerResponsesCheckpoint>,
        portable: Vec<ConversationItem>,
        mode_artifacts: Vec<ConversationItem>,
        token_seed: u64,
    },
    Builtin {
        history: Vec<ConversationItem>,
        mode_artifacts: Vec<ConversationItem>,
        token_seed: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionExecutionOutcome {
    Committed,
    Cancelled,
    Superseded,
    Failed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ServerCheckpointSeedError {
    #[error("failed to serialize canonical server output: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("server checkpoint did not reduce the conversation")]
    DidNotShrink,
}

/// Canonical non-transcript portion of a final sealed compact body.
///
/// `input` is deliberately excluded: the server output token seed accounts
/// for the compacted transcript, while this envelope accounts only for prompt
/// semantics that remain live around it. Routing and transport-only fields do
/// not consume model context and are excluded as well.
pub fn resolved_prompt_envelope(
    final_body: &serde_json::Value,
) -> Result<serde_json::Value, serde_json::Error> {
    let mut envelope = serde_json::Map::new();
    for field in [
        "instructions",
        "tools",
        "tool_choice",
        "reasoning",
        "text",
        "parallel_tool_calls",
    ] {
        if let Some(value) = final_body.get(field).filter(|value| !value.is_null()) {
            envelope.insert(field.into(), value.clone());
        }
    }
    let envelope = serde_json::Value::Object(envelope);
    xai_grok_sampling_types::canonical_json_bytes(&envelope)?;
    Ok(envelope)
}

/// Approximate token cost of the non-transcript envelope in the final sealed
/// compact body. This never reconstructs checkpoint identity or history.
pub fn prompt_envelope_token_estimate(
    final_body: &serde_json::Value,
) -> Result<u64, serde_json::Error> {
    let envelope = resolved_prompt_envelope(final_body)?;
    Ok((xai_grok_sampling_types::canonical_json_bytes(&envelope)?.len() as u64).div_ceil(4))
}

fn canonical_compaction_item_token_estimate(
    compaction_item: &serde_json::Value,
) -> Result<u64, serde_json::Error> {
    fn replace_image_payloads(value: &mut serde_json::Value) -> u64 {
        match value {
            serde_json::Value::Array(values) => values.iter_mut().map(replace_image_payloads).sum(),
            serde_json::Value::Object(object) => {
                let is_image = matches!(
                    object.get("type").and_then(serde_json::Value::as_str),
                    Some("input_image" | "image")
                );
                let mut images = 0;
                if is_image {
                    let has_image = ["image_url", "url", "file_id"].into_iter().any(|field| {
                        object
                            .get(field)
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|value| !value.is_empty())
                    });
                    if has_image {
                        images = 1;
                        for field in ["image_url", "url"] {
                            if let Some(value) = object.get_mut(field)
                                && value.is_string()
                            {
                                *value = serde_json::Value::String(String::new());
                            }
                        }
                    }
                }
                images + object.values_mut().map(replace_image_payloads).sum::<u64>()
            }
            _ => 0,
        }
    }

    let mut value = compaction_item.clone();
    let images = replace_image_payloads(&mut value);
    let text_tokens =
        (xai_grok_sampling_types::canonical_json_bytes(&value)?.len() as u64).div_ceil(4);
    Ok(text_tokens.saturating_add(xai_token_estimation::estimate_image_tokens(images)))
}

/// Derive the checkpoint token seed from a remote-compaction response.
///
/// Prefers `usage_output_tokens`; if absent, falls back to a canonical estimate
/// of the single opaque compaction blob. `usage_total_tokens` is available on
/// the response for diagnostics but is not a seed source (it includes prompt).
pub fn server_checkpoint_token_seed(
    response: &ResponsesCompactResponse,
    prompt_envelope_tokens: u64,
    pre_compaction_tokens: u64,
) -> Result<(u64, TokenSeedSource), ServerCheckpointSeedError> {
    let (output_tokens, source) = match response.usage_output_tokens.filter(|value| *value > 0) {
        Some(value) => (value, TokenSeedSource::UsageOutputTokens),
        None => (
            canonical_compaction_item_token_estimate(&response.compaction_item)?.max(1),
            TokenSeedSource::EstimatedCanonicalOutput,
        ),
    };
    let seed = output_tokens.saturating_add(prompt_envelope_tokens).max(1);
    if pre_compaction_tokens > 0 && seed >= pre_compaction_tokens {
        return Err(ServerCheckpointSeedError::DidNotShrink);
    }
    Ok((seed, source))
}

/// Derive the identity used to bind the current live wrapper while preparing
/// a recompact. `successor_identity` points at the live checkpoint, while the
/// live wrapper still points at its own predecessor. All other freshly
/// recomputed compatibility fields remain in force.
pub fn current_identity_for_recompact_binding(
    successor_identity: &CheckpointIdentity,
    live_wrapper: &ServerResponsesCheckpoint,
) -> CheckpointIdentity {
    let mut current = successor_identity.clone();
    current.prior_checkpoint_id = live_wrapper.prior_checkpoint_id.clone();
    current
}

/// Build the one current server-compaction successor:
/// `[ResponsesCompactionCheckpoint(wrapper)] ++ tail`.
///
/// Replacement history shape (compaction_trigger contract):
/// retained typed prefix + single opaque compaction blob inside the wrapper;
/// the live conversation still stores the wrapper at index 0 with the typed
/// mode/transcript tail after it. Replay expands to
/// `[retained_prefix…, compaction_item, typed_tail…]`.
///
/// The caller supplies the digest of the portable history that will be
/// embedded in the sidecar. The sidecar constructor independently
/// recomputes it and fills `portable_history_bytes` before persistence.
#[allow(clippy::too_many_arguments)]
pub fn build_server_successor(
    checkpoint_id: &str,
    operation_id: &str,
    prompt_index: usize,
    auto_continue: bool,
    mode: ResponsesCompactionMode,
    branch_id: &str,
    mut identity: CheckpointIdentity,
    retained_prefix: Vec<ConversationItem>,
    compaction_item: serde_json::Value,
    portable_history_path: &str,
    portable_history_sha256: String,
    token_seed: u64,
    token_seed_source: TokenSeedSource,
    prior_checkpoint_id: Option<String>,
    memory_revision: Option<u64>,
    tail: Vec<ConversationItem>,
) -> Vec<ConversationItem> {
    identity.prior_checkpoint_id = prior_checkpoint_id.clone();
    let wrapper = Box::new(ServerResponsesCheckpoint {
        checkpoint_id: checkpoint_id.to_string(),
        operation_id: operation_id.to_string(),
        prompt_index,
        created_at: chrono::Utc::now(),
        auto_continue,
        mode,
        branch_id: branch_id.to_string(),
        identity,
        retained_prefix,
        compaction_item,
        portable_history_path: portable_history_path.to_string(),
        portable_history_sha256,
        portable_history_bytes: 0,
        checkpoint_token_seed: token_seed,
        token_seed_source,
        prior_checkpoint_id,
        memory_revision,
    });
    std::iter::once(ConversationItem::ResponsesCompactionCheckpoint(wrapper))
        .chain(tail)
        .collect()
}

/// Inline provider compaction is never mixed with the explicit Responses
/// compaction endpoint. When explicit remote compaction is disabled,
/// grok-build falls back to its builtin compactor instead of silently
/// enabling a second server-side mechanism through headers.
pub fn should_send_inline_compaction_headers(
    backend: &xai_grok_sampling_types::ApiBackend,
) -> bool {
    *backend != xai_grok_sampling_types::ApiBackend::Responses
}

pub fn resolve_server_compaction_layers(
    env: Option<bool>,
    config: Option<bool>,
    remote: Option<bool>,
) -> bool {
    env.or(config).or(remote).unwrap_or(true)
}

pub fn resolve_compact_model_layers(
    env: Option<&str>,
    config: Option<&str>,
    remote: Option<&str>,
    current_model: &str,
    is_known_model: impl Fn(&str) -> bool,
) -> String {
    let selected = env
        .filter(|value| !value.trim().is_empty())
        .or_else(|| config.filter(|value| !value.trim().is_empty()))
        .or_else(|| remote.filter(|value| !value.trim().is_empty()));
    match selected.map(str::trim) {
        Some(model) if is_known_model(model) => model.to_string(),
        _ => current_model.to_string(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerCompactionFailureReason {
    Unsupported,
    Auth,
    Quota,
    RateLimited,
    Timeout,
    Transport,
    Server,
    InvalidResponse,
    ContextOverflow,
    RequestTooLarge,
}

impl ServerCompactionFailureReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Auth => "auth",
            Self::Quota => "quota",
            Self::RateLimited => "rate_limited",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::Server => "server",
            Self::InvalidResponse => "invalid_response",
            Self::ContextOverflow => "context_overflow",
            Self::RequestTooLarge => "request_too_large",
        }
    }
}

pub fn classify_compact_failure(
    failure: ResponsesCompactFailure,
    status: Option<u16>,
    error_code: Option<&str>,
) -> Option<ServerCompactionFailureReason> {
    use ServerCompactionFailureReason as Reason;
    if failure == ResponsesCompactFailure::Cancelled {
        return None;
    }
    if matches!(
        error_code,
        Some("insufficient_quota" | "usage_limit_reached" | "credits_exhausted" | "spending_limit")
    ) {
        return Some(Reason::Quota);
    }
    if matches!(
        error_code,
        Some("context_length_exceeded" | "max_context_length")
    ) {
        return Some(Reason::ContextOverflow);
    }
    match failure {
        ResponsesCompactFailure::Cancelled => None,
        ResponsesCompactFailure::IdentityChanged | ResponsesCompactFailure::MissingCredential => {
            Some(Reason::Auth)
        }
        ResponsesCompactFailure::RequestTooLarge => Some(Reason::RequestTooLarge),
        ResponsesCompactFailure::Timeout => Some(Reason::Timeout),
        ResponsesCompactFailure::Transport => Some(Reason::Transport),
        ResponsesCompactFailure::ResponseTooLarge | ResponsesCompactFailure::InvalidResponse => {
            Some(Reason::InvalidResponse)
        }
        // D6: completed without exactly one compaction item is unsupported
        // (after sampler retries) and feeds the negative capability cache.
        ResponsesCompactFailure::CompletedWithoutCompaction => Some(Reason::Unsupported),
        ResponsesCompactFailure::HttpStatus => Some(match status {
            // D6: 400/404/405/422/501 are endpoint-unsupported.
            Some(400 | 404 | 405 | 422 | 501) => Reason::Unsupported,
            Some(401) => Reason::Auth,
            Some(408) => Reason::Timeout,
            Some(429) => Reason::RateLimited,
            Some(500..=599) => Reason::Server,
            _ => Reason::InvalidResponse,
        }),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityKey {
    pub endpoint_fingerprint: String,
    pub model: String,
    pub auth_principal_fingerprint: String,
    pub contract_version: String,
}

pub struct NegativeCapabilityCache {
    ttl: Duration,
    unsupported_until: HashMap<CapabilityKey, Duration>,
}

impl NegativeCapabilityCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            unsupported_until: HashMap::new(),
        }
    }

    /// HTTP statuses that, after retries, mean the endpoint does not support
    /// the compaction_trigger contract (D6). Prefer
    /// [`xai_grok_sampler::ResponsesCompactError::is_unsupported_capability`]
    /// which also covers `CompletedWithoutCompaction`.
    pub fn status_is_unsupported(status: u16) -> bool {
        matches!(status, 400 | 404 | 405 | 422 | 501)
    }

    pub fn record_unsupported(&mut self, key: CapabilityKey, now: Duration) {
        self.unsupported_until
            .insert(key, now.saturating_add(self.ttl));
    }

    pub fn is_unsupported(&mut self, key: &CapabilityKey, now: Duration) -> bool {
        self.unsupported_until.retain(|_, expires| now < *expires);
        self.unsupported_until.contains_key(key)
    }
}

fn process_cache() -> &'static Mutex<NegativeCapabilityCache> {
    static CACHE: OnceLock<Mutex<NegativeCapabilityCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(NegativeCapabilityCache::new(CAPABILITY_CACHE_TTL)))
}

fn monotonic_now() -> Duration {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed()
}

pub fn process_cache_is_unsupported(key: &CapabilityKey) -> bool {
    process_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_unsupported(key, monotonic_now())
}

pub fn process_cache_record_unsupported(key: CapabilityKey) {
    process_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .record_unsupported(key, monotonic_now());
}

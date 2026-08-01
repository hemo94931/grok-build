use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use sha2::Digest as _;
use xai_grok_sampler::{ResponsesCompactFailure, ResponsesCompactResponse};
use xai_grok_sampling_types::{
    CheckpointIdentityV1, ConversationItem, FinalResponsesRequest,
    RESPONSES_COMPACTION_CONTRACT_V1, ResponsesCompactionModeV1, ServerResponsesCheckpointV1,
    TokenSeedSource,
};

pub const CAPABILITY_CACHE_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Clone)]
pub struct ResponsesRequestSnapshot {
    pub chat_revision: u64,
    pub request_identity_generation: u64,
    pub prompt_index: usize,
    pub pre_compaction_tokens: u64,
    /// Flattened provider-visible body at snapshot time. Raw JSON: the
    /// snapshot feeds identity/token accounting and the (currently
    /// disabled) V1 writer, never a normal POST gate.
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
    pub identity: CheckpointIdentityV1,
    pub request_bytes: Vec<u8>,
    pub trigger: xai_grok_telemetry::events::CompactionTrigger,
    pub mode: ResponsesCompactionModeV1,
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
        wrapper: Box<ServerResponsesCheckpointV1>,
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

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Stage D1b kill switch: new V1 server checkpoints are disabled globally
/// and compaction falls back to builtin, so no new unsafe checkpoints are
/// produced. V2 writers (stage D4) supersede this switch; the env var
/// exists only for integration harnesses verifying the legacy path.
pub fn v1_server_compaction_writers_enabled() -> bool {
    std::env::var("GROK_RESPONSES_V1_SERVER_COMPACTION")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
}

/// Stage-D1b migration cohort from `GROK_V1_MIGRATION_PERCENT`
/// (1 → 10 → 50 → 100 during the rollout; defaults to 100).
pub fn v1_migration_cohort(session_id: &str) -> bool {
    let percent = std::env::var("GROK_V1_MIGRATION_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(100);
    v1_migration_cohort_percent(session_id, percent)
}

/// Pure cohort decision: stable session-hash bucket in `[0, 100)`.
pub fn v1_migration_cohort_percent(session_id: &str, percent: u32) -> bool {
    let percent = percent.min(100);
    if percent >= 100 {
        return true;
    }
    if percent == 0 {
        return false;
    }
    let digest = sha2::Sha256::digest(session_id.as_bytes());
    let bucket = u64::from_be_bytes(digest[..8].try_into().expect("sha256 prefix")) % 100;
    bucket < u64::from(percent)
}

/// Canonical normal-request semantics that are outside the compactable transcript.
///
/// Takes the flattened provider-visible body as raw JSON (see
/// `FinalResponsesRequest::replay_projection_body`): identity computation
/// never needs a sendable request.
pub fn canonical_prompt_envelope(
    body: &serde_json::Value,
) -> Result<(serde_json::Value, serde_json::Value), serde_json::Error> {
    let prompt_projection = serde_json::Value::Array(
        body.get("input")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter(|item| {
                matches!(
                    item.get("role").and_then(serde_json::Value::as_str),
                    Some("system" | "developer")
                )
            })
            .cloned()
            .collect(),
    );
    let mut envelope = serde_json::Map::new();
    envelope.insert(
        "canonical_prompt_projection".into(),
        prompt_projection.clone(),
    );
    for field in [
        "instructions",
        "tools",
        "tool_choice",
        "reasoning",
        "text",
        "parallel_tool_calls",
    ] {
        if let Some(value) = body.get(field).filter(|value| !value.is_null()) {
            envelope.insert(field.into(), value.clone());
        }
    }
    // Validate that the value remains canonically serializable here; callers
    // use the exact same serializer for both identity and token accounting.
    let envelope = serde_json::Value::Object(envelope);
    xai_grok_sampling_types::canonical_json_bytes(&envelope)?;
    Ok((prompt_projection, envelope))
}

pub fn build_checkpoint_identity(
    provider_id: &str,
    endpoint_fingerprint: &str,
    auth_principal_fingerprint: &str,
    final_body: &serde_json::Value,
) -> Result<CheckpointIdentityV1, serde_json::Error> {
    build_checkpoint_identity_with_projection(
        provider_id,
        endpoint_fingerprint,
        auth_principal_fingerprint,
        final_body,
        None,
    )
}

pub fn build_checkpoint_identity_with_projection(
    provider_id: &str,
    endpoint_fingerprint: &str,
    auth_principal_fingerprint: &str,
    final_body: &serde_json::Value,
    prompt_projection: Option<serde_json::Value>,
) -> Result<CheckpointIdentityV1, serde_json::Error> {
    let model = final_body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let (default_projection, mut envelope) = canonical_prompt_envelope(final_body)?;
    let prompt_projection = prompt_projection.unwrap_or(default_projection);
    envelope["canonical_prompt_projection"] = prompt_projection.clone();
    let prompt_envelope_fingerprint =
        sha256_hex(&xai_grok_sampling_types::canonical_json_bytes(&envelope)?);
    Ok(CheckpointIdentityV1 {
        provider_id: provider_id.to_string(),
        api: "responses".into(),
        endpoint_fingerprint: endpoint_fingerprint.to_string(),
        model,
        auth_principal_fingerprint: auth_principal_fingerprint.to_string(),
        contract_version: RESPONSES_COMPACTION_CONTRACT_V1.into(),
        prompt_envelope_fingerprint,
        canonical_prompt_projection: Some(prompt_projection),
    })
}

pub fn prompt_envelope_token_estimate(
    final_body: &serde_json::Value,
) -> Result<u64, serde_json::Error> {
    prompt_envelope_token_estimate_with_projection(final_body, None)
}

pub fn prompt_envelope_token_estimate_with_projection(
    final_body: &serde_json::Value,
    prompt_projection: Option<serde_json::Value>,
) -> Result<u64, serde_json::Error> {
    let (default_projection, mut envelope) = canonical_prompt_envelope(final_body)?;
    envelope["canonical_prompt_projection"] = prompt_projection.unwrap_or(default_projection);
    let bytes = xai_grok_sampling_types::canonical_json_bytes(&envelope)?.len() as u64;
    Ok(bytes.div_ceil(4))
}

#[derive(Debug, thiserror::Error)]
pub enum ServerCheckpointSeedError {
    #[error("failed to serialize canonical server output: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("server checkpoint did not reduce the conversation")]
    DidNotShrink,
}

fn canonical_output_token_estimate(output: &[serde_json::Value]) -> Result<u64, serde_json::Error> {
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

    let mut value = serde_json::Value::Array(output.to_vec());
    let images = replace_image_payloads(&mut value);
    let text_tokens =
        (xai_grok_sampling_types::canonical_json_bytes(&value)?.len() as u64).div_ceil(4);
    Ok(text_tokens.saturating_add(xai_token_estimation::estimate_image_tokens(images)))
}

pub fn server_checkpoint_token_seed(
    response: &ResponsesCompactResponse,
    prompt_envelope_tokens: u64,
    pre_compaction_tokens: u64,
) -> Result<(u64, TokenSeedSource), ServerCheckpointSeedError> {
    let (output_tokens, source) = match response.usage_output_tokens.filter(|value| *value > 0) {
        Some(value) => (value, TokenSeedSource::UsageOutputTokens),
        None => (
            canonical_output_token_estimate(&response.output)?.max(1),
            TokenSeedSource::EstimatedCanonicalOutput,
        ),
    };
    let seed = output_tokens.saturating_add(prompt_envelope_tokens).max(1);
    if pre_compaction_tokens > 0 && seed >= pre_compaction_tokens {
        return Err(ServerCheckpointSeedError::DidNotShrink);
    }
    Ok((seed, source))
}

#[allow(clippy::too_many_arguments)]
pub fn build_server_successor(
    checkpoint_id: &str,
    operation_id: &str,
    prompt_index: usize,
    auto_continue: bool,
    mode: ResponsesCompactionModeV1,
    branch_id: &str,
    identity: CheckpointIdentityV1,
    output: Vec<serde_json::Value>,
    portable_history_path: &str,
    checkpoint_token_seed: u64,
    token_seed_source: TokenSeedSource,
    tail: Vec<ConversationItem>,
) -> Vec<ConversationItem> {
    let server_output_item_count = output.len();
    let wrapper = Box::new(ServerResponsesCheckpointV1 {
        schema_version: 1,
        checkpoint_id: checkpoint_id.to_string(),
        operation_id: operation_id.to_string(),
        prompt_index,
        created_at: chrono::Utc::now(),
        auto_continue,
        mode,
        branch_id: branch_id.to_string(),
        identity,
        output,
        portable_history_path: portable_history_path.to_string(),
        portable_history_sha256: String::new(),
        portable_history_bytes: 0,
        checkpoint_token_seed,
        token_seed_source,
        server_output_item_count,
    });
    std::iter::once(ConversationItem::ResponsesCompactionCheckpoint(wrapper))
        .chain(tail)
        .collect()
}

pub fn should_send_inline_compaction_headers(
    server_compaction_enabled: bool,
    backend: &xai_grok_sampling_types::ApiBackend,
) -> bool {
    !(server_compaction_enabled && *backend == xai_grok_sampling_types::ApiBackend::Responses)
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
        ResponsesCompactFailure::HttpStatus => Some(match status {
            Some(404 | 405 | 501) => Reason::Unsupported,
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

    pub fn status_is_unsupported(status: u16) -> bool {
        matches!(status, 404 | 405 | 501)
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

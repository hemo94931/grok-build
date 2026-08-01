use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::{
    ACCEPT, AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue,
};
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::{SamplingClient, extract_retry_after, extract_should_retry};

pub const RESPONSES_COMPACT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const RESPONSES_COMPACT_MAX_BYTES: usize = 52_428_800;
pub const RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES: usize = 10_485_760;
pub use xai_grok_sampling_types::USER_CONTEXT_DELIMITER;
const RESPONSES_COMPACT_TOTAL_TIMEOUT: Duration = Duration::from_secs(120);
const RESPONSES_COMPACT_MAX_ATTEMPTS: u8 = 2;
const MIN_RETRY_BUDGET: Duration = Duration::from_secs(1);
const DEFAULT_RETRY_MAX_DELAY: Duration = Duration::from_millis(200);

#[derive(Clone, Debug, Default)]
pub struct CompactCorrelationHeaders {
    pub conversation_id: Option<String>,
    pub request_id: Option<String>,
    pub session_id: Option<String>,
    pub turn_index: Option<String>,
    pub agent_id: Option<String>,
    pub deployment_id: Option<String>,
    pub user_id: Option<String>,
}

/// Standalone `/responses/compact` request body.
///
/// Fields are private: the only way to build one is the typed
/// [`ResponsesCompactRequest::from_final`] conversion (or future canonical
/// constructors), so the compact POST point can trust that no caller
/// injected arbitrary raw JSON.
#[derive(Clone, Serialize)]
pub struct ResponsesCompactRequest {
    model: String,
    input: Vec<Value>,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<Value>,
    #[serde(skip)]
    correlation: CompactCorrelationHeaders,
}

impl std::fmt::Debug for ResponsesCompactRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesCompactRequest")
            .field("model", &self.model)
            .field("input_items", &self.input.len())
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("has_instructions", &self.instructions.is_some())
            .field("has_tools", &self.tools.is_some())
            .field("has_reasoning", &self.reasoning.is_some())
            .field("has_service_tier", &self.service_tier.is_some())
            .field("has_prompt_cache_key", &self.prompt_cache_key.is_some())
            .field("has_text", &self.text.is_some())
            .finish()
    }
}

impl ResponsesCompactRequest {
    pub fn from_final(
        final_request: &xai_grok_sampling_types::FinalResponsesRequest,
        user_context: Option<&str>,
    ) -> Result<Self, ResponsesCompactError> {
        let body = final_request.body();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?
            .to_string();
        let source_input = body
            .get("input")
            .and_then(Value::as_array)
            .filter(|input| !input.is_empty())
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        let mut input = Vec::with_capacity(source_input.len());
        let mut instruction_parts = Vec::new();
        for item in source_input {
            if item.get("role").and_then(Value::as_str) == Some("system") {
                let content = item
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse)
                    })?;
                instruction_parts.push(content.to_owned());
            } else {
                input.push(item.clone());
            }
        }
        if input.is_empty() {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::InvalidResponse,
            ));
        }
        let string_field = |name: &str| {
            body.get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        if let Some(normal) = string_field("instructions") {
            instruction_parts.push(normal);
        }
        let mut instructions =
            (!instruction_parts.is_empty()).then(|| instruction_parts.join("\n\n"));
        if let Some(context) = user_context.filter(|context| !context.is_empty()) {
            let value = instructions.get_or_insert_with(String::new);
            value.push_str(USER_CONTEXT_DELIMITER);
            value.push_str(context);
        }
        let optional_value = |name: &str| {
            body.get(name)
                .filter(|value| !value.is_null())
                .filter(|value| !value.as_array().is_some_and(Vec::is_empty))
                .cloned()
        };
        Ok(Self {
            model,
            input,
            parallel_tool_calls: body
                .get("parallel_tool_calls")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            instructions,
            tools: optional_value("tools"),
            reasoning: optional_value("reasoning"),
            service_tier: string_field("service_tier"),
            prompt_cache_key: string_field("prompt_cache_key"),
            text: optional_value("text"),
            correlation: CompactCorrelationHeaders::default(),
        })
    }

    /// Build a compact request from a sealed [`ResolvedCompactRequest`]
    /// (V2 contract: first compact via `try_normal`, continuous compact via
    /// `from_validated_recompact`). The frozen body is reused verbatim so
    /// sampler retries never re-read session state.
    pub fn from_resolved(
        resolved: &xai_grok_sampling_types::ResolvedCompactRequest,
    ) -> Result<Self, ResponsesCompactError> {
        let body = resolved.body();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?
            .to_string();
        let input = body
            .get("input")
            .and_then(Value::as_array)
            .filter(|input| !input.is_empty())
            .cloned()
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        let string_field = |name: &str| {
            body.get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        let optional_value = |name: &str| {
            body.get(name)
                .filter(|value| !value.is_null())
                .filter(|value| !value.as_array().is_some_and(Vec::is_empty))
                .cloned()
        };
        Ok(Self {
            model,
            input,
            parallel_tool_calls: body
                .get("parallel_tool_calls")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            instructions: string_field("instructions"),
            tools: optional_value("tools"),
            reasoning: optional_value("reasoning"),
            service_tier: string_field("service_tier"),
            prompt_cache_key: string_field("prompt_cache_key"),
            text: optional_value("text"),
            correlation: CompactCorrelationHeaders {
                conversation_id: resolved.correlation().x_grok_conv_id.clone(),
                request_id: resolved.correlation().x_grok_req_id.clone(),
                session_id: resolved.correlation().x_grok_session_id.clone(),
                turn_index: resolved.correlation().x_grok_turn_idx.clone(),
                agent_id: resolved.correlation().x_grok_agent_id.clone(),
                deployment_id: resolved.correlation().x_grok_deployment_id.clone(),
                user_id: resolved.correlation().x_grok_user_id.clone(),
            },
        })
    }

    pub fn with_correlation(mut self, correlation: CompactCorrelationHeaders) -> Self {
        self.correlation = correlation;
        self
    }

    /// Replace the top-level instructions. Instructions are plain text, so
    /// this cannot weaken the typed-input gate; it exists for callers (and
    /// tests) that adjust the compaction directive.
    pub fn with_instructions(mut self, instructions: Option<String>) -> Self {
        self.instructions = instructions;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn to_bounded_bytes(&self) -> Result<Vec<u8>, ResponsesCompactError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|_| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        if bytes.len() > RESPONSES_COMPACT_MAX_BYTES {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::RequestTooLarge,
            ));
        }
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactAuthKind {
    Bearer,
    XApiKey,
}

#[derive(Clone)]
pub struct CompactCredential {
    kind: CompactAuthKind,
    secret: String,
    principal_fingerprint: String,
}

impl CompactCredential {
    pub fn bearer(secret: impl Into<String>, principal_fingerprint: impl Into<String>) -> Self {
        Self {
            kind: CompactAuthKind::Bearer,
            secret: secret.into(),
            principal_fingerprint: principal_fingerprint.into(),
        }
    }

    pub fn x_api_key(secret: impl Into<String>, principal_fingerprint: impl Into<String>) -> Self {
        Self {
            kind: CompactAuthKind::XApiKey,
            secret: secret.into(),
            principal_fingerprint: principal_fingerprint.into(),
        }
    }

    pub fn principal_fingerprint(&self) -> &str {
        &self.principal_fingerprint
    }
}

impl std::fmt::Debug for CompactCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactCredential")
            .field("kind", &self.kind)
            .field("has_secret", &!self.secret.is_empty())
            .field("principal_fingerprint", &"[REDACTED]")
            .finish()
    }
}

pub trait CompactCredentialResolver: Send + Sync + std::fmt::Debug {
    fn current_credential(&self) -> Option<CompactCredential>;
}

#[derive(Clone)]
pub struct RequestCredentialSnapshot {
    initial: CompactCredential,
    resolver: Option<Arc<dyn CompactCredentialResolver>>,
}

impl RequestCredentialSnapshot {
    pub fn bearer(secret: impl Into<String>, principal_fingerprint: impl Into<String>) -> Self {
        Self {
            initial: CompactCredential::bearer(secret, principal_fingerprint),
            resolver: None,
        }
    }

    pub fn x_api_key(secret: impl Into<String>, principal_fingerprint: impl Into<String>) -> Self {
        Self {
            initial: CompactCredential::x_api_key(secret, principal_fingerprint),
            resolver: None,
        }
    }

    pub fn with_resolver(mut self, resolver: Arc<dyn CompactCredentialResolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    pub fn principal_fingerprint(&self) -> &str {
        self.initial.principal_fingerprint()
    }

    fn current(&self) -> Result<CompactCredential, ResponsesCompactError> {
        let credential = match &self.resolver {
            Some(resolver) => resolver.current_credential().ok_or_else(|| {
                ResponsesCompactError::new(ResponsesCompactFailure::MissingCredential)
            })?,
            None => self.initial.clone(),
        };
        if credential.principal_fingerprint != self.initial.principal_fingerprint
            || credential.kind != self.initial.kind
        {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::IdentityChanged,
            ));
        }
        if credential.secret.is_empty() {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::MissingCredential,
            ));
        }
        Ok(credential)
    }
}

impl std::fmt::Debug for RequestCredentialSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestCredentialSnapshot")
            .field("kind", &self.initial.kind)
            .field("has_secret", &!self.initial.secret.is_empty())
            .field("has_resolver", &self.resolver.is_some())
            .field("principal_fingerprint", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesCompactFailure {
    Cancelled,
    IdentityChanged,
    MissingCredential,
    RequestTooLarge,
    ResponseTooLarge,
    Timeout,
    Transport,
    HttpStatus,
    InvalidResponse,
}

#[derive(Debug, Clone)]
pub struct ResponsesCompactError {
    failure: ResponsesCompactFailure,
    status: Option<u16>,
    error_code: Option<String>,
    attempts: u8,
}

impl ResponsesCompactError {
    fn new(failure: ResponsesCompactFailure) -> Self {
        Self {
            failure,
            status: None,
            error_code: None,
            attempts: 0,
        }
    }

    fn http(status: reqwest::StatusCode, error_code: Option<String>, attempts: u8) -> Self {
        Self {
            failure: ResponsesCompactFailure::HttpStatus,
            status: Some(status.as_u16()),
            error_code,
            attempts,
        }
    }

    fn with_attempts(mut self, attempts: u8) -> Self {
        self.attempts = attempts;
        self
    }

    pub fn failure(&self) -> ResponsesCompactFailure {
        self.failure
    }

    pub fn status(&self) -> Option<u16> {
        self.status
    }

    pub fn error_code(&self) -> Option<&str> {
        self.error_code.as_deref()
    }

    pub fn attempts(&self) -> u8 {
        self.attempts
    }
}

impl std::fmt::Display for ResponsesCompactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "responses compact failed: {:?}", self.failure)
    }
}

impl std::error::Error for ResponsesCompactError {}

#[derive(Debug, Clone)]
pub struct ResponsesCompactResponse {
    pub output: Vec<Value>,
    pub usage_output_tokens: Option<u64>,
    pub usage_total_tokens: Option<u64>,
    pub response_bytes: usize,
    pub attempts: u8,
}

pub fn validate_responses_compact_response(
    value: Value,
) -> Result<ResponsesCompactResponse, ResponsesCompactError> {
    let object = value
        .as_object()
        .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
    let output = object
        .get("output")
        .and_then(Value::as_array)
        .filter(|output| !output.is_empty())
        .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
    let mut has_compaction = false;
    for item in output {
        let item = item
            .as_object()
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        let item_type = item
            .get("type")
            .and_then(Value::as_str)
            .filter(|item_type| !item_type.is_empty())
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        if matches!(
            item_type,
            "compaction_trigger" | "summary" | "context" | "context_summary"
        ) {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::InvalidResponse,
            ));
        }
        if matches!(item_type, "compaction" | "compaction_summary") {
            let encrypted = item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse)
                })?;
            if encrypted.is_empty() || encrypted.len() > RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES {
                return Err(ResponsesCompactError::new(
                    ResponsesCompactFailure::InvalidResponse,
                ));
            }
            has_compaction = true;
        }
    }
    if !has_compaction {
        return Err(ResponsesCompactError::new(
            ResponsesCompactFailure::InvalidResponse,
        ));
    }
    let usage = object.get("usage").and_then(Value::as_object);
    let parse_usage = |field: &str| -> Result<Option<u64>, ResponsesCompactError> {
        match usage.and_then(|usage| usage.get(field)) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value.as_u64().map(Some).ok_or_else(|| {
                ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse)
            }),
        }
    };
    let response_bytes = serde_json::to_vec(&value)
        .map_err(|_| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?
        .len();
    Ok(ResponsesCompactResponse {
        output: output.clone(),
        usage_output_tokens: parse_usage("output_tokens")?,
        usage_total_tokens: parse_usage("total_tokens")?,
        response_bytes,
        attempts: 0,
    })
}

impl SamplingClient {
    pub async fn compact_responses(
        &self,
        request: &ResponsesCompactRequest,
        credential: &RequestCredentialSnapshot,
        cancellation: &CancellationToken,
    ) -> Result<ResponsesCompactResponse, ResponsesCompactError> {
        if cancellation.is_cancelled() {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::Cancelled,
            ));
        }
        let body = request.to_bounded_bytes()?;
        if cancellation.is_cancelled() {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::Cancelled,
            ));
        }

        let deadline = Instant::now() + RESPONSES_COMPACT_TOTAL_TIMEOUT;
        let mut attempts = 0;
        loop {
            attempts += 1;
            let credential = credential
                .current()
                .map_err(|error| error.with_attempts(attempts))?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(ResponsesCompactError::new(ResponsesCompactFailure::Timeout)
                    .with_attempts(attempts));
            }

            let headers = self.compact_headers(&credential, &request.correlation)?;
            let send = self
                .compact_http
                .post(self.endpoint("responses/compact"))
                .headers(headers)
                .timeout(remaining)
                .body(body.clone())
                .send();
            let response = tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    return Err(ResponsesCompactError::new(ResponsesCompactFailure::Cancelled)
                        .with_attempts(attempts));
                }
                result = tokio::time::timeout_at(deadline, send) => match result {
                    Ok(Ok(response)) => response,
                    Ok(Err(_)) => {
                        if self.can_retry(attempts, deadline, cancellation).await? {
                            continue;
                        }
                        return Err(ResponsesCompactError::new(ResponsesCompactFailure::Transport)
                            .with_attempts(attempts));
                    }
                    Err(_) => {
                        return Err(ResponsesCompactError::new(ResponsesCompactFailure::Timeout)
                            .with_attempts(attempts));
                    }
                }
            };

            let status = response.status();
            let retry_after = extract_retry_after(response.headers());
            let should_retry = extract_should_retry(response.headers());
            let bytes = read_bounded_response(response, deadline, cancellation)
                .await
                .map_err(|error| error.with_attempts(attempts))?;
            if !status.is_success() {
                let error_code = structured_error_code(&bytes);
                let retryable = (status == reqwest::StatusCode::REQUEST_TIMEOUT
                    || status.is_server_error())
                    && should_retry != Some(false);
                if retryable
                    && attempts < RESPONSES_COMPACT_MAX_ATTEMPTS
                    && deadline.saturating_duration_since(Instant::now()) >= MIN_RETRY_BUDGET
                {
                    let delay = retry_after
                        .map(Duration::from_secs)
                        .unwrap_or_else(full_jitter_delay);
                    if wait_for_retry(delay, deadline, cancellation).await? {
                        continue;
                    }
                }
                return Err(ResponsesCompactError::http(status, error_code, attempts));
            }
            let value: Value = serde_json::from_slice(&bytes).map_err(|_| {
                ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse)
                    .with_attempts(attempts)
            })?;
            let mut parsed = validate_responses_compact_response(value)
                .map_err(|error| error.with_attempts(attempts))?;
            parsed.response_bytes = bytes.len();
            parsed.attempts = attempts;
            return Ok(parsed);
        }
    }

    async fn can_retry(
        &self,
        attempts: u8,
        deadline: Instant,
        cancellation: &CancellationToken,
    ) -> Result<bool, ResponsesCompactError> {
        if attempts >= RESPONSES_COMPACT_MAX_ATTEMPTS
            || deadline.saturating_duration_since(Instant::now()) < MIN_RETRY_BUDGET
        {
            return Ok(false);
        }
        wait_for_retry(full_jitter_delay(), deadline, cancellation).await
    }

    fn compact_headers(
        &self,
        credential: &CompactCredential,
        correlation: &CompactCorrelationHeaders,
    ) -> Result<HeaderMap, ResponsesCompactError> {
        let mut headers = self.default_headers.clone();
        if let Some(injector) = &self.header_injector {
            injector.inject(&mut headers);
        }
        strip_compact_denied_headers(&mut headers);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        match credential.kind {
            CompactAuthKind::Bearer => {
                let value = HeaderValue::from_str(&format!("Bearer {}", credential.secret))
                    .map_err(|_| {
                        ResponsesCompactError::new(ResponsesCompactFailure::MissingCredential)
                    })?;
                headers.insert(AUTHORIZATION, value);
            }
            CompactAuthKind::XApiKey => {
                let value = HeaderValue::from_str(&credential.secret).map_err(|_| {
                    ResponsesCompactError::new(ResponsesCompactFailure::MissingCredential)
                })?;
                headers.insert(HeaderName::from_static("x-api-key"), value);
            }
        }
        apply_correlation_headers(&mut headers, correlation)?;
        Ok(headers)
    }
}

fn strip_compact_denied_headers(headers: &mut HeaderMap) {
    for name in [
        "authorization",
        "x-api-key",
        "x-compaction-at",
        "x-compactions-remaining",
        "x-codex-turn-state",
        "x-codex-attestation",
        "x-codex-turn-metadata",
        "x-grok-doom-loop-check",
        "last-event-id",
    ] {
        headers.remove(name);
    }
    headers.remove(CONTENT_LENGTH);
}

fn apply_correlation_headers(
    headers: &mut HeaderMap,
    correlation: &CompactCorrelationHeaders,
) -> Result<(), ResponsesCompactError> {
    let fields = [
        ("x-grok-conv-id", correlation.conversation_id.as_deref()),
        ("x-grok-req-id", correlation.request_id.as_deref()),
        ("x-grok-session-id", correlation.session_id.as_deref()),
        ("x-grok-turn-idx", correlation.turn_index.as_deref()),
        ("x-grok-agent-id", correlation.agent_id.as_deref()),
        ("x-grok-deployment-id", correlation.deployment_id.as_deref()),
        ("x-grok-user-id", correlation.user_id.as_deref()),
    ];
    for (name, value) in fields {
        if let Some(value) = value.filter(|value| !value.is_empty()) {
            let value = HeaderValue::from_str(value).map_err(|_| {
                ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse)
            })?;
            headers.insert(HeaderName::from_static(name), value);
        }
    }
    Ok(())
}

async fn read_bounded_response(
    response: reqwest::Response,
    deadline: Instant,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, ResponsesCompactError> {
    if response
        .content_length()
        .is_some_and(|length| length > RESPONSES_COMPACT_MAX_BYTES as u64)
    {
        return Err(ResponsesCompactError::new(
            ResponsesCompactFailure::ResponseTooLarge,
        ));
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    loop {
        let chunk = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                return Err(ResponsesCompactError::new(ResponsesCompactFailure::Cancelled));
            }
            result = tokio::time::timeout_at(deadline, stream.next()) => match result {
                Ok(result) => result,
                Err(_) => return Err(ResponsesCompactError::new(ResponsesCompactFailure::Timeout)),
            }
        };
        let Some(chunk) = chunk else {
            break;
        };
        let chunk =
            chunk.map_err(|_| ResponsesCompactError::new(ResponsesCompactFailure::Transport))?;
        if bytes.len().saturating_add(chunk.len()) > RESPONSES_COMPACT_MAX_BYTES {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::ResponseTooLarge,
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn structured_error_code(bytes: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(bytes).ok()?;
    let code = value.pointer("/error/code")?.as_str()?;
    (code.len() <= 64
        && code
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')))
    .then(|| code.to_string())
}

fn full_jitter_delay() -> Duration {
    let sample = u64::from(uuid::Uuid::new_v4().as_bytes()[0]);
    let max_ms = DEFAULT_RETRY_MAX_DELAY.as_millis() as u64;
    Duration::from_millis(sample * max_ms / u8::MAX as u64)
}

async fn wait_for_retry(
    delay: Duration,
    deadline: Instant,
    cancellation: &CancellationToken,
) -> Result<bool, ResponsesCompactError> {
    if delay >= deadline.saturating_duration_since(Instant::now()) {
        return Ok(false);
    }
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            Err(ResponsesCompactError::new(ResponsesCompactFailure::Cancelled))
        }
        _ = tokio::time::sleep(delay) => Ok(true),
    }
}

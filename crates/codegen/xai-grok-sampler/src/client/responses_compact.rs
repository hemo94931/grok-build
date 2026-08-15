//! Streaming Responses compaction transport (compaction_trigger contract).
//!
//! Remote compaction is a normal streaming `POST /responses` request whose
//! frozen body ends with `{"type":"compaction_trigger"}`. Collection requires
//! `response.completed` with exactly one `compaction`/`compaction_summary`
//! item carrying non-empty `encrypted_content`.

use std::sync::Arc;
use std::time::Duration;

use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::SamplingClient;

/// Request body size cap shared with the prior unary path (50 MiB + headroom).
pub const RESPONSES_COMPACT_MAX_BYTES: usize = 52_428_800;
/// Cap on a single compaction item's `encrypted_content` (10 MiB).
pub const RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES: usize = 10_485_760;
pub use xai_grok_sampling_types::USER_CONTEXT_DELIMITER;

const RESPONSES_COMPACT_TOTAL_TIMEOUT: Duration = Duration::from_secs(120);
/// Aligns with upstream `MAX_REMOTE_COMPACTION_V2_STREAM_RETRIES` (2 attempts).
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

/// Sealed streaming compaction request.
///
/// Fields are private: the only construction path consumes a frozen
/// [`xai_grok_sampling_types::ResolvedCompactRequest`], so the transport
/// cannot accept caller-supplied raw checkpoint JSON.
#[derive(Clone)]
pub struct ResponsesCompactRequest {
    /// Full streaming Responses body (includes trailing compaction_trigger).
    body: Value,
    model: String,
    correlation: CompactCorrelationHeaders,
}

impl std::fmt::Debug for ResponsesCompactRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesCompactRequest")
            .field("model", &self.model)
            .field(
                "input_items",
                &self
                    .body
                    .get("input")
                    .and_then(Value::as_array)
                    .map(Vec::len)
                    .unwrap_or(0),
            )
            .field("has_instructions", &self.body.get("instructions").is_some())
            .field("stream", &self.body.get("stream"))
            .finish()
    }
}

impl ResponsesCompactRequest {
    /// Build a compact request from a sealed [`ResolvedCompactRequest`]
    /// (first compact via `try_normal`, continuous compact via
    /// `from_validated_recompact`). The frozen body is reused verbatim so
    /// sampler retries never re-read session state.
    pub fn from_resolved(
        resolved: &xai_grok_sampling_types::ResolvedCompactRequest,
    ) -> Result<Self, ResponsesCompactError> {
        let body = resolved.body().clone();
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
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        // Sealed constructor must leave the trigger last.
        let last_is_trigger = input
            .last()
            .and_then(|item| item.get("type"))
            .and_then(Value::as_str)
            == Some("compaction_trigger");
        if !last_is_trigger {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::InvalidResponse,
            ));
        }
        if body.get("stream").and_then(Value::as_bool) != Some(true) {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::InvalidResponse,
            ));
        }
        Ok(Self {
            body,
            model,
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
        match instructions {
            Some(value) => {
                self.body["instructions"] = Value::String(value);
            }
            None => {
                if let Some(object) = self.body.as_object_mut() {
                    object.remove("instructions");
                }
            }
        }
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn body(&self) -> &Value {
        &self.body
    }

    pub fn to_bounded_bytes(&self) -> Result<Vec<u8>, ResponsesCompactError> {
        let bytes = serde_json::to_vec(&self.body)
            .map_err(|_| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        if bytes.len() > RESPONSES_COMPACT_MAX_BYTES {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::RequestTooLarge,
            ));
        }
        Ok(bytes)
    }
}

/// Hex SHA-256 of the compact-only user-context suffix, for recording as
/// `compact_directive_hash`. `None` when no suffix is present (absent or empty).
pub fn compact_directive_hash(compact_user_context: Option<&str>) -> Option<String> {
    let suffix = compact_user_context.filter(|context| !context.is_empty())?;
    xai_grok_sampling_types::canonical_value_digest(&Value::String(suffix.to_owned())).ok()
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
    /// Malformed SSE / JSON / usage that is not a completed-without-item case.
    InvalidResponse,
    /// Stream reached `response.completed` without exactly one valid
    /// compaction item. After the retry budget is exhausted, shell classifies
    /// this as unsupported (D6 negative capability cache).
    CompletedWithoutCompaction,
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

    /// Whether shell should treat this (after retries) as endpoint-unsupported
    /// for the negative capability cache.
    pub fn is_unsupported_capability(&self) -> bool {
        match self.failure {
            ResponsesCompactFailure::CompletedWithoutCompaction => true,
            ResponsesCompactFailure::HttpStatus => {
                matches!(self.status, Some(400 | 404 | 405 | 422 | 501))
            }
            _ => false,
        }
    }
}

impl std::fmt::Display for ResponsesCompactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "responses compact failed: {:?}", self.failure)
    }
}

impl std::error::Error for ResponsesCompactError {}

/// Successful remote-compaction collection result.
#[derive(Debug, Clone)]
pub struct ResponsesCompactResponse {
    /// Exactly one compaction item (`compaction` or `compaction_summary`)
    /// with non-empty `encrypted_content`.
    pub compaction_item: Value,
    pub usage_output_tokens: Option<u64>,
    pub usage_total_tokens: Option<u64>,
    pub response_bytes: usize,
    pub attempts: u8,
}

/// Dedup key for a compaction item seen across `response.output_item.done`
/// and `response.completed.response.output` frames: prefer the item id, fall
/// back to the encrypted payload itself.
fn compaction_dedup_key(item: &Value) -> Option<String> {
    item.get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(|id| format!("id:{id}"))
        .or_else(|| {
            item.get("encrypted_content")
                .and_then(Value::as_str)
                .map(|content| format!("enc:{content}"))
        })
}

/// Accumulates compaction items across streamed frames. The live ChatGPT
/// Codex backend delivers all output items via `response.output_item.done`
/// events and leaves `response.completed.response.output` EMPTY (see
/// `codex-backend-quirks.md`), so collecting only from the terminal frame
/// would never find the compaction item there. Both sources are merged with
/// dedup; the terminal frame must yield exactly one distinct compaction item.
#[derive(Default)]
struct CompactionCollector {
    items: Vec<Value>,
    keys: Vec<Option<String>>,
}

impl CompactionCollector {
    /// Record one candidate item. Invalid compaction payloads (empty or
    /// oversized `encrypted_content`) fail immediately, matching the sealed
    /// transport's validation semantics.
    fn note(&mut self, item: &Value) -> Result<(), ResponsesCompactError> {
        let Some(object) = item.as_object() else {
            return Ok(());
        };
        let Some(item_type) = object.get("type").and_then(Value::as_str) else {
            return Ok(());
        };
        if !matches!(item_type, "compaction" | "compaction_summary") {
            return Ok(());
        }
        let encrypted = object
            .get("encrypted_content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if encrypted.is_empty() || encrypted.len() > RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::InvalidResponse,
            ));
        }
        let key = compaction_dedup_key(item);
        if key.is_some() && self.keys.contains(&key) {
            return Ok(());
        }
        self.items.push(item.clone());
        self.keys.push(key);
        Ok(())
    }

    /// Merge any compaction items embedded in the terminal frame's
    /// `response.output` (some backends populate it) with the streamed ones.
    fn note_completed_output(&mut self, completed: &Value) -> Result<(), ResponsesCompactError> {
        if let Some(output) = completed
            .get("response")
            .or(Some(completed))
            .and_then(|response| response.get("output"))
            .and_then(Value::as_array)
        {
            for item in output {
                self.note(item)?;
            }
        }
        Ok(())
    }

    /// Terminal validation at `response.completed`: exactly one distinct
    /// compaction item; usage read from the completed frame.
    fn finish(
        mut self,
        completed: &Value,
    ) -> Result<(Value, Option<u64>, Option<u64>), ResponsesCompactError> {
        self.note_completed_output(completed)?;
        if self.items.len() != 1 {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::CompletedWithoutCompaction,
            ));
        }
        let compaction_item = self.items.pop().ok_or_else(|| {
            ResponsesCompactError::new(ResponsesCompactFailure::CompletedWithoutCompaction)
        })?;

        let response = completed
            .get("response")
            .or(Some(completed))
            .and_then(Value::as_object)
            .ok_or_else(|| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        let usage = response.get("usage").and_then(Value::as_object);
        let parse_usage = |field: &str| -> Result<Option<u64>, ResponsesCompactError> {
            match usage.and_then(|usage| usage.get(field)) {
                None | Some(Value::Null) => Ok(None),
                Some(value) => value.as_u64().map(Some).ok_or_else(|| {
                    ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse)
                }),
            }
        };

        Ok((
            compaction_item,
            parse_usage("output_tokens")?,
            parse_usage("total_tokens")?,
        ))
    }
}

/// Validate a completed response payload on its own: exactly one compaction
/// item with non-empty encrypted_content (≤ 10 MiB) in `response.output`.
/// Unit-test helper; the streaming transport uses [`CompactionCollector`]
/// across frames instead, because the live backend leaves
/// `response.completed.response.output` empty.
pub fn collect_compaction_from_completed(
    completed: &Value,
) -> Result<(Value, Option<u64>, Option<u64>), ResponsesCompactError> {
    CompactionCollector::default().finish(completed)
}

impl SamplingClient {
    /// Run remote compaction as a streaming Responses request and collect
    /// exactly one compaction item from `response.completed`.
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
        let body = self.compact_body_bytes(request)?;
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
                .http
                .post(self.endpoint("responses"))
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
            let retry_after = super::extract_retry_after(response.headers());
            let should_retry = super::extract_should_retry(response.headers());
            if !status.is_success() {
                let bytes = read_bounded_response(response, deadline, cancellation)
                    .await
                    .map_err(|error| error.with_attempts(attempts))?;
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

            match collect_sse_compaction(response, deadline, cancellation).await {
                Ok(mut parsed) => {
                    parsed.attempts = attempts;
                    return Ok(parsed);
                }
                Err(error) => {
                    let error = error.with_attempts(attempts);
                    // Retry stream/transport failures and completed-without-item
                    // within the attempt budget; surface the final classification.
                    let retryable = matches!(
                        error.failure(),
                        ResponsesCompactFailure::Transport
                            | ResponsesCompactFailure::Timeout
                            | ResponsesCompactFailure::CompletedWithoutCompaction
                            | ResponsesCompactFailure::InvalidResponse
                    );
                    if retryable
                        && attempts < RESPONSES_COMPACT_MAX_ATTEMPTS
                        && deadline.saturating_duration_since(Instant::now()) >= MIN_RETRY_BUDGET
                        && wait_for_retry(full_jitter_delay(), deadline, cancellation).await?
                    {
                        continue;
                    }
                    return Err(error);
                }
            }
        }
    }

    /// Serialize the sealed body, applying ordinary provider wire sanitization
    /// (model rewrite etc.) used by turn requests.
    fn compact_body_bytes(
        &self,
        request: &ResponsesCompactRequest,
    ) -> Result<Vec<u8>, ResponsesCompactError> {
        let mut value = request.body.clone();
        if let Some(route) = &self.provider_wire {
            route.sanitize_body(&mut value, &self.defaults.api_backend);
        }
        // Compaction is always streaming.
        if let Some(object) = value.as_object_mut() {
            object.insert("stream".into(), Value::Bool(true));
            if object
                .get("store")
                .map(|v| v.is_null() || v.as_bool() == Some(true))
                .unwrap_or(true)
            {
                object.insert("store".into(), Value::Bool(false));
            }
        }
        let bytes = serde_json::to_vec(&value)
            .map_err(|_| ResponsesCompactError::new(ResponsesCompactFailure::InvalidResponse))?;
        if bytes.len() > RESPONSES_COMPACT_MAX_BYTES {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::RequestTooLarge,
            ));
        }
        Ok(bytes)
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
        // Inherit the same headers as turn requests (default_headers +
        // header_injector), then apply the same per-provider allowlist the
        // turn pipeline applies (`post_with_headers`), so compaction never
        // leaks headers a turn would strip (e.g. x-compactions-remaining).
        // Correlation headers are applied after sanitization, matching the
        // prior compact path which always emitted them.
        let mut headers = self.default_headers.clone();
        if let Some(injector) = &self.header_injector {
            injector.inject(&mut headers);
        }
        if let Some(route) = &self.provider_wire {
            route.sanitize_headers(&mut headers, self.defaults.auth_scheme);
        }
        // Drop auth headers before re-applying the snapshot credential so a
        // stale injected Authorization never wins over the live principal.
        headers.remove(AUTHORIZATION);
        headers.remove(HeaderName::from_static("x-api-key"));
        headers.remove(HeaderName::from_static("api-key"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
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

async fn collect_sse_compaction(
    response: reqwest::Response,
    deadline: Instant,
    cancellation: &CancellationToken,
) -> Result<ResponsesCompactResponse, ResponsesCompactError> {
    if response
        .content_length()
        .is_some_and(|length| length > RESPONSES_COMPACT_MAX_BYTES as u64)
    {
        return Err(ResponsesCompactError::new(
            ResponsesCompactFailure::ResponseTooLarge,
        ));
    }

    let mut response_bytes = 0usize;
    let mut collector = CompactionCollector::default();
    let mut stream = response.bytes_stream().eventsource();
    loop {
        let event = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                return Err(ResponsesCompactError::new(ResponsesCompactFailure::Cancelled));
            }
            result = tokio::time::timeout_at(deadline, stream.next()) => match result {
                Ok(Some(Ok(event))) => event,
                Ok(Some(Err(_))) => {
                    return Err(ResponsesCompactError::new(ResponsesCompactFailure::Transport));
                }
                Ok(None) => {
                    return Err(ResponsesCompactError::new(
                        ResponsesCompactFailure::CompletedWithoutCompaction,
                    ));
                }
                Err(_) => {
                    return Err(ResponsesCompactError::new(ResponsesCompactFailure::Timeout));
                }
            }
        };

        let data = event.data;
        if data == "[DONE]" {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::CompletedWithoutCompaction,
            ));
        }
        response_bytes = response_bytes.saturating_add(data.len());
        if response_bytes > RESPONSES_COMPACT_MAX_BYTES {
            return Err(ResponsesCompactError::new(
                ResponsesCompactFailure::ResponseTooLarge,
            ));
        }

        let value: Value = match serde_json::from_str(&data) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let event_type = value.get("type").and_then(Value::as_str).or_else(|| {
            let name = event.event.as_str();
            if name.is_empty() || name == "message" {
                None
            } else {
                Some(name)
            }
        });

        match event_type {
            // The live backend delivers items (including the compaction
            // blob) exclusively via output_item.done frames.
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item") {
                    collector.note(item)?;
                }
                continue;
            }
            Some("response.completed") => {}
            _ => continue,
        }

        let (compaction_item, usage_output_tokens, usage_total_tokens) =
            collector.finish(&value)?;
        return Ok(ResponsesCompactResponse {
            compaction_item,
            usage_output_tokens,
            usage_total_tokens,
            response_bytes,
            attempts: 0,
        });
    }
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

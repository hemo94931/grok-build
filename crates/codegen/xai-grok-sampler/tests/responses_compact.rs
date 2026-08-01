use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, Method, Uri},
    response::IntoResponse,
    routing::post,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use tokio_util::sync::CancellationToken;
use tracing::instrument::WithSubscriber as _;
use xai_grok_sampler::{
    ApiBackend, CompactCorrelationHeaders, CompactCredential, CompactCredentialResolver,
    RESPONSES_COMPACT_CONNECT_TIMEOUT, RESPONSES_COMPACT_MAX_BYTES,
    RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES, RequestCredentialSnapshot, ResponsesCompactFailure,
    ResponsesCompactRequest, SamplerConfig, SamplingClient, USER_CONTEXT_DELIMITER,
    compact_directive_hash, validate_responses_compact_response,
};
use xai_grok_sampling_types::{
    CanonicalResponsesContext, INSTRUCTIONS_MEMORY_SEPARATOR,
};

#[derive(Clone, Debug)]
struct Captured {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Value,
}

type CredentialHandlerState = (Arc<AtomicUsize>, Arc<Mutex<Vec<String>>>);

#[derive(Clone, Default)]
struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

struct CapturedLogWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for CapturedLogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
    type Writer = CapturedLogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        CapturedLogWriter(Arc::clone(&self.0))
    }
}

fn canonical_context() -> CanonicalResponsesContext {
    CanonicalResponsesContext {
        model: "grok-test".into(),
        base_instructions: "normal instructions".into(),
        memory_context: None,
        tools: json!([]),
        tool_choice: None,
        reasoning: Some(json!({ "effort": "medium" })),
        text: None,
        parallel_tool_calls: true,
        prompt_cache_key: Some("cache-key".into()),
        prompt_cache_options: None,
        prompt_cache_retention: None,
        service_tier: Some("priority".into()),
    }
}

fn canonical_input() -> Vec<Value> {
    vec![json!({ "role": "user", "content": "hello" })]
}

fn compact_request() -> ResponsesCompactRequest {
    ResponsesCompactRequest::from_canonical(&canonical_context(), canonical_input(), None).unwrap()
}

fn valid_response() -> Value {
    json!({
        "id": "cmp_1",
        "object": "response.compaction",
        "created_at": 123,
        "output": [
            {"type": "future_provider_item", "nested": {"z": 1}},
            {"type": "compaction_summary", "encrypted_content": "opaque"}
        ],
        "usage": {"output_tokens": 17, "total_tokens": 99},
        "ignored_top_level": "discard"
    })
}

#[test]
fn from_canonical_sends_full_allowlist_and_never_tool_choice() {
    let context = CanonicalResponsesContext {
        model: "grok-test".into(),
        base_instructions: "base instructions".into(),
        memory_context: Some("memory".into()),
        tools: json!([{ "type": "function", "name": "read_file" }]),
        tool_choice: Some(json!({ "type": "auto" })),
        reasoning: Some(json!({ "effort": "high" })),
        text: Some(json!({ "format": { "type": "text" } })),
        parallel_tool_calls: false,
        prompt_cache_key: Some("cache-key".into()),
        prompt_cache_options: Some(json!({ "scope": "session" })),
        prompt_cache_retention: Some("24h".into()),
        service_tier: Some("priority".into()),
    };
    let body = serde_json::to_value(
        ResponsesCompactRequest::from_canonical(&context, canonical_input(), Some("directive"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        body["instructions"],
        format!(
            "base instructions{INSTRUCTIONS_MEMORY_SEPARATOR}memory{USER_CONTEXT_DELIMITER}directive"
        )
    );
    assert_eq!(body["model"], "grok-test");
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["tools"], context.tools);
    assert_eq!(body["reasoning"], json!({ "effort": "high" }));
    assert_eq!(body["text"], json!({ "format": { "type": "text" } }));
    assert_eq!(body["prompt_cache_key"], "cache-key");
    assert_eq!(body["prompt_cache_options"], json!({ "scope": "session" }));
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["service_tier"], "priority");
    assert_eq!(body["input"], json!(canonical_input()));
    assert!(
        body.get("tool_choice").is_none(),
        "tool_choice is create-only and must never reach the compact endpoint"
    );
}

#[test]
fn from_canonical_omits_fields_the_compact_endpoint_does_not_support() {
    let context = CanonicalResponsesContext {
        model: "grok-test".into(),
        base_instructions: String::new(),
        memory_context: None,
        tools: json!([]),
        tool_choice: Some(json!({ "type": "auto" })),
        reasoning: None,
        text: None,
        parallel_tool_calls: true,
        prompt_cache_key: None,
        prompt_cache_options: None,
        prompt_cache_retention: None,
        service_tier: None,
    };
    let body = serde_json::to_value(
        ResponsesCompactRequest::from_canonical(&context, canonical_input(), None).unwrap(),
    )
    .unwrap();
    assert_eq!(
        body.as_object().unwrap().keys().cloned().collect::<Vec<_>>(),
        vec!["model", "input", "parallel_tool_calls"]
    );
    for absent in [
        "instructions",
        "tools",
        "reasoning",
        "text",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "service_tier",
        "tool_choice",
    ] {
        assert!(body.get(absent).is_none(), "unexpected field {absent}");
    }
}

#[test]
fn from_canonical_preserves_explicit_false_parallel_tool_calls() {
    let context = CanonicalResponsesContext {
        parallel_tool_calls: false,
        ..canonical_context()
    };
    let body = serde_json::to_value(
        ResponsesCompactRequest::from_canonical(&context, canonical_input(), None).unwrap(),
    )
    .unwrap();
    assert_eq!(body["parallel_tool_calls"], false);
    assert!(body.get("parallel_tool_calls").is_some());
}

#[test]
fn from_canonical_rejects_empty_input_and_system_items() {
    let context = canonical_context();
    let empty = ResponsesCompactRequest::from_canonical(&context, Vec::new(), None);
    assert_eq!(
        empty.unwrap_err().failure(),
        ResponsesCompactFailure::InvalidResponse
    );
    let with_system = vec![
        json!({ "role": "system", "content": "system prompt" }),
        json!({ "role": "user", "content": "hello" }),
    ];
    let rejected =
        ResponsesCompactRequest::from_canonical(&context, with_system, None);
    assert_eq!(
        rejected.unwrap_err().failure(),
        ResponsesCompactFailure::InvalidResponse
    );
    // The user item alone (no system) is accepted.
    assert!(ResponsesCompactRequest::from_canonical(
        &context,
        canonical_input(),
        None
    )
    .is_ok());
}

#[test]
fn from_canonical_appends_compact_only_user_context_suffix() {
    let context = canonical_context();
    let body = serde_json::to_value(
        ResponsesCompactRequest::from_canonical(&context, canonical_input(), Some("preserve this"))
            .unwrap(),
    )
    .unwrap();
    let instructions = body["instructions"].as_str().unwrap();
    assert!(instructions.contains(USER_CONTEXT_DELIMITER));
    assert!(instructions.ends_with("preserve this"));
    // An empty suffix is treated as absent: no delimiter, plain canonical instructions.
    let body = serde_json::to_value(
        ResponsesCompactRequest::from_canonical(&context, canonical_input(), Some(""))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["instructions"], "normal instructions");
}

#[test]
fn compact_directive_hash_is_stable_and_absent_without_context() {
    assert_eq!(compact_directive_hash(None), None);
    assert_eq!(compact_directive_hash(Some("")), None);
    let a = compact_directive_hash(Some("preserve this")).unwrap();
    let b = compact_directive_hash(Some("preserve this")).unwrap();
    assert_eq!(a, b, "hash must be stable for identical directives");
    assert_eq!(a.len(), 64);
    assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(
        a,
        compact_directive_hash(Some("preserve this!")).unwrap(),
        "different directives must hash differently"
    );
    // Digest of the canonical JSON bytes of the string literal (matches
    // xai_grok_sampling_types::canonical_value_digest(Value::String)).
    use sha2::Digest as _;
    let expected = format!(
        "{:x}",
        sha2::Sha256::digest(serde_json::to_vec(&json!("preserve this")).unwrap())
    );
    assert_eq!(a, expected);
}

async fn spawn_server(app: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

#[derive(Debug)]
struct CredentialSequence(Mutex<VecDeque<Option<CompactCredential>>>);

impl CompactCredentialResolver for CredentialSequence {
    fn current_credential(&self) -> Option<CompactCredential> {
        self.0.lock().unwrap().pop_front().flatten()
    }
}

#[derive(Debug)]
struct TestHeaderInjector;

impl xai_grok_sampler::config::HeaderInjector for TestHeaderInjector {
    fn inject(&self, headers: &mut HeaderMap) {
        headers.insert("traceparent", "00-safe-trace-01".parse().unwrap());
        headers.insert("authorization", "Bearer injected-again".parse().unwrap());
        headers.insert("x-compactions-remaining", "9".parse().unwrap());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_uses_real_endpoint_exact_body_and_metadata_only_auth() {
    assert_eq!(
        RESPONSES_COMPACT_CONNECT_TIMEOUT,
        std::time::Duration::from_secs(10)
    );
    let captured = Arc::new(Mutex::new(None::<Captured>));
    let sink = Arc::clone(&captured);
    let app = Router::new().route(
        "/v1/responses/compact",
        post(
            move |method: Method, uri: Uri, headers: HeaderMap, body: Bytes| {
                let sink = Arc::clone(&sink);
                async move {
                    *sink.lock().unwrap() = Some(Captured {
                        method,
                        uri,
                        headers,
                        body: serde_json::from_slice(&body).unwrap(),
                    });
                    axum::Json(valid_response())
                }
            },
        ),
    );
    let base = spawn_server(app).await;
    let mut config = SamplerConfig {
        api_key: Some("seed-must-not-win".into()),
        base_url: format!("{base}/v1/?dup=one&dup=two&encoded=%2Fkeep%2f&space=a%20b"),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    };
    config.query_params.insert("tenant".into(), "a b".into());
    config
        .extra_headers
        .insert("authorization".into(), "Bearer injected".into());
    config
        .extra_headers
        .insert("x-api-key".into(), "injected-key".into());
    config
        .extra_headers
        .insert("x-compaction-at".into(), "123".into());
    config
        .extra_headers
        .insert("x-provider-safe".into(), "kept".into());
    config.header_injector = Some(Arc::new(TestHeaderInjector));

    let client = SamplingClient::new(config).unwrap();
    let request = ResponsesCompactRequest::from_canonical(&canonical_context(), canonical_input(), Some("preserve this"))
        .unwrap()
        .with_correlation(CompactCorrelationHeaders {
            conversation_id: Some("conv".into()),
            request_id: Some("req".into()),
            session_id: Some("session".into()),
            agent_id: Some("agent".into()),
            ..Default::default()
        });
    let credential = RequestCredentialSnapshot::bearer("live-token", "principal-a");
    let response = client
        .compact_responses(&request, &credential, &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(response.output[0]["type"], "future_provider_item");
    assert_eq!(response.output[1]["encrypted_content"], "opaque");
    assert_eq!(response.usage_output_tokens, Some(17));

    let captured = captured.lock().unwrap().take().unwrap();
    assert_eq!(captured.method, Method::POST);
    assert_eq!(captured.uri.path(), "/v1/responses/compact");
    assert_eq!(
        captured.uri.query().unwrap(),
        "dup=one&dup=two&encoded=%2Fkeep%2f&space=a%20b&tenant=a+b"
    );
    assert_eq!(captured.headers["content-type"], "application/json");
    assert_eq!(captured.headers["authorization"], "Bearer live-token");
    assert!(!captured.headers.contains_key("x-api-key"));
    assert!(!captured.headers.contains_key("x-compaction-at"));
    assert!(!captured.headers.contains_key("x-compactions-remaining"));
    assert_eq!(captured.headers["x-provider-safe"], "kept");
    assert_eq!(captured.headers["traceparent"], "00-safe-trace-01");
    assert_eq!(captured.headers["x-grok-conv-id"], "conv");

    let object = captured.body.as_object().unwrap();
    assert_eq!(
        object.keys().cloned().collect::<Vec<_>>(),
        vec![
            "model",
            "input",
            "parallel_tool_calls",
            "instructions",
            "reasoning",
            "service_tier",
            "prompt_cache_key",
        ]
    );
    assert!(
        captured.body["instructions"]
            .as_str()
            .unwrap()
            .contains(USER_CONTEXT_DELIMITER)
    );
    assert!(
        captured.body["instructions"]
            .as_str()
            .unwrap()
            .ends_with("preserve this")
    );
    for forbidden in [
        "previous_response_id",
        "prompt_cache_options",
        "prompt_cache_retention",
        "tool_choice",
        "store",
        "stream",
        "stream_options",
        "include",
        "max_output_tokens",
        "temperature",
        "top_p",
        "metadata",
        "context_management",
    ] {
        assert!(
            !object.contains_key(forbidden),
            "forbidden field {forbidden}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retries_408_and_5xx_but_never_429_or_should_retry_false() {
    async fn handler(
        State((calls, mode)): State<(Arc<AtomicUsize>, &'static str)>,
    ) -> impl IntoResponse {
        let call = calls.fetch_add(1, Ordering::SeqCst);
        match (mode, call) {
            ("retry", 0) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                HeaderMap::new(),
                "{}",
            )
                .into_response(),
            ("timeout", 0) => {
                let mut headers = HeaderMap::new();
                headers.insert("retry-after", "0".parse().unwrap());
                (axum::http::StatusCode::REQUEST_TIMEOUT, headers, "{}").into_response()
            }
            ("no-retry", 0) => {
                let mut headers = HeaderMap::new();
                headers.insert("x-should-retry", "false".parse().unwrap());
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, headers, "{}").into_response()
            }
            ("rate", 0) => (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                HeaderMap::new(),
                "{}",
            )
                .into_response(),
            _ => axum::Json(valid_response()).into_response(),
        }
    }

    for (mode, expected_calls, succeeds) in [
        ("retry", 2, true),
        ("timeout", 2, true),
        ("no-retry", 1, false),
        ("rate", 1, false),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route("/custom/responses/compact", post(handler))
            .with_state((Arc::clone(&calls), mode));
        let base = spawn_server(app).await;
        let client = SamplingClient::new(SamplerConfig {
            base_url: format!("{base}/custom"),
            model: "grok-test".into(),
            api_backend: ApiBackend::Responses,
            ..Default::default()
        })
        .unwrap();
        let request = compact_request();
        let result = client
            .compact_responses(
                &request,
                &RequestCredentialSnapshot::bearer("token", "principal"),
                &CancellationToken::new(),
            )
            .await;
        assert_eq!(result.is_ok(), succeeds, "mode={mode}: {result:?}");
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls, "mode={mode}");
        if succeeds {
            assert_eq!(result.as_ref().unwrap().attempts, 2, "mode={mode}");
        }
        if mode == "rate" {
            assert_eq!(result.unwrap_err().status(), Some(429));
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn credential_refresh_keeps_principal_and_rotation_fails_before_replay() {
    async fn handler(
        State((calls, auth)): State<CredentialHandlerState>,
        headers: HeaderMap,
    ) -> impl IntoResponse {
        let call = calls.fetch_add(1, Ordering::SeqCst);
        auth.lock()
            .unwrap()
            .push(headers["authorization"].to_str().unwrap().to_string());
        if call == 0 {
            axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response()
        } else {
            axum::Json(valid_response()).into_response()
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let auth = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/responses/compact", post(handler))
        .with_state((Arc::clone(&calls), Arc::clone(&auth)));
    let base = spawn_server(app).await;
    let client = SamplingClient::new(SamplerConfig {
        base_url: format!("{base}/v1"),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    })
    .unwrap();
    let request = compact_request();
    let credential = RequestCredentialSnapshot::bearer("stale", "principal-a").with_resolver(
        Arc::new(CredentialSequence(Mutex::new(VecDeque::from([
            Some(CompactCredential::bearer("token-1", "principal-a")),
            Some(CompactCredential::bearer("token-2", "principal-a")),
        ])))),
    );
    let response = client
        .compact_responses(&request, &credential, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(response.attempts, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        auth.lock().unwrap().as_slice(),
        ["Bearer token-1", "Bearer token-2"]
    );

    let rotated = RequestCredentialSnapshot::bearer("old", "principal-a").with_resolver(Arc::new(
        CredentialSequence(Mutex::new(VecDeque::from([Some(
            CompactCredential::bearer("new", "principal-b"),
        )]))),
    ));
    let error = client
        .compact_responses(&request, &rotated, &CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(error.failure(), ResponsesCompactFailure::IdentityChanged);
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let missing = RequestCredentialSnapshot::bearer("", "principal-a");
    let error = client
        .compact_responses(&request, &missing, &CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(error.failure(), ResponsesCompactFailure::MissingCredential);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn x_api_key_is_the_only_wire_auth_header() {
    let captured = Arc::new(Mutex::new(None::<HeaderMap>));
    let sink = Arc::clone(&captured);
    let app = Router::new().route(
        "/v1/responses/compact",
        post(move |headers: HeaderMap| {
            let sink = Arc::clone(&sink);
            async move {
                *sink.lock().unwrap() = Some(headers);
                axum::Json(valid_response())
            }
        }),
    );
    let base = spawn_server(app).await;
    let client = SamplingClient::new(SamplerConfig {
        base_url: format!("{base}/v1"),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    })
    .unwrap();
    let request = compact_request();
    client
        .compact_responses(
            &request,
            &RequestCredentialSnapshot::x_api_key("wire-key", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap();

    let headers = captured.lock().unwrap().take().unwrap();
    assert_eq!(headers["x-api-key"], "wire-key");
    assert!(!headers.contains_key("authorization"));
}

#[test]
fn response_validation_accepts_codex_compaction_summary_without_rewriting_it() {
    let parsed = validate_responses_compact_response(valid_response()).unwrap();
    assert_eq!(parsed.output[1]["type"], "compaction_summary");
    assert_eq!(parsed.output[1]["encrypted_content"], "opaque");
}

#[test]
fn response_validation_preserves_unknown_items_and_rejects_aliases_and_triggers() {
    let parsed = validate_responses_compact_response(valid_response()).unwrap();
    assert_eq!(parsed.output.len(), 2);
    assert_eq!(parsed.output[0]["nested"]["z"], 1);

    let mut codex_v1 = valid_response();
    codex_v1.as_object_mut().unwrap().remove("object");
    assert_eq!(
        validate_responses_compact_response(codex_v1)
            .unwrap()
            .output
            .len(),
        2
    );

    let mut compatible_compaction = valid_response();
    compatible_compaction["output"][1]["type"] = json!("compaction");
    assert_eq!(
        validate_responses_compact_response(compatible_compaction)
            .unwrap()
            .output[1]["type"],
        "compaction"
    );

    for bad in [
        json!({"output": [{"type": "summary", "encrypted_content": "opaque"}]}),
        json!({"output": [{"type": "context", "encrypted_content": "opaque"}]}),
        json!({"output": [{"type": "compaction_trigger"}, {"type": "compaction", "encrypted_content": "opaque"}]}),
        json!({"output": [{"type": "compaction", "encrypted_content": ""}]}),
        json!({"output": [{"type": "compaction_summary", "encrypted_content": ""}]}),
        json!({
            "object": "response.compaction",
            "output": [{
                "type": "compaction",
                "encrypted_content": "x".repeat(RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES + 1)
            }]
        }),
    ] {
        assert_eq!(
            validate_responses_compact_response(bad)
                .unwrap_err()
                .failure(),
            ResponsesCompactFailure::InvalidResponse
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_logs_never_expose_prompt_blob_query_or_credentials() {
    const PROMPT: &str = "prompt-sentinel-never-log";
    const BLOB: &str = "opaque-blob-sentinel-never-log";
    const QUERY: &str = "query-sentinel-never-log";
    const API_KEY: &str = "api-key-sentinel-never-log";
    const TOKEN: &str = "bearer-token-sentinel-never-log";

    let app = Router::new().route(
        "/v1/responses/compact",
        post(|| async {
            (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(json!({
                    "error": {
                        "code": "invalid_request",
                        "message": BLOB
                    }
                })),
            )
        }),
    );
    let base = spawn_server(app).await;
    let logs = CapturedLogs::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .without_time()
        .with_writer(logs.clone())
        .finish();

    async move {
        let client = SamplingClient::new(SamplerConfig {
            api_key: Some(API_KEY.into()),
            base_url: format!("{base}/v1?secret={QUERY}"),
            model: "grok-test".into(),
            api_backend: ApiBackend::Responses,
            ..Default::default()
        })
        .unwrap();
        let request = compact_request()
            .with_instructions(Some(PROMPT.into()));
        let error = client
            .compact_responses(
                &request,
                &RequestCredentialSnapshot::bearer(TOKEN, "principal"),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.status(), Some(400));
    }
    .with_subscriber(subscriber)
    .await;

    let captured = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    for sentinel in [PROMPT, BLOB, QUERY, API_KEY, TOKEN] {
        assert!(!captured.contains(sentinel), "leaked sentinel: {sentinel}");
    }
}

#[tokio::test]
async fn oversized_request_is_rejected_before_transport() {
    let client = SamplingClient::new(SamplerConfig {
        base_url: "http://127.0.0.1:9/v1".into(),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    })
    .unwrap();
    let request = compact_request()
        .with_instructions(Some("x".repeat(RESPONSES_COMPACT_MAX_BYTES)));

    let error = client
        .compact_responses(
            &request,
            &RequestCredentialSnapshot::bearer("token", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.failure(), ResponsesCompactFailure::RequestTooLarge);
    assert_eq!(error.attempts(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_content_length_is_rejected_before_body_read() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 64 * 1024];
        let _ = socket.read(&mut request).await.unwrap();
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    RESPONSES_COMPACT_MAX_BYTES as u64 + 1
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    let client = SamplingClient::new(SamplerConfig {
        base_url: format!("http://{addr}/v1"),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    })
    .unwrap();
    let request = compact_request();

    let error = client
        .compact_responses(
            &request,
            &RequestCredentialSnapshot::bearer("token", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.failure(), ResponsesCompactFailure::ResponseTooLarge);
    assert_eq!(error.attempts(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_interrupts_a_stalled_response_body() {
    let body_polled = Arc::new(tokio::sync::Notify::new());
    let signal = Arc::clone(&body_polled);
    let app = Router::new().route(
        "/v1/responses/compact",
        post(move || {
            let signal = Arc::clone(&signal);
            async move {
                let stream = futures_util::stream::unfold(false, move |sent| {
                    let signal = Arc::clone(&signal);
                    async move {
                        if sent {
                            std::future::pending::<
                                Option<(Result<Bytes, std::convert::Infallible>, bool)>,
                            >()
                            .await
                        } else {
                            signal.notify_one();
                            Some((Ok(Bytes::from_static(b"{")), true))
                        }
                    }
                });
                axum::response::Response::new(Body::from_stream(stream))
            }
        }),
    );
    let base = spawn_server(app).await;
    let client = SamplingClient::new(SamplerConfig {
        base_url: format!("{base}/v1"),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    })
    .unwrap();
    let request = compact_request();
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let task = tokio::spawn(async move {
        client
            .compact_responses(
                &request,
                &RequestCredentialSnapshot::bearer("token", "principal"),
                &task_cancellation,
            )
            .await
    });

    body_polled.notified().await;
    cancellation.cancel();
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(error.failure(), ResponsesCompactFailure::Cancelled);
    assert_eq!(error.attempts(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_interrupts_retry_after_backoff() {
    let first_attempt = Arc::new(tokio::sync::Notify::new());
    let signal = Arc::clone(&first_attempt);
    let calls = Arc::new(AtomicUsize::new(0));
    let call_count = Arc::clone(&calls);
    let app = Router::new().route(
        "/v1/responses/compact",
        post(move || {
            let signal = Arc::clone(&signal);
            let call = call_count.fetch_add(1, Ordering::SeqCst);
            async move {
                assert_eq!(call, 0, "cancelled backoff must not start attempt two");
                let mut headers = HeaderMap::new();
                headers.insert("retry-after", "30".parse().unwrap());
                signal.notify_one();
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, headers, "{}")
            }
        }),
    );
    let base = spawn_server(app).await;
    let client = SamplingClient::new(SamplerConfig {
        base_url: format!("{base}/v1"),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    })
    .unwrap();
    let request = compact_request();
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let task = tokio::spawn(async move {
        client
            .compact_responses(
                &request,
                &RequestCredentialSnapshot::bearer("token", "principal"),
                &task_cancellation,
            )
            .await
    });

    first_attempt.notified().await;
    cancellation.cancel();
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(error.failure(), ResponsesCompactFailure::Cancelled);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn pre_cancelled_request_does_not_reach_transport() {
    let app = Router::new().route(
        "/v1/responses/compact",
        post(|| async {
            panic!("cancelled request reached server");
            #[allow(unreachable_code)]
            "unreachable"
        }),
    );
    let base = spawn_server(app).await;
    let client = SamplingClient::new(SamplerConfig {
        base_url: format!("{base}/v1"),
        model: "grok-test".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    })
    .unwrap();
    let request = compact_request();
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let error = client
        .compact_responses(
            &request,
            &RequestCredentialSnapshot::x_api_key("key", "principal"),
            &cancellation,
        )
        .await
        .unwrap_err();
    assert_eq!(error.failure(), ResponsesCompactFailure::Cancelled);
}

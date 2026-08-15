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
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::instrument::WithSubscriber as _;
use xai_grok_sampler::{
    ApiBackend, CompactCorrelationHeaders, CompactCredential, CompactCredentialResolver,
    KnownProvider, ProviderRouteHint, RESPONSES_COMPACT_MAX_BYTES,
    RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES, RequestCredentialSnapshot, ResponsesCompactFailure,
    ResponsesCompactRequest, SamplerConfig, SamplingClient, USER_CONTEXT_DELIMITER,
    collect_compaction_from_completed, compact_directive_hash,
};
use xai_grok_sampling_types::{
    ConversationItem, ConversationRequest, ReasoningEffort, ResolvedCompactRequest,
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

fn compact_request_with_context(user_context: Option<&str>) -> ResponsesCompactRequest {
    let request = ConversationRequest {
        items: vec![ConversationItem::user("hello")],
        model: Some("grok-test".into()),
        reasoning_effort: Some(ReasoningEffort::Medium),
        instructions: Some("normal instructions".into()),
        prompt_cache_key: Some("cache-key".into()),
        service_tier: Some("priority".into()),
        parallel_tool_calls: Some(true),
        ..Default::default()
    };
    let resolved = ResolvedCompactRequest::try_normal(&request, user_context).unwrap();
    ResponsesCompactRequest::from_resolved(&resolved).unwrap()
}

fn compact_request() -> ResponsesCompactRequest {
    compact_request_with_context(None)
}

fn completed_sse(output: Value, usage: Value) -> String {
    let payload = json!({
        "type": "response.completed",
        "response": {
            "id": "cmp_1",
            "output": output,
            "usage": usage
        }
    });
    format!("event: response.completed\ndata: {payload}\n\n")
}

fn valid_sse() -> String {
    completed_sse(
        json!([
            {"type": "future_provider_item", "nested": {"z": 1}},
            {"type": "compaction_summary", "encrypted_content": "opaque"}
        ]),
        json!({"output_tokens": 17, "total_tokens": 99}),
    )
}

fn sse_response(body: String) -> impl IntoResponse {
    (
        [
            (
                axum::http::header::CONTENT_TYPE,
                "text/event-stream".to_owned(),
            ),
            (axum::http::header::CACHE_CONTROL, "no-cache".to_owned()),
        ],
        body,
    )
}

#[test]
fn resolved_constructor_appends_compact_only_user_context() {
    let body = compact_request_with_context(Some("preserve this"))
        .body()
        .clone();
    let instructions = body["instructions"].as_str().unwrap();
    assert!(instructions.contains(USER_CONTEXT_DELIMITER));
    assert!(instructions.ends_with("preserve this"));
    assert_eq!(body["model"], "grok-test");
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["stream"], true);
    assert_eq!(body["store"], false);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input.last().unwrap()["type"], "compaction_trigger");
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
        headers.insert("x-codex-turn-state", "sticky-token".parse().unwrap());
        headers.insert("x-compactions-remaining", "9".parse().unwrap());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_uses_responses_endpoint_with_trigger_last_and_turn_headers() {
    let captured = Arc::new(Mutex::new(None::<Captured>));
    let sink = Arc::clone(&captured);
    let app = Router::new().route(
        "/v1/responses",
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
                    sse_response(valid_sse())
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
        .insert("x-provider-safe".into(), "kept".into());
    config.header_injector = Some(Arc::new(TestHeaderInjector));

    let client = SamplingClient::new(config).unwrap();
    let request = compact_request_with_context(Some("preserve this")).with_correlation(
        CompactCorrelationHeaders {
            conversation_id: Some("conv".into()),
            request_id: Some("req".into()),
            session_id: Some("session".into()),
            agent_id: Some("agent".into()),
            ..Default::default()
        },
    );
    let credential = RequestCredentialSnapshot::bearer("live-token", "principal-a");
    let response = client
        .compact_responses(&request, &credential, &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(response.compaction_item["type"], "compaction_summary");
    assert_eq!(response.compaction_item["encrypted_content"], "opaque");
    assert_eq!(response.usage_output_tokens, Some(17));
    assert_eq!(response.usage_total_tokens, Some(99));

    let captured = captured.lock().unwrap().take().unwrap();
    assert_eq!(captured.method, Method::POST);
    assert_eq!(captured.uri.path(), "/v1/responses");
    assert_eq!(
        captured.uri.query().unwrap(),
        "dup=one&dup=two&encoded=%2Fkeep%2f&space=a%20b&tenant=a+b"
    );
    assert_eq!(captured.headers["content-type"], "application/json");
    assert_eq!(captured.headers["accept"], "text/event-stream");
    assert_eq!(captured.headers["authorization"], "Bearer live-token");
    assert!(!captured.headers.contains_key("x-api-key"));
    // Turn-state headers are inherited (no longer stripped).
    assert_eq!(captured.headers["x-codex-turn-state"], "sticky-token");
    assert_eq!(captured.headers["x-provider-safe"], "kept");
    assert_eq!(captured.headers["traceparent"], "00-safe-trace-01");
    assert_eq!(captured.headers["x-grok-conv-id"], "conv");

    let input = captured.body["input"].as_array().unwrap();
    assert_eq!(input.last().unwrap()["type"], "compaction_trigger");
    assert_eq!(captured.body["stream"], true);
    assert_eq!(captured.body["store"], false);
    assert!(
        captured.body["instructions"]
            .as_str()
            .unwrap()
            .contains(USER_CONTEXT_DELIMITER)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_codex_route_uses_codex_path_and_allowlisted_headers() {
    let captured = Arc::new(Mutex::new(None::<Captured>));
    let sink = Arc::clone(&captured);
    let app = Router::new().route(
        "/backend-api/codex/responses",
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
                    sse_response(valid_sse())
                }
            },
        ),
    );
    let base = spawn_server(app).await;
    let mut config = SamplerConfig {
        // Known-provider routes require a configured credential at
        // construction; the compact path replaces auth with the snapshot
        // credential below, so this seed never reaches the wire.
        api_key: Some("seed-must-not-win".into()),
        base_url: format!("{base}/backend-api"),
        model: "gpt-5.6-codex".into(),
        api_backend: ApiBackend::Responses,
        ..Default::default()
    };
    // Shell injects the compaction-only codex headers via extra_headers.
    config.extra_headers.insert(
        "x-codex-beta-features".into(),
        "remote_compaction_v2".into(),
    );
    config.extra_headers.insert(
        "x-codex-turn-metadata".into(),
        json!({"request_kind": "compaction"}).to_string(),
    );
    config.header_injector = Some(Arc::new(TestHeaderInjector));

    let client = SamplingClient::new_with_route(
        config,
        ProviderRouteHint::Known(KnownProvider::OpenaiCodex),
    )
    .unwrap();
    let request = compact_request().with_correlation(CompactCorrelationHeaders {
        conversation_id: Some("conv".into()),
        ..Default::default()
    });
    let credential = RequestCredentialSnapshot::bearer("live-token", "principal-a");
    client
        .compact_responses(&request, &credential, &CancellationToken::new())
        .await
        .unwrap();

    let captured = captured.lock().unwrap().take().unwrap();
    assert_eq!(captured.method, Method::POST);
    assert_eq!(captured.uri.path(), "/backend-api/codex/responses");
    // Compaction-only codex headers survive the provider allowlist.
    assert_eq!(
        captured.headers["x-codex-beta-features"],
        "remote_compaction_v2"
    );
    assert!(
        captured.headers["x-codex-turn-metadata"]
            .to_str()
            .unwrap()
            .contains("compaction")
    );
    // Sticky routing token inherited like a turn request.
    assert_eq!(captured.headers["x-codex-turn-state"], "sticky-token");
    // Legacy inline enablement headers are stripped, as on turn requests.
    assert!(!captured.headers.contains_key("x-compactions-remaining"));
    // Correlation headers are still emitted (applied post-sanitize).
    assert_eq!(captured.headers["x-grok-conv-id"], "conv");
    assert_eq!(captured.headers["authorization"], "Bearer live-token");

    let input = captured.body["input"].as_array().unwrap();
    assert_eq!(input.last().unwrap()["type"], "compaction_trigger");
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
            _ => sse_response(valid_sse()).into_response(),
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
            .route("/custom/responses", post(handler))
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
            sse_response(valid_sse()).into_response()
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let auth = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/responses", post(handler))
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
        "/v1/responses",
        post(move |headers: HeaderMap| {
            let sink = Arc::clone(&sink);
            async move {
                *sink.lock().unwrap() = Some(headers);
                sse_response(valid_sse())
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
fn completed_collection_accepts_alias_and_tolerates_other_items() {
    let parsed = collect_compaction_from_completed(&json!({
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "future_provider_item", "nested": {"z": 1}},
                {"type": "compaction_summary", "encrypted_content": "opaque"}
            ],
            "usage": {"output_tokens": 17, "total_tokens": 99}
        }
    }))
    .unwrap();
    assert_eq!(parsed.0["type"], "compaction_summary");
    assert_eq!(parsed.1, Some(17));

    let compaction = collect_compaction_from_completed(&json!({
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "message", "role": "assistant"},
                {"type": "compaction", "encrypted_content": "blob"}
            ],
            "usage": {}
        }
    }))
    .unwrap();
    assert_eq!(compaction.0["encrypted_content"], "blob");
}

#[test]
fn completed_collection_rejects_zero_or_two_compaction_items() {
    let zero = collect_compaction_from_completed(&json!({
        "type": "response.completed",
        "response": {
            "output": [{"type": "message", "role": "assistant"}],
            "usage": {}
        }
    }))
    .unwrap_err();
    assert_eq!(
        zero.failure(),
        ResponsesCompactFailure::CompletedWithoutCompaction
    );

    let two = collect_compaction_from_completed(&json!({
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "compaction", "encrypted_content": "a"},
                {"type": "compaction_summary", "encrypted_content": "b"}
            ],
            "usage": {}
        }
    }))
    .unwrap_err();
    assert_eq!(
        two.failure(),
        ResponsesCompactFailure::CompletedWithoutCompaction
    );

    let empty_encrypted = collect_compaction_from_completed(&json!({
        "type": "response.completed",
        "response": {
            "output": [{"type": "compaction", "encrypted_content": ""}],
            "usage": {}
        }
    }))
    .unwrap_err();
    assert_eq!(
        empty_encrypted.failure(),
        ResponsesCompactFailure::InvalidResponse
    );

    let oversized = collect_compaction_from_completed(&json!({
        "type": "response.completed",
        "response": {
            "output": [{
                "type": "compaction",
                "encrypted_content": "x".repeat(RESPONSES_COMPACT_MAX_ENCRYPTED_BYTES + 1)
            }],
            "usage": {}
        }
    }))
    .unwrap_err();
    assert_eq!(
        oversized.failure(),
        ResponsesCompactFailure::InvalidResponse
    );
}

/// SSE body where output items arrive via `response.output_item.done` frames
/// and the terminal `response.completed` carries `completed_output` — the
/// live ChatGPT Codex backend leaves that terminal `output` array EMPTY and
/// delivers all items exclusively through the streamed frames
/// (`codex-backend-quirks.md` quirk 1).
fn streamed_items_sse(streamed: Vec<Value>, completed_output: Value, usage: Value) -> String {
    let mut body = String::new();
    for item in streamed {
        let frame = json!({"type": "response.output_item.done", "output_index": 0, "item": item});
        body.push_str(&format!(
            "event: response.output_item.done\ndata: {frame}\n\n"
        ));
    }
    body.push_str(&completed_sse(completed_output, usage));
    body
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compaction_item_from_output_item_done_with_empty_completed_output() {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            sse_response(streamed_items_sse(
                vec![
                    json!({"type": "message", "role": "assistant"}),
                    json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "streamed-blob"}),
                ],
                // Live backend: terminal frame carries an empty output array.
                json!([]),
                json!({"output_tokens": 42, "total_tokens": 100}),
            ))
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
    let response = client
        .compact_responses(
            &compact_request(),
            &RequestCredentialSnapshot::bearer("token", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.compaction_item["encrypted_content"],
        "streamed-blob"
    );
    assert_eq!(response.usage_output_tokens, Some(42));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compaction_item_duplicated_across_frames_counts_once() {
    // Same item (same id) in output_item.done AND completed.output must not
    // be counted twice.
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            sse_response(streamed_items_sse(
                vec![json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "blob"})],
                json!([{"type": "compaction", "id": "cmp_1", "encrypted_content": "blob"}]),
                json!({"output_tokens": 7, "total_tokens": 70}),
            ))
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
    let response = client
        .compact_responses(
            &compact_request(),
            &RequestCredentialSnapshot::bearer("token", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(response.compaction_item["id"], "cmp_1");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_distinct_streamed_compaction_items_rejected() {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            sse_response(streamed_items_sse(
                vec![
                    json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "a"}),
                    json!({"type": "compaction", "id": "cmp_2", "encrypted_content": "b"}),
                ],
                json!([]),
                json!({"output_tokens": 1, "total_tokens": 2}),
            ))
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
    let error = client
        .compact_responses(
            &compact_request(),
            &RequestCredentialSnapshot::bearer("token", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.failure(),
        ResponsesCompactFailure::CompletedWithoutCompaction
    );
    assert!(error.is_unsupported_capability());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zero_compaction_items_classifies_completed_without_compaction() {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            sse_response(completed_sse(
                json!([{"type": "message", "role": "assistant", "content": "hi"}]),
                json!({"output_tokens": 1, "total_tokens": 2}),
            ))
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
    let error = client
        .compact_responses(
            &compact_request(),
            &RequestCredentialSnapshot::bearer("token", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.failure(),
        ResponsesCompactFailure::CompletedWithoutCompaction
    );
    assert!(error.is_unsupported_capability());
    // Attempt budget of 2 for stream-level completed-without-item.
    assert_eq!(error.attempts(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_compaction_items_are_rejected() {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            sse_response(completed_sse(
                json!([
                    {"type": "compaction", "encrypted_content": "a"},
                    {"type": "compaction", "encrypted_content": "b"}
                ]),
                json!({}),
            ))
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
    let error = client
        .compact_responses(
            &compact_request(),
            &RequestCredentialSnapshot::bearer("token", "principal"),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.failure(),
        ResponsesCompactFailure::CompletedWithoutCompaction
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_logs_never_expose_prompt_blob_query_or_credentials() {
    const PROMPT: &str = "prompt-sentinel-never-log";
    const BLOB: &str = "opaque-blob-sentinel-never-log";
    const QUERY: &str = "query-sentinel-never-log";
    const API_KEY: &str = "api-key-sentinel-never-log";
    const TOKEN: &str = "bearer-token-sentinel-never-log";

    let app = Router::new().route(
        "/v1/responses",
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
        let request = compact_request().with_instructions(Some(PROMPT.into()));
        let error = client
            .compact_responses(
                &request,
                &RequestCredentialSnapshot::bearer(TOKEN, "principal"),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.status(), Some(400));
        assert!(error.is_unsupported_capability());
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
    let request =
        compact_request().with_instructions(Some("x".repeat(RESPONSES_COMPACT_MAX_BYTES)));

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
async fn cancellation_interrupts_a_stalled_response_body() {
    let body_polled = Arc::new(tokio::sync::Notify::new());
    let signal = Arc::clone(&body_polled);
    let app = Router::new().route(
        "/v1/responses",
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
                            Some((
                                Ok(Bytes::from_static(b"event: response.created\ndata: {}\n\n")),
                                true,
                            ))
                        }
                    }
                });
                axum::response::Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
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
        "/v1/responses",
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
        "/v1/responses",
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

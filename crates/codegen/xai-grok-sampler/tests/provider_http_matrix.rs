//! Traffic-level provider contract matrix.
//!
//! Each case crosses the public shell-owned route sidecar into a real
//! `SamplingClient`, posts to an Axum capture server, and asserts the final
//! URL, merged/sanitized headers, upstream model, compatibility body, stream
//! order, and wire-derived 401 credential classification.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::{io, str};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::post,
};
use futures_util::stream;
use indexmap::IndexMap;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tracing::{Instrument as _, instrument::WithSubscriber as _};
use xai_grok_sampler::{
    ApiBackend, AuthScheme, HeaderInjector, KnownProvider, ProviderRouteHint, SamplerConfig,
    SamplingClient,
};
use xai_grok_sampling_types::{
    ConversationItem, ConversationRequest, ConversationResponse, ReasoningEffort, SamplingError,
    SentCredential, ToolSpec,
};
use xai_grok_test_support::scripted::ScriptedBody;
use xai_grok_test_support::{ScriptedResponse, SseEvent, sse};

#[derive(Clone, Default)]
struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

struct CapturedLogWriter(Arc<Mutex<Vec<u8>>>);

impl io::Write for CapturedLogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
    type Writer = CapturedLogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        CapturedLogWriter(Arc::clone(&self.0))
    }
}

#[derive(Clone, Debug)]
struct CapturedRequest {
    path: String,
    headers: BTreeMap<String, String>,
    body: Value,
}

#[derive(Clone)]
struct CaptureState {
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
    response: ScriptedResponse,
}

#[derive(Debug)]
struct ProviderHeaderInjector;

impl HeaderInjector for ProviderHeaderInjector {
    fn inject(&self, headers: &mut HeaderMap) {
        headers.insert("traceparent", "00-provider-matrix-safe".parse().unwrap());
        headers.insert("x-grok-conv-id", "must-be-removed".parse().unwrap());
        headers.insert("x-xai-token-auth", "must-be-removed".parse().unwrap());
    }
}

struct CaptureServer {
    origin: String,
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
}

impl CaptureServer {
    async fn start(paths: &[&str], response: ScriptedResponse) -> Self {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let state = CaptureState {
            captured: Arc::clone(&captured),
            response,
        };
        let mut router = Router::new();
        for path in paths {
            router = router.route(path, post(capture));
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router.with_state(state))
                .await
                .unwrap();
        });
        Self {
            origin: format!("http://{addr}"),
            captured,
        }
    }

    fn take_one(&self) -> CapturedRequest {
        let mut requests = self.captured.lock().unwrap();
        assert_eq!(requests.len(), 1, "expected exactly one captured request");
        requests.remove(0)
    }
}

async fn capture(
    State(state): State<CaptureState>,
    uri: axum::http::Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let headers = headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_owned(), value.to_owned()))
        })
        .collect();
    state.captured.lock().unwrap().push(CapturedRequest {
        path: uri.path().to_owned(),
        headers,
        body,
    });
    scripted_response(state.response.clone())
}

fn scripted_response(script: ScriptedResponse) -> Response {
    let mut response = match script.body {
        ScriptedBody::Json(body) => Json(body).into_response(),
        ScriptedBody::Raw(body) => body.into_response(),
        ScriptedBody::Sse(events) => {
            let events = events.into_iter().map(|event| {
                let rendered = Event::default().data(event.data);
                Ok::<_, std::convert::Infallible>(match event.event {
                    Some(name) => rendered.event(name),
                    None => rendered,
                })
            });
            Sse::new(stream::iter(events)).into_response()
        }
    };
    *response.status_mut() = StatusCode::from_u16(script.status).expect("valid status");
    for (name, value) in script.headers {
        response.headers_mut().insert(
            axum::http::HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
            axum::http::HeaderValue::from_str(&value).expect("valid header value"),
        );
    }
    response
}

#[derive(Clone, Copy, Debug)]
enum BodyContract {
    Anthropic,
    Copilot,
    Openrouter,
    Kimi,
    Radius,
    Deepseek,
    Zai,
}

#[derive(Clone, Debug)]
struct ProviderCase {
    name: &'static str,
    provider: KnownProvider,
    namespace: &'static str,
    upstream_model: &'static str,
    base_path: &'static str,
    endpoint_path: &'static str,
    api_backend: ApiBackend,
    auth_scheme: AuthScheme,
    expected_provider_headers: &'static [(&'static str, &'static str)],
    body: BodyContract,
}

const CASES: &[ProviderCase] = &[
    ProviderCase {
        name: "anthropic",
        provider: KnownProvider::Anthropic,
        namespace: "anthropic",
        upstream_model: "claude-fable-5",
        base_path: "",
        endpoint_path: "/v1/messages",
        api_backend: ApiBackend::Messages,
        auth_scheme: AuthScheme::XApiKey,
        expected_provider_headers: &[("anthropic-version", "2023-06-01")],
        body: BodyContract::Anthropic,
    },
    ProviderCase {
        name: "github-copilot",
        provider: KnownProvider::GithubCopilot,
        namespace: "github-copilot",
        upstream_model: "gpt-4.1",
        base_path: "",
        endpoint_path: "/chat/completions",
        api_backend: ApiBackend::ChatCompletions,
        auth_scheme: AuthScheme::Bearer,
        expected_provider_headers: &[
            ("editor-version", "vscode/1.107.0"),
            ("editor-plugin-version", "copilot-chat/0.35.0"),
            ("copilot-integration-id", "vscode-chat"),
            ("x-initiator", "user"),
            ("openai-intent", "conversation-edits"),
        ],
        body: BodyContract::Copilot,
    },
    ProviderCase {
        name: "openrouter",
        provider: KnownProvider::Openrouter,
        namespace: "openrouter",
        upstream_model: "openai/gpt-5.2",
        base_path: "/api/v1",
        endpoint_path: "/api/v1/chat/completions",
        api_backend: ApiBackend::ChatCompletions,
        auth_scheme: AuthScheme::Bearer,
        expected_provider_headers: &[],
        body: BodyContract::Openrouter,
    },
    ProviderCase {
        name: "kimi-coding",
        provider: KnownProvider::KimiCoding,
        namespace: "kimi-coding",
        upstream_model: "k3",
        base_path: "/coding",
        endpoint_path: "/coding/v1/messages",
        api_backend: ApiBackend::Messages,
        auth_scheme: AuthScheme::Bearer,
        expected_provider_headers: &[("anthropic-version", "2023-06-01")],
        body: BodyContract::Kimi,
    },
    ProviderCase {
        name: "radius",
        provider: KnownProvider::Radius,
        namespace: "radius",
        upstream_model: "radius-1",
        base_path: "",
        endpoint_path: "/messages",
        api_backend: ApiBackend::Messages,
        auth_scheme: AuthScheme::Bearer,
        expected_provider_headers: &[],
        body: BodyContract::Radius,
    },
    ProviderCase {
        name: "deepseek",
        provider: KnownProvider::Deepseek,
        namespace: "deepseek",
        upstream_model: "deepseek-v4-flash",
        base_path: "",
        endpoint_path: "/chat/completions",
        api_backend: ApiBackend::ChatCompletions,
        auth_scheme: AuthScheme::Bearer,
        expected_provider_headers: &[],
        body: BodyContract::Deepseek,
    },
    ProviderCase {
        name: "zai",
        provider: KnownProvider::Zai,
        namespace: "zai",
        upstream_model: "glm-5.2",
        base_path: "/api/coding/paas/v4",
        endpoint_path: "/api/coding/paas/v4/chat/completions",
        api_backend: ApiBackend::ChatCompletions,
        auth_scheme: AuthScheme::Bearer,
        expected_provider_headers: &[],
        body: BodyContract::Zai,
    },
    ProviderCase {
        name: "zai-coding-cn",
        provider: KnownProvider::ZaiCodingCn,
        namespace: "zai-coding-cn",
        upstream_model: "glm-5.2",
        base_path: "/api/coding/paas/v4",
        endpoint_path: "/api/coding/paas/v4/chat/completions",
        api_backend: ApiBackend::ChatCompletions,
        auth_scheme: AuthScheme::Bearer,
        expected_provider_headers: &[],
        body: BodyContract::Zai,
    },
];

fn model_id(case: &ProviderCase) -> String {
    format!("{}/{}", case.namespace, case.upstream_model)
}

fn config(case: &ProviderCase, origin: &str, key: &str) -> SamplerConfig {
    let mut extra_headers = IndexMap::new();
    for (name, value) in case.expected_provider_headers {
        if !matches!(*name, "x-initiator" | "openai-intent") {
            extra_headers.insert((*name).to_owned(), (*value).to_owned());
        }
    }
    extra_headers.insert("x-grok-conv-id".to_owned(), "must-be-removed".to_owned());
    extra_headers.insert("x-xai-token-auth".to_owned(), "must-be-removed".to_owned());
    extra_headers.insert("anthropic-version".to_owned(), "foreign-value".to_owned());
    match case.auth_scheme {
        AuthScheme::Bearer => {
            extra_headers.insert("x-api-key".to_owned(), "foreign-auth".to_owned());
            extra_headers.insert("api-key".to_owned(), "foreign-auth".to_owned());
        }
        AuthScheme::XApiKey => {
            extra_headers.insert("authorization".to_owned(), "Bearer foreign-auth".to_owned());
            extra_headers.insert("api-key".to_owned(), "foreign-auth".to_owned());
        }
    }
    if matches!(case.body, BodyContract::Anthropic | BodyContract::Kimi) {
        extra_headers.insert("anthropic-version".to_owned(), "2023-06-01".to_owned());
    }

    SamplerConfig {
        api_key: Some(key.to_owned()),
        base_url: format!("{origin}{}", case.base_path),
        model: model_id(case),
        max_completion_tokens: Some(match case.body {
            BodyContract::Deepseek => 384_000,
            BodyContract::Zai => 131_072,
            _ => 4_096,
        }),
        api_backend: case.api_backend.clone(),
        auth_scheme: case.auth_scheme,
        extra_headers,
        max_retries: Some(0),
        reasoning_effort: Some(match case.body {
            BodyContract::Openrouter => ReasoningEffort::Xhigh,
            BodyContract::Deepseek | BodyContract::Zai | BodyContract::Radius => {
                ReasoningEffort::High
            }
            _ => ReasoningEffort::None,
        }),
        header_injector: Some(Arc::new(ProviderHeaderInjector)),
        ..SamplerConfig::default()
    }
}

fn request(case: &ProviderCase) -> ConversationRequest {
    let reasoning_effort = match case.body {
        BodyContract::Openrouter => ReasoningEffort::Xhigh,
        BodyContract::Deepseek | BodyContract::Zai | BodyContract::Radius => ReasoningEffort::High,
        _ => ReasoningEffort::None,
    };
    let mut request = ConversationRequest {
        items: vec![
            ConversationItem::system("system prompt"),
            ConversationItem::user("hello"),
        ],
        x_grok_conv_id: Some("private-conv".to_owned()),
        x_grok_req_id: Some("provider-matrix-request".to_owned()),
        x_grok_session_id: Some("private-session".to_owned()),
        reasoning_effort: Some(reasoning_effort),
        ..ConversationRequest::default()
    };
    if matches!(case.body, BodyContract::Zai) {
        request.tools.push(ToolSpec {
            name: "read".to_owned(),
            description: Some("Read a file".to_owned()),
            parameters: json!({"type":"object"}),
        });
    }
    request
}

fn response(case: &ProviderCase) -> ScriptedResponse {
    match case.api_backend {
        ApiBackend::ChatCompletions => ScriptedResponse::sse(
            sse::chat_completions_reasoning_then_tool_call_events(
                "reasoning first",
                "call_1",
                "read",
                "{}",
                case.upstream_model,
            ),
        ),
        ApiBackend::Messages if matches!(case.body, BodyContract::Radius) => {
            ScriptedResponse::sse(vec![
                SseEvent::data(json!({"type":"start"}).to_string()),
                SseEvent::data(json!({"type":"thinking_start","contentIndex":0}).to_string()),
                SseEvent::data(
                    json!({"type":"thinking_delta","contentIndex":0,"delta":"reasoning first"})
                        .to_string(),
                ),
                SseEvent::data(
                    json!({"type":"thinking_end","contentIndex":0,"content":"reasoning first","contentSignature":"sig"})
                        .to_string(),
                ),
                SseEvent::data(json!({"type":"text_start","contentIndex":1}).to_string()),
                SseEvent::data(
                    json!({"type":"text_delta","contentIndex":1,"delta":"hello"}).to_string(),
                ),
                SseEvent::data(
                    json!({"type":"text_end","contentIndex":1,"content":"hello"}).to_string(),
                ),
                SseEvent::data(
                    json!({"type":"done","reason":"stop","usage":{"inputTokens":2,"outputTokens":3}})
                        .to_string(),
                ),
            ])
        }
        ApiBackend::Messages => ScriptedResponse::sse(sse::messages_api_script(
            "hello",
            case.upstream_model,
            "end_turn",
        )),
        ApiBackend::Responses => unreachable!(),
    }
}

fn assert_auth(case: &ProviderCase, headers: &BTreeMap<String, String>, key: &str) {
    match case.auth_scheme {
        AuthScheme::Bearer => {
            let expected = format!("Bearer {key}");
            assert_eq!(
                headers.get("authorization").map(String::as_str),
                Some(expected.as_str())
            );
            assert!(!headers.contains_key("x-api-key"));
            assert!(!headers.contains_key("api-key"));
        }
        AuthScheme::XApiKey => {
            assert_eq!(headers.get("x-api-key").map(String::as_str), Some(key));
            assert!(!headers.contains_key("authorization"));
            assert!(!headers.contains_key("api-key"));
        }
    }
}

fn assert_common_headers(case: &ProviderCase, captured: &CapturedRequest, key: &str) {
    assert_auth(case, &captured.headers, key);
    assert_eq!(
        captured.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        captured.headers.get("accept").map(String::as_str),
        Some("text/event-stream")
    );
    assert_eq!(
        captured.headers.get("traceparent").map(String::as_str),
        Some("00-provider-matrix-safe")
    );
    for private in [
        "x-grok-conv-id",
        "x-xai-token-auth",
        "x-grok-client-identifier",
    ] {
        assert!(
            !captured.headers.contains_key(private),
            "{} leaked {private}",
            case.name
        );
    }
    for (name, value) in case.expected_provider_headers {
        assert_eq!(
            captured.headers.get(*name).map(String::as_str),
            Some(*value),
            "{} header {name}",
            case.name
        );
    }
    if matches!(case.body, BodyContract::Copilot) {
        assert_eq!(
            captured
                .headers
                .get("anthropic-version")
                .map(String::as_str),
            Some("foreign-value")
        );
    }
    if !matches!(
        case.body,
        BodyContract::Anthropic | BodyContract::Kimi | BodyContract::Copilot
    ) {
        assert!(
            !captured.headers.contains_key("anthropic-version"),
            "{} kept foreign Anthropic header",
            case.name
        );
    }

    let mut expected = BTreeMap::from([
        ("accept".to_owned(), "text/event-stream".to_owned()),
        ("content-type".to_owned(), "application/json".to_owned()),
        (
            "traceparent".to_owned(),
            "00-provider-matrix-safe".to_owned(),
        ),
    ]);
    expected.insert(
        match case.auth_scheme {
            AuthScheme::Bearer => "authorization".to_owned(),
            AuthScheme::XApiKey => "x-api-key".to_owned(),
        },
        match case.auth_scheme {
            AuthScheme::Bearer => format!("Bearer {key}"),
            AuthScheme::XApiKey => key.to_owned(),
        },
    );
    for (name, value) in case.expected_provider_headers {
        expected.insert((*name).to_owned(), (*value).to_owned());
    }
    if matches!(case.body, BodyContract::Copilot) {
        expected.insert("anthropic-version".to_owned(), "foreign-value".to_owned());
    }

    let actual = captured
        .headers
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                "host" | "content-length" | "user-agent" | "accept-encoding"
            )
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(actual, expected, "{} complete final header set", case.name);
}

fn assert_body(case: &ProviderCase, body: &Value) {
    assert_eq!(
        body["model"], case.upstream_model,
        "{} upstream model",
        case.name
    );
    for private in [
        "x_grok_conv_id",
        "x_grok_req_id",
        "x_grok_session_id",
        "x_grok_turn_idx",
        "x_grok_agent_id",
    ] {
        assert!(
            body.get(private).is_none(),
            "{} body leaked {private}",
            case.name
        );
    }
    match case.body {
        BodyContract::Anthropic | BodyContract::Kimi => {
            assert_eq!(body["messages"][0]["role"], "user");
            assert_eq!(body["max_tokens"], 4_096);
            assert!(body.get("reasoning_effort").is_none());
        }
        BodyContract::Copilot => {
            assert_eq!(body["messages"][0]["role"], "system");
            assert_eq!(body["max_tokens"], 4_096);
            assert!(body.get("reasoning").is_none());
        }
        BodyContract::Openrouter => {
            assert_eq!(body["messages"][0]["role"], "developer");
            assert_eq!(body["max_completion_tokens"], 4_096);
            assert!(body.get("max_tokens").is_none());
            assert_eq!(body["reasoning"], json!({"effort":"xhigh"}));
            assert!(body.get("reasoning_effort").is_none());
        }
        BodyContract::Radius => {
            assert_eq!(body["context"]["systemPrompt"], "system prompt");
            assert_eq!(body["context"]["messages"][0]["role"], "user");
            assert_eq!(body["options"]["maxTokens"], 4_096);
            assert_eq!(body["options"]["reasoning"], "high");
            assert_eq!(body["options"]["sessionId"], "private-session");
        }
        BodyContract::Deepseek => {
            assert_eq!(body["messages"][0]["role"], "system");
            assert_eq!(body["max_completion_tokens"], 384_000);
            assert!(body.get("max_tokens").is_none());
            assert_eq!(body["thinking"], json!({"type":"enabled"}));
            assert_eq!(body["reasoning_effort"], "high");
            assert!(body.get("store").is_none());
        }
        BodyContract::Zai => {
            assert_eq!(body["messages"][0]["role"], "system");
            assert_eq!(body["max_tokens"], 131_072);
            assert!(body.get("max_completion_tokens").is_none());
            assert_eq!(
                body["thinking"],
                json!({"type":"enabled","clear_thinking":false})
            );
            assert_eq!(body["reasoning_effort"], "high");
            assert_eq!(body["tool_stream"], true);
            assert!(body.get("store").is_none());
        }
    }
}

fn assert_stream_order(case: &ProviderCase, response: &ConversationResponse) {
    if matches!(case.api_backend, ApiBackend::ChatCompletions) {
        let reasoning_index = response
            .items
            .iter()
            .position(|item| matches!(item, ConversationItem::Reasoning(_)))
            .expect("reasoning item");
        let assistant_index = response
            .items
            .iter()
            .position(|item| matches!(item, ConversationItem::Assistant(_)))
            .expect("assistant item");
        assert!(
            reasoning_index < assistant_index,
            "{} stream order",
            case.name
        );
        let assistant = response.assistant().unwrap();
        assert_eq!(assistant.tool_calls.len(), 1);
        assert_eq!(assistant.tool_calls[0].name, "read");
    } else {
        assert_eq!(
            response.assistant_text(),
            "hello",
            "{} message stream",
            case.name
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_batch_providers_route_wire_and_stream_through_http() {
    for case in CASES.iter().cloned() {
        let server = CaptureServer::start(&[case.endpoint_path], response(&case)).await;
        let key = format!("matrix-secret-{}", case.name);
        let cfg = config(&case, &server.origin, &key);
        let client = SamplingClient::new_with_route(cfg, ProviderRouteHint::Known(case.provider))
            .unwrap_or_else(|error| panic!("{} client: {error}", case.name));
        assert_eq!(
            client.endpoint_url(match case.api_backend {
                ApiBackend::ChatCompletions => "chat/completions",
                ApiBackend::Messages => "messages",
                ApiBackend::Responses => unreachable!(),
            }),
            format!("{}{}", server.origin, case.endpoint_path),
            "{} final URL",
            case.name
        );

        let response = client
            .conversation_collect(request(&case))
            .await
            .unwrap_or_else(|error| panic!("{} stream: {error}", case.name));
        assert_stream_order(&case, &response);
        let captured = server.take_one();
        assert_eq!(captured.path, case.endpoint_path);
        assert_common_headers(&case, &captured, &key);
        assert_body(&case, &captured.body);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_wire_scheme_truth_table_reaches_http() {
    const API_HEADERS: &[(&str, &str)] = &[("anthropic-version", "2023-06-01")];
    const OAUTH_HEADERS: &[(&str, &str)] = &[
        ("anthropic-version", "2023-06-01"),
        ("anthropic-beta", "claude-code-20250219,oauth-2025-04-20"),
        ("x-app", "cli"),
    ];
    let base = CASES
        .iter()
        .find(|case| case.name == "anthropic")
        .expect("Anthropic case");

    for (name, key, auth_scheme, expected_headers) in [
        (
            "x-api-key-wire",
            "plain-anthropic-key",
            AuthScheme::XApiKey,
            API_HEADERS,
        ),
        (
            "bearer-oauth-wire",
            "oauth-token",
            AuthScheme::Bearer,
            OAUTH_HEADERS,
        ),
        (
            "bearer-auth-token-wire",
            "opaque-token",
            AuthScheme::Bearer,
            OAUTH_HEADERS,
        ),
        (
            "bearer-oat-wire",
            "prefix-sk-ant-oat-middle",
            AuthScheme::Bearer,
            OAUTH_HEADERS,
        ),
    ] {
        let mut case = base.clone();
        case.name = name;
        case.auth_scheme = auth_scheme;
        case.expected_provider_headers = expected_headers;
        let server = CaptureServer::start(
            &["/v1/messages"],
            ScriptedResponse::sse(sse::messages_api_script(
                "anthropic answer",
                case.upstream_model,
                "end_turn",
            )),
        )
        .await;
        let mut cfg = config(&case, &server.origin, key);
        if auth_scheme == AuthScheme::Bearer {
            cfg.extra_headers.insert(
                "anthropic-beta".to_owned(),
                "claude-code-20250219,oauth-2025-04-20".to_owned(),
            );
            cfg.extra_headers
                .insert("x-app".to_owned(), "cli".to_owned());
            cfg.extra_headers
                .insert("user-agent".to_owned(), "claude-cli/2.1.75".to_owned());
        }
        let client =
            SamplingClient::new_with_route(cfg, ProviderRouteHint::Known(KnownProvider::Anthropic))
                .expect("Anthropic client");
        let response = client
            .conversation_collect(request(&case))
            .await
            .expect("Anthropic response");
        assert_eq!(response.assistant_text(), "anthropic answer");
        let captured = server.captured.lock().unwrap();
        assert_eq!(captured.len(), 1, "{name}");
        assert_common_headers(&case, &captured[0], key);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_batch_provider_401s_classify_sent_credentials() {
    for case in CASES.iter().cloned() {
        let unauthorized = ScriptedResponse::json(
            StatusCode::UNAUTHORIZED.as_u16(),
            json!({"error":{"type":"authentication_error","message":"rejected"}}),
        );
        let server = CaptureServer::start(&[case.endpoint_path], unauthorized).await;
        let cfg = config(&case, &server.origin, "matrix-rejected-key");
        let client = SamplingClient::new_with_route(cfg, ProviderRouteHint::Known(case.provider))
            .unwrap_or_else(|error| panic!("{} client: {error}", case.name));
        let error = client
            .conversation_collect(request(&case))
            .await
            .expect_err("401 must fail");
        match error {
            SamplingError::Auth { credential, .. } => {
                assert_eq!(credential, SentCredential::Sent, "{} 401", case.name);
            }
            other => panic!("{} expected Auth, got {other:?}", case.name),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_provider_credentials_fail_locally_without_sending_parent_fallback() {
    for case in CASES.iter().cloned() {
        let server = CaptureServer::start(
            &[case.endpoint_path],
            ScriptedResponse::json(StatusCode::OK.as_u16(), json!({})),
        )
        .await;
        let mut cfg = config(&case, &server.origin, "unused");
        cfg.api_key = None;
        let error = SamplingClient::new_with_route(cfg, ProviderRouteHint::Known(case.provider))
            .expect_err("known provider without credentials must fail before HTTP");
        match error {
            SamplingError::Auth { credential, .. } => {
                assert_eq!(
                    credential,
                    SentCredential::Unknown,
                    "{} local auth",
                    case.name
                );
            }
            other => panic!("{} expected Auth, got {other:?}", case.name),
        }
        assert_eq!(
            server.captured.lock().unwrap().len(),
            0,
            "{} must not send an unauthenticated or parent-authenticated request",
            case.name
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_http_errors_and_stream_frames_never_enter_trace_logs() {
    const RAW: &str = "sentinel-prefix-raw-value-sentinel-suffix";
    const PREFIX: &str = "sentinel-prefix";
    const SUFFIX: &str = "sentinel-suffix";
    const EDGE_PREFIX: &str = "sentinel";
    const EDGE_SUFFIX: &str = "l-suffix";
    const SHORT: &str = "xy";
    const QUERY: &str = "unlabeled-query-secret";
    const ENCODED_QUERY: &str = "encoded%2Fquery-secret";
    const DECODED_QUERY: &str = "encoded/query-secret";
    const URL_USER: &str = "url-user-secret";
    const URL_PASSWORD: &str = "url-password-secret";
    const EXTRA_HEADER: &str = "extra-header-secret";

    let logs = CapturedLogs::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .without_time()
        .with_writer(logs.clone())
        .finish();

    async {
        let case = CASES
            .iter()
            .find(|case| case.name == "deepseek")
            .expect("DeepSeek case")
            .clone();
        let error_response = ScriptedResponse::json(
            StatusCode::BAD_REQUEST.as_u16(),
            json!({
                "error": {
                    "type": "provider_error",
                    "message": format!(
                        "upstream rejected {RAW}; retained_prefix={EDGE_PREFIX}; retained_suffix={EDGE_SUFFIX}; model={}; URL https://{URL_USER}:{URL_PASSWORD}@host/v1?key={RAW}&tenant={QUERY}&opaque={ENCODED_QUERY}; decoded={DECODED_QUERY}; header={EXTRA_HEADER}",
                        case.upstream_model
                    )
                }
            }),
        );
        let server = CaptureServer::start(&["/chat/completions"], error_response).await;
        let mut error_config = config(&case, &server.origin, RAW);
        error_config
            .query_params
            .insert("tenant".to_owned(), QUERY.to_owned());
        error_config.base_url = format!(
            "{}?opaque={ENCODED_QUERY}",
            error_config
                .base_url
                .replacen("http://", &format!("http://{URL_USER}:{URL_PASSWORD}@"), 1)
        );
        error_config
            .extra_headers
            .insert("user-agent".to_owned(), EXTRA_HEADER.to_owned());
        let client =
            SamplingClient::new_with_route(error_config, ProviderRouteHint::Known(case.provider))
                .expect("client");
        let error = client
            .conversation_collect(request(&case))
            .await
            .expect_err("HTTP error");
        for fragment in [
            RAW,
            PREFIX,
            SUFFIX,
            EDGE_PREFIX,
            EDGE_SUFFIX,
            QUERY,
            ENCODED_QUERY,
            DECODED_QUERY,
            URL_USER,
            URL_PASSWORD,
            EXTRA_HEADER,
            case.upstream_model,
        ] {
            assert!(
                !error.to_string().contains(fragment),
                "returned HTTP error leaked {fragment:?}: {error}"
            );
        }

        let stream_response = ScriptedResponse::sse(vec![SseEvent::data(
            json!({
                "error": {
                    "type": "provider_error",
                    "message": format!("key sent was {RAW}")
                }
            })
            .to_string(),
        )]);
        let stream_server = CaptureServer::start(&["/chat/completions"], stream_response).await;
        let stream_client = SamplingClient::new_with_route(
            config(&case, &stream_server.origin, RAW),
            ProviderRouteHint::Known(case.provider),
        )
        .expect("stream client");
        let stream_error = stream_client
            .conversation_collect(request(&case))
            .await
            .expect_err("stream error");
        for fragment in [RAW, PREFIX, SUFFIX] {
            assert!(
                !stream_error.to_string().contains(fragment),
                "returned stream error leaked {fragment:?}: {stream_error}"
            );
        }

        let short_response = ScriptedResponse::json(
            StatusCode::BAD_REQUEST.as_u16(),
            json!({"error":{"type":"provider_error","message":format!("bad credential: {SHORT}")}}),
        );
        let short_server = CaptureServer::start(&["/chat/completions"], short_response).await;
        let short_client = SamplingClient::new_with_route(
            config(&case, &short_server.origin, SHORT),
            ProviderRouteHint::Known(case.provider),
        )
        .expect("short-key client");
        let short_error = short_client
            .conversation_collect(request(&case))
            .await
            .expect_err("short-key HTTP error");
        assert!(
            !short_error.to_string().contains(SHORT),
            "returned short-key error leaked: {short_error}"
        );
    }
    .instrument(tracing::info_span!("provider_secret_log_test"))
    .with_subscriber(subscriber)
    .await;

    let captured = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    for fragment in [
        RAW,
        PREFIX,
        SUFFIX,
        EDGE_PREFIX,
        EDGE_SUFFIX,
        SHORT,
        QUERY,
        ENCODED_QUERY,
        DECODED_QUERY,
        URL_USER,
        URL_PASSWORD,
        EXTRA_HEADER,
        "deepseek-v4-flash",
    ] {
        assert!(
            !captured.contains(fragment),
            "trace logs leaked {fragment:?}: {captured}"
        );
    }
}

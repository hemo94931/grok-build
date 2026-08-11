use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
use serde_json::Value;
use xai_grok_sampling_types::{ApiBackend, messages};

use crate::config::AuthScheme;

mod compat;
mod radius;
pub(crate) use radius::{PiMessagesEventDecoder, radius_payload};

use compat::{
    BaseUrlProfile, ResolvedProviderModelWireFacts, apply_chat_completions_compat,
    base_url_profile, generated_wire_fact, resolve_wire_facts,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnownProvider {
    Anthropic,
    OpenaiCodex,
    GithubCopilot,
    Openrouter,
    KimiCoding,
    Radius,
}

impl KnownProvider {
    fn namespace(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenaiCodex => "openai-codex",
            Self::GithubCopilot => "github-copilot",
            Self::Openrouter => "openrouter",
            Self::KimiCoding => "kimi-coding",
            Self::Radius => "radius",
        }
    }
}

/// Provider-route provenance supplied alongside `SamplerConfig` without
/// adding provider fields to shared sampling types.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProviderRouteHint {
    /// Safe compatibility fallback for sampler-only callers: a namespaced
    /// model is considered known only when its parsed base URL matches that
    /// provider. Otherwise it remains a custom third-party route.
    #[default]
    Auto,
    Known(KnownProvider),
    CustomThirdParty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProviderKind {
    Anthropic,
    OpenaiCodex,
    GithubCopilot,
    Openrouter,
    KimiCoding,
    Radius,
    CustomThirdParty,
}

impl From<KnownProvider> for ProviderKind {
    fn from(provider: KnownProvider) -> Self {
        match provider {
            KnownProvider::Anthropic => Self::Anthropic,
            KnownProvider::OpenaiCodex => Self::OpenaiCodex,
            KnownProvider::GithubCopilot => Self::GithubCopilot,
            KnownProvider::Openrouter => Self::Openrouter,
            KnownProvider::KimiCoding => Self::KimiCoding,
            KnownProvider::Radius => Self::Radius,
        }
    }
}

impl ProviderKind {
    fn from_namespace(namespace: &str) -> Option<Self> {
        match namespace {
            "anthropic" => Some(Self::Anthropic),
            "openai-codex" => Some(Self::OpenaiCodex),
            "github-copilot" => Some(Self::GithubCopilot),
            "openrouter" => Some(Self::Openrouter),
            "kimi-coding" => Some(Self::KimiCoding),
            "radius" => Some(Self::Radius),
            _ => None,
        }
    }

    fn provider_name(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("anthropic"),
            Self::OpenaiCodex => Some("openai-codex"),
            Self::GithubCopilot => Some("github-copilot"),
            Self::Openrouter => Some("openrouter"),
            Self::KimiCoding => Some("kimi-coding"),
            Self::Radius => Some("radius"),
            Self::CustomThirdParty => None,
        }
    }

    fn matches_base_url(self, base_url: &str) -> bool {
        let Ok(url) = reqwest::Url::parse(base_url) else {
            return false;
        };
        let Some(host) = url.host_str().map(str::to_ascii_lowercase) else {
            return false;
        };
        match self {
            Self::Anthropic => host == "api.anthropic.com" || host.ends_with(".anthropic.com"),
            Self::OpenaiCodex => host == "chatgpt.com" || host.ends_with(".chatgpt.com"),
            Self::GithubCopilot => {
                host == "githubcopilot.com" || host.ends_with(".githubcopilot.com")
            }
            Self::Openrouter => host == "openrouter.ai" || host.ends_with(".openrouter.ai"),
            Self::KimiCoding => host == "api.kimi.com" || host.ends_with(".kimi.com"),
            Self::Radius => host == "radius.pi.dev" || host.ends_with(".radius.pi.dev"),
            Self::CustomThirdParty => false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ProviderWireRoute {
    kind: ProviderKind,
    upstream_model: String,
    base_ends_in_v1: bool,
    facts: ResolvedProviderModelWireFacts,
}

impl ProviderWireRoute {
    #[cfg(test)]
    pub(crate) fn from_config(model: &str, base_url: &str) -> Option<Self> {
        Self::from_config_with_hint(model, base_url, ProviderRouteHint::Auto)
    }

    pub(crate) fn from_config_with_hint(
        model: &str,
        base_url: &str,
        hint: ProviderRouteHint,
    ) -> Option<Self> {
        let (kind, upstream_model) = match hint {
            ProviderRouteHint::Known(provider) => {
                let upstream = model
                    .split_once('/')
                    .filter(|(namespace, upstream)| {
                        *namespace == provider.namespace() && !upstream.is_empty()
                    })
                    .map_or(model, |(_, upstream)| upstream);
                (ProviderKind::from(provider), upstream)
            }
            ProviderRouteHint::CustomThirdParty => (ProviderKind::CustomThirdParty, model),
            ProviderRouteHint::Auto => {
                if is_first_party_xai_url(base_url) {
                    return None;
                }
                match model.split_once('/') {
                    Some((namespace, upstream))
                        if !upstream.is_empty()
                            && ProviderKind::from_namespace(namespace)
                                .is_some_and(|kind| kind.matches_base_url(base_url)) =>
                    {
                        (
                            ProviderKind::from_namespace(namespace)
                                .expect("guard checked known provider namespace"),
                            upstream,
                        )
                    }
                    _ => (ProviderKind::CustomThirdParty, model),
                }
            }
        };
        let base_ends_in_v1 = reqwest::Url::parse(base_url)
            .ok()
            .is_some_and(|url| url.path().trim_end_matches('/').ends_with("/v1"));
        let fact_provider = match (kind.provider_name(), base_url_profile(base_url)) {
            (_, Some(BaseUrlProfile::Openrouter)) => Some("openrouter"),
            (provider, _) => provider,
        };
        let explicit =
            fact_provider.and_then(|provider| generated_wire_fact(provider, upstream_model));
        let facts = resolve_wire_facts(kind.provider_name(), base_url, upstream_model, explicit);
        Some(Self {
            kind,
            upstream_model: upstream_model.to_owned(),
            base_ends_in_v1,
            facts,
        })
    }

    pub(crate) fn upstream_model(&self) -> &str {
        &self.upstream_model
    }

    #[cfg(test)]
    fn kind(&self) -> ProviderKind {
        self.kind
    }

    pub(crate) fn is_known_provider(&self) -> bool {
        self.kind != ProviderKind::CustomThirdParty
    }

    pub(crate) fn is_radius(&self) -> bool {
        self.kind == ProviderKind::Radius
    }

    pub(crate) fn endpoint_path<'a>(&self, default: &'a str, backend: &ApiBackend) -> &'a str {
        match (self.kind, backend, default) {
            (ProviderKind::OpenaiCodex, ApiBackend::Responses, "responses") => "codex/responses",
            // The standalone compaction endpoint lives under the same
            // `codex/` prefix on the ChatGPT backend; the unprefixed path
            // answers 404 with the ChatGPT front-end error page.
            (ProviderKind::OpenaiCodex, ApiBackend::Responses, "responses/compact") => {
                "codex/responses/compact"
            }
            (
                ProviderKind::Anthropic | ProviderKind::GithubCopilot | ProviderKind::KimiCoding,
                ApiBackend::Messages,
                "messages",
            ) if !self.base_ends_in_v1 => "v1/messages",
            _ => default,
        }
    }

    pub(crate) fn sanitize_headers(&self, headers: &mut HeaderMap) {
        let rejected = headers
            .keys()
            .filter(|name| !self.header_allowed(name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        for name in rejected {
            headers.remove(name);
        }
    }

    fn header_allowed(&self, name: &str) -> bool {
        let name = name.to_ascii_lowercase();
        if matches!(
            name.as_str(),
            "content-type"
                | "accept"
                | "authorization"
                | "x-api-key"
                | "api-key"
                | "user-agent"
                | "traceparent"
                | "tracestate"
                | "baggage"
        ) {
            return true;
        }
        match self.kind {
            ProviderKind::Anthropic => matches!(
                name.as_str(),
                "anthropic-version" | "anthropic-beta" | "x-app"
            ),
            ProviderKind::OpenaiCodex => matches!(
                name.as_str(),
                "chatgpt-account-id" | "originator" | "openai-beta" | "session-id"
            ),
            ProviderKind::GithubCopilot => matches!(
                name.as_str(),
                "editor-version"
                    | "editor-plugin-version"
                    | "copilot-integration-id"
                    | "x-initiator"
                    | "openai-intent"
                    | "copilot-vision-request"
                    | "anthropic-version"
                    | "anthropic-beta"
            ),
            ProviderKind::Openrouter => {
                matches!(name.as_str(), "http-referer" | "x-title")
            }
            ProviderKind::KimiCoding => name == "anthropic-version",
            ProviderKind::Radius => false,
            ProviderKind::CustomThirdParty => !is_xai_private_header(&name),
        }
    }

    pub(crate) fn add_dynamic_headers<T: Serialize>(&self, body: &T, headers: &mut HeaderMap) {
        if self.kind != ProviderKind::GithubCopilot {
            return;
        }
        let Ok(value) = serde_json::to_value(body) else {
            return;
        };
        let role = value
            .get("messages")
            .or_else(|| value.get("input"))
            .and_then(Value::as_array)
            .and_then(|items| items.last())
            .and_then(|item| item.get("role"))
            .and_then(Value::as_str);
        headers.insert(
            HeaderName::from_static("x-initiator"),
            HeaderValue::from_static(if role == Some("user") {
                "user"
            } else {
                "agent"
            }),
        );
        headers.insert(
            HeaderName::from_static("openai-intent"),
            HeaderValue::from_static("conversation-edits"),
        );
        if contains_image(&value) {
            headers.insert(
                HeaderName::from_static("copilot-vision-request"),
                HeaderValue::from_static("true"),
            );
        }
    }

    pub(crate) fn sanitize_body(&self, body: &mut Value, backend: &ApiBackend) {
        let Some(object) = body.as_object_mut() else {
            return;
        };
        object.insert(
            "model".to_owned(),
            Value::String(self.upstream_model.clone()),
        );
        for key in [
            "x_grok_conv_id",
            "x_grok_req_id",
            "x_grok_session_id",
            "x_grok_turn_idx",
            "x_grok_agent_id",
            "x_grok_deployment_id",
            "x_grok_user_id",
            "trace",
            "stream_tool_calls",
            "cache_family",
        ] {
            object.remove(key);
        }
        if *backend == ApiBackend::ChatCompletions {
            apply_chat_completions_compat(object, &self.upstream_model, &self.facts);
        }
        if *backend == ApiBackend::Responses {
            if self.kind == ProviderKind::OpenaiCodex {
                object.insert("store".to_owned(), Value::Bool(false));
                object.insert(
                    "include".to_owned(),
                    serde_json::json!(["reasoning.encrypted_content"]),
                );
                object.remove("previous_response_id");
                // The ChatGPT Codex backend rejects each of these with HTTP
                // 400 ("Unsupported parameter: <name>"; verified against the
                // live endpoint). System-role items in `input` are likewise
                // refused ("System messages are not allowed") and must travel
                // in the top-level `instructions` field instead.
                for key in [
                    "includeSystemPrompt",
                    "max_output_tokens",
                    "max_tool_calls",
                    "temperature",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stream_options",
                    "truncation",
                    "metadata",
                    "safety_identifier",
                    "service_tier",
                    "background",
                ] {
                    object.remove(key);
                }
                lift_system_messages_into_instructions(object);
            }
            if let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) {
                input.retain(|item| item.get("type").and_then(Value::as_str) != Some("compaction"));
            }
        }
        if let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) {
            tools.retain(|tool| tool.get("type").and_then(Value::as_str) != Some("x_search"));
        }
    }

    /// Standalone-compaction bodies bypass [`sanitize_body`](Self::sanitize_body)
    /// (they are sealed upstream), so provider body fixes for
    /// `responses/compact` live here. The `model` rewrite mirrors
    /// `sanitize_body`: the sealed body carries the namespaced catalog id,
    /// which the ChatGPT backend rejects
    /// ("not supported when using Codex with a ChatGPT account").
    /// Additionally, verified live: `service_tier`, `prompt_cache_retention`,
    /// and `prompt_cache_options` each answer HTTP 400 there.
    pub(crate) fn sanitize_compact_body(&self, body: &mut Value) {
        let Some(object) = body.as_object_mut() else {
            return;
        };
        object.insert(
            "model".to_owned(),
            Value::String(self.upstream_model.clone()),
        );
        if self.kind != ProviderKind::OpenaiCodex {
            return;
        }
        for key in [
            "service_tier",
            "prompt_cache_retention",
            "prompt_cache_options",
        ] {
            object.remove(key);
        }
    }

    pub(crate) fn prepare_messages(
        &self,
        request: &mut messages::MessagesRequest,
        auth_scheme: AuthScheme,
    ) -> Vec<(String, String)> {
        request.model.clone_from(&self.upstream_model);
        if self.kind != ProviderKind::Anthropic || auth_scheme != AuthScheme::Bearer {
            return Vec::new();
        }
        prepend_claude_code_identity(&mut request.system);
        let mut names = Vec::new();
        if let Some(tools) = request.tools.as_mut() {
            for tool in tools {
                let original = tool.name.clone();
                let canonical = claude_code_tool_name(&original).to_owned();
                if canonical != original {
                    names.push((canonical.clone(), original));
                    tool.name = canonical;
                }
            }
        }
        if let Some(messages::ToolChoiceParam::Tool { name }) = request.tool_choice.as_mut() {
            *name = claude_code_tool_name(name).to_owned();
        }
        for message in &mut request.messages {
            if let messages::MessageContent::Blocks(blocks) = &mut message.content {
                canonicalize_content_blocks(blocks);
            }
        }
        names
    }

    pub(crate) fn restore_message_response(
        &self,
        response: &mut messages::MessagesResponse,
        names: &[(String, String)],
    ) {
        for block in &mut response.content {
            restore_content_block(block, names);
        }
    }

    pub(crate) fn restore_message_event(
        &self,
        event: &mut messages::MessageStreamEvent,
        names: &[(String, String)],
    ) {
        match event {
            messages::MessageStreamEvent::MessageStart { message } => {
                self.restore_message_response(message, names);
            }
            messages::MessageStreamEvent::ContentBlockStart { content_block, .. } => {
                restore_content_block(content_block, names);
            }
            _ => {}
        }
    }
}

/// Move `input` items of `{"type": "message", "role": "system"}` into the
/// top-level `instructions` string, which is the only channel for system
/// prompts the ChatGPT Codex backend accepts. Pre-existing `instructions`
/// keep first position; hoisted bodies join with "\n\n".
fn lift_system_messages_into_instructions(object: &mut serde_json::Map<String, Value>) {
    let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) else {
        return;
    };
    let mut hoisted: Vec<String> = Vec::new();
    input.retain(|item| {
        let is_system = item.get("type").and_then(Value::as_str) == Some("message")
            && item.get("role").and_then(Value::as_str) == Some("system");
        if is_system {
            hoisted.push(system_message_text(item));
        }
        !is_system
    });
    let mut parts: Vec<String> = Vec::new();
    if let Some(existing) = object.get("instructions").and_then(Value::as_str) {
        if !existing.trim().is_empty() {
            parts.push(existing.to_owned());
        }
    }
    parts.extend(hoisted.into_iter().filter(|text| !text.trim().is_empty()));
    if parts.is_empty() {
        return;
    }
    object.insert("instructions".to_owned(), Value::String(parts.join("\n\n")));
}

/// System message content arrives either as a plain string or as an array of
/// content parts; flatten both into text.
fn system_message_text(item: &Value) -> String {
    match item.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn prepend_claude_code_identity(system: &mut Option<messages::SystemParam>) {
    const IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
    let identity = messages::TextBlock {
        r#type: "text".to_owned(),
        text: IDENTITY.to_owned(),
        cache_control: None,
    };
    match system.take() {
        None => *system = Some(messages::SystemParam::Blocks(vec![identity])),
        Some(messages::SystemParam::Text(text)) => {
            *system = Some(messages::SystemParam::Blocks(vec![
                identity,
                messages::TextBlock {
                    r#type: "text".to_owned(),
                    text,
                    cache_control: None,
                },
            ]));
        }
        Some(messages::SystemParam::Blocks(mut blocks)) => {
            if blocks.first().is_none_or(|block| block.text != IDENTITY) {
                blocks.insert(0, identity);
            }
            *system = Some(messages::SystemParam::Blocks(blocks));
        }
    }
}

fn canonicalize_content_blocks(blocks: &mut [messages::ContentBlock]) {
    for block in blocks {
        match block {
            messages::ContentBlock::ToolUse { name, .. } => {
                *name = claude_code_tool_name(name).to_owned();
            }
            messages::ContentBlock::ToolResult {
                content: messages::ToolResultContent::Blocks(nested),
                ..
            } => canonicalize_content_blocks(nested),
            _ => {}
        }
    }
}

fn restore_content_block(block: &mut messages::ContentBlock, names: &[(String, String)]) {
    if let messages::ContentBlock::ToolUse { name, .. } = block
        && let Some((_, original)) = names
            .iter()
            .find(|(canonical, _)| canonical.eq_ignore_ascii_case(name))
    {
        name.clone_from(original);
    }
}

fn claude_code_tool_name(name: &str) -> &str {
    const NAMES: [&str; 17] = [
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Grep",
        "Glob",
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
        "KillShell",
        "NotebookEdit",
        "Skill",
        "Task",
        "TaskOutput",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
    ];
    NAMES
        .into_iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(name))
        .unwrap_or(name)
}

fn is_xai_private_header(name: &str) -> bool {
    name.starts_with("x-grok-")
        || name.starts_with("x-xai-")
        || matches!(
            name,
            "x-authenticateresponse"
                | "x-compaction-at"
                | "x-compactions-remaining"
                | "x-grok-doom-loop-check"
        )
}

fn contains_image(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(contains_image),
        Value::Object(object) => {
            matches!(
                object.get("type").and_then(Value::as_str),
                Some("image" | "image_url" | "input_image")
            ) || object.values().any(contains_image)
        }
        _ => false,
    }
}

fn is_first_party_xai_url(value: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(value) else {
        return false;
    };
    let Some(host) = url.host_str().map(str::to_ascii_lowercase) else {
        return false;
    };
    host == "x.ai" || host.ends_with(".x.ai") || host == "grok.com" || host.ends_with(".grok.com")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaced_routes_keep_slashes_in_upstream_model() {
        let route = ProviderWireRoute::from_config(
            "openrouter/anthropic/claude-sonnet",
            "https://openrouter.ai/api/v1",
        )
        .unwrap();
        assert_eq!(route.upstream_model(), "anthropic/claude-sonnet");
    }

    #[test]
    fn unknown_custom_route_drops_xai_headers_after_trace_injection() {
        let route = ProviderWireRoute::from_config("custom", "https://example.com/v1").unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer safe"));
        headers.insert("traceparent", HeaderValue::from_static("00-test"));
        headers.insert("x-grok-conv-id", HeaderValue::from_static("secret"));
        headers.insert("x-xai-token-auth", HeaderValue::from_static("secret"));
        route.sanitize_headers(&mut headers);
        assert!(headers.contains_key("authorization"));
        assert!(headers.contains_key("traceparent"));
        assert!(!headers.contains_key("x-grok-conv-id"));
        assert!(!headers.contains_key("x-xai-token-auth"));
    }

    fn openrouter_route(model: &str) -> ProviderWireRoute {
        ProviderWireRoute::from_config_with_hint(
            &format!("openrouter/{model}"),
            "https://openrouter.ai/api/v1",
            ProviderRouteHint::Known(KnownProvider::Openrouter),
        )
        .expect("OpenRouter route")
    }

    fn chat_body(model: &str, effort: Option<&str>) -> Value {
        let mut body = serde_json::json!({
            "model": format!("openrouter/{model}"),
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"}
            ]
        });
        if let Some(effort) = effort {
            body.as_object_mut().unwrap().insert(
                "reasoning_effort".to_owned(),
                Value::String(effort.to_owned()),
            );
        }
        body
    }

    #[test]
    fn openrouter_reasoning_off_unset_and_explicit_none_match() {
        let route = openrouter_route("openai/gpt-5.2");
        for effort in [None, Some("none")] {
            let mut body = chat_body("openai/gpt-5.2", effort);
            route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
            assert_eq!(body["reasoning"], serde_json::json!({"effort": "none"}));
            assert!(body.get("reasoning_effort").is_none());
        }
    }

    #[test]
    fn openrouter_reasoning_uses_thinking_level_map() {
        let route = openrouter_route("openai/gpt-5.2");
        let mut body = chat_body("openai/gpt-5.2", Some("xhigh"));
        route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
        assert_eq!(body["reasoning"], serde_json::json!({"effort": "xhigh"}));
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn openrouter_explicit_null_off_omits_reasoning() {
        let route = openrouter_route("anthropic/claude-fable-5");
        for effort in [None, Some("none")] {
            let mut body = chat_body("anthropic/claude-fable-5", effort);
            route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
            assert!(body.get("reasoning").is_none());
            assert!(body.get("reasoning_effort").is_none());
        }
    }

    #[test]
    fn openrouter_nonreasoning_model_omits_reasoning() {
        let route = openrouter_route("ai21/jamba-large-1.7");
        let mut body = chat_body("ai21/jamba-large-1.7", Some("high"));
        route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
        assert!(body.get("reasoning").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn openrouter_uses_only_max_completion_tokens() {
        let route = openrouter_route("openai/gpt-5.2");
        let mut body = chat_body("openai/gpt-5.2", None);
        route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
        assert_eq!(body["max_completion_tokens"], 4096);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn openrouter_developer_role_truth_table_is_effort_independent() {
        let cases = [
            ("anthropic/claude-opus-4.6", true),
            ("openai/gpt-5.2", true),
            ("deepseek/deepseek-r1", false),
            ("anthropic/claude-3-haiku", false),
            ("openai/gpt-3.5-turbo", false),
        ];
        for (model, developer) in cases {
            for effort in [None, Some("high")] {
                let route = openrouter_route(model);
                let mut body = chat_body(model, effort);
                route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
                let role = body["messages"][0]["role"].as_str().unwrap();
                assert_eq!(
                    role,
                    if developer { "developer" } else { "system" },
                    "{model} effort={effort:?}"
                );
                assert_eq!(body["messages"][1]["role"], "user");
            }
        }
    }

    #[test]
    fn custom_openrouter_slash_prefix_keeps_custom_route_and_headers() {
        for model in ["anthropic/claude-opus-4.6", "openai/gpt-5.2"] {
            let route = ProviderWireRoute::from_config_with_hint(
                model,
                "https://OPENROUTER.ai/api/v1/",
                ProviderRouteHint::CustomThirdParty,
            )
            .unwrap();
            assert_eq!(route.kind(), ProviderKind::CustomThirdParty);
            assert!(!route.is_known_provider());
            assert_eq!(route.upstream_model(), model);

            let mut headers = HeaderMap::new();
            headers.insert("authorization", HeaderValue::from_static("Bearer custom"));
            headers.insert(
                "anthropic-version",
                HeaderValue::from_static("custom-value"),
            );
            headers.insert("x-grok-conv-id", HeaderValue::from_static("private"));
            route.sanitize_headers(&mut headers);
            assert_eq!(
                headers.get("authorization").unwrap(),
                HeaderValue::from_static("Bearer custom")
            );
            assert!(headers.contains_key("anthropic-version"));
            assert!(!headers.contains_key("x-grok-conv-id"));

            let mut body = chat_body(model, Some("high"));
            route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
            assert_eq!(body["model"], model);
            assert!(body.get("max_completion_tokens").is_some());
        }
    }

    #[test]
    fn auto_route_does_not_promote_openrouter_upstream_prefixes() {
        for model in ["anthropic/claude-opus-4.6", "openai/gpt-5.2"] {
            let route =
                ProviderWireRoute::from_config(model, "https://edge.openrouter.ai/custom/v1/")
                    .unwrap();
            assert_eq!(route.kind(), ProviderKind::CustomThirdParty);
            assert_eq!(route.upstream_model(), model);
        }
    }

    #[test]
    fn non_openrouter_chat_body_keeps_generic_fields_and_system_role() {
        let route = ProviderWireRoute::from_config_with_hint(
            "github-copilot/gpt-4.1",
            "https://api.individual.githubcopilot.com",
            ProviderRouteHint::Known(KnownProvider::GithubCopilot),
        )
        .unwrap();
        let mut body = serde_json::json!({
            "model": "github-copilot/gpt-4.1",
            "max_tokens": 2048,
            "reasoning_effort": "high",
            "messages": [{"role": "system", "content": "prompt"}]
        });
        route.sanitize_body(&mut body, &ApiBackend::ChatCompletions);
        assert_eq!(body["model"], "gpt-4.1");
        assert_eq!(body["max_tokens"], 2048);
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn codex_compact_endpoint_uses_codex_prefix() {
        let route = ProviderWireRoute::from_config(
            "openai-codex/gpt-5.6-luna",
            "https://chatgpt.com/backend-api",
        )
        .unwrap();
        assert_eq!(
            route.endpoint_path("responses", &ApiBackend::Responses),
            "codex/responses"
        );
        assert_eq!(
            route.endpoint_path("responses/compact", &ApiBackend::Responses),
            "codex/responses/compact"
        );
    }

    #[test]
    fn multi_provider_regression_codex_body_matches_contract() {
        let route = ProviderWireRoute::from_config(
            "openai-codex/gpt-5.4",
            "https://chatgpt.com/backend-api",
        )
        .unwrap();
        let mut body = serde_json::json!({
            "model": "openai-codex/gpt-5.4",
            "store": true,
            "previous_response_id": "response-1",
            "max_output_tokens": 128000,
            "includeSystemPrompt": true,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stream_options": {"include_usage": true},
            "truncation": "disabled",
            "max_tool_calls": 5,
            "metadata": {"k": "v"},
            "safety_identifier": "id",
            "service_tier": "auto",
            "background": false,
            "input": [
                {"type": "message", "role": "system", "content": "base prompt"},
                {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "extra"}]},
                {"type": "compaction"},
                {"role": "user", "content": "hello"}
            ],
            "tools": [{"type": "x_search"}, {"type": "function", "name": "bash"}],
            "x_grok_conv_id": "conv"
        });
        route.sanitize_body(&mut body, &ApiBackend::Responses);
        assert_eq!(body["model"], "gpt-5.4");
        assert_eq!(body["store"], false);
        // The Codex backend HTTP-400s on `includeSystemPrompt`, sampling
        // knobs (`temperature`, `top_p`, penalties, ...), and system-role
        // `input` items; system prompts must be lifted into top-level
        // `instructions`. Verified against the live endpoint.
        for key in [
            "includeSystemPrompt",
            "max_output_tokens",
            "max_tool_calls",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stream_options",
            "truncation",
            "metadata",
            "safety_identifier",
            "service_tier",
            "background",
        ] {
            assert!(body.get(key).is_none(), "{key} must be stripped");
        }
        assert_eq!(body["instructions"], "base prompt\n\nextra");
        assert_eq!(
            body["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
        assert!(body.get("previous_response_id").is_none());
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert!(body.get("x_grok_conv_id").is_none());
    }

    #[test]
    fn first_party_xai_route_is_untouched() {
        assert!(ProviderWireRoute::from_config("grok-4", "https://api.x.ai/v1").is_none());
        assert!(
            ProviderWireRoute::from_config("grok-4", "https://cli-chat-proxy.grok.com/v1")
                .is_none()
        );
    }
}

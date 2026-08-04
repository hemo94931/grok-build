use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
use serde_json::Value;
use xai_grok_sampling_types::{ApiBackend, messages};

use crate::config::AuthScheme;

mod radius;
pub(crate) use radius::{PiMessagesEventDecoder, radius_payload};

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

#[derive(Clone, Debug)]
pub(crate) struct ProviderWireRoute {
    kind: ProviderKind,
    upstream_model: String,
    base_ends_in_v1: bool,
}

impl ProviderWireRoute {
    pub(crate) fn from_config(model: &str, base_url: &str) -> Option<Self> {
        let (kind, upstream_model) = match model.split_once('/') {
            Some(("anthropic", upstream)) if !upstream.is_empty() => {
                (ProviderKind::Anthropic, upstream)
            }
            Some(("openai-codex", upstream)) if !upstream.is_empty() => {
                (ProviderKind::OpenaiCodex, upstream)
            }
            Some(("github-copilot", upstream)) if !upstream.is_empty() => {
                (ProviderKind::GithubCopilot, upstream)
            }
            Some(("openrouter", upstream)) if !upstream.is_empty() => {
                (ProviderKind::Openrouter, upstream)
            }
            Some(("kimi-coding", upstream)) if !upstream.is_empty() => {
                (ProviderKind::KimiCoding, upstream)
            }
            Some(("radius", upstream)) if !upstream.is_empty() => (ProviderKind::Radius, upstream),
            _ if !is_first_party_xai_url(base_url) => (ProviderKind::CustomThirdParty, model),
            _ => return None,
        };
        let base_ends_in_v1 = reqwest::Url::parse(base_url)
            .ok()
            .is_some_and(|url| url.path().trim_end_matches('/').ends_with("/v1"));
        Some(Self {
            kind,
            upstream_model: upstream_model.to_owned(),
            base_ends_in_v1,
        })
    }

    pub(crate) fn upstream_model(&self) -> &str {
        &self.upstream_model
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
        if *backend == ApiBackend::Responses {
            if self.kind == ProviderKind::OpenaiCodex {
                object.insert("store".to_owned(), Value::Bool(false));
                object.insert("includeSystemPrompt".to_owned(), Value::Bool(false));
                object.insert(
                    "include".to_owned(),
                    serde_json::json!(["reasoning.encrypted_content"]),
                );
                object.remove("previous_response_id");
            }
            if let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) {
                input.retain(|item| item.get("type").and_then(Value::as_str) != Some("compaction"));
            }
        }
        if let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) {
            tools.retain(|tool| tool.get("type").and_then(Value::as_str) != Some("x_search"));
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
            "input": [{"type": "compaction"}, {"role": "user", "content": "hello"}],
            "tools": [{"type": "x_search"}, {"type": "function", "name": "bash"}],
            "x_grok_conv_id": "conv"
        });
        route.sanitize_body(&mut body, &ApiBackend::Responses);
        assert_eq!(body["model"], "gpt-5.4");
        assert_eq!(body["store"], false);
        assert_eq!(body["includeSystemPrompt"], false);
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

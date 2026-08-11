use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MaxTokensField {
    MaxTokens,
    MaxCompletionTokens,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ThinkingFormat {
    Openai,
    Openrouter,
    Deepseek,
    Zai,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProviderCompat {
    pub(crate) supports_store: bool,
    pub(crate) supports_developer_role: bool,
    pub(crate) supports_reasoning_effort: bool,
    pub(crate) max_tokens_field: MaxTokensField,
    pub(crate) requires_reasoning_content_on_assistant_messages: bool,
    pub(crate) thinking_format: ThinkingFormat,
    pub(crate) zai_tool_stream: bool,
}

impl ProviderCompat {
    fn conservative_default() -> Self {
        Self {
            supports_store: false,
            supports_developer_role: false,
            supports_reasoning_effort: false,
            max_tokens_field: MaxTokensField::MaxTokens,
            requires_reasoning_content_on_assistant_messages: false,
            thinking_format: ThinkingFormat::Openai,
            zai_tool_stream: false,
        }
    }

    fn apply(&mut self, partial: &ProviderCompatFields) {
        if let Some(value) = partial.supports_store {
            self.supports_store = value;
        }
        if let Some(value) = partial.supports_developer_role {
            self.supports_developer_role = value;
        }
        if let Some(value) = partial.supports_reasoning_effort {
            self.supports_reasoning_effort = value;
        }
        if let Some(value) = partial.max_tokens_field {
            self.max_tokens_field = value;
        }
        if let Some(value) = partial.requires_reasoning_content_on_assistant_messages {
            self.requires_reasoning_content_on_assistant_messages = value;
        }
        if let Some(value) = partial.thinking_format {
            self.thinking_format = value;
        }
        if let Some(value) = partial.zai_tool_stream {
            self.zai_tool_stream = value;
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ProviderCompatFields {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) supports_store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) supports_developer_role: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) supports_reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens_field: Option<MaxTokensField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) requires_reasoning_content_on_assistant_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_format: Option<ThinkingFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) zai_tool_stream: Option<bool>,
    /// Explicitly registered from pi-ai but intentionally not consumed by the
    /// ticket-02 body adapter. Cache-control conversion is a separate seam.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control_format: Option<String>,
}

/// A map value is `Some(string)` for an explicit wire mapping and `None` for
/// an explicit JSON null. Absence from the map remains distinguishable from
/// both, which is required for the OpenRouter `off` behavior.
pub(crate) type ThinkingLevelMap = BTreeMap<String, Option<String>>;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ProviderModelWireFacts {
    pub(crate) provider: String,
    pub(crate) id: String,
    pub(crate) reasoning: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) thinking_level_map: ThinkingLevelMap,
    #[serde(default, skip_serializing_if = "ProviderCompatFields::is_empty")]
    pub(crate) compat: ProviderCompatFields,
}

impl ProviderCompatFields {
    fn is_empty(&self) -> bool {
        self.supports_store.is_none()
            && self.supports_developer_role.is_none()
            && self.supports_reasoning_effort.is_none()
            && self.max_tokens_field.is_none()
            && self
                .requires_reasoning_content_on_assistant_messages
                .is_none()
            && self.thinking_format.is_none()
            && self.zai_tool_stream.is_none()
            && self.cache_control_format.is_none()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedProviderModelWireFacts {
    pub(crate) reasoning: bool,
    pub(crate) thinking_level_map: ThinkingLevelMap,
    pub(crate) compat: ProviderCompat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BaseUrlProfile {
    Openrouter,
    Deepseek,
    Zai,
}

/// Parse a base URL once and classify only exact hosts (or their subdomains).
/// Query strings, host case, and trailing slashes do not affect the profile;
/// lookalike suffixes such as `openrouter.ai.example.com` never match.
pub(crate) fn base_url_profile(base_url: &str) -> Option<BaseUrlProfile> {
    let url = reqwest::Url::parse(base_url).ok()?;
    let host = url.host_str()?.to_ascii_lowercase();
    if host_matches(&host, "openrouter.ai") {
        Some(BaseUrlProfile::Openrouter)
    } else if host_matches(&host, "api.deepseek.com") {
        Some(BaseUrlProfile::Deepseek)
    } else if host_matches(&host, "api.z.ai") || host_matches(&host, "open.bigmodel.cn") {
        Some(BaseUrlProfile::Zai)
    } else {
        None
    }
}

fn host_matches(host: &str, expected: &str) -> bool {
    host == expected || host.ends_with(&format!(".{expected}"))
}

fn provider_defaults(provider: Option<&str>, upstream_model: &str) -> ProviderCompatFields {
    match provider {
        Some("openrouter") => openrouter_fields(upstream_model),
        _ => ProviderCompatFields::default(),
    }
}

fn profile_defaults(profile: Option<BaseUrlProfile>, upstream_model: &str) -> ProviderCompatFields {
    match profile {
        Some(BaseUrlProfile::Openrouter) => openrouter_fields(upstream_model),
        Some(BaseUrlProfile::Deepseek) => ProviderCompatFields {
            supports_store: Some(false),
            supports_developer_role: Some(false),
            supports_reasoning_effort: Some(true),
            max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
            requires_reasoning_content_on_assistant_messages: Some(true),
            thinking_format: Some(ThinkingFormat::Deepseek),
            zai_tool_stream: Some(false),
            ..Default::default()
        },
        Some(BaseUrlProfile::Zai) => ProviderCompatFields {
            supports_store: Some(false),
            supports_developer_role: Some(false),
            supports_reasoning_effort: Some(false),
            max_tokens_field: Some(MaxTokensField::MaxTokens),
            requires_reasoning_content_on_assistant_messages: Some(false),
            thinking_format: Some(ThinkingFormat::Zai),
            zai_tool_stream: Some(false),
            ..Default::default()
        },
        None => ProviderCompatFields::default(),
    }
}

fn openrouter_fields(upstream_model: &str) -> ProviderCompatFields {
    ProviderCompatFields {
        supports_store: Some(true),
        supports_developer_role: Some(
            upstream_model.starts_with("anthropic/") || upstream_model.starts_with("openai/"),
        ),
        supports_reasoning_effort: Some(true),
        max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
        requires_reasoning_content_on_assistant_messages: Some(false),
        thinking_format: Some(ThinkingFormat::Openrouter),
        zai_tool_stream: Some(false),
        ..Default::default()
    }
}

/// Resolve model wire facts with per-field precedence:
/// generated model facts > recognized base URL profile > namespaced provider
/// default > conservative generic default.
pub(crate) fn resolve_wire_facts(
    provider: Option<&str>,
    base_url: &str,
    upstream_model: &str,
    explicit: Option<&ProviderModelWireFacts>,
) -> ResolvedProviderModelWireFacts {
    let mut compat = ProviderCompat::conservative_default();
    compat.apply(&provider_defaults(provider, upstream_model));
    compat.apply(&profile_defaults(
        base_url_profile(base_url),
        upstream_model,
    ));
    if let Some(explicit) = explicit {
        compat.apply(&explicit.compat);
    }
    ResolvedProviderModelWireFacts {
        reasoning: explicit.is_some_and(|facts| facts.reasoning),
        thinking_level_map: explicit
            .map(|facts| facts.thinking_level_map.clone())
            .unwrap_or_default(),
        compat,
    }
}

pub(crate) fn generated_wire_fact(
    provider: &str,
    upstream_model: &str,
) -> Option<&'static ProviderModelWireFacts> {
    generated_wire_facts()
        .iter()
        .find(|facts| facts.provider == provider && facts.id == upstream_model)
}

pub(crate) fn generated_wire_facts() -> &'static [ProviderModelWireFacts] {
    static FACTS: OnceLock<Vec<ProviderModelWireFacts>> = OnceLock::new();
    FACTS.get_or_init(|| {
        serde_json::from_str(include_str!("generated_wire_facts.json"))
            .expect("generated provider wire facts must be valid")
    })
}

pub(crate) fn apply_chat_completions_compat(
    object: &mut Map<String, Value>,
    upstream_model: &str,
    facts: &ResolvedProviderModelWireFacts,
) {
    if facts.compat.max_tokens_field == MaxTokensField::MaxCompletionTokens
        && let Some(max_tokens) = object.remove("max_tokens")
    {
        object
            .entry("max_completion_tokens".to_owned())
            .or_insert(max_tokens);
    }

    if !facts.compat.supports_store {
        object.remove("store");
    }

    if facts.compat.thinking_format != ThinkingFormat::Openrouter {
        return;
    }

    let requested_effort = object
        .remove("reasoning_effort")
        .and_then(|value| value.as_str().map(ToOwned::to_owned));
    object.remove("reasoning");

    if facts.reasoning {
        let level = requested_effort
            .as_deref()
            .filter(|value| *value != "none")
            .unwrap_or("off");
        let mapped = match facts.thinking_level_map.get(level) {
            Some(Some(value)) => Some(value.clone()),
            Some(None) => None,
            None if level == "off" => Some("none".to_owned()),
            None => Some(level.to_owned()),
        };
        if let Some(effort) = mapped {
            object.insert(
                "reasoning".to_owned(),
                serde_json::json!({ "effort": effort }),
            );
        }
    }

    if facts.reasoning
        && facts.compat.supports_developer_role
        && (upstream_model.starts_with("anthropic/") || upstream_model.starts_with("openai/"))
        && let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut)
    {
        for message in messages {
            if message.get("role").and_then(Value::as_str) == Some("system")
                && let Some(message) = message.as_object_mut()
            {
                message.insert("role".to_owned(), Value::String("developer".to_owned()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn explicit() -> ProviderModelWireFacts {
        ProviderModelWireFacts {
            provider: "openrouter".to_owned(),
            id: "anthropic/example".to_owned(),
            reasoning: true,
            thinking_level_map: BTreeMap::from([
                ("off".to_owned(), None),
                ("high".to_owned(), Some("deep".to_owned())),
            ]),
            compat: ProviderCompatFields {
                supports_store: Some(false),
                supports_developer_role: Some(false),
                ..Default::default()
            },
        }
    }

    #[test]
    fn base_url_profile_uses_parsed_normalized_host() {
        assert_eq!(
            base_url_profile("HTTPS://OPENROUTER.AI/api/v1/?x=1"),
            Some(BaseUrlProfile::Openrouter)
        );
        assert_eq!(
            base_url_profile("https://edge.openrouter.ai/custom"),
            Some(BaseUrlProfile::Openrouter)
        );
        assert_eq!(
            base_url_profile("https://openrouter.ai.example.com/v1"),
            None
        );
        assert_eq!(base_url_profile("not a url"), None);
    }

    #[test]
    fn compat_precedence_is_per_field() {
        let explicit = explicit();
        let facts = resolve_wire_facts(
            Some("openrouter"),
            "https://OPENROUTER.ai/api/v1/",
            "anthropic/example",
            Some(&explicit),
        );
        assert!(facts.reasoning);
        assert!(!facts.compat.supports_store, "explicit overrides profile");
        assert!(
            facts.compat.supports_reasoning_effort,
            "profile is inherited"
        );
        assert!(
            !facts.compat.supports_developer_role,
            "explicit false overrides provider/profile true"
        );
        assert_eq!(
            facts.compat.max_tokens_field,
            MaxTokensField::MaxCompletionTokens
        );
        assert_eq!(facts.thinking_level_map.get("off"), Some(&None));
    }

    #[test]
    fn profile_overrides_namespaced_provider_default() {
        let facts = resolve_wire_facts(
            Some("openrouter"),
            "https://api.z.ai/api/coding/paas/v4",
            "anthropic/example",
            None,
        );
        assert_eq!(facts.compat.thinking_format, ThinkingFormat::Zai);
        assert_eq!(facts.compat.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(!facts.compat.supports_developer_role);
    }

    #[test]
    fn generic_defaults_are_conservative() {
        let facts = resolve_wire_facts(None, "https://example.com/v1", "model", None);
        assert!(!facts.reasoning);
        assert!(!facts.compat.supports_store);
        assert!(!facts.compat.supports_developer_role);
        assert!(!facts.compat.supports_reasoning_effort);
        assert_eq!(facts.compat.max_tokens_field, MaxTokensField::MaxTokens);
        assert_eq!(facts.compat.thinking_format, ThinkingFormat::Openai);
    }

    #[test]
    fn explicit_thinking_level_mapping_controls_wire_effort() {
        let explicit = explicit();
        let facts = resolve_wire_facts(
            Some("openrouter"),
            "https://openrouter.ai/api/v1",
            "anthropic/example",
            Some(&explicit),
        );
        let mut object = serde_json::json!({"reasoning_effort": "high"})
            .as_object()
            .unwrap()
            .clone();
        apply_chat_completions_compat(&mut object, "anthropic/example", &facts);
        assert_eq!(
            object.get("reasoning"),
            Some(&serde_json::json!({"effort": "deep"}))
        );
        assert!(!object.contains_key("reasoning_effort"));
    }

    #[test]
    fn generated_fact_schema_rejects_unknown_compat_keys() {
        let error = serde_json::from_value::<ProviderModelWireFacts>(serde_json::json!({
            "provider": "openrouter",
            "id": "example/model",
            "reasoning": true,
            "compat": {"silentFutureBehavior": true}
        }))
        .expect_err("unknown compat keys must be registered explicitly");
        assert!(error.to_string().contains("silentFutureBehavior"));
    }
}

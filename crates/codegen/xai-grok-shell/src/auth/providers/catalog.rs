use std::collections::BTreeMap;
use std::sync::OnceLock;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use xai_grok_sampler::ApiBackend;
use xai_grok_sampling_types::{ReasoningEffort, ReasoningEffortOption};

use super::{GATEWAY_CONFIG_METADATA_KEY, ProviderCredential, ProviderId, RadiusGatewayConfig};

pub(crate) type ThinkingLevelMap = BTreeMap<String, Option<String>>;

/// pi-ai provider compatibility schema. Every source key is either consumed by
/// sampler wire facts or explicitly registered here (`cache_control_format` is
/// intentionally catalog-only until prompt cache-control conversion is added).
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ProviderCatalogCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) supports_store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) supports_developer_role: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) supports_reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens_field: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) requires_reasoning_content_on_assistant_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) zai_tool_stream: Option<bool>,
    /// Explicitly registered but not consumed by ticket-02 body transforms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control_format: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProviderCatalogModel {
    pub(crate) provider: ProviderId,
    pub(crate) id: String,
    pub(crate) name: String,
    api: String,
    pub(crate) base_url: String,
    pub(crate) reasoning: bool,
    pub(crate) context_window: u64,
    pub(crate) max_tokens: u32,
    /// Per-model default reasoning effort; `None` lets the backend decide.
    #[serde(default)]
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    /// Advertised reasoning-effort menu. Empty = no catalog-level
    /// validation (legacy fallback menu in the UI).
    #[serde(default)]
    pub(crate) reasoning_efforts: Vec<ReasoningEffortOption>,
    /// String-or-null wire mapping. Missing keys and explicit nulls remain
    /// distinguishable for OpenRouter's `off` behavior.
    #[serde(default)]
    pub(crate) thinking_level_map: ThinkingLevelMap,
    #[serde(default)]
    pub(crate) compat: ProviderCatalogCompat,
    #[serde(default)]
    pub(crate) headers: IndexMap<String, String>,
}

impl ProviderCatalogModel {
    pub(crate) fn api_backend(&self) -> Option<ApiBackend> {
        match self.api.as_str() {
            "anthropic-messages" | "pi-messages" => Some(ApiBackend::Messages),
            "openai-completions" => Some(ApiBackend::ChatCompletions),
            "openai-codex-responses" | "openai-responses" => Some(ApiBackend::Responses),
            _ => None,
        }
    }
}

pub(crate) fn provider_models(
    provider: ProviderId,
    credential: Option<&ProviderCredential>,
) -> Vec<ProviderCatalogModel> {
    if provider == ProviderId::Radius {
        return radius_models(credential);
    }

    let available = (provider == ProviderId::GithubCopilot)
        .then(|| credential.and_then(|value| value.metadata_strings("availableModelIds")))
        .flatten();
    static_models()
        .iter()
        .filter(|model| model.provider == provider)
        .filter(|model| {
            available
                .as_ref()
                .is_none_or(|ids| ids.iter().any(|id| id == &model.id))
        })
        .cloned()
        .collect()
}

fn static_models() -> &'static [ProviderCatalogModel] {
    static MODELS: OnceLock<Vec<ProviderCatalogModel>> = OnceLock::new();
    MODELS.get_or_init(|| {
        serde_json::from_str(include_str!("models.json"))
            .expect("embedded provider model catalog must be valid")
    })
}

fn radius_models(credential: Option<&ProviderCredential>) -> Vec<ProviderCatalogModel> {
    let Some(config) = credential
        .and_then(|value| value.metadata(GATEWAY_CONFIG_METADATA_KEY))
        .and_then(RadiusGatewayConfig::from_metadata)
    else {
        return Vec::new();
    };
    radius_models_from_config(&config)
}

pub(crate) fn radius_models_from_config(config: &RadiusGatewayConfig) -> Vec<ProviderCatalogModel> {
    config
        .models
        .iter()
        .map(|model| ProviderCatalogModel {
            provider: ProviderId::Radius,
            id: model.id.clone(),
            name: model.name.clone(),
            api: "pi-messages".to_owned(),
            base_url: config.base_url.clone(),
            reasoning: model.reasoning,
            context_window: model.context_window,
            max_tokens: model.max_tokens,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
            thinking_level_map: ThinkingLevelMap::new(),
            compat: ProviderCatalogCompat::default(),
            headers: IndexMap::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_has_namespaced_provider_models_for_every_static_provider() {
        for provider in ProviderId::ALL
            .into_iter()
            .filter(|id| *id != ProviderId::Radius)
        {
            let models = provider_models(provider, None);
            assert!(!models.is_empty(), "{provider}");
            assert!(models.iter().all(|model| model.api_backend().is_some()));
        }
    }

    #[test]
    fn deepseek_catalog_matches_pi_0841_contract() {
        let models = provider_models(ProviderId::Deepseek, None);
        assert_eq!(models.len(), 2);
        for ((model, id), name) in models
            .iter()
            .zip(["deepseek-v4-flash", "deepseek-v4-pro"])
            .zip(["DeepSeek V4 Flash", "DeepSeek V4 Pro"])
        {
            assert_eq!(model.id, id);
            assert_eq!(model.name, name);
            assert_eq!(model.api_backend(), Some(ApiBackend::ChatCompletions));
            assert_eq!(model.base_url, "https://api.deepseek.com");
            assert!(model.reasoning);
            assert_eq!(model.context_window, 1_000_000);
            assert_eq!(model.max_tokens, 384_000);
            assert_eq!(
                model
                    .reasoning_efforts
                    .iter()
                    .map(|option| option.value)
                    .collect::<Vec<_>>(),
                vec![ReasoningEffort::High, ReasoningEffort::Max]
            );
            assert_eq!(
                model.thinking_level_map,
                BTreeMap::from([
                    ("minimal".to_owned(), None),
                    ("low".to_owned(), None),
                    ("medium".to_owned(), None),
                    ("high".to_owned(), Some("high".to_owned())),
                    ("max".to_owned(), Some("max".to_owned())),
                ])
            );
            assert_eq!(model.compat.supports_store, Some(false));
            assert_eq!(model.compat.supports_developer_role, Some(false));
            assert_eq!(model.compat.supports_reasoning_effort, Some(true));
            assert_eq!(
                model.compat.max_tokens_field.as_deref(),
                Some("max_completion_tokens")
            );
            assert_eq!(
                model
                    .compat
                    .requires_reasoning_content_on_assistant_messages,
                Some(true)
            );
            assert_eq!(model.compat.thinking_format.as_deref(), Some("deepseek"));
        }
    }

    #[test]
    fn copilot_catalog_honors_server_model_filter() {
        let mut credential = ProviderCredential::permanent("secret");
        credential.set_metadata("availableModelIds", serde_json::json!(["gpt-4.1"]));
        let models = provider_models(ProviderId::GithubCopilot, Some(&credential));
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4.1");
    }

    /// Live-verified against `chatgpt.com/backend-api/codex/responses`:
    /// all three GPT-5.6 codex models accept low/medium/high/xhigh/max and
    /// reject `minimal` with HTTP 400 ("Supported values: low, medium,
    /// high, xhigh, max"). The embedded menu must match that contract — the
    /// legacy UI fallback offers `minimal` (would 400) and omits `max`.
    #[test]
    fn codex_gpt56_models_declare_live_verified_effort_menu() {
        let expected = [
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max,
        ];
        for id in ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"] {
            let model = provider_models(ProviderId::OpenaiCodex, None)
                .into_iter()
                .find(|model| model.id == id)
                .unwrap_or_else(|| panic!("{id} missing from codex catalog"));
            let menu: Vec<ReasoningEffort> = model
                .reasoning_efforts
                .iter()
                .map(|option| option.value)
                .collect();
            assert_eq!(
                menu, expected,
                "{id} effort menu drifted from the live contract"
            );
            assert_eq!(
                model.reasoning_effort,
                Some(ReasoningEffort::High),
                "{id} default effort",
            );
            assert!(
                model
                    .reasoning_efforts
                    .iter()
                    .any(|option| option.default && option.value == ReasoningEffort::High),
                "{id} menu must mark high as the default entry",
            );
        }
    }

    #[test]
    fn multi_provider_regression_radius_catalog_comes_from_gateway_config() {
        let mut credential = ProviderCredential::permanent("secret");
        credential.set_metadata(
            "gatewayConfig",
            serde_json::json!({
                "baseUrl": "https://api.radius.example",
                "models": [{
                    "id": "radius-1",
                    "name": "Radius One",
                    "reasoning": true,
                    "contextWindow": 123_000,
                    "maxTokens": 4_096
                }]
            }),
        );
        let models = provider_models(ProviderId::Radius, Some(&credential));
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].base_url, "https://api.radius.example");
        assert_eq!(models[0].api_backend(), Some(ApiBackend::Messages));
    }

    #[test]
    fn catalog_schema_preserves_string_or_null_maps_and_registered_compat() {
        let model: ProviderCatalogModel = serde_json::from_value(serde_json::json!({
            "provider": "openrouter",
            "id": "anthropic/example",
            "name": "Example",
            "api": "openai-completions",
            "baseUrl": "https://openrouter.ai/api/v1",
            "reasoning": true,
            "contextWindow": 128000,
            "maxTokens": 4096,
            "thinkingLevelMap": {"off": null, "high": "deep"},
            "compat": {
                "thinkingFormat": "openrouter",
                "zaiToolStream": true,
                "cacheControlFormat": "anthropic"
            }
        }))
        .expect("catalog compat schema");
        assert_eq!(model.thinking_level_map.get("off"), Some(&None));
        assert_eq!(
            model.thinking_level_map.get("high"),
            Some(&Some("deep".to_owned()))
        );
        assert_eq!(model.compat.zai_tool_stream, Some(true));
        assert_eq!(
            model.compat.cache_control_format.as_deref(),
            Some("anthropic")
        );
    }

    #[test]
    fn catalog_rejects_unregistered_compat_keys() {
        let error = serde_json::from_value::<ProviderCatalogCompat>(serde_json::json!({
            "thinkingFormat": "openrouter",
            "futureSilentBehavior": true
        }))
        .expect_err("unknown compat keys must not be silently consumed");
        assert!(error.to_string().contains("futureSilentBehavior"));
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct GeneratedWireFact {
        provider: ProviderId,
        id: String,
        reasoning: bool,
        #[serde(default)]
        thinking_level_map: ThinkingLevelMap,
        #[serde(default)]
        compat: ProviderCatalogCompat,
    }

    fn expected_effort_menu(reasoning: bool, map: &ThinkingLevelMap) -> Vec<ReasoningEffort> {
        if !reasoning {
            return vec![ReasoningEffort::None];
        }
        [
            ("off", ReasoningEffort::None),
            ("minimal", ReasoningEffort::Minimal),
            ("low", ReasoningEffort::Low),
            ("medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
            ("xhigh", ReasoningEffort::Xhigh),
            ("max", ReasoningEffort::Max),
        ]
        .into_iter()
        .filter_map(|(level, effort)| match map.get(level) {
            Some(None) => None,
            None if matches!(level, "xhigh" | "max") => None,
            _ => Some(effort),
        })
        .collect()
    }

    #[test]
    fn openrouter_catalog_and_sampler_wire_facts_do_not_drift() {
        let facts: Vec<GeneratedWireFact> = serde_json::from_str(include_str!(
            "../../../../xai-grok-sampler/src/provider_wire/generated_wire_facts.json"
        ))
        .expect("generated sampler wire facts");
        let manifest: Value = serde_json::from_str(include_str!(
            "../../../../../../scripts/openrouter_compat_manifest.json"
        ))
        .expect("OpenRouter sync manifest");
        assert_eq!(static_models().len(), 360, "catalog member set changed");
        assert_eq!(manifest["source"]["version"], "0.84.1");
        assert_eq!(
            manifest["source"]["generatedAt"],
            "2026-08-07T05:53:06.539Z"
        );
        assert_eq!(manifest["counts"]["catalogMembers"], 360);
        assert_eq!(manifest["counts"]["catalogOpenRouterMembers"], 303);
        assert_eq!(manifest["counts"]["catalogDeepSeekMembers"], 2);
        assert_eq!(manifest["counts"]["generatedDeepSeekWireFacts"], 2);
        assert_eq!(
            facts.len(),
            manifest["counts"]["generatedWireFacts"]
                .as_u64()
                .expect("manifest generated fact count") as usize
        );

        for fact in facts {
            assert!(matches!(
                fact.provider,
                ProviderId::Openrouter | ProviderId::Deepseek
            ));
            let catalog = static_models()
                .iter()
                .find(|model| model.provider == fact.provider && model.id == fact.id)
                .unwrap_or_else(|| panic!("generated fact missing from catalog: {}", fact.id));
            assert_eq!(catalog.reasoning, fact.reasoning, "{} reasoning", fact.id);
            assert_eq!(
                catalog.thinking_level_map, fact.thinking_level_map,
                "{} thinkingLevelMap",
                fact.id
            );
            assert_eq!(catalog.compat, fact.compat, "{} compat", fact.id);

            let menu = catalog
                .reasoning_efforts
                .iter()
                .map(|option| option.value)
                .collect::<Vec<_>>();
            assert_eq!(
                menu,
                expected_effort_menu(fact.reasoning, &fact.thinking_level_map),
                "{} reasoning effort menu",
                fact.id
            );
        }
    }
}

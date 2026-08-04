use std::sync::OnceLock;

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;
use xai_grok_sampler::ApiBackend;

use super::{ProviderCredential, ProviderId};

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
    #[serde(default)]
    pub(crate) headers: IndexMap<String, String>,
}

impl ProviderCatalogModel {
    pub(crate) fn api_backend(&self) -> Option<ApiBackend> {
        match self.api.as_str() {
            "anthropic-messages" => Some(ApiBackend::Messages),
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
    let Some(config) = credential.and_then(|value| value.metadata("gatewayConfig")) else {
        return Vec::new();
    };
    let Some(base_url) = config.get("baseUrl").and_then(Value::as_str) else {
        return Vec::new();
    };
    config
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| {
            Some(ProviderCatalogModel {
                provider: ProviderId::Radius,
                id: model.get("id")?.as_str()?.to_owned(),
                name: model.get("name")?.as_str()?.to_owned(),
                api: "pi-messages".to_owned(),
                base_url: base_url.to_owned(),
                reasoning: model.get("reasoning")?.as_bool()?,
                context_window: model.get("contextWindow")?.as_u64()?,
                max_tokens: u32::try_from(model.get("maxTokens")?.as_u64()?).ok()?,
                headers: IndexMap::new(),
            })
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
    fn copilot_catalog_honors_server_model_filter() {
        let mut credential = ProviderCredential::permanent("secret");
        credential.set_metadata("availableModelIds", serde_json::json!(["gpt-4.1"]));
        let models = provider_models(ProviderId::GithubCopilot, Some(&credential));
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4.1");
    }

    #[test]
    fn radius_catalog_comes_from_gateway_config() {
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
    }
}

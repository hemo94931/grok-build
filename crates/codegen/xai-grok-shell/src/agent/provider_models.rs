use std::num::NonZeroU64;

use indexmap::IndexMap;
use xai_grok_sampler::AuthScheme;

use super::config::{ModelEntry, ModelInfo};
use crate::auth::providers::{
    ProviderId, ProviderSecret, ProviderSlotState, ProviderStore, ProviderStoredCredential,
    namespaced_model_id, provider_models,
};

pub(crate) fn append_available_provider_models(resolved: &mut IndexMap<String, ModelEntry>) {
    let store = ProviderStore::default();
    for provider in ProviderId::ALL {
        let slot = match store.get(provider) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(%provider, %error, "provider model catalog: credential read failed");
                continue;
            }
        };
        let credential = match &slot {
            ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)) => {
                Some(credential)
            }
            ProviderSlotState::Known(ProviderStoredCredential::ApiKey(_))
                if provider == ProviderId::OpenaiCodex =>
            {
                tracing::warn!(%provider, "provider model catalog: Codex API-key slot is unusable");
                continue;
            }
            ProviderSlotState::Known(ProviderStoredCredential::ApiKey(_)) => None,
            ProviderSlotState::PresentUnsupportedOrInvalid => continue,
            ProviderSlotState::Missing => {
                let has_environment_key = ProviderSecret::from_environment(provider)
                    .map(|value| value.is_some())
                    .unwrap_or_else(|error| {
                        tracing::warn!(%provider, %error, "provider model catalog: environment credential invalid");
                        false
                    });
                if !has_environment_key {
                    continue;
                }
                None
            }
        };

        for model in provider_models(provider, credential) {
            let Some(api_backend) = model.api_backend() else {
                continue;
            };
            let catalog_id = namespaced_model_id(provider, &model.id);
            let Some(context_window) = NonZeroU64::new(model.context_window) else {
                continue;
            };
            let mut info = ModelInfo::fallback(&catalog_id);
            info.id = Some(catalog_id.clone());
            info.model = catalog_id.clone();
            info.base_url = model.base_url;
            info.name = Some(model.name);
            info.max_completion_tokens = Some(model.max_tokens);
            info.api_backend = api_backend;
            info.auth_scheme = AuthScheme::Bearer;
            info.extra_headers = model.headers;
            info.context_window = context_window;
            info.supports_reasoning_effort = model.reasoning;
            info.reasoning_effort = model.reasoning_effort;
            info.reasoning_efforts = model.reasoning_efforts;
            resolved.insert(
                catalog_id,
                ModelEntry {
                    info,
                    api_key: None,
                    env_key: None,
                    auth_provider: None,
                    api_base_url: None,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_models_are_namespaced_without_changing_model_info_shape() {
        let mut resolved = IndexMap::new();
        let model = crate::auth::providers::provider_models(ProviderId::Anthropic, None)
            .into_iter()
            .next()
            .unwrap();
        let catalog_id = namespaced_model_id(ProviderId::Anthropic, &model.id);
        let mut info = ModelInfo::fallback(&catalog_id);
        info.id = Some(catalog_id.clone());
        info.model = catalog_id.clone();
        resolved.insert(
            catalog_id.clone(),
            ModelEntry {
                info,
                api_key: None,
                env_key: None,
                auth_provider: None,
                api_base_url: None,
            },
        );
        assert_eq!(resolved[&catalog_id].info.model, catalog_id);
    }
}

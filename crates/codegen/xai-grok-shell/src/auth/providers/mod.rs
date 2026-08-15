//! OAuth credentials and request routing for non-xAI model providers.
//!
//! xAI authentication remains owned by the parent `auth` module. Provider
//! credentials live in `providers.json` and never enter `AuthManager`.

mod anthropic;
mod catalog;
mod cli;
mod deepseek;
mod flow;
mod github_copilot;
mod kimi_coding;
mod openai_codex;
mod openrouter;
mod radius;
mod route;
mod store;
mod zai;

use std::fmt;
use std::str::FromStr;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub(crate) use catalog::{ProviderCatalogModel, provider_models, radius_models_from_config};
pub(crate) use cli::{
    CliAuthInteraction, ProviderLoginTarget, ProviderTarget, cli_login_options,
    report_provider_api_key_login, report_provider_login, select_login_target, select_target,
    terminal_is_interactive,
};
pub(crate) use flow::{
    AuthInteraction, AuthNotification, AuthPrompt, AuthPromptResponse, AuthSecret, DeviceCode,
    SelectOption,
};
pub(crate) use radius::{
    GATEWAY_CONFIG_METADATA_KEY, RadiusGatewayConfig, RadiusGatewayModel, gateway_cache_origin,
    load_gateway_config_for_catalog,
};
pub(crate) use route::{
    ProviderAuthRemedy, ProviderCredentialMethod, ProviderDescriptor, ProviderLoginOption,
    ProviderLoginTransport, ProviderRequestContext, ProviderSecret, ProviderSecretSource,
    ProviderWireDialect, namespaced_model_id, openai_codex_compaction_headers,
    parse_namespaced_model_id, provider_auth_remedy, provider_bearer_resolver, provider_descriptor,
    provider_login_options, resolve_fresh_provider_secret, resolve_provider_secret,
};
pub(crate) use store::{
    ProviderApiKeyCredential, ProviderCredential, ProviderSlotState, ProviderStore,
    ProviderStoredCredential, RefreshReason,
};

/// Provider IDs are stable storage keys and catalog namespace prefixes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ProviderId {
    Anthropic,
    OpenaiCodex,
    GithubCopilot,
    Openrouter,
    KimiCoding,
    Radius,
    Deepseek,
    Zai,
    ZaiCodingCn,
}

impl ProviderId {
    pub(crate) const ALL: [Self; 9] = [
        Self::Anthropic,
        Self::OpenaiCodex,
        Self::GithubCopilot,
        Self::Openrouter,
        Self::KimiCoding,
        Self::Radius,
        Self::Deepseek,
        Self::Zai,
        Self::ZaiCodingCn,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenaiCodex => "openai-codex",
            Self::GithubCopilot => "github-copilot",
            Self::Openrouter => "openrouter",
            Self::KimiCoding => "kimi-coding",
            Self::Radius => "radius",
            Self::Deepseek => "deepseek",
            Self::Zai => "zai",
            Self::ZaiCodingCn => "zai-coding-cn",
        }
    }

    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::OpenaiCodex => "OpenAI Codex",
            Self::GithubCopilot => "GitHub Copilot",
            Self::Openrouter => "OpenRouter",
            Self::KimiCoding => "Kimi Coding",
            Self::Radius => "Radius",
            Self::Deepseek => "DeepSeek",
            Self::Zai => "Z.AI",
            Self::ZaiCodingCn => "Z.AI Coding CN",
        }
    }

    /// Narrow shell-to-sampler identity mapping. Provider provenance stays a
    /// sidecar and never enters shared model or request structs.
    pub(crate) const fn sampler_provider(self) -> xai_grok_sampler::KnownProvider {
        match self {
            Self::Anthropic => xai_grok_sampler::KnownProvider::Anthropic,
            Self::OpenaiCodex => xai_grok_sampler::KnownProvider::OpenaiCodex,
            Self::GithubCopilot => xai_grok_sampler::KnownProvider::GithubCopilot,
            Self::Openrouter => xai_grok_sampler::KnownProvider::Openrouter,
            Self::KimiCoding => xai_grok_sampler::KnownProvider::KimiCoding,
            Self::Radius => xai_grok_sampler::KnownProvider::Radius,
            Self::Deepseek => xai_grok_sampler::KnownProvider::Deepseek,
            Self::Zai => xai_grok_sampler::KnownProvider::Zai,
            Self::ZaiCodingCn => xai_grok_sampler::KnownProvider::ZaiCodingCn,
        }
    }
}

/// Resolve sampler route provenance from the shell's catalog identity, not
/// from the slash-prefixed upstream model sent on the wire. A custom catalog
/// entry may legitimately route to `anthropic/*` or `openai/*` on OpenRouter.
pub(crate) fn sampler_route_hint(
    catalog_model_id: &str,
    upstream_model_id: &str,
) -> xai_grok_sampler::ProviderRouteHint {
    if let Some((provider, _)) = parse_namespaced_model_id(catalog_model_id) {
        xai_grok_sampler::ProviderRouteHint::Known(provider.sampler_provider())
    } else if upstream_model_id.contains('/') {
        xai_grok_sampler::ProviderRouteHint::CustomThirdParty
    } else {
        xai_grok_sampler::ProviderRouteHint::Auto
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProviderId {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|provider| provider.as_str() == normalized)
            .with_context(|| format!("unknown provider `{value}`"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoginMode {
    Browser,
    DeviceCode,
}

impl LoginMode {
    pub(crate) const fn as_transport(self) -> ProviderLoginTransport {
        match self {
            Self::Browser => ProviderLoginTransport::Browser,
            Self::DeviceCode => ProviderLoginTransport::Device,
        }
    }
}

pub(crate) enum ProviderRefreshOutcome {
    Save(ProviderCredential),
    Remove { message: String },
}

pub(crate) async fn login(
    provider: ProviderId,
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    match provider {
        ProviderId::Anthropic => anthropic::login(interaction, mode).await,
        ProviderId::OpenaiCodex => openai_codex::login(interaction, mode).await,
        ProviderId::GithubCopilot => github_copilot::login(interaction, mode).await,
        ProviderId::Openrouter => openrouter::login(interaction, mode).await,
        ProviderId::KimiCoding => kimi_coding::login(interaction, mode).await,
        ProviderId::Radius => radius::login(interaction, mode).await,
        ProviderId::Deepseek => deepseek::login(interaction, mode).await,
        ProviderId::Zai | ProviderId::ZaiCodingCn => zai::login(interaction, mode).await,
    }
}

async fn refresh(
    provider: ProviderId,
    credential: ProviderCredential,
    signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    match provider {
        ProviderId::Anthropic => anthropic::refresh(credential, signal).await,
        ProviderId::OpenaiCodex => openai_codex::refresh(credential, signal).await,
        ProviderId::GithubCopilot => github_copilot::refresh(credential, signal).await,
        ProviderId::Openrouter => openrouter::refresh(credential, signal).await,
        ProviderId::KimiCoding => kimi_coding::refresh(credential, signal).await,
        ProviderId::Radius => radius::refresh(credential, signal).await,
        ProviderId::Deepseek => deepseek::refresh(credential, signal).await,
        ProviderId::Zai | ProviderId::ZaiCodingCn => zai::refresh(credential, signal).await,
    }
}

pub(crate) async fn login_and_store(
    provider: ProviderId,
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    validate_oauth_login_request(provider, mode)?;
    let credential = login(provider, interaction, mode).await?;
    ProviderStore::default()
        .put(provider, credential.clone())
        .await?;
    Ok(credential)
}

pub(crate) fn validate_oauth_login_request(
    provider: ProviderId,
    mode: Option<LoginMode>,
) -> anyhow::Result<()> {
    let descriptor = provider_descriptor(provider);
    if !descriptor.supports_method(ProviderCredentialMethod::OAuth) {
        anyhow::bail!("{} does not support OAuth login", descriptor.display_name);
    }
    if let Some(mode) = mode {
        let transport = mode.as_transport();
        if !descriptor.supports_oauth_transport(transport) {
            anyhow::bail!(
                "{} does not support {} OAuth login",
                descriptor.display_name,
                match transport {
                    ProviderLoginTransport::Browser => "browser",
                    ProviderLoginTransport::Device => "device-code",
                }
            );
        }
    }
    Ok(())
}

pub(crate) async fn login_api_key_and_store(
    provider: ProviderId,
    interaction: &dyn AuthInteraction,
) -> anyhow::Result<ProviderApiKeyCredential> {
    login_api_key_and_store_with_store(provider, interaction, &ProviderStore::default()).await
}

async fn login_api_key_and_store_with_store(
    provider: ProviderId,
    interaction: &dyn AuthInteraction,
    store: &ProviderStore,
) -> anyhow::Result<ProviderApiKeyCredential> {
    let descriptor = provider_descriptor(provider);
    if !descriptor.supports_method(ProviderCredentialMethod::ApiKey) {
        anyhow::bail!("{} does not support API-key login", descriptor.display_name);
    }
    let signal = interaction.signal();
    let key = interaction
        .prompt(AuthPrompt::Secret {
            message: format!("Enter API key for {}:", descriptor.display_name),
            placeholder: "API key".to_owned(),
        })
        .await?
        .into_secret()?
        .into_zeroizing_string();
    let key = key.trim();
    if key.is_empty() {
        anyhow::bail!("API key cannot be empty");
    }
    let credential = ProviderApiKeyCredential::new(key.to_owned());
    if signal.is_cancelled() {
        anyhow::bail!("login cancelled");
    }
    // Cancellation owns the race until the credential-store commit begins.
    // In particular, a cancel that arrives with (or immediately after) the
    // reverse secret response must not persist the returned key.
    tokio::select! {
        biased;
        _ = signal.cancelled() => anyhow::bail!("login cancelled"),
        result = store.put(provider, credential.clone()) => result?,
    }
    Ok(credential)
}

pub(crate) async fn stored_slot(provider: ProviderId) -> anyhow::Result<ProviderSlotState> {
    ProviderStore::default().get(provider)
}

pub(crate) async fn fresh_stored_slot(
    provider: ProviderId,
    rejected_access: Option<&str>,
) -> anyhow::Result<ProviderSlotState> {
    let store = ProviderStore::default();
    let slot = store.get(provider)?;
    let ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)) = slot else {
        return Ok(slot);
    };
    if rejected_access.is_none() && !credential.expires_within(store::REFRESH_WINDOW) {
        return Ok(ProviderSlotState::Known(ProviderStoredCredential::OAuth(
            credential,
        )));
    }

    let reason = match rejected_access {
        Some(access) => RefreshReason::Rejected {
            access: access.to_owned(),
        },
        None => RefreshReason::Expiring,
    };
    let signal = CancellationToken::new();
    store
        .refresh(provider, reason, move |credential| {
            refresh(provider, credential, signal)
        })
        .await
        .map(|credential| ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)))
}

pub(crate) async fn fresh_stored_credential(
    provider: ProviderId,
    rejected_access: Option<&str>,
) -> anyhow::Result<Option<ProviderCredential>> {
    match fresh_stored_slot(provider, rejected_access).await? {
        ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)) => {
            Ok(Some(credential))
        }
        ProviderSlotState::Missing => Ok(None),
        ProviderSlotState::Known(ProviderStoredCredential::ApiKey(_)) => {
            anyhow::bail!("{provider} stores an API key, which cannot be refreshed")
        }
        ProviderSlotState::PresentUnsupportedOrInvalid => {
            anyhow::bail!("{provider} credential slot is unsupported or invalid")
        }
    }
}

pub(crate) async fn logout(provider: ProviderId) -> anyhow::Result<bool> {
    ProviderStore::default().remove(provider).await
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;

    struct ApiKeyInteraction {
        value: Option<&'static str>,
    }

    struct CancelledAfterSecretInteraction {
        signal: CancellationToken,
    }

    #[async_trait(?Send)]
    impl AuthInteraction for ApiKeyInteraction {
        fn signal(&self) -> CancellationToken {
            CancellationToken::new()
        }

        async fn notify(&self, _notification: AuthNotification) -> anyhow::Result<()> {
            Ok(())
        }

        async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<AuthPromptResponse> {
            assert!(matches!(prompt, AuthPrompt::Secret { .. }));
            match self.value {
                Some(value) => Ok(AuthPromptResponse::Secret(AuthSecret::new(value))),
                None => anyhow::bail!("login cancelled"),
            }
        }
    }

    #[async_trait(?Send)]
    impl AuthInteraction for CancelledAfterSecretInteraction {
        fn signal(&self) -> CancellationToken {
            self.signal.clone()
        }

        async fn notify(&self, _notification: AuthNotification) -> anyhow::Result<()> {
            Ok(())
        }

        async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<AuthPromptResponse> {
            assert!(matches!(prompt, AuthPrompt::Secret { .. }));
            self.signal.cancel();
            Ok(AuthPromptResponse::Secret(AuthSecret::new(
                "sk-cancelled-race",
            )))
        }
    }

    #[tokio::test]
    async fn api_key_login_blindly_overwrites_oauth_and_cancel_does_not_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = ProviderStore::with_paths(
            tmp.path().join("providers.json"),
            tmp.path().join("auth.json"),
        );
        store
            .put(
                ProviderId::Anthropic,
                ProviderCredential::oauth("old-access", "old-refresh", u64::MAX),
            )
            .await
            .unwrap();

        login_api_key_and_store_with_store(
            ProviderId::Anthropic,
            &ApiKeyInteraction {
                value: Some("  sk-blind-store  "),
            },
            &store,
        )
        .await
        .unwrap();
        match store.get(ProviderId::Anthropic).unwrap() {
            ProviderSlotState::Known(ProviderStoredCredential::ApiKey(credential)) => {
                assert_eq!(credential.key, "sk-blind-store");
            }
            other => panic!("expected API-key slot, got {other:?}"),
        }

        let cancelled_store = ProviderStore::with_paths(
            tmp.path().join("cancelled-providers.json"),
            tmp.path().join("cancelled-auth.json"),
        );
        assert!(
            login_api_key_and_store_with_store(
                ProviderId::Anthropic,
                &ApiKeyInteraction { value: None },
                &cancelled_store,
            )
            .await
            .is_err()
        );
        assert!(matches!(
            cancelled_store.get(ProviderId::Anthropic).unwrap(),
            ProviderSlotState::Missing
        ));

        let raced_store = ProviderStore::with_paths(
            tmp.path().join("cancelled-after-secret-providers.json"),
            tmp.path().join("cancelled-after-secret-auth.json"),
        );
        let interaction = CancelledAfterSecretInteraction {
            signal: CancellationToken::new(),
        };
        let error =
            login_api_key_and_store_with_store(ProviderId::Anthropic, &interaction, &raced_store)
                .await
                .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        assert!(matches!(
            raced_store.get(ProviderId::Anthropic).unwrap(),
            ProviderSlotState::Missing
        ));
    }

    #[test]
    fn provider_ids_round_trip() {
        for provider in ProviderId::ALL {
            assert_eq!(provider.as_str().parse::<ProviderId>().unwrap(), provider);
            assert_eq!(
                serde_json::to_string(&provider).unwrap(),
                format!("\"{provider}\"")
            );
        }
        assert_eq!(
            "deepseek".parse::<ProviderId>().unwrap(),
            ProviderId::Deepseek
        );
        assert_eq!(ProviderId::Deepseek.display_name(), "DeepSeek");
        assert_eq!("zai".parse::<ProviderId>().unwrap(), ProviderId::Zai);
        assert_eq!(
            "zai-coding-cn".parse::<ProviderId>().unwrap(),
            ProviderId::ZaiCodingCn
        );
        assert_eq!(ProviderId::Zai.display_name(), "Z.AI");
        assert_eq!(ProviderId::ZaiCodingCn.display_name(), "Z.AI Coding CN");
    }

    #[test]
    fn deepseek_is_api_key_only_and_routes_to_sampler_identity() {
        let descriptor = provider_descriptor(ProviderId::Deepseek);
        assert_eq!(descriptor.base_url, "https://api.deepseek.com");
        assert_eq!(
            descriptor.api_backend,
            xai_grok_sampler::ApiBackend::ChatCompletions
        );
        assert_eq!(
            descriptor.wire_dialect,
            ProviderWireDialect::OpenaiChatCompletions
        );
        assert_eq!(descriptor.env_keys, &["DEEPSEEK_API_KEY"]);
        assert_eq!(descriptor.oauth_transports(), Vec::new());
        assert!(descriptor.supports_method(ProviderCredentialMethod::ApiKey));
        assert!(!descriptor.supports_method(ProviderCredentialMethod::OAuth));
        assert_eq!(
            sampler_route_hint("deepseek/deepseek-v4-flash", "deepseek-v4-flash"),
            xai_grok_sampler::ProviderRouteHint::Known(xai_grok_sampler::KnownProvider::Deepseek)
        );
        assert!(validate_oauth_login_request(ProviderId::Deepseek, None).is_err());
        let remedy = provider_auth_remedy(ProviderId::Deepseek, ProviderSecretSource::StoredApiKey);
        assert_eq!(remedy.method, ProviderCredentialMethod::ApiKey);
        assert!(remedy.advice().contains("--api-key"));
    }

    #[test]
    fn zai_variants_are_api_key_only_bearer_chat_providers() {
        for (provider, base_url, env_key, known) in [
            (
                ProviderId::Zai,
                "https://api.z.ai/api/coding/paas/v4",
                "ZAI_API_KEY",
                xai_grok_sampler::KnownProvider::Zai,
            ),
            (
                ProviderId::ZaiCodingCn,
                "https://open.bigmodel.cn/api/coding/paas/v4",
                "ZAI_CODING_CN_API_KEY",
                xai_grok_sampler::KnownProvider::ZaiCodingCn,
            ),
        ] {
            let descriptor = provider_descriptor(provider);
            assert_eq!(descriptor.base_url, base_url);
            assert_eq!(
                descriptor.api_backend,
                xai_grok_sampler::ApiBackend::ChatCompletions
            );
            assert_eq!(
                descriptor.wire_dialect,
                ProviderWireDialect::OpenaiChatCompletions
            );
            assert_eq!(descriptor.env_keys, &[env_key]);
            assert_eq!(descriptor.oauth_transports(), Vec::new());
            assert!(descriptor.supports_method(ProviderCredentialMethod::ApiKey));
            assert!(!descriptor.supports_method(ProviderCredentialMethod::OAuth));
            assert!(validate_oauth_login_request(provider, None).is_err());
            assert_eq!(
                sampler_route_hint(&format!("{provider}/glm-4.7"), "glm-4.7"),
                xai_grok_sampler::ProviderRouteHint::Known(known)
            );
            let secret = ProviderSecret::from_api_key(provider, "key".to_owned()).unwrap();
            assert_eq!(secret.auth_scheme, xai_grok_sampler::AuthScheme::Bearer);
        }
    }

    #[test]
    fn provider_login_options_project_authoritative_method_matrix() {
        let options = provider_login_options();
        assert_eq!(options.len(), 14);

        let methods = |provider| {
            options
                .iter()
                .filter(|option| option.provider == provider)
                .map(|option| option.method)
                .collect::<Vec<_>>()
        };
        for provider in [
            ProviderId::Anthropic,
            ProviderId::GithubCopilot,
            ProviderId::Openrouter,
            ProviderId::KimiCoding,
            ProviderId::Radius,
        ] {
            assert_eq!(
                methods(provider),
                vec![
                    ProviderCredentialMethod::OAuth,
                    ProviderCredentialMethod::ApiKey
                ],
                "{provider} should offer OAuth and API-key login"
            );
        }
        assert_eq!(
            methods(ProviderId::OpenaiCodex),
            vec![ProviderCredentialMethod::OAuth]
        );
        for provider in [
            ProviderId::Deepseek,
            ProviderId::Zai,
            ProviderId::ZaiCodingCn,
        ] {
            assert_eq!(methods(provider), vec![ProviderCredentialMethod::ApiKey]);
        }
        assert_eq!(cli_login_options().len(), 15);
    }

    #[test]
    fn sampler_route_hint_uses_catalog_provenance_not_upstream_prefix() {
        assert_eq!(
            sampler_route_hint(
                "openrouter/anthropic/claude-sonnet-4.6",
                "openrouter/anthropic/claude-sonnet-4.6"
            ),
            xai_grok_sampler::ProviderRouteHint::Known(xai_grok_sampler::KnownProvider::Openrouter)
        );
        assert_eq!(
            sampler_route_hint("custom-openrouter", "anthropic/claude-sonnet-4.6"),
            xai_grok_sampler::ProviderRouteHint::CustomThirdParty
        );
        assert_eq!(
            sampler_route_hint("custom-openrouter", "openai/gpt-5.4"),
            xai_grok_sampler::ProviderRouteHint::CustomThirdParty
        );
        assert_eq!(
            sampler_route_hint("grok-4.5", "grok-4.5"),
            xai_grok_sampler::ProviderRouteHint::Auto
        );
    }
}

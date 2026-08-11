//! OAuth credentials and request routing for non-xAI model providers.
//!
//! xAI authentication remains owned by the parent `auth` module. Provider
//! credentials live in `providers.json` and never enter `AuthManager`.

mod anthropic;
mod catalog;
mod cli;
mod flow;
mod github_copilot;
mod kimi_coding;
mod openai_codex;
mod openrouter;
mod radius;
mod route;
mod store;

use std::fmt;
use std::str::FromStr;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub(crate) use catalog::{ProviderCatalogModel, provider_models};
pub(crate) use cli::{
    CliAuthInteraction, ProviderTarget, report_provider_login, select_target,
    terminal_is_interactive,
};
pub(crate) use flow::{AuthInteraction, AuthNotification, AuthPrompt, DeviceCode, SelectOption};
pub(crate) use route::{
    ProviderAuthRemedy, ProviderCredentialMethod, ProviderDescriptor, ProviderRequestContext,
    ProviderSecret, ProviderSecretSource, ProviderWireDialect, namespaced_model_id,
    parse_namespaced_model_id, provider_auth_remedy, provider_bearer_resolver, provider_descriptor,
    resolve_fresh_provider_secret, resolve_provider_secret,
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
}

impl ProviderId {
    pub(crate) const ALL: [Self; 6] = [
        Self::Anthropic,
        Self::OpenaiCodex,
        Self::GithubCopilot,
        Self::Openrouter,
        Self::KimiCoding,
        Self::Radius,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenaiCodex => "openai-codex",
            Self::GithubCopilot => "github-copilot",
            Self::Openrouter => "openrouter",
            Self::KimiCoding => "kimi-coding",
            Self::Radius => "radius",
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
        }
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
    }
}

pub(crate) async fn login_and_store(
    provider: ProviderId,
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    let credential = login(provider, interaction, mode).await?;
    ProviderStore::default()
        .put(provider, credential.clone())
        .await?;
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
    use super::*;

    #[test]
    fn provider_ids_round_trip() {
        for provider in ProviderId::ALL {
            assert_eq!(provider.as_str().parse::<ProviderId>().unwrap(), provider);
            assert_eq!(
                serde_json::to_string(&provider).unwrap(),
                format!("\"{provider}\"")
            );
        }
    }
}

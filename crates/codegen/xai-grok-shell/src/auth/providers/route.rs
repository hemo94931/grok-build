use std::fmt;

use anyhow::{Context, bail};
use indexmap::IndexMap;
use sha2::{Digest, Sha256};
use xai_grok_sampler::{ApiBackend, AuthScheme};

use super::flow::validate_http_url;
use super::{
    ProviderCredential, ProviderId, ProviderSlotState, ProviderStore, ProviderStoredCredential,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderLoginFlow {
    None,
    Browser,
    DeviceCode,
    BrowserOrDevice,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProviderLoginTransport {
    Browser,
    Device,
}

impl ProviderLoginTransport {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Browser => "browser",
            Self::Device => "device",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProviderLoginOption {
    pub(crate) provider: ProviderId,
    pub(crate) display_name: &'static str,
    pub(crate) method: ProviderCredentialMethod,
    pub(crate) oauth_transports: Vec<ProviderLoginTransport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderWireDialect {
    AnthropicMessages,
    OpenaiCodexResponses,
    GithubCopilot,
    OpenaiChatCompletions,
    KimiAnthropicMessages,
    PiMessages,
}

#[derive(Clone, Debug)]
pub(crate) struct ProviderDescriptor {
    pub(crate) id: ProviderId,
    pub(crate) display_name: &'static str,
    pub(crate) login_flow: ProviderLoginFlow,
    pub(crate) api_key_login: bool,
    pub(crate) base_url: &'static str,
    pub(crate) api_backend: ApiBackend,
    pub(crate) wire_dialect: ProviderWireDialect,
    pub(crate) env_keys: &'static [&'static str],
}

impl ProviderDescriptor {
    pub(crate) fn supports_method(&self, method: ProviderCredentialMethod) -> bool {
        match method {
            ProviderCredentialMethod::OAuth => self.supports_oauth(),
            ProviderCredentialMethod::ApiKey => self.supports_api_key(),
        }
    }

    pub(crate) fn supports_oauth(&self) -> bool {
        !self.oauth_transports().is_empty()
    }

    pub(crate) const fn supports_api_key(&self) -> bool {
        self.api_key_login
    }

    pub(crate) fn default_login_method(&self) -> ProviderCredentialMethod {
        if self.supports_oauth() {
            ProviderCredentialMethod::OAuth
        } else {
            ProviderCredentialMethod::ApiKey
        }
    }

    pub(crate) fn oauth_transports(&self) -> Vec<ProviderLoginTransport> {
        match self.login_flow {
            ProviderLoginFlow::None => Vec::new(),
            ProviderLoginFlow::Browser => vec![ProviderLoginTransport::Browser],
            ProviderLoginFlow::DeviceCode => vec![ProviderLoginTransport::Device],
            ProviderLoginFlow::BrowserOrDevice => {
                vec![
                    ProviderLoginTransport::Browser,
                    ProviderLoginTransport::Device,
                ]
            }
        }
    }

    pub(crate) fn supports_oauth_transport(&self, transport: ProviderLoginTransport) -> bool {
        self.oauth_transports().contains(&transport)
    }

    pub(crate) fn login_options(&self) -> Vec<ProviderLoginOption> {
        let mut options = Vec::new();
        if self.supports_oauth() {
            options.push(ProviderLoginOption {
                provider: self.id,
                display_name: self.display_name,
                method: ProviderCredentialMethod::OAuth,
                oauth_transports: self.oauth_transports(),
            });
        }
        if self.supports_api_key() {
            options.push(ProviderLoginOption {
                provider: self.id,
                display_name: self.display_name,
                method: ProviderCredentialMethod::ApiKey,
                oauth_transports: Vec::new(),
            });
        }
        options
    }
}

pub(crate) fn provider_login_options() -> Vec<ProviderLoginOption> {
    ProviderId::ALL
        .into_iter()
        .flat_map(|provider| provider_descriptor(provider).login_options())
        .collect()
}

pub(crate) fn provider_descriptor(id: ProviderId) -> ProviderDescriptor {
    match id {
        ProviderId::Anthropic => ProviderDescriptor {
            id,
            display_name: "Anthropic",
            login_flow: ProviderLoginFlow::Browser,
            api_key_login: true,
            base_url: "https://api.anthropic.com",
            api_backend: ApiBackend::Messages,
            wire_dialect: ProviderWireDialect::AnthropicMessages,
            env_keys: &[
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_OAUTH_TOKEN",
                "ANTHROPIC_API_KEY",
            ],
        },
        ProviderId::OpenaiCodex => ProviderDescriptor {
            id,
            display_name: "OpenAI Codex",
            login_flow: ProviderLoginFlow::BrowserOrDevice,
            api_key_login: false,
            base_url: "https://chatgpt.com/backend-api",
            api_backend: ApiBackend::Responses,
            wire_dialect: ProviderWireDialect::OpenaiCodexResponses,
            env_keys: &[],
        },
        ProviderId::GithubCopilot => ProviderDescriptor {
            id,
            display_name: "GitHub Copilot",
            login_flow: ProviderLoginFlow::DeviceCode,
            api_key_login: true,
            base_url: "https://api.individual.githubcopilot.com",
            api_backend: ApiBackend::ChatCompletions,
            wire_dialect: ProviderWireDialect::GithubCopilot,
            env_keys: &["COPILOT_GITHUB_TOKEN"],
        },
        ProviderId::Openrouter => ProviderDescriptor {
            id,
            display_name: "OpenRouter",
            login_flow: ProviderLoginFlow::Browser,
            api_key_login: true,
            base_url: "https://openrouter.ai/api/v1",
            api_backend: ApiBackend::ChatCompletions,
            wire_dialect: ProviderWireDialect::OpenaiChatCompletions,
            env_keys: &["OPENROUTER_API_KEY"],
        },
        ProviderId::KimiCoding => ProviderDescriptor {
            id,
            display_name: "Kimi Coding",
            login_flow: ProviderLoginFlow::DeviceCode,
            api_key_login: true,
            base_url: "https://api.kimi.com/coding",
            api_backend: ApiBackend::Messages,
            wire_dialect: ProviderWireDialect::KimiAnthropicMessages,
            env_keys: &["KIMI_API_KEY"],
        },
        ProviderId::Radius => ProviderDescriptor {
            id,
            display_name: "Radius",
            login_flow: ProviderLoginFlow::BrowserOrDevice,
            api_key_login: true,
            base_url: "https://radius.pi.dev",
            // The provider wire dispatcher intercepts this dialect before the
            // generic Messages conversion. No fourth shared ApiBackend variant
            // is needed.
            api_backend: ApiBackend::Messages,
            wire_dialect: ProviderWireDialect::PiMessages,
            env_keys: &["RADIUS_API_KEY"],
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderSecretSource {
    Model,
    StoredOAuth,
    StoredApiKey,
    Environment(&'static str),
}

impl ProviderSecretSource {
    pub(crate) const fn method(self) -> ProviderCredentialMethod {
        match self {
            Self::StoredOAuth => ProviderCredentialMethod::OAuth,
            Self::Model | Self::StoredApiKey | Self::Environment(_) => {
                ProviderCredentialMethod::ApiKey
            }
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::StoredOAuth => "stored_oauth",
            Self::StoredApiKey => "stored_api_key",
            Self::Environment(_) => "environment",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) enum ProviderCredentialMethod {
    #[serde(rename = "oauth")]
    OAuth,
    #[serde(rename = "api_key")]
    ApiKey,
}

impl ProviderCredentialMethod {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::OAuth => "oauth",
            Self::ApiKey => "api_key",
        }
    }

    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::OAuth => "OAuth",
            Self::ApiKey => "API key",
        }
    }
}

impl std::fmt::Display for ProviderCredentialMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProviderAuthRemedy {
    pub(crate) provider: ProviderId,
    pub(crate) source: ProviderSecretSource,
    pub(crate) method: ProviderCredentialMethod,
}

impl ProviderAuthRemedy {
    pub(crate) fn advice(self) -> String {
        match self.source {
            ProviderSecretSource::StoredOAuth => format!(
                "Run `grok login --provider {}` to refresh the stored OAuth credential.",
                self.provider
            ),
            ProviderSecretSource::StoredApiKey => format!(
                "Run `grok login --provider {} --api-key` to replace the stored API key.",
                self.provider
            ),
            ProviderSecretSource::Environment(key) => format!(
                "Update `{key}`, or run `grok login --provider {} --api-key` to store a credential that owns this provider.",
                self.provider
            ),
            ProviderSecretSource::Model => {
                "Update the model's `api_key`/`env_key` credential; model-scoped BYOK takes precedence over stored provider credentials.".to_owned()
            }
        }
    }
}

pub(crate) fn provider_auth_remedy(
    provider: ProviderId,
    source: ProviderSecretSource,
) -> ProviderAuthRemedy {
    ProviderAuthRemedy {
        provider,
        source,
        method: source.method(),
    }
}

#[derive(Clone)]
pub(crate) struct ProviderSecret {
    pub(crate) token: String,
    pub(crate) auth_scheme: AuthScheme,
    pub(crate) source: ProviderSecretSource,
}

impl ProviderSecret {
    pub(crate) fn from_model(provider: ProviderId, token: String) -> anyhow::Result<Self> {
        if provider == ProviderId::OpenaiCodex {
            bail!("OpenAI Codex requires OAuth login and does not accept BYOK credentials");
        }
        secret(provider, token, ProviderSecretSource::Model)
    }

    pub(crate) fn from_oauth(provider: ProviderId, credential: &ProviderCredential) -> Self {
        Self {
            token: credential.access.clone(),
            auth_scheme: oauth_scheme(provider),
            source: ProviderSecretSource::StoredOAuth,
        }
    }

    pub(crate) fn from_api_key(provider: ProviderId, key: String) -> anyhow::Result<Self> {
        secret(provider, key, ProviderSecretSource::StoredApiKey)
    }

    /// Resolve only provider-scoped variables. This intentionally never reads
    /// `XAI_API_KEY` or the xAI session bearer.
    pub(crate) fn from_environment(provider: ProviderId) -> anyhow::Result<Option<Self>> {
        for key in provider_descriptor(provider).env_keys {
            let Ok(value) = std::env::var(key) else {
                continue;
            };
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            return secret(
                provider,
                value.to_owned(),
                ProviderSecretSource::Environment(key),
            )
            .map(Some);
        }
        Ok(None)
    }
}

#[derive(Debug)]
struct ProviderBearerResolver {
    provider: ProviderId,
}

impl xai_grok_sampler::BearerResolver for ProviderBearerResolver {
    fn current_bearer(&self) -> Option<String> {
        match ProviderStore::default().get(self.provider) {
            Ok(ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential))) => {
                Some(credential.access)
            }
            Ok(ProviderSlotState::Missing) => None,
            Ok(ProviderSlotState::Known(ProviderStoredCredential::ApiKey(_))) => {
                tracing::warn!(provider = %self.provider, "provider bearer reload found an API-key slot; OAuth resolver disabled");
                None
            }
            Ok(ProviderSlotState::PresentUnsupportedOrInvalid) => {
                tracing::warn!(provider = %self.provider, "provider bearer reload found an unsupported or invalid slot");
                None
            }
            Err(error) => {
                let error = xai_acp_lib::redact_provider_auth_error(&error.to_string());
                tracing::warn!(provider = %self.provider, %error, "provider bearer reload failed");
                None
            }
        }
    }
}

pub(crate) fn provider_bearer_resolver(
    provider: ProviderId,
) -> xai_grok_sampler::SharedBearerResolver {
    std::sync::Arc::new(ProviderBearerResolver { provider })
}

fn model_provider_secret(
    provider: ProviderId,
    model_secret: Option<String>,
) -> anyhow::Result<Option<(ProviderSecret, Option<ProviderStoredCredential>)>> {
    model_secret
        .filter(|token| !token.trim().is_empty())
        .map(|token| ProviderSecret::from_model(provider, token).map(|secret| (secret, None)))
        .transpose()
}

fn stored_or_environment_secret_with<F>(
    provider: ProviderId,
    slot: ProviderSlotState,
    environment: F,
) -> anyhow::Result<(ProviderSecret, Option<ProviderStoredCredential>)>
where
    F: FnOnce(ProviderId) -> anyhow::Result<Option<ProviderSecret>>,
{
    match slot {
        ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)) => {
            let secret = ProviderSecret::from_oauth(provider, &credential);
            Ok((secret, Some(ProviderStoredCredential::OAuth(credential))))
        }
        ProviderSlotState::Known(ProviderStoredCredential::ApiKey(credential)) => {
            let secret = ProviderSecret::from_api_key(provider, credential.key.clone())?;
            Ok((secret, Some(ProviderStoredCredential::ApiKey(credential))))
        }
        ProviderSlotState::PresentUnsupportedOrInvalid => {
            bail!(
                "{provider} credential slot is unsupported or invalid; remove or replace the stored credential"
            )
        }
        ProviderSlotState::Missing => {
            if let Some(secret) = environment(provider)? {
                return Ok((secret, None));
            }
            let env_keys = provider_descriptor(provider).env_keys;
            if env_keys.is_empty() {
                bail!("{provider} credentials are missing; run `grok login --provider {provider}`");
            }
            bail!(
                "{provider} credentials are missing; run `grok login --provider {provider}` or set one of: {}",
                env_keys.join(", ")
            )
        }
    }
}

fn stored_or_environment_secret(
    provider: ProviderId,
    slot: ProviderSlotState,
) -> anyhow::Result<(ProviderSecret, Option<ProviderStoredCredential>)> {
    stored_or_environment_secret_with(provider, slot, ProviderSecret::from_environment)
}

pub(crate) fn resolve_provider_secret(
    provider: ProviderId,
    model_secret: Option<String>,
) -> anyhow::Result<(ProviderSecret, Option<ProviderStoredCredential>)> {
    if let Some(secret) = model_provider_secret(provider, model_secret)? {
        return Ok(secret);
    }
    stored_or_environment_secret(provider, ProviderStore::default().get(provider)?)
}

pub(crate) async fn resolve_fresh_provider_secret(
    provider: ProviderId,
    model_secret: Option<String>,
) -> anyhow::Result<(ProviderSecret, Option<ProviderStoredCredential>)> {
    if let Some(secret) = model_provider_secret(provider, model_secret)? {
        return Ok(secret);
    }
    stored_or_environment_secret(provider, super::fresh_stored_slot(provider, None).await?)
}

impl fmt::Debug for ProviderSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderSecret")
            .field("token", &"[REDACTED]")
            .field("auth_scheme", &self.auth_scheme)
            .field("source", &self.source)
            .finish()
    }
}

fn secret(
    provider: ProviderId,
    token: String,
    source: ProviderSecretSource,
) -> anyhow::Result<ProviderSecret> {
    if token.trim().is_empty() {
        bail!("{provider} credential is empty");
    }
    if provider == ProviderId::OpenaiCodex {
        bail!("OpenAI Codex requires OAuth login and does not accept API-key credentials");
    }
    let auth_scheme = if provider == ProviderId::Anthropic
        && !matches!(
            source,
            ProviderSecretSource::Environment("ANTHROPIC_AUTH_TOKEN")
        )
        && !token.contains("sk-ant-oat")
    {
        AuthScheme::XApiKey
    } else {
        AuthScheme::Bearer
    };
    Ok(ProviderSecret {
        token,
        auth_scheme,
        source,
    })
}

fn oauth_scheme(provider: ProviderId) -> AuthScheme {
    match provider {
        ProviderId::Anthropic
        | ProviderId::OpenaiCodex
        | ProviderId::GithubCopilot
        | ProviderId::Openrouter
        | ProviderId::KimiCoding
        | ProviderId::Radius => AuthScheme::Bearer,
    }
}

#[derive(Clone)]
pub(crate) struct ProviderRequestContext {
    pub(crate) provider_id: ProviderId,
    pub(crate) principal: String,
    pub(crate) token: String,
    pub(crate) auth_scheme: AuthScheme,
    pub(crate) base_url: String,
    pub(crate) headers: IndexMap<String, String>,
    pub(crate) api_backend: ApiBackend,
    pub(crate) wire_dialect: ProviderWireDialect,
    pub(crate) catalog_model_id: String,
    pub(crate) upstream_model_id: String,
    pub(crate) credential_source: ProviderSecretSource,
}

impl ProviderRequestContext {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        provider_id: ProviderId,
        upstream_model_id: impl Into<String>,
        secret: ProviderSecret,
        credential: Option<&ProviderStoredCredential>,
        base_url_override: Option<&str>,
        api_backend_override: Option<ApiBackend>,
        wire_dialect_override: Option<ProviderWireDialect>,
    ) -> anyhow::Result<Self> {
        let descriptor = provider_descriptor(provider_id);
        let upstream_model_id = upstream_model_id.into();
        if upstream_model_id.trim().is_empty() {
            bail!("{provider_id} model ID is empty");
        }
        let base_url = dynamic_base_url(provider_id, credential, base_url_override)
            .unwrap_or(descriptor.base_url)
            .trim_end_matches('/')
            .to_owned();
        validate_http_url(&base_url)?;
        let headers = provider_headers(provider_id, credential, &secret)?;
        let principal = credential
            .and_then(|credential| {
                credential
                    .metadata_str("accountId")
                    .or_else(|| credential.metadata_str("enterpriseUrl"))
            })
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| token_principal(provider_id, &secret.token));

        Ok(Self {
            provider_id,
            principal,
            token: secret.token,
            auth_scheme: secret.auth_scheme,
            base_url,
            headers,
            api_backend: api_backend_override.unwrap_or(descriptor.api_backend),
            wire_dialect: wire_dialect_override.unwrap_or(descriptor.wire_dialect),
            catalog_model_id: namespaced_model_id(provider_id, &upstream_model_id),
            upstream_model_id,
            credential_source: secret.source,
        })
    }
}

impl fmt::Debug for ProviderRequestContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderRequestContext")
            .field("provider_id", &self.provider_id)
            .field("principal", &self.principal)
            .field("token", &"[REDACTED]")
            .field("auth_scheme", &self.auth_scheme)
            .field("base_url", &self.base_url)
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .field("api_backend", &self.api_backend)
            .field("wire_dialect", &self.wire_dialect)
            .field("catalog_model_id", &self.catalog_model_id)
            .field("upstream_model_id", &self.upstream_model_id)
            .field("credential_source", &self.credential_source)
            .finish()
    }
}

fn dynamic_base_url<'a>(
    provider: ProviderId,
    credential: Option<&'a ProviderStoredCredential>,
    override_url: Option<&'a str>,
) -> Option<&'a str> {
    override_url.or_else(|| match provider {
        ProviderId::GithubCopilot => credential.and_then(|value| value.metadata_str("baseUrl")),
        ProviderId::Radius => credential
            .and_then(|value| value.metadata("gatewayConfig"))
            .and_then(|value| value.get("baseUrl"))
            .and_then(serde_json::Value::as_str)
            .or_else(|| credential.and_then(|value| value.metadata_str("baseUrl"))),
        _ => None,
    })
}

fn provider_headers(
    provider: ProviderId,
    credential: Option<&ProviderStoredCredential>,
    secret: &ProviderSecret,
) -> anyhow::Result<IndexMap<String, String>> {
    let mut headers = IndexMap::new();
    match provider {
        ProviderId::Anthropic => {
            headers.insert("anthropic-version".to_owned(), "2023-06-01".to_owned());
            if secret.auth_scheme == AuthScheme::Bearer {
                headers.insert(
                    "anthropic-beta".to_owned(),
                    "claude-code-20250219,oauth-2025-04-20".to_owned(),
                );
                headers.insert("user-agent".to_owned(), "claude-cli/2.1.75".to_owned());
                headers.insert("x-app".to_owned(), "cli".to_owned());
            }
        }
        ProviderId::OpenaiCodex => {
            let account_id = credential
                .and_then(|value| value.metadata_str("accountId"))
                .context("OpenAI Codex credential is missing accountId")?;
            headers.insert("chatgpt-account-id".to_owned(), account_id.to_owned());
            headers.insert("originator".to_owned(), "pi".to_owned());
            headers.insert(
                "openai-beta".to_owned(),
                "responses=experimental".to_owned(),
            );
        }
        ProviderId::GithubCopilot => {
            headers.insert("anthropic-version".to_owned(), "2023-06-01".to_owned());
            headers.insert(
                "user-agent".to_owned(),
                "GitHubCopilotChat/0.35.0".to_owned(),
            );
            headers.insert("editor-version".to_owned(), "vscode/1.107.0".to_owned());
            headers.insert(
                "editor-plugin-version".to_owned(),
                "copilot-chat/0.35.0".to_owned(),
            );
            headers.insert(
                "copilot-integration-id".to_owned(),
                "vscode-chat".to_owned(),
            );
        }
        ProviderId::KimiCoding => {
            headers.insert("anthropic-version".to_owned(), "2023-06-01".to_owned());
            headers.insert("user-agent".to_owned(), "KimiCLI/1.5".to_owned());
        }
        ProviderId::Openrouter | ProviderId::Radius => {}
    }
    Ok(headers)
}

fn token_principal(provider: ProviderId, token: &str) -> String {
    let digest = Sha256::digest(token.as_bytes());
    let suffix = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{provider}:{suffix}")
}

pub(crate) fn namespaced_model_id(provider: ProviderId, upstream_model_id: &str) -> String {
    format!("{provider}/{upstream_model_id}")
}

pub(crate) fn parse_namespaced_model_id(value: &str) -> Option<(ProviderId, &str)> {
    let (provider, upstream_model_id) = value.split_once('/')?;
    let provider = provider.parse().ok()?;
    (!upstream_model_id.is_empty()).then_some((provider, upstream_model_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaced_model_ids_preserve_provider_model_slashes() {
        let id = namespaced_model_id(ProviderId::Openrouter, "anthropic/claude-sonnet");
        assert_eq!(
            parse_namespaced_model_id(&id),
            Some((ProviderId::Openrouter, "anthropic/claude-sonnet"))
        );
        assert!(parse_namespaced_model_id("grok-4").is_none());
    }

    fn anthropic_context(secret: ProviderSecret) -> ProviderRequestContext {
        ProviderRequestContext::build(
            ProviderId::Anthropic,
            "claude-test",
            secret,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn anthropic_scheme_and_oauth_header_truth_table() {
        let oauth = ProviderCredential::oauth("stored-oauth", "refresh", u64::MAX);
        let cases = [
            ProviderSecret::from_oauth(ProviderId::Anthropic, &oauth),
            secret(
                ProviderId::Anthropic,
                "opaque-auth-token".to_owned(),
                ProviderSecretSource::Environment("ANTHROPIC_AUTH_TOKEN"),
            )
            .unwrap(),
            secret(
                ProviderId::Anthropic,
                "plain-stored-key".to_owned(),
                ProviderSecretSource::StoredApiKey,
            )
            .unwrap(),
            secret(
                ProviderId::Anthropic,
                "prefix-sk-ant-oat-middle".to_owned(),
                ProviderSecretSource::StoredApiKey,
            )
            .unwrap(),
            secret(
                ProviderId::Anthropic,
                "plain-oauth-env-value".to_owned(),
                ProviderSecretSource::Environment("ANTHROPIC_OAUTH_TOKEN"),
            )
            .unwrap(),
            secret(
                ProviderId::Anthropic,
                "prefix-sk-ant-oat-middle".to_owned(),
                ProviderSecretSource::Environment("ANTHROPIC_API_KEY"),
            )
            .unwrap(),
        ];
        let expected = [
            AuthScheme::Bearer,
            AuthScheme::Bearer,
            AuthScheme::XApiKey,
            AuthScheme::Bearer,
            AuthScheme::XApiKey,
            AuthScheme::Bearer,
        ];
        for (secret, expected_scheme) in cases.into_iter().zip(expected) {
            let context = anthropic_context(secret);
            assert_eq!(context.auth_scheme, expected_scheme);
            assert_eq!(
                context.headers.contains_key("anthropic-beta"),
                expected_scheme == AuthScheme::Bearer
            );
            assert_eq!(
                context.headers.contains_key("x-app"),
                expected_scheme == AuthScheme::Bearer
            );
        }
    }

    #[test]
    fn present_invalid_slot_blocks_environment_fallback() {
        let calls = std::cell::Cell::new(0);
        let result = stored_or_environment_secret_with(
            ProviderId::Anthropic,
            ProviderSlotState::PresentUnsupportedOrInvalid,
            |_| {
                calls.set(calls.get() + 1);
                Ok(Some(secret(
                    ProviderId::Anthropic,
                    "environment-key".to_owned(),
                    ProviderSecretSource::Environment("ANTHROPIC_API_KEY"),
                )?))
            },
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn codex_rejects_all_api_key_sources() {
        for source in [
            ProviderSecretSource::Model,
            ProviderSecretSource::StoredApiKey,
            ProviderSecretSource::Environment("OPENAI_API_KEY"),
        ] {
            assert!(secret(ProviderId::OpenaiCodex, "key".to_owned(), source).is_err());
        }
    }

    #[test]
    fn provider_auth_remedies_are_source_and_method_aware() {
        let stored =
            provider_auth_remedy(ProviderId::Anthropic, ProviderSecretSource::StoredApiKey);
        assert_eq!(stored.method, ProviderCredentialMethod::ApiKey);
        assert_eq!(stored.source.as_str(), "stored_api_key");
        assert!(stored.advice().contains("--api-key"));

        let environment = provider_auth_remedy(
            ProviderId::Openrouter,
            ProviderSecretSource::Environment("OPENROUTER_API_KEY"),
        );
        assert!(environment.advice().contains("OPENROUTER_API_KEY"));

        let model = provider_auth_remedy(ProviderId::Radius, ProviderSecretSource::Model);
        assert!(model.advice().contains("model-scoped BYOK"));
    }

    #[test]
    fn provider_context_never_uses_xai_environment_fallback() {
        // Resolver has a closed provider-specific list; xAI is absent even if
        // callers happen to have an xAI credential in their process.
        for provider in ProviderId::ALL {
            assert!(
                !provider_descriptor(provider)
                    .env_keys
                    .contains(&"XAI_API_KEY")
            );
        }
    }

    #[test]
    fn context_uses_dynamic_copilot_base_and_redacts_token() {
        let mut credential = ProviderCredential::oauth("oauth-secret", "github-token", u64::MAX);
        credential.set_metadata("baseUrl", "https://api.enterprise.example");
        let stored = ProviderStoredCredential::OAuth(credential.clone());
        let context = ProviderRequestContext::build(
            ProviderId::GithubCopilot,
            "gpt-4.1",
            ProviderSecret::from_oauth(ProviderId::GithubCopilot, &credential),
            Some(&stored),
            None,
            Some(ApiBackend::Responses),
            None,
        )
        .unwrap();
        assert_eq!(context.base_url, "https://api.enterprise.example");
        assert_eq!(context.api_backend, ApiBackend::Responses);
        assert!(!format!("{context:?}").contains("oauth-secret"));
    }
}

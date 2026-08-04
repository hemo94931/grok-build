use std::fmt;

use anyhow::{Context, bail};
use indexmap::IndexMap;
use sha2::{Digest, Sha256};
use xai_grok_sampler::{ApiBackend, AuthScheme};

use super::flow::validate_http_url;
use super::{ProviderCredential, ProviderId, ProviderStore};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderLoginFlow {
    Browser,
    DeviceCode,
    BrowserOrDevice,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderCapabilities {
    pub(crate) supports_remote_compaction: bool,
    pub(crate) accepts_responses_checkpoint: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ProviderDescriptor {
    pub(crate) id: ProviderId,
    pub(crate) display_name: &'static str,
    pub(crate) login_flow: ProviderLoginFlow,
    pub(crate) base_url: &'static str,
    pub(crate) api_backend: ApiBackend,
    pub(crate) wire_dialect: ProviderWireDialect,
    pub(crate) env_keys: &'static [&'static str],
    pub(crate) capabilities: ProviderCapabilities,
}

pub(crate) fn provider_descriptor(id: ProviderId) -> ProviderDescriptor {
    let capabilities = ProviderCapabilities::default();
    match id {
        ProviderId::Anthropic => ProviderDescriptor {
            id,
            display_name: "Anthropic",
            login_flow: ProviderLoginFlow::Browser,
            base_url: "https://api.anthropic.com",
            api_backend: ApiBackend::Messages,
            wire_dialect: ProviderWireDialect::AnthropicMessages,
            env_keys: &[
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_OAUTH_TOKEN",
                "ANTHROPIC_API_KEY",
            ],
            capabilities,
        },
        ProviderId::OpenaiCodex => ProviderDescriptor {
            id,
            display_name: "OpenAI Codex",
            login_flow: ProviderLoginFlow::BrowserOrDevice,
            base_url: "https://chatgpt.com/backend-api",
            api_backend: ApiBackend::Responses,
            wire_dialect: ProviderWireDialect::OpenaiCodexResponses,
            env_keys: &[],
            capabilities,
        },
        ProviderId::GithubCopilot => ProviderDescriptor {
            id,
            display_name: "GitHub Copilot",
            login_flow: ProviderLoginFlow::DeviceCode,
            base_url: "https://api.individual.githubcopilot.com",
            api_backend: ApiBackend::ChatCompletions,
            wire_dialect: ProviderWireDialect::GithubCopilot,
            env_keys: &["COPILOT_GITHUB_TOKEN"],
            capabilities,
        },
        ProviderId::Openrouter => ProviderDescriptor {
            id,
            display_name: "OpenRouter",
            login_flow: ProviderLoginFlow::Browser,
            base_url: "https://openrouter.ai/api/v1",
            api_backend: ApiBackend::ChatCompletions,
            wire_dialect: ProviderWireDialect::OpenaiChatCompletions,
            env_keys: &["OPENROUTER_API_KEY"],
            capabilities,
        },
        ProviderId::KimiCoding => ProviderDescriptor {
            id,
            display_name: "Kimi Coding",
            login_flow: ProviderLoginFlow::DeviceCode,
            base_url: "https://api.kimi.com/coding",
            api_backend: ApiBackend::Messages,
            wire_dialect: ProviderWireDialect::KimiAnthropicMessages,
            env_keys: &["KIMI_API_KEY"],
            capabilities,
        },
        ProviderId::Radius => ProviderDescriptor {
            id,
            display_name: "Radius",
            login_flow: ProviderLoginFlow::BrowserOrDevice,
            base_url: "https://radius.pi.dev",
            // The provider wire dispatcher intercepts this dialect before the
            // generic Messages conversion. No fourth shared ApiBackend variant
            // is needed.
            api_backend: ApiBackend::Messages,
            wire_dialect: ProviderWireDialect::PiMessages,
            env_keys: &["RADIUS_API_KEY"],
            capabilities,
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderSecretSource {
    Model,
    StoredOAuth,
    Environment(&'static str),
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
        ProviderStore::default()
            .get(self.provider)
            .map_err(|error| {
                tracing::warn!(provider = %self.provider, %error, "provider bearer reload failed");
                error
            })
            .ok()
            .flatten()
            .map(|credential| credential.access)
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
) -> anyhow::Result<Option<(ProviderSecret, Option<ProviderCredential>)>> {
    model_secret
        .filter(|token| !token.trim().is_empty())
        .map(|token| ProviderSecret::from_model(provider, token).map(|secret| (secret, None)))
        .transpose()
}

fn stored_or_environment_secret(
    provider: ProviderId,
    credential: Option<ProviderCredential>,
) -> anyhow::Result<(ProviderSecret, Option<ProviderCredential>)> {
    if let Some(credential) = credential {
        let secret = ProviderSecret::from_oauth(provider, &credential);
        return Ok((secret, Some(credential)));
    }
    if let Some(secret) = ProviderSecret::from_environment(provider)? {
        return Ok((secret, None));
    }
    bail!(
        "{provider} credentials are missing; run `grok login --provider {provider}` or set one of: {}",
        provider_descriptor(provider).env_keys.join(", ")
    )
}

pub(crate) fn resolve_provider_secret(
    provider: ProviderId,
    model_secret: Option<String>,
) -> anyhow::Result<(ProviderSecret, Option<ProviderCredential>)> {
    if let Some(secret) = model_provider_secret(provider, model_secret)? {
        return Ok(secret);
    }
    stored_or_environment_secret(provider, ProviderStore::default().get(provider)?)
}

pub(crate) async fn resolve_fresh_provider_secret(
    provider: ProviderId,
    model_secret: Option<String>,
) -> anyhow::Result<(ProviderSecret, Option<ProviderCredential>)> {
    if let Some(secret) = model_provider_secret(provider, model_secret)? {
        return Ok(secret);
    }
    stored_or_environment_secret(
        provider,
        super::fresh_stored_credential(provider, None).await?,
    )
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
    let auth_scheme = if provider == ProviderId::Anthropic
        && !token.starts_with("sk-ant-oat")
        && !matches!(
            source,
            ProviderSecretSource::Environment("ANTHROPIC_AUTH_TOKEN" | "ANTHROPIC_OAUTH_TOKEN")
        ) {
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
    pub(crate) capabilities: ProviderCapabilities,
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
        credential: Option<&ProviderCredential>,
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
            capabilities: descriptor.capabilities,
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
            .field("capabilities", &self.capabilities)
            .field("catalog_model_id", &self.catalog_model_id)
            .field("upstream_model_id", &self.upstream_model_id)
            .field("credential_source", &self.credential_source)
            .finish()
    }
}

fn dynamic_base_url<'a>(
    provider: ProviderId,
    credential: Option<&'a ProviderCredential>,
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
    credential: Option<&ProviderCredential>,
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

    #[test]
    fn every_non_xai_route_fails_closed_for_first_party_capabilities() {
        for provider in ProviderId::ALL {
            let capabilities = provider_descriptor(provider).capabilities;
            assert!(!capabilities.supports_remote_compaction);
            assert!(!capabilities.accepts_responses_checkpoint);
        }
    }

    #[test]
    fn multi_provider_regression_anthropic_oauth_env_is_bearer() {
        let oauth = secret(
            ProviderId::Anthropic,
            "opaque-oauth-token".to_owned(),
            ProviderSecretSource::Environment("ANTHROPIC_OAUTH_TOKEN"),
        )
        .unwrap();
        let api_key = secret(
            ProviderId::Anthropic,
            "sk-ant-api".to_owned(),
            ProviderSecretSource::Environment("ANTHROPIC_API_KEY"),
        )
        .unwrap();
        assert_eq!(oauth.auth_scheme, AuthScheme::Bearer);
        assert_eq!(api_key.auth_scheme, AuthScheme::XApiKey);
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
        let context = ProviderRequestContext::build(
            ProviderId::GithubCopilot,
            "gpt-4.1",
            ProviderSecret::from_oauth(ProviderId::GithubCopilot, &credential),
            Some(&credential),
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

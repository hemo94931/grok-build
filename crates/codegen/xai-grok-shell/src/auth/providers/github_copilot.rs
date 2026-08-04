use std::time::Duration;

use anyhow::{Context, bail};
use futures_util::{StreamExt, stream};
use reqwest::RequestBuilder;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::flow::{DevicePoll, DevicePollOptions};
use super::{
    AuthInteraction, AuthNotification, AuthPrompt, DeviceCode, LoginMode, ProviderCredential,
    ProviderId, ProviderRefreshOutcome,
};

const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";
const COPILOT_USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
const COPILOT_API_VERSION: &str = "2026-06-01";
const DEFAULT_BASE_URL: &str = "https://api.individual.githubcopilot.com";
const EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;
const MODEL_LIST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Deserialize)]
struct DeviceAuthorizationResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: Option<u64>,
    expires_in: u64,
}

#[derive(Deserialize)]
struct DeviceTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    interval: Option<u64>,
}

#[derive(Deserialize)]
struct CopilotTokenResponse {
    token: String,
    expires_at: u64,
}

pub(super) async fn login(
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    if matches!(mode, Some(LoginMode::Browser)) {
        bail!("GitHub Copilot only supports device-code login");
    }

    let signal = interaction.signal();
    if signal.is_cancelled() {
        bail!("login cancelled");
    }
    let input = interaction
        .prompt(AuthPrompt::Text {
            message: "GitHub Enterprise URL/domain (blank for github.com)".to_owned(),
            placeholder: "company.ghe.com".to_owned(),
        })
        .await?;
    if signal.is_cancelled() {
        bail!("login cancelled");
    }

    let domain = super::flow::normalize_domain(&input)?;
    let enterprise_domain = (!input.trim().is_empty()).then_some(domain.as_str());
    let client = crate::http::shared_client();
    let device = tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        result = start_device_flow(&client, &domain) => result?,
    };
    tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        result = super::flow::notify_device_code(interaction, &device) => result?,
    }

    let github_access_token =
        poll_github_access_token(&client, &domain, &device, signal.clone()).await?;
    let mut credential = tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        result = issue_copilot_token(&client, &github_access_token, enterprise_domain) => result?,
    };
    interaction
        .notify(AuthNotification::Progress {
            message: "Enabling GitHub Copilot models...".to_owned(),
        })
        .await?;
    enable_known_models(&client, &credential, &signal).await;
    if let Some(ids) = try_fetch_available_model_ids(&client, &credential, &signal).await? {
        credential.set_metadata("availableModelIds", ids);
    }
    Ok(credential)
}

pub(super) async fn refresh(
    credential: ProviderCredential,
    signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    if signal.is_cancelled() {
        bail!("refresh cancelled");
    }
    let enterprise_domain = credential
        .metadata_str("enterpriseUrl")
        .filter(|value| !value.trim().is_empty())
        .map(super::flow::normalize_domain)
        .transpose()?;
    let previous_models = credential.metadata("availableModelIds").cloned();
    let github_access_token = credential.refresh;
    if github_access_token.trim().is_empty() {
        bail!("GitHub Copilot credential is missing its GitHub access token");
    }

    let client = crate::http::shared_client();
    let mut refreshed = tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("refresh cancelled"),
        result = issue_copilot_token(&client, &github_access_token, enterprise_domain.as_deref()) => result?,
    };
    match try_fetch_available_model_ids(&client, &refreshed, &signal).await? {
        Some(ids) => refreshed.set_metadata("availableModelIds", ids),
        None => {
            if let Some(ids) = previous_models {
                refreshed.set_metadata("availableModelIds", ids);
            }
        }
    }
    Ok(ProviderRefreshOutcome::Save(refreshed))
}

async fn start_device_flow(client: &reqwest::Client, domain: &str) -> anyhow::Result<DeviceCode> {
    let response = client
        .post(format!("https://{domain}/login/device/code"))
        .header(reqwest::header::ACCEPT, "application/json")
        .header(reqwest::header::USER_AGENT, COPILOT_USER_AGENT)
        .form(&[("client_id", CLIENT_ID), ("scope", "read:user")])
        .send()
        .await
        .context("GitHub device-code request failed")?;
    let raw: DeviceAuthorizationResponse =
        decode_json(response, "GitHub device-code request").await?;
    if raw.device_code.trim().is_empty() || raw.user_code.trim().is_empty() {
        bail!("GitHub returned an invalid device-code response");
    }
    let verification_uri = super::flow::validate_http_url(&raw.verification_uri)
        .context("GitHub returned an untrusted verification_uri")?
        .to_string();

    Ok(DeviceCode {
        device_code: raw.device_code,
        user_code: raw.user_code,
        verification_uri,
        verification_uri_complete: None,
        interval_seconds: raw.interval.unwrap_or(5),
        expires_in_seconds: raw.expires_in,
    })
}

async fn poll_github_access_token(
    client: &reqwest::Client,
    domain: &str,
    device: &DeviceCode,
    signal: CancellationToken,
) -> anyhow::Result<String> {
    let client = client.clone();
    let token_url = format!("https://{domain}/login/oauth/access_token");
    let device_code = device.device_code.clone();
    let request_signal = signal.clone();

    super::flow::poll_device_code(
        DevicePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: device.expires_in_seconds,
            wait_before_first_poll: true,
        },
        signal,
        move || {
            let client = client.clone();
            let token_url = token_url.clone();
            let device_code = device_code.clone();
            let request_signal = request_signal.clone();
            async move {
                let request = client
                    .post(token_url)
                    .header(reqwest::header::ACCEPT, "application/json")
                    .header(reqwest::header::USER_AGENT, COPILOT_USER_AGENT)
                    .form(&[
                        ("client_id", CLIENT_ID),
                        ("device_code", device_code.as_str()),
                        ("grant_type", DEVICE_GRANT_TYPE),
                    ]);
                let response = tokio::select! {
                    biased;
                    _ = request_signal.cancelled() => bail!("login cancelled"),
                    result = request.send() => result.context("GitHub device token request failed")?,
                };
                let raw: DeviceTokenResponse =
                    decode_json(response, "GitHub device token request").await?;
                if let Some(token) = raw.access_token.filter(|token| !token.trim().is_empty()) {
                    return Ok(DevicePoll::Complete(token));
                }
                match raw.error.as_deref() {
                    Some("authorization_pending") => Ok(DevicePoll::Pending),
                    Some("slow_down") => Ok(DevicePoll::SlowDown {
                        interval_seconds: raw.interval,
                    }),
                    Some(_) => bail!("GitHub device authorization failed"),
                    None => bail!("GitHub returned an invalid device token response"),
                }
            }
        },
    )
    .await
}

async fn issue_copilot_token(
    client: &reqwest::Client,
    github_access_token: &str,
    enterprise_domain: Option<&str>,
) -> anyhow::Result<ProviderCredential> {
    let domain = enterprise_domain.unwrap_or("github.com");
    let request = with_copilot_headers(
        client
            .get(format!("https://api.{domain}/copilot_internal/v2/token"))
            .bearer_auth(github_access_token),
    );
    let response = request
        .send()
        .await
        .context("GitHub Copilot token request failed")?;
    let raw: CopilotTokenResponse = decode_json(response, "GitHub Copilot token request").await?;
    if raw.token.trim().is_empty() {
        bail!("GitHub returned an invalid Copilot token response");
    }

    let base_url = github_copilot_base_url(&raw.token, enterprise_domain);
    let mut credential = ProviderCredential::oauth(
        raw.token,
        github_access_token,
        raw.expires_at
            .saturating_mul(1000)
            .saturating_sub(EXPIRY_SKEW_MS),
    );
    if let Some(domain) = enterprise_domain {
        credential.set_metadata("enterpriseUrl", domain);
    }
    credential.set_metadata("baseUrl", base_url);
    Ok(credential)
}

async fn enable_known_models(
    client: &reqwest::Client,
    credential: &ProviderCredential,
    signal: &CancellationToken,
) {
    let Some(base_url) = credential.metadata_str("baseUrl") else {
        return;
    };
    let models = super::provider_models(ProviderId::GithubCopilot, None);
    let access = credential.access.clone();
    let base_url = base_url.trim_end_matches('/').to_owned();
    let signal = signal.clone();
    let enabled = stream::iter(models.into_iter().map(|model| {
        let client = client.clone();
        let access = access.clone();
        let base_url = base_url.clone();
        let signal = signal.clone();
        async move {
            if signal.is_cancelled() {
                return false;
            }
            with_copilot_headers(
                client
                    .post(format!("{base_url}/models/{}/policy", model.id))
                    .bearer_auth(access)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .header("openai-intent", "chat-policy")
                    .header("x-interaction-type", "chat-policy")
                    .timeout(MODEL_LIST_TIMEOUT)
                    .json(&serde_json::json!({ "state": "enabled" })),
            )
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        }
    }))
    .buffer_unordered(4)
    .filter(|enabled| std::future::ready(*enabled))
    .count()
    .await;
    tracing::debug!(
        enabled,
        "provider auth: enabled GitHub Copilot model policies"
    );
}

async fn try_fetch_available_model_ids(
    client: &reqwest::Client,
    credential: &ProviderCredential,
    signal: &CancellationToken,
) -> anyhow::Result<Option<Vec<String>>> {
    let base_url = credential
        .metadata_str("baseUrl")
        .context("GitHub Copilot credential is missing baseUrl")?;
    tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("authentication cancelled"),
        result = fetch_available_model_ids(client, &credential.access, base_url) => Ok(result.ok()),
    }
}

async fn fetch_available_model_ids(
    client: &reqwest::Client,
    copilot_token: &str,
    base_url: &str,
) -> anyhow::Result<Vec<String>> {
    let request = with_copilot_headers(
        client
            .get(format!("{}/models", base_url.trim_end_matches('/')))
            .bearer_auth(copilot_token),
    )
    .header("X-GitHub-Api-Version", COPILOT_API_VERSION)
    .timeout(MODEL_LIST_TIMEOUT);
    let response = request
        .send()
        .await
        .context("GitHub Copilot model-list request failed")?;
    let raw: Value = decode_json(response, "GitHub Copilot model-list request").await?;
    available_model_ids(&raw)
}

fn with_copilot_headers(request: RequestBuilder) -> RequestBuilder {
    request
        .header(reqwest::header::ACCEPT, "application/json")
        .header(reqwest::header::USER_AGENT, COPILOT_USER_AGENT)
        .header("Editor-Version", "vscode/1.107.0")
        .header("Editor-Plugin-Version", "copilot-chat/0.35.0")
        .header("Copilot-Integration-Id", "vscode-chat")
}

async fn decode_json<T: DeserializeOwned>(
    response: reqwest::Response,
    operation: &str,
) -> anyhow::Result<T> {
    super::flow::read_json(response, operation).await
}

fn github_copilot_base_url(token: &str, enterprise_url: Option<&str>) -> String {
    base_url_from_token(token)
        .or_else(|| {
            let enterprise_url = enterprise_url?.trim();
            if enterprise_url.is_empty() {
                return None;
            }
            let domain = super::flow::normalize_domain(enterprise_url).ok()?;
            let candidate = format!("https://copilot-api.{domain}");
            super::flow::validate_http_url(&candidate).ok()?;
            Some(candidate)
        })
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
}

fn base_url_from_token(token: &str) -> Option<String> {
    let endpoint = token
        .split(';')
        .map(str::trim)
        .find_map(|field| field.strip_prefix("proxy-ep="))?
        .trim();
    if endpoint.is_empty() {
        return None;
    }

    let candidate = if endpoint.contains("://") {
        endpoint.to_owned()
    } else {
        format!("https://{endpoint}")
    };
    let mut url = super::flow::validate_http_url(&candidate).ok()?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return None;
    }

    let host = url.host_str()?.to_owned();
    if let Some(suffix) = host.strip_prefix("proxy.") {
        if suffix.is_empty() {
            return None;
        }
        url.set_host(Some(&format!("api.{suffix}"))).ok()?;
    }
    Some(url.as_str().trim_end_matches('/').to_owned())
}

fn available_model_ids(raw: &Value) -> anyhow::Result<Vec<String>> {
    let models = raw
        .get("data")
        .and_then(Value::as_array)
        .context("invalid GitHub Copilot models response")?;
    Ok(models
        .iter()
        .filter_map(|model| {
            let model = model.as_object()?;
            if model.get("model_picker_enabled").and_then(Value::as_bool) != Some(true)
                || model
                    .get("policy")
                    .and_then(Value::as_object)
                    .and_then(|policy| policy.get("state"))
                    .and_then(Value::as_str)
                    == Some("disabled")
                || model
                    .get("capabilities")
                    .and_then(Value::as_object)
                    .and_then(|capabilities| capabilities.get("supports"))
                    .and_then(Value::as_object)
                    .and_then(|supports| supports.get("tool_calls"))
                    .and_then(Value::as_bool)
                    == Some(false)
            {
                return None;
            }
            model.get("id")?.as_str().map(ToOwned::to_owned)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn copilot_base_url_uses_valid_proxy_endpoint() {
        assert_eq!(
            github_copilot_base_url(
                "tid=test;proxy-ep=proxy.business.githubcopilot.com;exp=1",
                None,
            ),
            "https://api.business.githubcopilot.com"
        );
        assert_eq!(
            github_copilot_base_url("proxy-ep=http://proxy.localhost:3000", None),
            "http://api.localhost:3000"
        );
    }

    #[test]
    fn copilot_base_url_rejects_unsafe_proxy_and_falls_back() {
        assert_eq!(
            github_copilot_base_url(
                "proxy-ep=https://proxy.example.com/path",
                Some("https://GHE.Example.com/path"),
            ),
            "https://copilot-api.ghe.example.com"
        );
        assert_eq!(
            github_copilot_base_url("proxy-ep=https://proxy.example.com@evil.example", None),
            DEFAULT_BASE_URL
        );
        assert_eq!(
            github_copilot_base_url("proxy-ep=ftp://proxy.example.com", None),
            DEFAULT_BASE_URL
        );
    }

    #[test]
    fn model_filter_keeps_only_selectable_tool_models() {
        let raw = json!({
            "data": [
                {
                    "id": "ready",
                    "model_picker_enabled": true,
                    "policy": { "state": "enabled" },
                    "capabilities": { "supports": { "tool_calls": true } }
                },
                { "id": "defaults", "model_picker_enabled": true },
                {
                    "id": "disabled",
                    "model_picker_enabled": true,
                    "policy": { "state": "disabled" }
                },
                {
                    "id": "no-tools",
                    "model_picker_enabled": true,
                    "capabilities": { "supports": { "tool_calls": false } }
                },
                { "id": "hidden", "model_picker_enabled": false },
                { "model_picker_enabled": true },
                null
            ]
        });

        assert_eq!(
            available_model_ids(&raw).unwrap(),
            vec!["ready".to_owned(), "defaults".to_owned()]
        );
        assert!(available_model_ids(&json!({ "data": {} })).is_err());
    }
}

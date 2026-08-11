use std::future::Future;
use std::time::Duration;

use anyhow::{Context, bail};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

use super::flow::{
    DevicePoll, DevicePollOptions, LoopbackServer, notify_device_code, poll_device_code,
    truncate_error_body, validate_http_url,
};
use super::{
    AuthInteraction, AuthNotification, AuthPrompt, DeviceCode, LoginMode, ProviderCredential,
    ProviderRefreshOutcome, SelectOption,
};

const DEFAULT_GATEWAY: &str = "https://radius.pi.dev";
const CLIENT_ID: &str = "pi-gateway";
const SCOPE: &str = "gateway offline_access";
const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";
const REDIRECT_URI: &str = "http://127.0.0.1:1456/oauth/callback";
const CALLBACK_PATH: &str = "/oauth/callback";
const CALLBACK_PORT: u16 = 1456;
const EXPIRY_SKEW_MS: u64 = 60_000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const LOGIN_TIMEOUT: Duration = Duration::from_secs(10 * 60);
pub(crate) const GATEWAY_CONFIG_METADATA_KEY: &str = "gatewayConfig";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RadiusGatewayConfig {
    pub(crate) base_url: String,
    pub(crate) models: Vec<RadiusGatewayModel>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RadiusGatewayModel {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) reasoning: bool,
    pub(crate) context_window: u64,
    pub(crate) max_tokens: u32,
}

impl RadiusGatewayConfig {
    pub(crate) fn from_metadata(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }
}

struct HttpPayload {
    status: StatusCode,
    json: Value,
    body: String,
}

pub(super) async fn login(
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    let signal = interaction.signal();
    let gateway = gateway_url()?;
    let mode = match mode {
        Some(mode) => mode,
        None => select_login_mode(interaction, &signal).await?,
    };
    let mut credential = match mode {
        LoginMode::Browser => browser_login(interaction, &gateway, &signal).await?,
        LoginMode::DeviceCode => device_login(interaction, &gateway, &signal).await?,
    };
    let config = load_gateway_config(&gateway, &credential.access, &signal).await?;
    credential.set_metadata(GATEWAY_CONFIG_METADATA_KEY, serde_json::to_value(config)?);
    Ok(credential)
}

pub(super) async fn refresh(
    credential: ProviderCredential,
    signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    if credential.refresh.trim().is_empty() {
        bail!("Radius credential is missing a refresh token");
    }
    let gateway = gateway_url()?;
    let previous_config = credential.metadata(GATEWAY_CONFIG_METADATA_KEY).cloned();
    let response = post_form(
        endpoint(&gateway, "/v1/oauth/token")?,
        &[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", credential.refresh.as_str()),
        ],
        &signal,
        "Radius OAuth token refresh",
    )
    .await?;
    let mut refreshed = token_credential(&response, "Radius OAuth token refresh")?;

    // The refresh token may rotate. A transient catalog failure must not lose
    // the freshly issued credential, so keep the previous validated catalog.
    match load_gateway_config(&gateway, &refreshed.access, &signal).await {
        Ok(config) => {
            refreshed.set_metadata(GATEWAY_CONFIG_METADATA_KEY, serde_json::to_value(config)?)
        }
        Err(error) => {
            tracing::warn!(%error, "provider auth: Radius model catalog refresh failed");
            if let Some(config) = previous_config {
                refreshed.set_metadata(GATEWAY_CONFIG_METADATA_KEY, config);
            }
        }
    }
    Ok(ProviderRefreshOutcome::Save(refreshed))
}

async fn select_login_mode(
    interaction: &dyn AuthInteraction,
    signal: &CancellationToken,
) -> anyhow::Result<LoginMode> {
    let selected = await_or_cancel(
        signal,
        interaction.prompt(AuthPrompt::Select {
            message: "Sign in to Radius:".to_owned(),
            options: vec![
                SelectOption {
                    id: "browser".to_owned(),
                    label: "Sign in with browser (recommended)".to_owned(),
                },
                SelectOption {
                    id: "device-code".to_owned(),
                    label: "Sign in with device code".to_owned(),
                },
            ],
        }),
    )
    .await?
    .into_text()?;
    match selected.trim() {
        "browser" => Ok(LoginMode::Browser),
        "device-code" => Ok(LoginMode::DeviceCode),
        value => bail!("unknown Radius login method `{value}`"),
    }
}

async fn browser_login(
    interaction: &dyn AuthInteraction,
    gateway: &str,
    signal: &CancellationToken,
) -> anyhow::Result<ProviderCredential> {
    let authorization_endpoint = load_discovery(gateway, signal).await?;
    let pkce = crate::auth::oidc::protocol::generate_pkce();
    let state = Uuid::new_v4().to_string();
    let server = await_or_cancel(
        signal,
        LoopbackServer::bind(
            "127.0.0.1",
            "127.0.0.1",
            CALLBACK_PORT,
            CALLBACK_PATH,
            Some(state.clone()),
        ),
    )
    .await
    .with_context(|| {
        format!(
            "Radius browser login could not bind 127.0.0.1:{CALLBACK_PORT}; choose device-code login"
        )
    })?;
    debug_assert_eq!(server.redirect_uri(), REDIRECT_URI);
    let authorize_url = authorization_url(&authorization_endpoint, &pkce.code_challenge, &state)?;
    interaction
        .notify(AuthNotification::Progress {
            message: format!("Listening for OAuth callback on {REDIRECT_URI}"),
        })
        .await?;
    interaction
        .notify(AuthNotification::AuthUrl {
            url: authorize_url,
            instructions: "Continue in your browser.".to_owned(),
        })
        .await?;
    let code = server
        .wait(
            interaction,
            "Complete login in your browser, or paste the authorization code / redirect URL here:",
            Some(&state),
            LOGIN_TIMEOUT,
        )
        .await?;
    let response = post_form(
        endpoint(gateway, "/v1/oauth/token")?,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", CLIENT_ID),
            ("redirect_uri", REDIRECT_URI),
            ("code", code.code.as_str()),
            ("code_verifier", pkce.code_verifier.as_str()),
        ],
        signal,
        "Radius OAuth token exchange",
    )
    .await?;
    token_credential(&response, "Radius OAuth token exchange")
}

async fn device_login(
    interaction: &dyn AuthInteraction,
    gateway: &str,
    signal: &CancellationToken,
) -> anyhow::Result<ProviderCredential> {
    let response = post_form(
        endpoint(gateway, "/v1/oauth/device")?,
        &[("client_id", CLIENT_ID), ("scope", SCOPE)],
        signal,
        "Radius OAuth device authorization",
    )
    .await?;
    if !response.status.is_success() {
        return oauth_error(&response, "Radius OAuth device authorization");
    }
    let device = device_code(&response.json)?;
    notify_device_code(interaction, &device).await?;

    let poll_gateway = gateway.to_owned();
    let poll_device = device.clone();
    let poll_signal = signal.clone();
    poll_device_code(
        DevicePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: device.expires_in_seconds,
            wait_before_first_poll: false,
        },
        signal.clone(),
        move || {
            let gateway = poll_gateway.clone();
            let device = poll_device.clone();
            let signal = poll_signal.clone();
            async move { poll_device_token(&gateway, &device, &signal).await }
        },
    )
    .await
}

async fn poll_device_token(
    gateway: &str,
    device: &DeviceCode,
    signal: &CancellationToken,
) -> anyhow::Result<DevicePoll<ProviderCredential>> {
    let response = post_form(
        endpoint(gateway, "/v1/oauth/token")?,
        &[
            ("grant_type", DEVICE_GRANT_TYPE),
            ("client_id", CLIENT_ID),
            ("device_code", device.device_code.as_str()),
        ],
        signal,
        "Radius OAuth device token request",
    )
    .await?;
    if response.status.is_success() {
        return Ok(DevicePoll::Complete(token_credential(
            &response,
            "Radius OAuth device token request",
        )?));
    }
    match response.json.get("error").and_then(Value::as_str) {
        Some("authorization_pending") => Ok(DevicePoll::Pending),
        Some("slow_down") => Ok(DevicePoll::SlowDown {
            interval_seconds: positive_u64(response.json.get("interval")),
        }),
        Some("expired_token") => bail!("Radius device authorization expired"),
        Some("access_denied") => bail!("Radius device authorization was denied"),
        _ => oauth_error(&response, "Radius OAuth device token request"),
    }
}

async fn load_discovery(gateway: &str, signal: &CancellationToken) -> anyhow::Result<String> {
    let response = get(
        endpoint(gateway, "/v1/oauth")?,
        None,
        signal,
        "Radius OAuth discovery",
    )
    .await?;
    if !response.status.is_success() {
        return response_error("Radius OAuth discovery", &response);
    }
    let value = response
        .json
        .get("authorizationEndpoint")
        .and_then(Value::as_str)
        .context("Radius OAuth discovery is missing authorizationEndpoint")?;
    validate_http_url(value)
        .context("Radius OAuth discovery returned an untrusted authorization endpoint")?;
    Ok(value.to_owned())
}

pub(crate) async fn load_gateway_config_for_catalog(
    access_token: &str,
    signal: &CancellationToken,
) -> anyhow::Result<RadiusGatewayConfig> {
    let gateway = gateway_url()?;
    load_gateway_config(&gateway, access_token, signal).await
}

pub(crate) fn gateway_cache_origin() -> anyhow::Result<String> {
    gateway_url()
}

async fn load_gateway_config(
    gateway: &str,
    access_token: &str,
    signal: &CancellationToken,
) -> anyhow::Result<RadiusGatewayConfig> {
    let response = get(
        endpoint(gateway, "/v1/config")?,
        Some(access_token),
        signal,
        "Radius gateway config",
    )
    .await?;
    if !response.status.is_success() {
        return response_error_redacted("Radius gateway config", &response, &[access_token]);
    }
    sanitize_gateway_config(&response.json)
}

fn sanitize_gateway_config(value: &Value) -> anyhow::Result<RadiusGatewayConfig> {
    let base_url = value
        .get("baseUrl")
        .and_then(Value::as_str)
        .context("Radius gateway config is missing baseUrl")?;
    validate_http_url(base_url).context("Radius gateway config has an invalid baseUrl")?;
    let models = value
        .get("models")
        .and_then(Value::as_array)
        .context("Radius gateway config is missing models")?;
    let models = models
        .iter()
        .filter_map(sanitized_gateway_model)
        .collect::<Vec<_>>();
    Ok(RadiusGatewayConfig {
        base_url: base_url.to_owned(),
        models,
    })
}

fn sanitized_gateway_model(value: &Value) -> Option<RadiusGatewayModel> {
    if !valid_gateway_model(value) {
        return None;
    }
    Some(RadiusGatewayModel {
        id: value.get("id")?.as_str()?.to_owned(),
        name: value.get("name")?.as_str()?.to_owned(),
        reasoning: value.get("reasoning")?.as_bool()?,
        context_window: value.get("contextWindow")?.as_u64()?,
        max_tokens: u32::try_from(value.get("maxTokens")?.as_u64()?).ok()?,
    })
}

fn valid_gateway_model(value: &Value) -> bool {
    let Some(model) = value.as_object() else {
        return false;
    };
    string_field(model, "id")
        && string_field(model, "name")
        && model.get("reasoning").and_then(Value::as_bool).is_some()
        && model.get("input").and_then(Value::as_array).is_some()
        && model.get("cost").and_then(Value::as_object).is_some()
        && model.get("contextWindow").and_then(Value::as_f64).is_some()
        && model.get("maxTokens").and_then(Value::as_f64).is_some()
}

fn string_field(model: &Map<String, Value>, key: &str) -> bool {
    model
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty())
}

fn device_code(json: &Value) -> anyhow::Result<DeviceCode> {
    let device_code = required_string(json, "device_code", "Radius device authorization")?;
    let user_code = required_string(json, "user_code", "Radius device authorization")?;
    let verification_uri =
        required_string(json, "verification_uri", "Radius device authorization")?;
    validate_http_url(&verification_uri)
        .context("Radius returned an untrusted verification_uri")?;
    let expires_in_seconds = positive_u64(json.get("expires_in"))
        .context("Radius device authorization is missing expires_in")?;
    Ok(DeviceCode {
        device_code,
        user_code,
        verification_uri,
        verification_uri_complete: None,
        interval_seconds: positive_u64(json.get("interval")).unwrap_or(5),
        expires_in_seconds,
    })
}

fn token_credential(response: &HttpPayload, operation: &str) -> anyhow::Result<ProviderCredential> {
    if !response.status.is_success() {
        return oauth_error(response, operation);
    }
    let access = required_string(&response.json, "access_token", operation)?;
    let refresh = required_string(&response.json, "refresh_token", operation)?;
    let expires_in = positive_u64(response.json.get("expires_in"))
        .with_context(|| format!("{operation} response is missing expires_in"))?;
    let mut credential = ProviderCredential::oauth(
        access,
        refresh,
        super::store::unix_millis()
            .saturating_add(expires_in.saturating_mul(1_000))
            .saturating_sub(EXPIRY_SKEW_MS),
    );
    if let Some(scope) = response.json.get("scope").and_then(Value::as_str) {
        credential.set_metadata("scope", scope);
    }
    Ok(credential)
}

fn required_string(json: &Value, key: &str, operation: &str) -> anyhow::Result<String> {
    json.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .with_context(|| format!("{operation} response is missing {key}"))
}

fn positive_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(value) => value.as_u64().filter(|value| *value > 0),
        Value::String(value) => value.trim().parse().ok().filter(|value| *value > 0),
        _ => None,
    }
}

fn authorization_url(endpoint: &str, challenge: &str, state: &str) -> anyhow::Result<String> {
    let mut url = validate_http_url(endpoint)?;
    url.set_query(None);
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPE)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("handoff", "url")
        .append_pair("state", state);
    Ok(url.into())
}

fn gateway_url() -> anyhow::Result<String> {
    normalize_gateway_url(
        &std::env::var("GROK_RADIUS_GATEWAY").unwrap_or_else(|_| DEFAULT_GATEWAY.to_owned()),
    )
}

fn normalize_gateway_url(value: &str) -> anyhow::Result<String> {
    let value = value.trim();
    let candidate = if value.starts_with("http://") || value.starts_with("https://") {
        value.to_owned()
    } else {
        format!("https://{value}")
    };
    let url = validate_http_url(&candidate)?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        bail!(
            "Radius gateway must be an http(s) origin without credentials, path, query, or fragment"
        );
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

fn endpoint(gateway: &str, path: &str) -> anyhow::Result<String> {
    Ok(Url::parse(&format!("{gateway}/"))?.join(path)?.into())
}

async fn post_form(
    url: String,
    form: &[(&str, &str)],
    signal: &CancellationToken,
    operation: &'static str,
) -> anyhow::Result<HttpPayload> {
    let response = await_or_cancel(
        signal,
        crate::http::shared_client()
            .post(url)
            .header(reqwest::header::ACCEPT, "application/json")
            .timeout(REQUEST_TIMEOUT)
            .form(form)
            .send(),
    )
    .await
    .with_context(|| format!("{operation} request failed"))?;
    read_payload(response, signal).await
}

async fn get(
    url: String,
    bearer: Option<&str>,
    signal: &CancellationToken,
    operation: &'static str,
) -> anyhow::Result<HttpPayload> {
    let mut request = crate::http::shared_client()
        .get(url)
        .header(reqwest::header::ACCEPT, "application/json")
        .timeout(REQUEST_TIMEOUT);
    if let Some(bearer) = bearer {
        request = request.bearer_auth(bearer);
    }
    let response = await_or_cancel(signal, request.send())
        .await
        .with_context(|| format!("{operation} request failed"))?;
    read_payload(response, signal).await
}

async fn read_payload(
    response: reqwest::Response,
    signal: &CancellationToken,
) -> anyhow::Result<HttpPayload> {
    let status = response.status();
    let body = await_or_cancel(signal, response.text())
        .await
        .context("failed to read Radius OAuth response")?;
    let json = serde_json::from_str(&body).unwrap_or(Value::Null);
    Ok(HttpPayload { status, json, body })
}

async fn await_or_cancel<T, E>(
    signal: &CancellationToken,
    future: impl Future<Output = Result<T, E>>,
) -> anyhow::Result<T>
where
    E: Into<anyhow::Error>,
{
    tokio::select! {
        _ = signal.cancelled() => bail!("Radius login cancelled"),
        result = future => result.map_err(Into::into),
    }
}

fn oauth_error<T>(response: &HttpPayload, operation: &str) -> anyhow::Result<T> {
    let code = response.json.get("error").and_then(Value::as_str);
    let description = response
        .json
        .get("error_description")
        .and_then(Value::as_str);
    match (code, description) {
        (Some(code), Some(description)) => {
            bail!(
                "{operation} failed ({}): {code}: {description}",
                response.status
            )
        }
        (Some(code), None) => bail!("{operation} failed ({}): {code}", response.status),
        _ => response_error(operation, response),
    }
}

fn response_error<T>(operation: &str, response: &HttpPayload) -> anyhow::Result<T> {
    response_error_redacted(operation, response, &[])
}

fn response_error_redacted<T>(
    operation: &str,
    response: &HttpPayload,
    redactions: &[&str],
) -> anyhow::Result<T> {
    let mut body = response.body.trim().to_owned();
    for secret in redactions.iter().filter(|secret| !secret.is_empty()) {
        body = body.replace(secret, "[REDACTED]");
    }
    let body = truncate_error_body(&body);
    if body.is_empty() {
        bail!("{operation} failed ({})", response.status)
    }
    bail!("{operation} failed ({}): {body}", response.status)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn browser_authorization_contract_is_exact() {
        let url = Url::parse(
            &authorization_url(
                "https://login.radius.test/authorize?old=value",
                "challenge",
                "state",
            )
            .unwrap(),
        )
        .unwrap();
        let query: HashMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(
            url.as_str().split('?').next(),
            Some("https://login.radius.test/authorize")
        );
        assert_eq!(query.get("response_type").map(String::as_str), Some("code"));
        assert_eq!(query.get("client_id").map(String::as_str), Some(CLIENT_ID));
        assert_eq!(
            query.get("redirect_uri").map(String::as_str),
            Some(REDIRECT_URI)
        );
        assert_eq!(query.get("scope").map(String::as_str), Some(SCOPE));
        assert_eq!(
            query.get("code_challenge").map(String::as_str),
            Some("challenge")
        );
        assert_eq!(
            query.get("code_challenge_method").map(String::as_str),
            Some("S256")
        );
        assert_eq!(query.get("handoff").map(String::as_str), Some("url"));
        assert_eq!(query.get("state").map(String::as_str), Some("state"));
        assert_eq!(query.len(), 8);
    }

    #[test]
    fn gateway_and_config_validation_are_fail_closed() {
        assert_eq!(
            normalize_gateway_url("radius.example").unwrap(),
            "https://radius.example"
        );
        assert!(normalize_gateway_url("ftp://radius.example").is_err());
        assert!(normalize_gateway_url("https://user:pass@radius.example").is_err());
        assert!(normalize_gateway_url("https://radius.example/path").is_err());

        let config = serde_json::json!({
            "baseUrl": "https://api.radius.example",
            "models": [
                {
                    "id": "valid",
                    "name": "Valid",
                    "reasoning": true,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100
                },
                {"id": "invalid"}
            ]
        });
        let sanitized = sanitize_gateway_config(&config).unwrap();
        assert_eq!(sanitized.models.len(), 1);
        assert_eq!(sanitized.models[0].id, "valid");
        assert!(sanitize_gateway_config(&serde_json::json!({"models": []})).is_err());
    }

    #[test]
    fn gateway_config_errors_redact_bearer_and_truncate_body() {
        let secret = "radius-secret-token";
        let response = HttpPayload {
            status: StatusCode::UNAUTHORIZED,
            json: Value::Null,
            body: format!("echoed {secret} {}", "x".repeat(5000)),
        };
        let error = response_error_redacted::<()>("Radius gateway config", &response, &[secret])
            .expect_err("gateway errors should fail");
        let text = error.to_string();
        assert!(text.contains("[REDACTED]"));
        assert!(!text.contains(secret));
        assert!(text.len() < 4300, "error body must stay bounded");

        let boundary_secret = "sk_PARTIAL_SECRET_MUST_NOT_LEAK";
        let boundary_response = HttpPayload {
            status: StatusCode::UNAUTHORIZED,
            json: Value::Null,
            body: format!("{}{}", "x".repeat(4090), boundary_secret),
        };
        let boundary_error = response_error_redacted::<()>(
            "Radius gateway config",
            &boundary_response,
            &[boundary_secret],
        )
        .expect_err("gateway errors should fail");
        assert!(
            !boundary_error.to_string().contains("sk_PAR"),
            "redaction must happen before truncation so a key prefix cannot survive at the boundary"
        );
    }

    #[test]
    fn token_expiry_uses_sixty_second_skew_and_requires_fields() {
        let now = super::super::store::unix_millis();
        let response = HttpPayload {
            status: StatusCode::OK,
            json: serde_json::json!({
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 120,
                "scope": SCOPE
            }),
            body: String::new(),
        };
        let credential = token_credential(&response, "test").unwrap();
        assert!(credential.expires >= now + 59_000);
        assert!(credential.expires <= now + 61_000);
        assert_eq!(credential.metadata_str("scope"), Some(SCOPE));
        let missing = HttpPayload {
            status: StatusCode::OK,
            json: Value::Null,
            body: String::new(),
        };
        assert!(token_credential(&missing, "test").is_err());
    }
}

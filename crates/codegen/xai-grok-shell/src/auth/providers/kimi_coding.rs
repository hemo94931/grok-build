use std::future::Future;
use std::time::Duration;

use anyhow::{Context, bail};
use reqwest::StatusCode;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::flow::{
    DevicePoll, DevicePollOptions, notify_device_code, poll_device_code, truncate_error_body,
    validate_http_url,
};
use super::{AuthInteraction, DeviceCode, LoginMode, ProviderCredential, ProviderRefreshOutcome};

const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const DEFAULT_OAUTH_HOST: &str = "https://auth.kimi.com";
const DEVICE_TIMEOUT_SECONDS: u64 = 15 * 60;
const DEFAULT_INTERVAL_SECONDS: u64 = 5;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const REFRESH_RETRIES: usize = 3;
const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

pub(super) async fn login(
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    if matches!(mode, Some(LoginMode::Browser)) {
        bail!("Kimi Coding only supports device-code login");
    }
    let signal = interaction.signal();
    let host = oauth_host()?;
    let device = start_device_authorization(&host, &signal).await?;
    notify_device_code(interaction, &device).await?;

    let poll_host = host.clone();
    let poll_device = device.clone();
    let poll_signal = signal.clone();
    poll_device_code(
        DevicePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: device.expires_in_seconds,
            wait_before_first_poll: true,
        },
        signal,
        move || {
            let host = poll_host.clone();
            let device = poll_device.clone();
            let signal = poll_signal.clone();
            async move { poll_token(&host, &device, &signal).await }
        },
    )
    .await
}

pub(super) async fn refresh(
    credential: ProviderCredential,
    signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    if credential.refresh.trim().is_empty() {
        bail!("Kimi Coding credential is missing a refresh token");
    }
    let host = oauth_host()?;
    let form = [
        ("client_id", CLIENT_ID),
        ("grant_type", "refresh_token"),
        ("refresh_token", credential.refresh.as_str()),
    ];
    let mut last_error = None;

    for attempt in 0..=REFRESH_RETRIES {
        if attempt > 0 {
            sleep_or_cancel(Duration::from_secs(1 << (attempt - 1)), &signal).await?;
        }
        let response = send_form(
            format!("{host}/api/oauth/token"),
            &form,
            &signal,
            "Kimi Coding token refresh",
        )
        .await;
        let response = match response {
            Ok(response) => response,
            Err(error) if attempt < REFRESH_RETRIES && !signal.is_cancelled() => {
                last_error = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };
        let status = response.status();
        let body = response_text(response, &signal).await?;
        let json = serde_json::from_str::<Value>(&body).unwrap_or(Value::Null);

        if status.is_success() {
            return Ok(ProviderRefreshOutcome::Save(token_credential(
                &json, "refresh",
            )?));
        }
        if refresh_credential_is_dead(status, &json) {
            return Ok(ProviderRefreshOutcome::Remove {
                message: format!(
                    "Kimi Coding authorization expired; run `grok login --provider kimi-coding`"
                ),
            });
        }
        if (status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error())
            && attempt < REFRESH_RETRIES
        {
            last_error = Some(anyhow::anyhow!(
                "Kimi Coding token refresh failed ({status})"
            ));
            continue;
        }
        return response_error("Kimi Coding token refresh", status, &body);
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Kimi Coding token refresh failed")))
}

async fn start_device_authorization(
    host: &str,
    signal: &CancellationToken,
) -> anyhow::Result<DeviceCode> {
    let response = send_form(
        format!("{host}/api/oauth/device_authorization"),
        &[("client_id", CLIENT_ID)],
        signal,
        "Kimi Coding device authorization",
    )
    .await?;
    let status = response.status();
    let body = response_text(response, signal).await?;
    if !status.is_success() {
        return response_error("Kimi Coding device authorization", status, &body);
    }
    let json: Value = serde_json::from_str(&body)
        .context("Kimi Coding device authorization returned invalid JSON")?;
    let device_code = required_string(&json, "device_code", "device authorization")?;
    let user_code = required_string(&json, "user_code", "device authorization")?;
    let verification_uri = required_string(&json, "verification_uri", "device authorization")?;
    let verification_uri_complete =
        required_string(&json, "verification_uri_complete", "device authorization")?;
    validate_http_url(&verification_uri)
        .context("Kimi Coding returned an untrusted verification_uri")?;
    validate_http_url(&verification_uri_complete)
        .context("Kimi Coding returned an untrusted verification_uri_complete")?;

    Ok(DeviceCode {
        device_code,
        user_code,
        verification_uri,
        verification_uri_complete: Some(verification_uri_complete),
        interval_seconds: positive_u64(json.get("interval")).unwrap_or(DEFAULT_INTERVAL_SECONDS),
        expires_in_seconds: positive_u64(json.get("expires_in")).unwrap_or(DEVICE_TIMEOUT_SECONDS),
    })
}

async fn poll_token(
    host: &str,
    device: &DeviceCode,
    signal: &CancellationToken,
) -> anyhow::Result<DevicePoll<ProviderCredential>> {
    let response = send_form(
        format!("{host}/api/oauth/token"),
        &[
            ("client_id", CLIENT_ID),
            ("device_code", device.device_code.as_str()),
            ("grant_type", DEVICE_GRANT_TYPE),
        ],
        signal,
        "Kimi Coding device token request",
    )
    .await?;
    let status = response.status();
    let body = response_text(response, signal).await?;
    if status.is_server_error() {
        return response_error("Kimi Coding device token request", status, &body);
    }
    let json = serde_json::from_str::<Value>(&body).unwrap_or(Value::Null);
    if status.is_success() && json.get("access_token").and_then(Value::as_str).is_some() {
        return Ok(DevicePoll::Complete(token_credential(&json, "poll")?));
    }

    match json.get("error").and_then(Value::as_str) {
        Some("authorization_pending") => Ok(DevicePoll::Pending),
        Some("slow_down") => Ok(DevicePoll::SlowDown {
            interval_seconds: positive_u64(json.get("interval")),
        }),
        Some("expired_token") => {
            bail!("Kimi Coding device authorization expired; restart login")
        }
        Some("access_denied") => bail!("Kimi Coding login was denied"),
        Some(error) => {
            let description = json
                .get("error_description")
                .and_then(Value::as_str)
                .unwrap_or("");
            if description.is_empty() {
                bail!("Kimi Coding device token request failed ({status}): {error}")
            }
            bail!("Kimi Coding device token request failed ({status}): {error}: {description}")
        }
        None => response_error("Kimi Coding device token request", status, &body),
    }
}

fn token_credential(json: &Value, operation: &str) -> anyhow::Result<ProviderCredential> {
    let access = required_string(json, "access_token", operation)?;
    let refresh = required_string(json, "refresh_token", operation)?;
    let expires_in = positive_u64(json.get("expires_in"))
        .with_context(|| format!("Kimi Coding token {operation} response is missing expires_in"))?;
    Ok(ProviderCredential::oauth(
        access,
        refresh,
        super::store::unix_millis().saturating_add(expires_in.saturating_mul(1_000)),
    ))
}

fn required_string(json: &Value, field: &str, operation: &str) -> anyhow::Result<String> {
    json.get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .with_context(|| format!("Kimi Coding {operation} response is missing {field}"))
}

fn positive_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(value) => value.as_u64().filter(|value| *value > 0),
        Value::String(value) => value.trim().parse().ok().filter(|value| *value > 0),
        _ => None,
    }
}

fn refresh_credential_is_dead(status: StatusCode, json: &Value) -> bool {
    matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
        || json.get("error").and_then(Value::as_str) == Some("invalid_grant")
}

fn oauth_host() -> anyhow::Result<String> {
    let configured = std::env::var("KIMI_CODE_OAUTH_HOST")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("KIMI_OAUTH_HOST")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| DEFAULT_OAUTH_HOST.to_owned());
    normalize_oauth_host(&configured)
}

fn normalize_oauth_host(value: &str) -> anyhow::Result<String> {
    let url = validate_http_url(value.trim())?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("Kimi OAuth host must not include credentials, query, or fragment");
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

async fn send_form(
    url: String,
    form: &[(&str, &str)],
    signal: &CancellationToken,
    operation: &'static str,
) -> anyhow::Result<reqwest::Response> {
    let request = crate::http::shared_client()
        .post(url)
        .header(reqwest::header::ACCEPT, "application/json")
        .timeout(REQUEST_TIMEOUT)
        .form(form)
        .send();
    await_or_cancel(signal, request)
        .await
        .with_context(|| format!("{operation} failed"))
}

async fn response_text(
    response: reqwest::Response,
    signal: &CancellationToken,
) -> anyhow::Result<String> {
    await_or_cancel(signal, response.text())
        .await
        .context("failed to read Kimi Coding OAuth response")
}

async fn sleep_or_cancel(duration: Duration, signal: &CancellationToken) -> anyhow::Result<()> {
    tokio::select! {
        _ = signal.cancelled() => bail!("Kimi Coding token refresh cancelled"),
        _ = tokio::time::sleep(duration) => Ok(()),
    }
}

async fn await_or_cancel<T, E>(
    signal: &CancellationToken,
    future: impl Future<Output = Result<T, E>>,
) -> anyhow::Result<T>
where
    E: Into<anyhow::Error>,
{
    tokio::select! {
        _ = signal.cancelled() => bail!("Kimi Coding OAuth cancelled"),
        result = future => result.map_err(Into::into),
    }
}

fn response_error<T>(operation: &str, status: StatusCode, body: &str) -> anyhow::Result<T> {
    let body = truncate_error_body(body.trim());
    if body.is_empty() {
        bail!("{operation} failed ({status})")
    }
    bail!("{operation} failed ({status}): {body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_validation_is_fail_closed() {
        assert_eq!(
            normalize_oauth_host("https://auth.kimi.com///").unwrap(),
            "https://auth.kimi.com"
        );
        assert!(normalize_oauth_host("javascript:alert(1)").is_err());
        assert!(normalize_oauth_host("https://user:pass@auth.kimi.com").is_err());
    }

    #[test]
    fn token_contract_and_terminal_refresh_errors_are_exact() {
        let token = serde_json::json!({
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 3600
        });
        let credential = token_credential(&token, "test").unwrap();
        assert_eq!(credential.access, "access");
        assert_eq!(credential.refresh, "refresh");
        assert!(token_credential(&serde_json::json!({}), "test").is_err());
        assert!(refresh_credential_is_dead(
            StatusCode::UNAUTHORIZED,
            &Value::Null
        ));
        assert!(refresh_credential_is_dead(
            StatusCode::BAD_REQUEST,
            &serde_json::json!({"error":"invalid_grant"})
        ));
        assert!(!refresh_credential_is_dead(
            StatusCode::INTERNAL_SERVER_ERROR,
            &Value::Null
        ));
    }
}

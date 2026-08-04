use std::fmt::Write as _;
use std::future::Future;
use std::time::Duration;

use anyhow::{Context, bail};
use base64::{Engine as _, engine::general_purpose::URL_SAFE};
use reqwest::{Response, StatusCode};
use serde::Deserialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::flow::{
    DevicePoll, DevicePollOptions, LoopbackServer, notify_device_code, poll_device_code,
    truncate_error_body,
};
use super::{
    AuthInteraction, AuthNotification, AuthPrompt, DeviceCode, LoginMode, ProviderCredential,
    ProviderRefreshOutcome, SelectOption,
};

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const CALLBACK_PATH: &str = "/auth/callback";
const CALLBACK_PORT: u16 = 1455;
const DEVICE_USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const DEVICE_VERIFICATION_URI: &str = "https://auth.openai.com/codex/device";
const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const SCOPE: &str = "openid profile email offline_access";
const JWT_AUTH_CLAIM: &str = "https://api.openai.com/auth";
const DEVICE_CODE_TIMEOUT_SECONDS: u64 = 15 * 60;
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const BROWSER_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const LOGIN_CANCELLED: &str = "OpenAI Codex login cancelled";
const REFRESH_CANCELLED: &str = "OpenAI Codex token refresh cancelled";

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
}

#[derive(Deserialize)]
struct DeviceStartResponse {
    device_auth_id: String,
    user_code: String,
    interval: Value,
}

struct DeviceGrant {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Deserialize)]
struct DeviceGrantResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Deserialize)]
struct JwtPayload {
    #[serde(rename = "https://api.openai.com/auth")]
    auth: Option<JwtAuthClaim>,
}

#[derive(Deserialize)]
struct JwtAuthClaim {
    chatgpt_account_id: Option<String>,
}

#[derive(Debug, Eq, PartialEq)]
enum DeviceErrorClass {
    Pending,
    SlowDown(Option<u64>),
    Failed,
}

pub(super) async fn login(
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    let signal = interaction.signal();
    ensure_active(&signal, LOGIN_CANCELLED)?;
    let mode = match mode {
        Some(mode) => mode,
        None => select_login_mode(interaction, &signal).await?,
    };

    match mode {
        LoginMode::Browser => browser_login(interaction, signal).await,
        LoginMode::DeviceCode => device_login(interaction, signal).await,
    }
}

pub(super) async fn refresh(
    credential: ProviderCredential,
    signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    ensure_active(&signal, REFRESH_CANCELLED)?;
    if credential.refresh.trim().is_empty() {
        bail!("OpenAI Codex credential is missing a refresh token");
    }
    let token = request_token(
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", credential.refresh.as_str()),
            ("client_id", CLIENT_ID),
        ],
        "refresh",
        &signal,
        REFRESH_CANCELLED,
    )
    .await?;
    Ok(ProviderRefreshOutcome::Save(credential_from_token(token)?))
}

async fn select_login_mode(
    interaction: &dyn AuthInteraction,
    signal: &CancellationToken,
) -> anyhow::Result<LoginMode> {
    let selected = await_or_cancel(
        signal,
        LOGIN_CANCELLED,
        interaction.prompt(AuthPrompt::Select {
            message: "Select OpenAI Codex login method:".to_owned(),
            options: vec![
                SelectOption {
                    id: "browser".to_owned(),
                    label: "Browser login (default)".to_owned(),
                },
                SelectOption {
                    id: "device_code".to_owned(),
                    label: "Device code login (headless)".to_owned(),
                },
            ],
        }),
    )
    .await?;

    match selected.trim() {
        "browser" => Ok(LoginMode::Browser),
        "device_code" => Ok(LoginMode::DeviceCode),
        unknown => bail!("unknown OpenAI Codex login method `{unknown}`"),
    }
}

async fn browser_login(
    interaction: &dyn AuthInteraction,
    signal: CancellationToken,
) -> anyhow::Result<ProviderCredential> {
    let pkce = crate::auth::oidc::protocol::generate_pkce();
    let state = create_state();
    let bind_host = callback_host();
    let server = await_or_cancel(&signal, LOGIN_CANCELLED, async {
        LoopbackServer::bind(
            &bind_host,
            "localhost",
            CALLBACK_PORT,
            CALLBACK_PATH,
            Some(state.clone()),
        )
        .await
        .with_context(|| {
            format!(
                "OpenAI Codex browser login could not bind {bind_host}:{CALLBACK_PORT}; if the port is occupied, choose device code login"
            )
        })
    })
    .await?;
    debug_assert_eq!(server.redirect_uri(), REDIRECT_URI);

    let url = authorization_url(&pkce.code_challenge, &state)?;
    await_or_cancel(
        &signal,
        LOGIN_CANCELLED,
        interaction.notify(AuthNotification::AuthUrl {
            url,
            instructions: "A browser window should open. Complete login to finish.".to_owned(),
        }),
    )
    .await?;

    let authorization = server
        .wait(
            interaction,
            "Complete login in your browser, or paste the authorization code / redirect URL here:",
            Some(&state),
            BROWSER_TIMEOUT,
        )
        .await?;
    exchange_authorization_code(
        &authorization.code,
        &pkce.code_verifier,
        REDIRECT_URI,
        &signal,
    )
    .await
}

async fn device_login(
    interaction: &dyn AuthInteraction,
    signal: CancellationToken,
) -> anyhow::Result<ProviderCredential> {
    let device = start_device_auth(&signal).await?;
    await_or_cancel(
        &signal,
        LOGIN_CANCELLED,
        notify_device_code(interaction, &device),
    )
    .await?;

    let poll_signal = signal.clone();
    let polled_device = device.clone();
    let grant = poll_device_code(
        DevicePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: DEVICE_CODE_TIMEOUT_SECONDS,
            wait_before_first_poll: false,
        },
        signal.clone(),
        move || {
            let signal = poll_signal.clone();
            let device = polled_device.clone();
            async move { poll_device_auth(&device, &signal).await }
        },
    )
    .await?;

    exchange_authorization_code(
        &grant.authorization_code,
        &grant.code_verifier,
        DEVICE_REDIRECT_URI,
        &signal,
    )
    .await
}

async fn start_device_auth(signal: &CancellationToken) -> anyhow::Result<DeviceCode> {
    let request = crate::http::shared_client()
        .post(DEVICE_USER_CODE_URL)
        .timeout(HTTP_TIMEOUT)
        .json(&serde_json::json!({ "client_id": CLIENT_ID }));
    let response = await_or_cancel(signal, LOGIN_CANCELLED, async move {
        request
            .send()
            .await
            .context("OpenAI Codex device code request failed")
    })
    .await?;
    let status = response.status();
    if status == StatusCode::NOT_FOUND {
        bail!(
            "OpenAI Codex device code login is not enabled for this server; use browser login or verify the server URL"
        );
    }
    if !status.is_success() {
        let body = response_text(response, signal, LOGIN_CANCELLED).await?;
        return http_error("OpenAI Codex device code request", status, &body);
    }

    let payload: DeviceStartResponse = await_or_cancel(signal, LOGIN_CANCELLED, async move {
        response
            .json()
            .await
            .context("OpenAI Codex device code response was invalid JSON")
    })
    .await?;
    let interval_seconds = json_u64(&payload.interval)
        .context("OpenAI Codex device code response has an invalid interval")?;
    if payload.device_auth_id.trim().is_empty() || payload.user_code.trim().is_empty() {
        bail!("OpenAI Codex device code response is missing required fields");
    }

    Ok(DeviceCode {
        device_code: payload.device_auth_id,
        user_code: payload.user_code,
        verification_uri: DEVICE_VERIFICATION_URI.to_owned(),
        verification_uri_complete: None,
        interval_seconds,
        expires_in_seconds: DEVICE_CODE_TIMEOUT_SECONDS,
    })
}

async fn poll_device_auth(
    device: &DeviceCode,
    signal: &CancellationToken,
) -> anyhow::Result<DevicePoll<DeviceGrant>> {
    let request = crate::http::shared_client()
        .post(DEVICE_TOKEN_URL)
        .timeout(HTTP_TIMEOUT)
        .json(&serde_json::json!({
            "device_auth_id": device.device_code,
            "user_code": device.user_code,
        }));
    let response = await_or_cancel(signal, LOGIN_CANCELLED, async move {
        request
            .send()
            .await
            .context("OpenAI Codex device authorization request failed")
    })
    .await?;
    let status = response.status();
    if status.is_success() {
        let payload: DeviceGrantResponse = await_or_cancel(signal, LOGIN_CANCELLED, async move {
            response
                .json()
                .await
                .context("OpenAI Codex device authorization returned invalid JSON")
        })
        .await?;
        if payload.authorization_code.trim().is_empty() || payload.code_verifier.trim().is_empty() {
            bail!("OpenAI Codex device authorization response is missing required fields");
        }
        return Ok(DevicePoll::Complete(DeviceGrant {
            authorization_code: payload.authorization_code,
            code_verifier: payload.code_verifier,
        }));
    }

    let body = response_text(response, signal, LOGIN_CANCELLED).await?;
    match classify_device_error(status, &body) {
        DeviceErrorClass::Pending => Ok(DevicePoll::Pending),
        DeviceErrorClass::SlowDown(interval_seconds) => {
            Ok(DevicePoll::SlowDown { interval_seconds })
        }
        DeviceErrorClass::Failed => http_error("OpenAI Codex device authorization", status, &body),
    }
}

async fn exchange_authorization_code(
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    signal: &CancellationToken,
) -> anyhow::Result<ProviderCredential> {
    let token = request_token(
        &[
            ("grant_type", "authorization_code"),
            ("client_id", CLIENT_ID),
            ("code", code),
            ("code_verifier", verifier),
            ("redirect_uri", redirect_uri),
        ],
        "exchange",
        signal,
        LOGIN_CANCELLED,
    )
    .await?;
    credential_from_token(token)
}

async fn request_token(
    form: &[(&str, &str)],
    operation: &'static str,
    signal: &CancellationToken,
    cancelled_message: &'static str,
) -> anyhow::Result<TokenResponse> {
    let request = crate::http::shared_client()
        .post(TOKEN_URL)
        .timeout(HTTP_TIMEOUT)
        .form(form);
    let response = await_or_cancel(signal, cancelled_message, async move {
        request
            .send()
            .await
            .with_context(|| format!("OpenAI Codex token {operation} request failed"))
    })
    .await?;
    read_token_response(response, operation, signal, cancelled_message).await
}

async fn read_token_response(
    response: Response,
    operation: &str,
    signal: &CancellationToken,
    cancelled_message: &'static str,
) -> anyhow::Result<TokenResponse> {
    let status = response.status();
    if !status.is_success() {
        let body = response_text(response, signal, cancelled_message).await?;
        return http_error(&format!("OpenAI Codex token {operation}"), status, &body);
    }
    let token: TokenResponse = await_or_cancel(signal, cancelled_message, async move {
        response
            .json()
            .await
            .with_context(|| format!("OpenAI Codex token {operation} response is invalid"))
    })
    .await?;
    if token.access_token.trim().is_empty() || token.refresh_token.trim().is_empty() {
        bail!("OpenAI Codex token {operation} response is missing required fields");
    }
    Ok(token)
}

fn credential_from_token(token: TokenResponse) -> anyhow::Result<ProviderCredential> {
    let account_id = extract_account_id(&token.access_token)?;
    let expires =
        super::store::unix_millis().saturating_add(token.expires_in.saturating_mul(1_000));
    let mut credential =
        ProviderCredential::oauth(token.access_token, token.refresh_token, expires);
    credential.set_metadata("accountId", account_id);
    Ok(credential)
}

fn extract_account_id(access_token: &str) -> anyhow::Result<String> {
    let mut parts = access_token.split('.');
    let (Some(_header), Some(payload), Some(_signature), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        bail!("OpenAI Codex access token is not a JWT");
    };
    let payload: JwtPayload = serde_json::from_slice(&decode_base64url(payload)?)
        .context("OpenAI Codex access token has an invalid JWT payload")?;
    payload
        .auth
        .and_then(|auth| auth.chatgpt_account_id)
        .filter(|account_id| !account_id.trim().is_empty())
        .context(format!(
            "OpenAI Codex access token is missing {JWT_AUTH_CLAIM}.chatgpt_account_id"
        ))
}

fn decode_base64url(value: &str) -> anyhow::Result<Vec<u8>> {
    let mut padded = value.to_owned();
    match padded.len() % 4 {
        0 => {}
        2 => padded.push_str("=="),
        3 => padded.push('='),
        _ => bail!("invalid base64url length"),
    }
    URL_SAFE
        .decode(padded)
        .context("invalid base64url encoding")
}

fn authorization_url(code_challenge: &str, state: &str) -> anyhow::Result<String> {
    let mut url = Url::parse(AUTHORIZE_URL)?;
    url.query_pairs_mut().extend_pairs([
        ("response_type", "code"),
        ("client_id", CLIENT_ID),
        ("redirect_uri", REDIRECT_URI),
        ("scope", SCOPE),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("originator", "pi"),
    ]);
    Ok(url.into())
}

fn create_state() -> String {
    let mut state = String::with_capacity(32);
    for byte in rand::random::<[u8; 16]>() {
        write!(&mut state, "{byte:02x}").expect("writing to a String cannot fail");
    }
    state
}

fn callback_host() -> String {
    std::env::var("PI_OAUTH_CALLBACK_HOST")
        .ok()
        .map(|host| host.trim().to_owned())
        .filter(|host| !host.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_owned())
}

fn classify_device_error(status: StatusCode, body: &str) -> DeviceErrorClass {
    if matches!(status, StatusCode::FORBIDDEN | StatusCode::NOT_FOUND) {
        return DeviceErrorClass::Pending;
    }
    let Ok(payload) = serde_json::from_str::<Value>(body) else {
        return DeviceErrorClass::Failed;
    };
    let error = payload.get("error");
    let code = match error {
        Some(Value::String(code)) => Some(code.as_str()),
        Some(Value::Object(error)) => error.get("code").and_then(Value::as_str),
        _ => None,
    };
    match code {
        Some("deviceauth_authorization_pending") => DeviceErrorClass::Pending,
        Some("slow_down") => {
            DeviceErrorClass::SlowDown(payload.get("interval").and_then(json_u64).or_else(|| {
                error
                    .and_then(|value| value.get("interval"))
                    .and_then(json_u64)
            }))
        }
        _ => DeviceErrorClass::Failed,
    }
}

fn json_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(number) => number.trim().parse().ok(),
        _ => None,
    }
}

fn http_error<T>(operation: &str, status: StatusCode, body: &str) -> anyhow::Result<T> {
    let body = truncate_error_body(body.trim());
    if body.is_empty() {
        bail!("{operation} failed ({status})");
    }
    bail!("{operation} failed ({status}): {body}")
}

async fn response_text(
    response: Response,
    signal: &CancellationToken,
    cancelled_message: &'static str,
) -> anyhow::Result<String> {
    await_or_cancel(signal, cancelled_message, async move {
        response
            .text()
            .await
            .context("failed to read OpenAI Codex error response")
    })
    .await
}

fn ensure_active(signal: &CancellationToken, message: &str) -> anyhow::Result<()> {
    if signal.is_cancelled() {
        bail!("{message}");
    }
    Ok(())
}

async fn await_or_cancel<T>(
    signal: &CancellationToken,
    cancelled_message: &'static str,
    future: impl Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
    tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("{cancelled_message}"),
        result = future => result,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use base64::engine::general_purpose::URL_SAFE_NO_PAD;

    use super::*;

    #[test]
    fn jwt_payload_uses_base64url_and_extracts_account_id() {
        assert_eq!(decode_base64url("-_8").unwrap(), vec![251, 255]);
        assert_eq!(decode_base64url("-_8=").unwrap(), vec![251, 255]);

        let payload = serde_json::json!({
            (JWT_AUTH_CLAIM): { "chatgpt_account_id": "account-123" }
        });
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
        let token = format!("e30.{payload}.signature");
        assert_eq!(extract_account_id(&token).unwrap(), "account-123");
        assert!(extract_account_id("e30.e30.signature").is_err());
    }

    #[test]
    fn authorization_url_contains_codex_parameters_and_random_hex_state() {
        let state = "00112233445566778899aabbccddeeff";
        let url = Url::parse(&authorization_url("challenge", state).unwrap()).unwrap();
        let query: HashMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(url.as_str().split('?').next(), Some(AUTHORIZE_URL));
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
        assert_eq!(query.get("state").map(String::as_str), Some(state));
        assert_eq!(
            query.get("id_token_add_organizations").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            query.get("codex_cli_simplified_flow").map(String::as_str),
            Some("true")
        );
        assert_eq!(query.get("originator").map(String::as_str), Some("pi"));
        assert_eq!(query.len(), 10);

        let generated = create_state();
        assert_eq!(generated.len(), 32);
        assert!(
            generated
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
    }

    #[test]
    fn device_errors_classify_pending_and_slow_down_shapes() {
        assert_eq!(
            classify_device_error(StatusCode::FORBIDDEN, ""),
            DeviceErrorClass::Pending
        );
        assert_eq!(
            classify_device_error(StatusCode::NOT_FOUND, ""),
            DeviceErrorClass::Pending
        );
        assert_eq!(
            classify_device_error(
                StatusCode::BAD_REQUEST,
                r#"{"error":"deviceauth_authorization_pending"}"#,
            ),
            DeviceErrorClass::Pending
        );
        assert_eq!(
            classify_device_error(StatusCode::TOO_MANY_REQUESTS, r#"{"error":"slow_down"}"#,),
            DeviceErrorClass::SlowDown(None)
        );
        assert_eq!(
            classify_device_error(
                StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"code":"slow_down","interval":"9"}}"#,
            ),
            DeviceErrorClass::SlowDown(Some(9))
        );
        assert_eq!(
            classify_device_error(StatusCode::BAD_REQUEST, r#"{"error":"denied"}"#),
            DeviceErrorClass::Failed
        );
    }
}

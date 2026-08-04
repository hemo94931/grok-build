use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, bail};
use async_trait::async_trait;
use axum::Router;
use axum::extract::{Query, State};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AuthNotification {
    AuthUrl {
        url: String,
        instructions: String,
    },
    DeviceCode {
        #[serde(rename = "userCode")]
        user_code: String,
        #[serde(rename = "verificationUri")]
        verification_uri: String,
        #[serde(rename = "intervalSeconds")]
        interval_seconds: u64,
        #[serde(rename = "expiresInSeconds")]
        expires_in_seconds: u64,
    },
    Progress {
        message: String,
    },
    Info {
        message: String,
        #[serde(default)]
        links: Vec<AuthLink>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct AuthLink {
    pub(crate) label: String,
    pub(crate) url: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AuthPrompt {
    Text {
        message: String,
        placeholder: String,
    },
    Select {
        message: String,
        options: Vec<SelectOption>,
    },
    ManualCode {
        message: String,
        placeholder: String,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct SelectOption {
    pub(crate) id: String,
    pub(crate) label: String,
}

#[async_trait(?Send)]
pub(crate) trait AuthInteraction: Send + Sync {
    fn signal(&self) -> CancellationToken;
    async fn notify(&self, notification: AuthNotification) -> anyhow::Result<()>;
    async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<String>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AuthorizationCode {
    pub(crate) code: String,
    pub(crate) state: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceCode {
    pub(crate) device_code: String,
    pub(crate) user_code: String,
    pub(crate) verification_uri: String,
    pub(crate) verification_uri_complete: Option<String>,
    pub(crate) interval_seconds: u64,
    pub(crate) expires_in_seconds: u64,
}

#[derive(Debug)]
pub(crate) enum DevicePoll<T> {
    Complete(T),
    Pending,
    SlowDown { interval_seconds: Option<u64> },
}

pub(crate) struct DevicePollOptions {
    pub(crate) interval_seconds: u64,
    pub(crate) expires_in_seconds: u64,
    pub(crate) wait_before_first_poll: bool,
}

pub(crate) async fn poll_device_code<T, F, Fut>(
    options: DevicePollOptions,
    signal: CancellationToken,
    mut poll: F,
) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<DevicePoll<T>>>,
{
    let deadline =
        tokio::time::Instant::now() + Duration::from_secs(options.expires_in_seconds.max(1));
    let mut interval = Duration::from_secs(options.interval_seconds.max(1));
    let mut slowed_down = false;

    if options.wait_before_first_poll {
        sleep_until_next_poll(interval, deadline, &signal, slowed_down).await?;
    }

    loop {
        if signal.is_cancelled() {
            bail!("login cancelled");
        }
        if tokio::time::Instant::now() >= deadline {
            return device_timeout(slowed_down);
        }

        match poll().await? {
            DevicePoll::Complete(value) => return Ok(value),
            DevicePoll::Pending => {}
            DevicePoll::SlowDown { interval_seconds } => {
                slowed_down = true;
                interval = interval_seconds
                    .map(|seconds| Duration::from_secs(seconds.max(1)))
                    .unwrap_or_else(|| interval + Duration::from_secs(5));
            }
        }
        sleep_until_next_poll(interval, deadline, &signal, slowed_down).await?;
    }
}

async fn sleep_until_next_poll(
    interval: Duration,
    deadline: tokio::time::Instant,
    signal: &CancellationToken,
    slowed_down: bool,
) -> anyhow::Result<()> {
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    if remaining.is_zero() {
        return device_timeout(slowed_down);
    }
    tokio::select! {
        _ = signal.cancelled() => bail!("login cancelled"),
        _ = tokio::time::sleep(interval.min(remaining)) => Ok(()),
    }
}

fn device_timeout<T>(slowed_down: bool) -> anyhow::Result<T> {
    if slowed_down {
        bail!(
            "device authorization timed out after repeated slow_down responses; check WSL/VM clock synchronization and try again"
        );
    }
    bail!("device authorization timed out; restart login and try again")
}

pub(crate) async fn notify_device_code(
    interaction: &dyn AuthInteraction,
    device: &DeviceCode,
) -> anyhow::Result<()> {
    let verification_uri = device
        .verification_uri_complete
        .as_deref()
        .unwrap_or(&device.verification_uri);
    validate_http_url(verification_uri)?;
    interaction
        .notify(AuthNotification::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: verification_uri.to_owned(),
            interval_seconds: device.interval_seconds.max(1),
            expires_in_seconds: device.expires_in_seconds,
        })
        .await
}

#[derive(Clone)]
struct CallbackState {
    expected_state: Option<String>,
    sender: Arc<Mutex<Option<oneshot::Sender<AuthorizationCode>>>>,
}

async fn callback(
    State(state): State<CallbackState>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    if let Some(error) = query.get("error") {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            &format!("Authorization failed: {error}"),
        );
    }
    if let Some(expected) = state.expected_state.as_deref()
        && query.get("state").map(String::as_str) != Some(expected)
    {
        return oauth_error(StatusCode::BAD_REQUEST, "OAuth state mismatch.");
    }
    let Some(code) = query.get("code").filter(|code| !code.is_empty()) else {
        return oauth_error(StatusCode::BAD_REQUEST, "Missing authorization code.");
    };
    let sender = state
        .sender
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    let Some(sender) = sender else {
        return oauth_error(
            StatusCode::CONFLICT,
            "Authorization callback was already claimed.",
        );
    };
    let _ = sender.send(AuthorizationCode {
        code: code.clone(),
        state: query.get("state").cloned(),
    });
    (
        StatusCode::OK,
        Html("Authentication completed. You can close this window.".to_owned()),
    )
        .into_response()
}

fn oauth_error(status: StatusCode, message: &str) -> Response {
    (status, Html(message.to_owned())).into_response()
}

pub(crate) struct LoopbackServer {
    redirect_uri: String,
    receiver: Option<oneshot::Receiver<AuthorizationCode>>,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

impl LoopbackServer {
    pub(crate) async fn bind(
        bind_host: &str,
        redirect_host: &str,
        port: u16,
        path: &str,
        expected_state: Option<String>,
    ) -> anyhow::Result<Self> {
        if !path.starts_with('/') || Path::new(path).components().count() == 0 {
            bail!("OAuth callback path must start with `/`");
        }
        let listener = TcpListener::bind((bind_host, port))
            .await
            .with_context(|| format!("failed to bind OAuth callback on {bind_host}:{port}"))?;
        let address = listener.local_addr()?;
        let (sender, receiver) = oneshot::channel();
        let state = CallbackState {
            expected_state,
            sender: Arc::new(Mutex::new(Some(sender))),
        };
        let router = Router::new().route(path, get(callback)).with_state(state);
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let task = tokio::spawn(async move {
            let result = axum::serve(listener, router)
                .with_graceful_shutdown(server_shutdown.cancelled_owned())
                .await;
            if let Err(error) = result {
                tracing::debug!(%error, "provider auth: OAuth callback server stopped");
            }
        });
        Ok(Self {
            redirect_uri: loopback_url(redirect_host, address, path),
            receiver: Some(receiver),
            shutdown,
            task,
        })
    }

    pub(crate) fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }

    pub(crate) async fn wait(
        mut self,
        interaction: &dyn AuthInteraction,
        manual_message: impl Into<String>,
        expected_state: Option<&str>,
        timeout: Duration,
    ) -> anyhow::Result<AuthorizationCode> {
        let signal = interaction.signal();
        let placeholder = self.redirect_uri.clone();
        let manual_message = manual_message.into();
        let mut receiver = self
            .receiver
            .take()
            .context("OAuth callback receiver missing")?;
        let wait = async {
            let prompt = interaction.prompt(AuthPrompt::ManualCode {
                message: manual_message,
                placeholder,
            });
            tokio::pin!(prompt);
            let first = tokio::select! {
                _ = signal.cancelled() => bail!("login cancelled"),
                callback = &mut receiver => return callback.context("OAuth callback server stopped"),
                manual = &mut prompt => manual?,
            };
            if !first.trim().is_empty() {
                return parse_authorization_input(&first, expected_state);
            }
            tokio::select! {
                _ = signal.cancelled() => bail!("login cancelled"),
                callback = &mut receiver => callback.context("OAuth callback server stopped"),
            }
        };
        let result = tokio::time::timeout(timeout, wait)
            .await
            .context("OAuth login timed out")?;
        self.shutdown.cancel();
        self.task.abort();
        result
    }
}

impl Drop for LoopbackServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.task.abort();
    }
}

fn loopback_url(redirect_host: &str, address: SocketAddr, path: &str) -> String {
    let host = if redirect_host.contains(':') && !redirect_host.starts_with('[') {
        format!("[{redirect_host}]")
    } else {
        redirect_host.to_owned()
    };
    format!("http://{host}:{}{path}", address.port())
}

pub(crate) fn parse_authorization_input(
    input: &str,
    expected_state: Option<&str>,
) -> anyhow::Result<AuthorizationCode> {
    let value = input.trim();
    if value.is_empty() {
        bail!("missing authorization code");
    }

    let (code, state) = if let Ok(url) = Url::parse(value) {
        let query: HashMap<_, _> = url.query_pairs().into_owned().collect();
        (query.get("code").cloned(), query.get("state").cloned())
    } else if let Some((code, state)) = value.split_once('#') {
        (Some(code.to_owned()), Some(state.to_owned()))
    } else if value.contains("code=") {
        let query: HashMap<_, _> = url::form_urlencoded::parse(value.as_bytes())
            .into_owned()
            .collect();
        (query.get("code").cloned(), query.get("state").cloned())
    } else {
        (Some(value.to_owned()), None)
    };

    let code = code
        .filter(|code| !code.is_empty())
        .context("missing authorization code")?;
    if let (Some(expected), Some(actual)) = (expected_state, state.as_deref())
        && expected != actual
    {
        bail!("OAuth state mismatch");
    }
    Ok(AuthorizationCode {
        code,
        state: state.or_else(|| expected_state.map(ToOwned::to_owned)),
    })
}

pub(crate) fn validate_http_url(value: &str) -> anyhow::Result<Url> {
    let url = Url::parse(value).with_context(|| format!("invalid URL `{value}`"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        bail!("URL must use http(s) and include a hostname");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("URL must not include credentials");
    }
    Ok(url)
}

pub(crate) fn normalize_domain(value: &str) -> anyhow::Result<String> {
    let value = value.trim();
    if value.is_empty() {
        return Ok("github.com".to_owned());
    }
    let candidate = if value.contains("://") {
        value.to_owned()
    } else {
        format!("https://{value}")
    };
    let url = validate_http_url(&candidate)?;
    if !url.username().is_empty() || url.password().is_some() {
        bail!("enterprise domain must not include credentials");
    }
    url.host_str()
        .map(str::to_ascii_lowercase)
        .context("enterprise domain is missing a hostname")
}

pub(crate) async fn read_json<T: DeserializeOwned>(
    response: reqwest::Response,
    operation: &str,
) -> anyhow::Result<T> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let body = truncate_error_body(&body);
        bail!("{operation} failed ({status}): {body}");
    }
    response
        .json::<T>()
        .await
        .with_context(|| format!("{operation} returned invalid JSON"))
}

pub(crate) fn truncate_error_body(body: &str) -> String {
    const LIMIT: usize = 4096;
    if body.len() <= LIMIT {
        return body.to_owned();
    }
    let mut end = LIMIT;
    while !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &body[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_input_accepts_url_hash_form_and_bare_code() {
        let expected = Some("expected");
        assert_eq!(
            parse_authorization_input("https://localhost/cb?code=a&state=expected", expected)
                .unwrap()
                .code,
            "a"
        );
        assert_eq!(
            parse_authorization_input("b#expected", expected)
                .unwrap()
                .code,
            "b"
        );
        assert_eq!(
            parse_authorization_input("code=c&state=expected", expected)
                .unwrap()
                .code,
            "c"
        );
        assert_eq!(
            parse_authorization_input("d", expected).unwrap(),
            AuthorizationCode {
                code: "d".to_owned(),
                state: Some("expected".to_owned())
            }
        );
        assert!(parse_authorization_input("a#wrong", expected).is_err());
    }

    #[test]
    fn url_validation_rejects_non_http_schemes_and_domain_credentials() {
        assert!(validate_http_url("javascript:alert(1)").is_err());
        assert!(normalize_domain("https://user:pass@example.com").is_err());
        assert_eq!(
            normalize_domain("GitHub.Example.com/path").unwrap(),
            "github.example.com"
        );
    }
}

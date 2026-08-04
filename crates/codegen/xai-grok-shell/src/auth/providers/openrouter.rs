use std::time::Duration;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

use super::flow::{LoopbackServer, read_json};
use super::{
    AuthInteraction, AuthNotification, LoginMode, ProviderCredential, ProviderRefreshOutcome,
};

const AUTHORIZE_URL: &str = "https://openrouter.ai/auth";
const TOKEN_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
const LOGIN_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const TOKEN_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Serialize)]
struct TokenRequest<'a> {
    code: &'a str,
    code_verifier: &'a str,
    code_challenge_method: &'static str,
}

#[derive(Deserialize)]
struct TokenResponse {
    key: Option<String>,
}

impl TokenResponse {
    fn into_credential(self) -> anyhow::Result<ProviderCredential> {
        let key = self
            .key
            .filter(|key| !key.trim().is_empty())
            .context("OpenRouter OAuth response carries no `key`")?;
        Ok(ProviderCredential::permanent(key))
    }
}

fn build_authorize_url(callback_url: &str, code_challenge: &str) -> anyhow::Result<Url> {
    let mut url = Url::parse(AUTHORIZE_URL).context("invalid OpenRouter authorization URL")?;
    url.query_pairs_mut()
        .append_pair("callback_url", callback_url)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256");
    Ok(url)
}

async fn exchange_authorization_code(
    code: &str,
    verifier: &str,
    signal: &CancellationToken,
) -> anyhow::Result<ProviderCredential> {
    if signal.is_cancelled() {
        bail!("login cancelled");
    }

    let exchange = async {
        let response = crate::http::shared_client()
            .post(TOKEN_URL)
            .timeout(TOKEN_EXCHANGE_TIMEOUT)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&TokenRequest {
                code,
                code_verifier: verifier,
                code_challenge_method: "S256",
            })
            .send()
            .await
            .context("OpenRouter OAuth key exchange request failed")?;
        read_json::<TokenResponse>(response, "OpenRouter OAuth key exchange")
            .await?
            .into_credential()
    };

    tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        result = exchange => result,
    }
}

pub(super) async fn login(
    interaction: &dyn AuthInteraction,
    mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    match mode {
        None | Some(LoginMode::Browser) => {}
        Some(LoginMode::DeviceCode) => bail!("OpenRouter supports browser login only"),
    }

    let signal = interaction.signal();
    if signal.is_cancelled() {
        bail!("login cancelled");
    }
    let pkce = crate::auth::oidc::protocol::generate_pkce();
    let callback_path = format!("/oauth/callback/{}", Uuid::new_v4());
    let server = tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        server = LoopbackServer::bind("127.0.0.1", "127.0.0.1", 0, &callback_path, None) => server?,
    };
    let callback_url = server.redirect_uri().to_owned();
    let authorize_url = build_authorize_url(&callback_url, &pkce.code_challenge)?;

    tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        notified = interaction.notify(AuthNotification::AuthUrl {
            url: authorize_url.to_string(),
            instructions: "Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here.".to_owned(),
        }) => notified?,
    }

    let authorization = server
        .wait(
            interaction,
            "Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
            None,
            LOGIN_TIMEOUT,
        )
        .await?;
    exchange_authorization_code(&authorization.code, &pkce.code_verifier, &signal).await
}

pub(super) async fn refresh(
    credential: ProviderCredential,
    _signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    Ok(ProviderRefreshOutcome::Save(credential))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorize_url_contains_only_openrouter_pkce_parameters() {
        let callback = "http://127.0.0.1:43210/oauth/callback/test-id";
        let url = build_authorize_url(callback, "test-challenge").unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("openrouter.ai"));
        assert_eq!(url.path(), "/auth");
        assert_eq!(
            url.query_pairs().into_owned().collect::<Vec<_>>(),
            vec![
                ("callback_url".to_owned(), callback.to_owned()),
                ("code_challenge".to_owned(), "test-challenge".to_owned()),
                ("code_challenge_method".to_owned(), "S256".to_owned()),
            ]
        );
    }

    #[test]
    fn token_response_requires_key_and_builds_redacted_permanent_credential() {
        let credential = serde_json::from_str::<TokenResponse>(r#"{"key":"sk-or-secret"}"#)
            .unwrap()
            .into_credential()
            .unwrap();
        assert_eq!(credential.access, "sk-or-secret");
        assert!(credential.refresh.is_empty());
        assert!(!format!("{credential:?}").contains("sk-or-secret"));
        assert!(
            serde_json::from_str::<TokenResponse>("{}")
                .unwrap()
                .into_credential()
                .is_err()
        );
    }
}

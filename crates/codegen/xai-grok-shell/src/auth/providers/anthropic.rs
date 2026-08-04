use std::time::Duration;

use anyhow::{Context, bail, ensure};
use serde::{Deserialize, Serialize};
use url::Url;

use super::flow::{LoopbackServer, read_json};
use super::store::unix_millis;
use crate::auth::oidc::protocol::generate_pkce;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CALLBACK_PORT: u16 = 53692;
const CALLBACK_PATH: &str = "/callback";
const REDIRECT_URI: &str = "http://localhost:53692/callback";
const SCOPE: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const EXPIRY_SKEW_MILLIS: u64 = 5 * 60 * 1_000;

#[derive(Serialize)]
struct AuthorizationCodeRequest<'a> {
    grant_type: &'static str,
    client_id: &'static str,
    code: &'a str,
    state: &'a str,
    redirect_uri: &'static str,
    code_verifier: &'a str,
}

#[derive(Serialize)]
struct RefreshTokenRequest<'a> {
    grant_type: &'static str,
    client_id: &'static str,
    refresh_token: &'a str,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
}

pub(super) async fn login(
    interaction: &dyn super::AuthInteraction,
    mode: Option<super::LoginMode>,
) -> anyhow::Result<super::ProviderCredential> {
    match mode.unwrap_or(super::LoginMode::Browser) {
        super::LoginMode::Browser => {}
        super::LoginMode::DeviceCode => {
            bail!("Anthropic does not support device-code login; use browser mode")
        }
    }

    let signal = interaction.signal();
    if signal.is_cancelled() {
        bail!("login cancelled");
    }
    let pkce = generate_pkce();
    let bind_host = callback_bind_host();
    let server = tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        server = LoopbackServer::bind(
            &bind_host,
            "localhost",
            CALLBACK_PORT,
            CALLBACK_PATH,
            Some(pkce.code_verifier.clone()),
        ) => server?,
    };
    let auth_url = authorization_url(&pkce.code_challenge, &pkce.code_verifier)?;

    notify(
        interaction,
        &signal,
        super::AuthNotification::AuthUrl {
            url: auth_url,
            instructions: "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here.".to_owned(),
        },
    )
    .await?;
    let authorization = server
        .wait(
            interaction,
            "Complete login in your browser, or paste the authorization code / redirect URL here:",
            Some(&pkce.code_verifier),
            CALLBACK_TIMEOUT,
        )
        .await?;
    let state = authorization
        .state
        .as_deref()
        .context("missing OAuth state")?;

    notify(
        interaction,
        &signal,
        super::AuthNotification::Progress {
            message: "Exchanging authorization code for tokens...".to_owned(),
        },
    )
    .await?;
    let request = authorization_code_request(&authorization.code, state, &pkce.code_verifier);
    let response = post_token(&request, &signal, "Anthropic token exchange").await?;
    credential_from_token(response, unix_millis())
}

pub(super) async fn refresh(
    credential: super::ProviderCredential,
    signal: tokio_util::sync::CancellationToken,
) -> anyhow::Result<super::ProviderRefreshOutcome> {
    ensure!(
        !credential.refresh.trim().is_empty(),
        "Anthropic credential is missing a refresh token"
    );
    let request = refresh_token_request(&credential.refresh);
    let response = post_token(&request, &signal, "Anthropic token refresh").await?;
    Ok(super::ProviderRefreshOutcome::Save(credential_from_token(
        response,
        unix_millis(),
    )?))
}

async fn notify(
    interaction: &dyn super::AuthInteraction,
    signal: &tokio_util::sync::CancellationToken,
    notification: super::AuthNotification,
) -> anyhow::Result<()> {
    tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("login cancelled"),
        result = interaction.notify(notification) => result,
    }
}

async fn post_token<T: Serialize + ?Sized>(
    body: &T,
    signal: &tokio_util::sync::CancellationToken,
    operation: &str,
) -> anyhow::Result<TokenResponse> {
    let request = async {
        let response = crate::http::shared_client()
            .post(TOKEN_URL)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(body)
            .send()
            .await
            .with_context(|| format!("{operation} request failed"))?;
        read_json(response, operation).await
    };
    tokio::select! {
        biased;
        _ = signal.cancelled() => bail!("{operation} cancelled"),
        result = tokio::time::timeout(HTTP_TIMEOUT, request) => {
            result.with_context(|| format!("{operation} timed out after 30 seconds"))?
        },
    }
}

fn callback_bind_host() -> String {
    std::env::var("PI_OAUTH_CALLBACK_HOST")
        .ok()
        .map(|host| host.trim().to_owned())
        .filter(|host| !host.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_owned())
}

fn authorization_url(challenge: &str, verifier: &str) -> anyhow::Result<String> {
    let mut url = Url::parse(AUTHORIZE_URL).context("invalid Anthropic authorization URL")?;
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPE)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", verifier);
    Ok(url.to_string())
}

fn authorization_code_request<'a>(
    code: &'a str,
    state: &'a str,
    verifier: &'a str,
) -> AuthorizationCodeRequest<'a> {
    AuthorizationCodeRequest {
        grant_type: "authorization_code",
        client_id: CLIENT_ID,
        code,
        state,
        redirect_uri: REDIRECT_URI,
        code_verifier: verifier,
    }
}

fn refresh_token_request(refresh_token: &str) -> RefreshTokenRequest<'_> {
    RefreshTokenRequest {
        grant_type: "refresh_token",
        client_id: CLIENT_ID,
        refresh_token,
    }
}

fn credential_from_token(
    response: TokenResponse,
    now_millis: u64,
) -> anyhow::Result<super::ProviderCredential> {
    ensure!(
        !response.access_token.trim().is_empty(),
        "Anthropic token response is missing access_token"
    );
    ensure!(
        !response.refresh_token.trim().is_empty(),
        "Anthropic token response is missing refresh_token"
    );
    let expires = now_millis
        .saturating_add(response.expires_in.saturating_mul(1_000))
        .saturating_sub(EXPIRY_SKEW_MILLIS);
    Ok(super::ProviderCredential::oauth(
        response.access_token,
        response.refresh_token,
        expires,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::super::flow::parse_authorization_input;
    use super::*;

    #[test]
    fn authorization_contract_and_manual_inputs_are_exact() {
        let value = authorization_url("challenge", "verifier").unwrap();
        let url = Url::parse(&value).unwrap();
        assert_eq!(value.split('?').next(), Some(AUTHORIZE_URL));
        assert_eq!(url.query_pairs().count(), 8);
        assert_eq!(
            url.query_pairs().into_owned().collect::<BTreeMap<_, _>>(),
            BTreeMap::from([
                ("client_id".to_owned(), CLIENT_ID.to_owned()),
                ("code".to_owned(), "true".to_owned()),
                ("code_challenge".to_owned(), "challenge".to_owned()),
                ("code_challenge_method".to_owned(), "S256".to_owned()),
                ("redirect_uri".to_owned(), REDIRECT_URI.to_owned()),
                ("response_type".to_owned(), "code".to_owned()),
                ("scope".to_owned(), SCOPE.to_owned()),
                ("state".to_owned(), "verifier".to_owned()),
            ])
        );

        for (input, code) in [
            (
                "http://localhost:53692/callback?code=url-code&state=verifier",
                "url-code",
            ),
            ("hash-code#verifier", "hash-code"),
            ("code=form-code&state=verifier", "form-code"),
            ("bare-code", "bare-code"),
        ] {
            let parsed = parse_authorization_input(input, Some("verifier")).unwrap();
            assert_eq!(parsed.code, code);
            assert_eq!(parsed.state.as_deref(), Some("verifier"));
        }
        assert!(parse_authorization_input("code#wrong", Some("verifier")).is_err());
    }

    #[test]
    fn token_request_response_and_expiry_contracts_are_exact() {
        assert_eq!(
            serde_json::to_value(authorization_code_request("code", "state", "verifier")).unwrap(),
            json!({
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": "code",
                "state": "state",
                "redirect_uri": REDIRECT_URI,
                "code_verifier": "verifier",
            })
        );
        assert_eq!(
            serde_json::to_value(refresh_token_request("refresh")).unwrap(),
            json!({
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": "refresh",
            })
        );

        let complete = json!({
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 600,
        });
        for required in ["access_token", "refresh_token", "expires_in"] {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(required);
            assert!(serde_json::from_value::<TokenResponse>(missing).is_err());
        }
        let credential =
            credential_from_token(serde_json::from_value(complete).unwrap(), 1_000_000).unwrap();
        assert_eq!(credential.access, "access");
        assert_eq!(credential.refresh, "refresh");
        assert_eq!(credential.expires, 1_300_000);
    }
}

//! Provider-scoped OAuth RPCs. These never mutate the global xAI auth method.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use agent_client_protocol as acp;
use agent_client_protocol::Client as _;
use anyhow::{Context, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use xai_acp_lib::AcpAgentGatewaySender;
use xai_grok_tools::implementations::grok_build::ask_user_question::{
    AskUserQuestionExtRequest, AskUserQuestionExtResponse, AskUserQuestionMode, Question,
    QuestionOption,
};

use super::{ExtResult, parse_params, to_raw_response};
use crate::agent::MvpAgent;
use crate::auth::providers::{
    AuthInteraction, AuthNotification, AuthPrompt, AuthPromptResponse, AuthSecret, LoginMode,
    ProviderCredentialMethod, ProviderId, SelectOption, provider_descriptor,
    validate_oauth_login_request,
};

#[tracing::instrument(skip_all, fields(method = %args.method))]
pub(crate) async fn handle(agent: &MvpAgent, args: &acp::ExtRequest) -> ExtResult {
    match args.method.as_ref() {
        "x.ai/providerAuth/info" => handle_info(agent).await,
        "x.ai/providerAuth/login" => handle_login(agent, args).await,
        "x.ai/providerAuth/logout" => handle_logout(agent, args).await,
        "x.ai/providerAuth/cancel" => handle_cancel(agent, args),
        _ => Err(acp::Error::method_not_found()),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LoginParams {
    provider: String,
    session_id: String,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    request_seq: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProviderLoginPlan {
    provider: ProviderId,
    method: ProviderCredentialMethod,
    mode: Option<LoginMode>,
}

fn plan_login(params: &LoginParams) -> anyhow::Result<ProviderLoginPlan> {
    let provider = params.provider.parse::<ProviderId>()?;
    let descriptor = provider_descriptor(provider);
    let method = params
        .method
        .as_deref()
        .map(parse_login_method)
        .transpose()?
        .unwrap_or_else(|| descriptor.default_login_method());
    if !descriptor.supports_method(method) {
        bail!(
            "{} does not support {} login",
            descriptor.display_name,
            method.as_str()
        );
    }
    let mode = params.mode.as_deref().map(parse_login_mode).transpose()?;
    match method {
        ProviderCredentialMethod::OAuth => validate_oauth_login_request(provider, mode)?,
        ProviderCredentialMethod::ApiKey if mode.is_some() => {
            bail!("mode is only valid when method is oauth")
        }
        ProviderCredentialMethod::ApiKey => {}
    }
    Ok(ProviderLoginPlan {
        provider,
        method,
        mode,
    })
}

async fn handle_login(agent: &MvpAgent, args: &acp::ExtRequest) -> ExtResult {
    let params: LoginParams = parse_params(args)?;
    if params.session_id.trim().is_empty() {
        return Err(acp::Error::invalid_params().data("sessionId is required"));
    }
    let plan = plan_login(&params)
        .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;

    let (signal, _guard) = agent
        .interactive_auth
        .begin_provider(plan.provider, params.request_seq);
    let interaction = AcpAuthInteraction::new(
        agent.gateway.clone(),
        params.session_id,
        plan.provider,
        params.request_seq,
        signal,
    );
    let login_result = match plan.method {
        ProviderCredentialMethod::OAuth => {
            crate::auth::providers::login_and_store(plan.provider, &interaction, plan.mode)
                .await
                .map(|_| ())
        }
        ProviderCredentialMethod::ApiKey => {
            crate::auth::providers::login_api_key_and_store(plan.provider, &interaction)
                .await
                .map(|_| ())
        }
    };
    login_result.map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
    let radius_catalog_refreshed = agent.models_manager.on_auth_changed().await;

    to_raw_response(&serde_json::json!({
        "ok": true,
        "provider": plan.provider.as_str(),
        "displayName": plan.provider.display_name(),
        "method": plan.method.as_str(),
        "message": login_success_message(plan.provider, plan.method, radius_catalog_refreshed),
        "catalogRefreshed": (plan.provider == ProviderId::Radius).then_some(radius_catalog_refreshed),
    }))
}

fn login_success_message(
    provider: ProviderId,
    method: ProviderCredentialMethod,
    radius_catalog_refreshed: bool,
) -> String {
    match method {
        ProviderCredentialMethod::OAuth => format!("Signed in to {}.", provider.display_name()),
        ProviderCredentialMethod::ApiKey
            if provider == ProviderId::Radius && !radius_catalog_refreshed =>
        {
            format!(
                "API key saved for {}; Radius catalog refresh failed and can be retried later.",
                provider.display_name()
            )
        }
        ProviderCredentialMethod::ApiKey => {
            format!("API key saved for {}.", provider.display_name())
        }
    }
}

fn parse_login_method(value: &str) -> anyhow::Result<ProviderCredentialMethod> {
    match value.trim().to_ascii_lowercase().as_str() {
        "oauth" => Ok(ProviderCredentialMethod::OAuth),
        "api_key" => Ok(ProviderCredentialMethod::ApiKey),
        value => bail!("unknown provider login method `{value}`"),
    }
}

fn parse_login_mode(value: &str) -> anyhow::Result<LoginMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "browser" | "oauth" => Ok(LoginMode::Browser),
        "device" | "device-code" | "device_code" => Ok(LoginMode::DeviceCode),
        value => bail!("unknown provider login mode `{value}`"),
    }
}

#[derive(Deserialize)]
struct ProviderParams {
    provider: String,
}

async fn handle_logout(agent: &MvpAgent, args: &acp::ExtRequest) -> ExtResult {
    let params: ProviderParams = parse_params(args)?;
    let provider = params
        .provider
        .parse::<ProviderId>()
        .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;
    agent.interactive_auth.cancel_provider(provider, None);
    let was_logged_in = crate::auth::providers::logout(provider)
        .await
        .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
    agent.models_manager.on_auth_changed().await;
    to_raw_response(&serde_json::json!({
        "ok": true,
        "provider": provider.as_str(),
        "wasLoggedIn": was_logged_in,
    }))
}

fn handle_cancel(agent: &MvpAgent, args: &acp::ExtRequest) -> ExtResult {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CancelParams {
        provider: String,
        #[serde(default)]
        request_seq: Option<u64>,
    }

    let params: CancelParams = parse_params(args)?;
    let provider = params
        .provider
        .parse::<ProviderId>()
        .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;
    let cancelled = agent
        .interactive_auth
        .cancel_provider(provider, params.request_seq);
    to_raw_response(&serde_json::json!({ "cancelled": cancelled }))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProviderInfo {
    id: &'static str,
    display_name: &'static str,
    supported_methods: Vec<ProviderCredentialMethod>,
    oauth_transports: Vec<crate::auth::providers::ProviderLoginTransport>,
    authenticated: bool,
    credential_type: Option<ProviderCredentialMethod>,
    expires_at: Option<u64>,
    model_count: usize,
}

fn provider_info(
    provider: ProviderId,
    slot: &crate::auth::providers::ProviderSlotState,
    model_count: usize,
) -> ProviderInfo {
    let descriptor = provider_descriptor(provider);
    let supported_methods = descriptor
        .login_options()
        .into_iter()
        .map(|option| option.method)
        .collect();
    let (credential_type, expires_at) = match slot {
        crate::auth::providers::ProviderSlotState::Known(
            crate::auth::providers::ProviderStoredCredential::OAuth(credential),
        ) => (
            Some(ProviderCredentialMethod::OAuth),
            Some(credential.expires),
        ),
        crate::auth::providers::ProviderSlotState::Known(
            crate::auth::providers::ProviderStoredCredential::ApiKey(_),
        ) => (Some(ProviderCredentialMethod::ApiKey), None),
        crate::auth::providers::ProviderSlotState::Missing
        | crate::auth::providers::ProviderSlotState::PresentUnsupportedOrInvalid => (None, None),
    };
    ProviderInfo {
        id: provider.as_str(),
        display_name: descriptor.display_name,
        supported_methods,
        oauth_transports: descriptor.oauth_transports(),
        authenticated: slot.is_known(),
        credential_type,
        expires_at,
        model_count,
    }
}

async fn handle_info(agent: &MvpAgent) -> ExtResult {
    let available = agent.models_manager.available();
    let mut providers = Vec::with_capacity(ProviderId::ALL.len());
    for provider in ProviderId::ALL {
        let slot = crate::auth::providers::stored_slot(provider)
            .await
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        let model_count = available
            .keys()
            .filter(|model_id| {
                crate::auth::providers::parse_namespaced_model_id(model_id.0.as_ref())
                    .is_some_and(|(model_provider, _)| model_provider == provider)
            })
            .count();
        providers.push(provider_info(provider, &slot, model_count));
    }
    to_raw_response(&serde_json::json!({ "providers": providers }))
}

#[derive(Clone)]
struct AcpAuthInteraction {
    gateway: AcpAgentGatewaySender,
    session_id: String,
    provider: ProviderId,
    request_seq: Option<u64>,
    signal: CancellationToken,
    prompt_seq: Arc<AtomicU64>,
    manual_input: Arc<Mutex<Option<String>>>,
}

impl AcpAuthInteraction {
    fn new(
        gateway: AcpAgentGatewaySender,
        session_id: String,
        provider: ProviderId,
        request_seq: Option<u64>,
        signal: CancellationToken,
    ) -> Self {
        Self {
            gateway,
            session_id,
            provider,
            request_seq,
            signal,
            prompt_seq: Arc::new(AtomicU64::new(0)),
            manual_input: Arc::new(Mutex::new(None)),
        }
    }

    async fn ask(&self, question: Question) -> anyhow::Result<PromptAnswer> {
        let sequence = self.prompt_seq.fetch_add(1, Ordering::Relaxed);
        let question_key = question.question.clone();
        let request = AskUserQuestionExtRequest {
            session_id: self.session_id.clone(),
            tool_call_id: format!("provider-auth-{}-{sequence}", self.provider.as_str()),
            questions: vec![question],
            mode: AskUserQuestionMode::Default,
        };
        let request = acp::ExtRequest::new(
            "x.ai/ask_user_question",
            serde_json::value::to_raw_value(&request)?.into(),
        );
        let response = tokio::select! {
            _ = self.signal.cancelled() => bail!("login cancelled"),
            response = self.gateway.ext_method(request) => response
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
        };
        let response: AskUserQuestionExtResponse =
            serde_json::from_str(response.0.get()).context("invalid authentication response")?;
        match response {
            AskUserQuestionExtResponse::Accepted {
                answers,
                annotations,
            } => Ok(PromptAnswer {
                selected: answers
                    .get(&question_key)
                    .and_then(|values| values.first())
                    .cloned(),
                notes: annotations
                    .and_then(|values| values.get(&question_key).cloned())
                    .and_then(|annotation| annotation.notes),
            }),
            AskUserQuestionExtResponse::Cancelled
            | AskUserQuestionExtResponse::ChatAboutThis { .. }
            | AskUserQuestionExtResponse::SkipInterview { .. } => bail!("login cancelled"),
        }
    }

    async fn prompt_secret(&self, prompt: String) -> anyhow::Result<AuthSecret> {
        let sequence = self.prompt_seq.fetch_add(1, Ordering::Relaxed);
        let request = xai_acp_lib::PromptSecretRequest {
            provider: self.provider.as_str().to_owned(),
            provider_display_name: self.provider.display_name().to_owned(),
            session_id: self.session_id.clone(),
            prompt,
            request_seq: self.request_seq.or(Some(sequence)),
        };
        let request = acp::ExtRequest::new(
            xai_acp_lib::PROMPT_SECRET_METHOD,
            serde_json::value::to_raw_value(&request)?.into(),
        );
        let response = tokio::select! {
            _ = self.signal.cancelled() => bail!("login cancelled"),
            response = self.gateway.ext_method(request) => response.map_err(|error| {
                if matches!(
                    error.code,
                    acp::ErrorCode::MethodNotFound | acp::ErrorCode::Other(-32004)
                ) {
                    anyhow::anyhow!("ACP client does not support secure provider API-key entry")
                } else {
                    anyhow::anyhow!("secure provider API-key prompt failed ({})", error.code)
                }
            })?,
        };
        decode_prompt_secret_response(response)
    }

    async fn show_url(
        &self,
        question: String,
        description: String,
        url: &str,
    ) -> anyhow::Result<()> {
        let _ = webbrowser::open(url);
        let answer = self
            .ask(Question {
                question,
                options: vec![
                    question_option("Continue", description),
                    question_option("Cancel", "Stop this login attempt."),
                ],
                multi_select: Some(false),
                id: None,
            })
            .await?;
        if answer.selected.as_deref() == Some("Cancel") {
            bail!("login cancelled");
        }
        if let Some(value) = answer.notes.filter(|value| !value.trim().is_empty()) {
            *self
                .manual_input
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(value);
        }
        Ok(())
    }

    async fn prompt_text(
        &self,
        message: String,
        placeholder: String,
        default_label: &str,
    ) -> anyhow::Result<String> {
        let answer = self
            .ask(Question {
                question: message,
                options: vec![
                    question_option(default_label, format!("Default: {placeholder}")),
                    question_option("Cancel", "Stop this login attempt."),
                ],
                multi_select: Some(false),
                id: None,
            })
            .await?;
        if answer.selected.as_deref() == Some("Cancel") {
            bail!("login cancelled");
        }
        Ok(answer.notes.unwrap_or_default())
    }
}

#[derive(Debug)]
struct PromptAnswer {
    selected: Option<String>,
    notes: Option<String>,
}

fn question_option(label: impl Into<String>, description: impl Into<String>) -> QuestionOption {
    QuestionOption {
        label: label.into(),
        description: description.into(),
        preview: None,
        id: None,
    }
}

#[async_trait(?Send)]
impl AuthInteraction for AcpAuthInteraction {
    fn signal(&self) -> CancellationToken {
        self.signal.clone()
    }

    async fn notify(&self, notification: AuthNotification) -> anyhow::Result<()> {
        match notification {
            AuthNotification::AuthUrl { url, instructions } => {
                self.show_url(
                    format!("Sign in to {}", self.provider.display_name()),
                    format!("{instructions}\n\n{url}\n\nPaste the final redirect URL into the free-form field when the browser runs elsewhere."),
                    &url,
                )
                .await
            }
            AuthNotification::DeviceCode {
                user_code,
                verification_uri,
                ..
            } => {
                self.show_url(
                    format!("Authorize {}", self.provider.display_name()),
                    format!("Open {verification_uri} and enter code {user_code}, then continue."),
                    &verification_uri,
                )
                .await
            }
            AuthNotification::Progress { message } | AuthNotification::Info { message, .. } => {
                tracing::info!(provider = %self.provider, %message, "provider auth progress");
                Ok(())
            }
        }
    }

    async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<AuthPromptResponse> {
        match prompt {
            AuthPrompt::ManualCode { .. } => Ok(AuthPromptResponse::Text(
                self.manual_input
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                    .unwrap_or_default(),
            )),
            AuthPrompt::Text {
                message,
                placeholder,
            } => self
                .prompt_text(message, placeholder, "Use default")
                .await
                .map(AuthPromptResponse::Text),
            AuthPrompt::Select { message, options } => {
                let answer = self
                    .ask(Question {
                        question: message,
                        options: options
                            .iter()
                            .map(|option| question_option(&option.label, ""))
                            .collect(),
                        multi_select: Some(false),
                        id: None,
                    })
                    .await?;
                selected_option_id(&options, answer).map(AuthPromptResponse::Text)
            }
            AuthPrompt::Secret { message, .. } => self
                .prompt_secret(message)
                .await
                .map(AuthPromptResponse::Secret),
        }
    }
}

fn decode_prompt_secret_response(response: acp::ExtResponse) -> anyhow::Result<AuthSecret> {
    let response: xai_acp_lib::PromptSecretResponse =
        serde_json::from_str(response.0.get()).context("invalid secure provider-auth response")?;
    match response {
        xai_acp_lib::PromptSecretResponse::Accepted { secret } => {
            let secret = secret.into_zeroizing_string();
            let trimmed = secret.trim();
            if trimmed.is_empty() {
                bail!("API key cannot be empty");
            }
            Ok(AuthSecret::new(trimmed.to_owned()))
        }
        xai_acp_lib::PromptSecretResponse::Cancelled => bail!("login cancelled"),
    }
}

fn selected_option_id(options: &[SelectOption], answer: PromptAnswer) -> anyhow::Result<String> {
    if let Some(notes) = answer.notes.filter(|value| !value.trim().is_empty()) {
        return Ok(notes);
    }
    let selected = answer
        .selected
        .context("no authentication option selected")?;
    options
        .iter()
        .find(|option| option.label == selected)
        .map(|option| option.id.clone())
        .with_context(|| format!("unknown authentication option `{selected}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_login_params_reject_inline_secret_material() {
        let result = serde_json::from_value::<LoginParams>(serde_json::json!({
            "provider": "anthropic",
            "sessionId": "session",
            "method": "api_key",
            "secret": "sk-must-use-reverse-request"
        }));
        assert!(result.is_err());
    }

    #[test]
    fn provider_login_plan_separates_method_from_oauth_transport() {
        let legacy = plan_login(&LoginParams {
            provider: "anthropic".to_owned(),
            session_id: "session".to_owned(),
            method: None,
            mode: Some("browser".to_owned()),
            request_seq: Some(3),
        })
        .unwrap();
        assert_eq!(legacy.method, ProviderCredentialMethod::OAuth);
        assert_eq!(legacy.mode, Some(LoginMode::Browser));

        let api_key = plan_login(&LoginParams {
            provider: "anthropic".to_owned(),
            session_id: "session".to_owned(),
            method: Some("api_key".to_owned()),
            mode: None,
            request_seq: None,
        })
        .unwrap();
        assert_eq!(api_key.method, ProviderCredentialMethod::ApiKey);
        assert_eq!(api_key.mode, None);

        let invalid_mode = plan_login(&LoginParams {
            provider: "anthropic".to_owned(),
            session_id: "session".to_owned(),
            method: Some("api_key".to_owned()),
            mode: Some("browser".to_owned()),
            request_seq: None,
        })
        .unwrap_err();
        assert!(invalid_mode.to_string().contains("mode"));

        let codex_key = plan_login(&LoginParams {
            provider: "openai-codex".to_owned(),
            session_id: "session".to_owned(),
            method: Some("api_key".to_owned()),
            mode: None,
            request_seq: None,
        })
        .unwrap_err();
        assert!(codex_key.to_string().contains("does not support"));
    }

    #[test]
    fn provider_login_modes_are_strict() {
        assert_eq!(parse_login_mode("oauth").unwrap(), LoginMode::Browser);
        assert_eq!(
            parse_login_mode("device-code").unwrap(),
            LoginMode::DeviceCode
        );
        assert!(parse_login_mode("magic").is_err());
    }

    #[test]
    fn radius_api_key_completion_distinguishes_saved_key_from_catalog_failure() {
        let message =
            login_success_message(ProviderId::Radius, ProviderCredentialMethod::ApiKey, false);
        assert!(message.contains("API key saved"));
        assert!(message.contains("catalog refresh failed"));
        assert!(!message.contains("authenticated"));
    }

    #[test]
    fn provider_info_reports_methods_and_api_key_type_without_secret() {
        let info = provider_info(
            ProviderId::Anthropic,
            &crate::auth::providers::ProviderSlotState::Known(
                crate::auth::providers::ProviderStoredCredential::ApiKey(
                    crate::auth::providers::ProviderApiKeyCredential::new(
                        "sk-info-must-not-leak".to_owned(),
                    ),
                ),
            ),
            15,
        );
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"supportedMethods\":[\"oauth\",\"api_key\"]"));
        assert!(json.contains("\"oauthTransports\":[\"browser\"]"));
        assert!(json.contains("\"credentialType\":\"api_key\""));
        assert!(json.contains("\"expiresAt\":null"));
        assert!(!json.contains("sk-info-must-not-leak"));
    }

    #[test]
    fn prompt_secret_response_is_dedicated_and_cancel_safe() {
        let accepted = acp::ExtResponse::new(
            serde_json::value::to_raw_value(&xai_acp_lib::PromptSecretResponse::Accepted {
                secret: xai_acp_lib::RedactedSecret::new("  sk-provider-secret  "),
            })
            .unwrap()
            .into(),
        );
        let accepted = decode_prompt_secret_response(accepted)
            .unwrap()
            .into_zeroizing_string();
        assert_eq!(accepted.as_str(), "sk-provider-secret");

        let cancelled = acp::ExtResponse::new(
            serde_json::value::to_raw_value(&xai_acp_lib::PromptSecretResponse::Cancelled)
                .unwrap()
                .into(),
        );
        assert!(
            decode_prompt_secret_response(cancelled)
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
    }

    #[test]
    fn select_answers_map_labels_back_to_opaque_ids() {
        let options = vec![SelectOption {
            id: "device_code".to_owned(),
            label: "Device code".to_owned(),
        }];
        assert_eq!(
            selected_option_id(
                &options,
                PromptAnswer {
                    selected: Some("Device code".to_owned()),
                    notes: None,
                }
            )
            .unwrap(),
            "device_code"
        );
    }
}

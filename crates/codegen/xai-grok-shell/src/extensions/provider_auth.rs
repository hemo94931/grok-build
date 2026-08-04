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
    AuthInteraction, AuthNotification, AuthPrompt, LoginMode, ProviderId, SelectOption,
};

#[tracing::instrument(skip_all, fields(method = %args.method))]
pub async fn handle(agent: &MvpAgent, args: &acp::ExtRequest) -> ExtResult {
    match args.method.as_ref() {
        "x.ai/providerAuth/info" => handle_info().await,
        "x.ai/providerAuth/login" => handle_login(agent, args).await,
        "x.ai/providerAuth/logout" => handle_logout(agent, args).await,
        "x.ai/providerAuth/cancel" => handle_cancel(agent, args),
        _ => Err(acp::Error::method_not_found()),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoginParams {
    provider: String,
    session_id: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    request_seq: Option<u64>,
}

async fn handle_login(agent: &MvpAgent, args: &acp::ExtRequest) -> ExtResult {
    let params: LoginParams = parse_params(args)?;
    if params.session_id.trim().is_empty() {
        return Err(acp::Error::invalid_params().data("sessionId is required"));
    }
    let provider = params
        .provider
        .parse::<ProviderId>()
        .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;
    let mode = params
        .mode
        .as_deref()
        .map(parse_login_mode)
        .transpose()
        .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;

    let (signal, _guard) = agent
        .interactive_auth
        .begin_provider(provider, params.request_seq);
    let interaction =
        AcpAuthInteraction::new(agent.gateway.clone(), params.session_id, provider, signal);
    crate::auth::providers::login_and_store(provider, &interaction, mode)
        .await
        .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
    agent.models_manager.on_auth_changed().await;

    to_raw_response(&serde_json::json!({
        "ok": true,
        "provider": provider.as_str(),
        "displayName": provider.display_name(),
    }))
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
    authenticated: bool,
    expires_at: Option<u64>,
    model_count: usize,
}

async fn handle_info() -> ExtResult {
    let mut providers = Vec::with_capacity(ProviderId::ALL.len());
    for provider in ProviderId::ALL {
        let credential = crate::auth::providers::stored_credential(provider)
            .await
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        providers.push(ProviderInfo {
            id: provider.as_str(),
            display_name: provider.display_name(),
            authenticated: credential.is_some(),
            expires_at: credential.as_ref().map(|credential| credential.expires),
            model_count: crate::auth::providers::provider_models(provider, credential.as_ref())
                .len(),
        });
    }
    to_raw_response(&serde_json::json!({ "providers": providers }))
}

#[derive(Clone)]
struct AcpAuthInteraction {
    gateway: AcpAgentGatewaySender,
    session_id: String,
    provider: ProviderId,
    signal: CancellationToken,
    prompt_seq: Arc<AtomicU64>,
    manual_input: Arc<Mutex<Option<String>>>,
}

impl AcpAuthInteraction {
    fn new(
        gateway: AcpAgentGatewaySender,
        session_id: String,
        provider: ProviderId,
        signal: CancellationToken,
    ) -> Self {
        Self {
            gateway,
            session_id,
            provider,
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

    async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<String> {
        match prompt {
            AuthPrompt::ManualCode { .. } => Ok(self
                .manual_input
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
                .unwrap_or_default()),
            AuthPrompt::Text {
                message,
                placeholder,
            } => self.prompt_text(message, placeholder, "Use default").await,
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
                selected_option_id(&options, answer)
            }
        }
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
    fn provider_login_modes_are_strict() {
        assert_eq!(parse_login_mode("oauth").unwrap(), LoginMode::Browser);
        assert_eq!(
            parse_login_mode("device-code").unwrap(),
            LoginMode::DeviceCode
        );
        assert!(parse_login_mode("magic").is_err());
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

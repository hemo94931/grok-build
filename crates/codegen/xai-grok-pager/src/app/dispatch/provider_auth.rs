use crate::app::actions::Effect;
use crate::app::app_view::{ActiveView, AppView};
use crate::app::provider_auth::{
    ProviderAuthInfo, ProviderLoginIntent, ProviderLoginSuccess, login_selector_items,
    provider_completion_items, resolve_provider_login,
};
use crate::scrollback::block::RenderBlock;
use xai_acp_lib::ProviderAuthMethod;

pub(super) fn dispatch_login(app: &mut AppView, intent: ProviderLoginIntent) -> Vec<Effect> {
    let ActiveView::Agent(agent_id) = app.active_view else {
        app.show_toast("Open a session before signing in to a model provider.");
        return vec![];
    };
    let Some(agent) = app.agents.get(&agent_id) else {
        return vec![];
    };
    if agent.session.session_id.is_none() {
        app.show_toast("The active session is still starting.");
        return vec![];
    }
    vec![Effect::ProviderAuthInfo { agent_id, intent }]
}

pub(super) fn handle_info_complete(
    app: &mut AppView,
    agent_id: crate::app::agent::AgentId,
    intent: ProviderLoginIntent,
    result: Result<Vec<ProviderAuthInfo>, String>,
) -> Vec<Effect> {
    let Some(agent) = app.agents.get_mut(&agent_id) else {
        return vec![];
    };
    let providers = match result {
        Ok(providers) => providers,
        Err(error) => {
            let message = crate::app::effects::sanitize_provider_auth_error(&format!(
                "Could not load provider login options: {error}"
            ));
            agent.scrollback.push_block(RenderBlock::system(message));
            return vec![];
        }
    };
    crate::slash::commands::providers::replace_provider_items(provider_completion_items(
        &providers,
    ));

    match intent {
        ProviderLoginIntent::Menu => {
            let items = login_selector_items(&providers);
            agent.active_modal = Some(crate::views::modal::ActiveModal::ArgPicker {
                command: "login".to_owned(),
                args_query: String::new(),
                items: items.clone(),
                original_items: items,
                state: crate::views::picker::PickerState::input_active(),
                previous_palette: None,
                window: crate::views::modal_window::ModalWindowState::new(),
            });
            vec![]
        }
        ProviderLoginIntent::Provider { provider, method } => {
            let resolved = match resolve_provider_login(&providers, &provider, method) {
                Ok(resolved) => resolved,
                Err(error) => {
                    agent.scrollback.push_block(RenderBlock::system(error));
                    return vec![];
                }
            };
            start_provider_login(
                app,
                agent_id,
                resolved.provider,
                resolved.display_name,
                resolved.method,
            )
        }
    }
}

pub(super) fn start_provider_login(
    app: &mut AppView,
    agent_id: crate::app::agent::AgentId,
    provider: String,
    display_name: String,
    method: ProviderAuthMethod,
) -> Vec<Effect> {
    let Some(session_id) = app
        .agents
        .get(&agent_id)
        .and_then(|agent| agent.session.session_id.clone())
    else {
        return vec![];
    };
    let request_seq = app.next_auth_request_seq;
    app.next_auth_request_seq += 1;
    let mut effects = Vec::new();
    if let Some(agent) = app.agents.get_mut(&agent_id) {
        // Replacing an API-key attempt must close its reverse-response state
        // and issue a sequence-scoped shell cancel before installing the new
        // attempt. The cancel may race the new login effect, but its old
        // requestSeq cannot cancel the successor.
        agent.cancel_provider_secret();
        if let Some((old_provider, old_request_seq)) = agent.request_provider_login_cancel() {
            effects.push(Effect::ProviderLoginCancel {
                agent_id,
                provider: old_provider,
                request_seq: old_request_seq,
            });
        }
        agent.pending_provider_login = Some(crate::app::provider_auth::PendingProviderLogin {
            provider: provider.clone(),
            display_name: display_name.clone(),
            method,
            request_seq,
            owns_input: method == ProviderAuthMethod::ApiKey,
            cancelled: false,
        });
        let message = match method {
            ProviderAuthMethod::OAuth => format!(
                "Signing in to {display_name} with OAuth… Run /logout {provider} to cancel."
            ),
            ProviderAuthMethod::ApiKey => {
                format!("Starting secure {display_name} API-key entry… Press Esc to cancel.")
            }
        };
        agent.scrollback.push_block(RenderBlock::system(message));
    }
    effects.push(Effect::ProviderLogin {
        agent_id,
        session_id,
        provider,
        display_name,
        method,
        request_seq,
    });
    effects
}

pub(super) fn dispatch_login_cancel(
    app: &mut AppView,
    provider: String,
    request_seq: u64,
) -> Vec<Effect> {
    let agent_id = match app.active_view {
        ActiveView::Agent(agent_id) => Some(agent_id),
        ActiveView::AgentDashboard => app
            .dashboard
            .as_ref()
            .and_then(|dashboard| dashboard.attached_agent),
        ActiveView::Welcome => None,
    };
    let Some(agent_id) = agent_id else {
        return vec![];
    };
    let Some(agent) = app.agents.get_mut(&agent_id) else {
        return vec![];
    };
    let matching_secret = agent.provider_secret.as_ref().is_some_and(|secret| {
        secret.request.provider == provider
            && secret
                .request
                .request_seq
                .is_none_or(|secret_seq| secret_seq == request_seq)
    });
    if matching_secret {
        if let Some(pending) = agent.pending_provider_login.as_mut()
            && pending.provider == provider
            && pending.request_seq == request_seq
        {
            pending.cancelled = true;
        }
        agent.cancel_provider_secret();
        agent
            .scrollback
            .push_block(RenderBlock::system("Provider login cancelled."));
        return vec![Effect::ProviderLoginCancel {
            agent_id,
            provider,
            request_seq,
        }];
    }
    let Some(pending) = agent.pending_provider_login.as_mut() else {
        return vec![];
    };
    if pending.provider != provider || pending.request_seq != request_seq {
        return vec![];
    }
    pending.cancelled = true;
    agent.scrollback.push_block(RenderBlock::system(format!(
        "Cancelling {} {}…",
        pending.display_name,
        pending.method.display_name()
    )));
    vec![Effect::ProviderLoginCancel {
        agent_id,
        provider,
        request_seq,
    }]
}

pub(super) fn dispatch_logout(app: &mut AppView, provider: String) -> Vec<Effect> {
    let ActiveView::Agent(agent_id) = app.active_view else {
        app.show_toast("Open a session before removing a model provider credential.");
        return vec![];
    };
    vec![Effect::ProviderLogout { agent_id, provider }]
}

pub(super) fn handle_login_complete(
    app: &mut AppView,
    agent_id: crate::app::agent::AgentId,
    provider: String,
    display_name: String,
    method: ProviderAuthMethod,
    request_seq: u64,
    result: Result<ProviderLoginSuccess, String>,
) -> Vec<Effect> {
    let Some(agent) = app.agents.get_mut(&agent_id) else {
        return vec![];
    };
    let Some(pending) = agent.pending_provider_login.as_ref() else {
        return vec![];
    };
    if pending.request_seq != request_seq
        || pending.provider != provider
        || pending.method != method
    {
        return vec![];
    }
    let cancelled = pending.cancelled;
    agent.pending_provider_login = None;
    agent.pending_provider_reauth = None;
    agent.cancel_provider_secret();
    if cancelled {
        agent.reauth_stashed_prompt = None;
        let action = match method {
            ProviderAuthMethod::OAuth => "OAuth sign-in",
            ProviderAuthMethod::ApiKey => "API-key entry",
        };
        agent.scrollback.push_block(RenderBlock::system(format!(
            "{action} for {display_name} was cancelled."
        )));
        return vec![];
    }
    match result {
        Ok(success) => {
            let message = crate::app::effects::sanitize_provider_auth_error(&success.message);
            agent.scrollback.push_block(RenderBlock::system(message));
            super::auth::strip_trailing_auth_error_blocks(agent);
            if let Some(prompt) = agent.reauth_stashed_prompt.take() {
                agent.session.enqueue_in_flight_prompt_front(prompt);
            } else {
                return vec![];
            }
        }
        Err(error) if provider_login_was_cancelled(&error) => {
            agent.reauth_stashed_prompt = None;
            let action = match method {
                ProviderAuthMethod::OAuth => "OAuth sign-in",
                ProviderAuthMethod::ApiKey => "API-key entry",
            };
            agent.scrollback.push_block(RenderBlock::system(format!(
                "{action} for {display_name} was cancelled."
            )));
            return vec![];
        }
        Err(error) => {
            agent.reauth_stashed_prompt = None;
            agent
                .scrollback
                .push_block(RenderBlock::system(provider_login_failure_message(
                    &display_name,
                    method,
                    &error,
                )));
            return vec![];
        }
    }
    super::queue::maybe_drain_queue_and_note_peek(app, agent_id)
}

fn provider_login_failure_message(
    display_name: &str,
    method: ProviderAuthMethod,
    error: &str,
) -> String {
    crate::app::effects::sanitize_provider_auth_error(&format!(
        "Could not configure {} {}: {error}",
        display_name,
        method.display_name()
    ))
}

fn provider_login_was_cancelled(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("cancelled") || lower.contains("canceled")
}

pub(super) fn handle_logout_complete(
    app: &mut AppView,
    agent_id: crate::app::agent::AgentId,
    provider: String,
    result: Result<bool, String>,
) -> Vec<Effect> {
    let Some(agent) = app.agents.get_mut(&agent_id) else {
        return vec![];
    };
    let message = match result {
        Ok(true) => format!("Removed the stored credential for {provider}."),
        Ok(false) => format!("No stored credential for {provider} was found."),
        Err(error) => crate::app::effects::sanitize_provider_auth_error(&format!(
            "Could not remove the stored credential for {provider}: {error}"
        )),
    };
    agent.scrollback.push_block(RenderBlock::system(message));
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_login_errors_redact_credential_shaped_material_before_display() {
        let raw = "backend echoed api_key=plain-secret-value and sk-provider-secret-sentinel";
        let rendered = provider_login_failure_message("Anthropic", ProviderAuthMethod::ApiKey, raw);
        assert!(!rendered.contains("plain-secret-value"));
        assert!(!rendered.contains("sk-provider-secret-sentinel"));
        assert!(rendered.contains("[REDACTED]"));
    }

    #[test]
    fn cancelled_provider_login_is_not_reported_as_failure() {
        assert!(provider_login_was_cancelled("login cancelled"));
        assert!(provider_login_was_cancelled(
            "Authentication CANCELED by user"
        ));
        assert!(!provider_login_was_cancelled("connection timed out"));
    }
}

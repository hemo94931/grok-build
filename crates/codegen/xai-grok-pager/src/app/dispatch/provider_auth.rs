use crate::app::actions::Effect;
use crate::app::app_view::{ActiveView, AppView};
use crate::scrollback::block::RenderBlock;

pub(super) fn dispatch_login(app: &mut AppView, provider: String) -> Vec<Effect> {
    let ActiveView::Agent(agent_id) = app.active_view else {
        app.show_toast("Open a session before signing in to a model provider.");
        return vec![];
    };
    let Some(session_id) = app
        .agents
        .get(&agent_id)
        .and_then(|agent| agent.session.session_id.clone())
    else {
        app.show_toast("The active session is still starting.");
        return vec![];
    };

    let request_seq = app.next_auth_request_seq;
    app.next_auth_request_seq += 1;
    if let Some(agent) = app.agents.get_mut(&agent_id) {
        agent.scrollback.push_block(RenderBlock::system(format!(
            "Signing in to {provider}… Run /logout {provider} to cancel."
        )));
    }
    vec![Effect::ProviderLogin {
        agent_id,
        session_id,
        provider,
        request_seq,
    }]
}

pub(super) fn dispatch_logout(app: &mut AppView, provider: String) -> Vec<Effect> {
    let ActiveView::Agent(agent_id) = app.active_view else {
        app.show_toast("Open a session before signing out of a model provider.");
        return vec![];
    };
    vec![Effect::ProviderLogout { agent_id, provider }]
}

pub(super) fn handle_login_complete(
    app: &mut AppView,
    agent_id: crate::app::agent::AgentId,
    provider: String,
    result: Result<(), String>,
) -> Vec<Effect> {
    let Some(agent) = app.agents.get_mut(&agent_id) else {
        return vec![];
    };
    match result {
        Ok(()) => {
            agent
                .scrollback
                .push_block(RenderBlock::system(format!("Signed in to {provider}.")));
            super::auth::strip_trailing_auth_error_blocks(agent);
            if let Some(prompt) = agent.reauth_stashed_prompt.take() {
                agent.session.enqueue_in_flight_prompt_front(prompt);
            } else {
                return vec![];
            }
        }
        Err(error) if provider_login_was_cancelled(&error) => {
            agent.reauth_stashed_prompt = None;
            agent.scrollback.push_block(RenderBlock::system(format!(
                "Sign-in to {provider} was cancelled."
            )));
            return vec![];
        }
        Err(error) => {
            agent.scrollback.push_block(RenderBlock::system(format!(
                "Could not sign in to {provider}: {error}"
            )));
            return vec![];
        }
    }
    super::queue::maybe_drain_queue_and_note_peek(app, agent_id)
}

fn provider_login_was_cancelled(error: &str) -> bool {
    error.to_ascii_lowercase().contains("cancel")
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
        Ok(true) => format!("Signed out of {provider}."),
        Ok(false) => format!("No cached {provider} session was found."),
        Err(error) => format!("Could not sign out of {provider}: {error}"),
    };
    agent.scrollback.push_block(RenderBlock::system(message));
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_provider_regression_cancelled_login_is_not_reported_as_failure() {
        assert!(provider_login_was_cancelled("login cancelled"));
        assert!(provider_login_was_cancelled(
            "Authentication CANCELED by user"
        ));
        assert!(!provider_login_was_cancelled("connection timed out"));
    }
}

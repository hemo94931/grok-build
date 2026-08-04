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
        agent
            .scrollback
            .push_block(RenderBlock::system(format!("Signing in to {provider}…")));
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
        Err(error) => {
            agent.scrollback.push_block(RenderBlock::system(format!(
                "Could not sign in to {provider}: {error}"
            )));
            return vec![];
        }
    }
    super::queue::maybe_drain_queue_and_note_peek(app, agent_id)
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

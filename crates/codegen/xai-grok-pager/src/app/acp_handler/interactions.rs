use super::*;

/// Handle the dedicated provider API-key reverse request.
///
/// The secret bytes live only inside `ProviderSecretState` and its ACP response
/// channel; the ordinary prompt/question/annotation path is never involved.
pub(crate) fn handle_prompt_secret(
    ext: xai_acp_lib::AcpArgs<acp::ExtRequest>,
    app: &mut AppView,
) -> bool {
    let mut request: xai_acp_lib::PromptSecretRequest =
        match serde_json::from_str(ext.request.params.get()) {
            Ok(request) => request,
            Err(error) => {
                tracing::error!(%error, "failed to parse provider promptSecret metadata");
                ext.response_tx
                    .send(Err(acp::Error::invalid_params()
                        .data("invalid secure provider API-key prompt metadata")))
                    .ok();
                return false;
            }
        };
    if request.provider.trim().is_empty()
        || request.provider_display_name.trim().is_empty()
        || request.session_id.trim().is_empty()
    {
        ext.response_tx
            .send(Err(acp::Error::invalid_params().data(
                "provider, providerDisplayName, and sessionId are required",
            )))
            .ok();
        return false;
    }

    let Some(id) = interaction_target_agent(app, &request.session_id) else {
        tracing::info!(
            session_id = %request.session_id,
            provider = %request.provider,
            "provider promptSecret has no local view; cancelling"
        );
        send_prompt_secret_cancelled(ext.response_tx);
        return false;
    };
    let is_active = is_matched_agent_active(app, id);
    if !is_active {
        tracing::info!(
            session_id = %request.session_id,
            provider = %request.provider,
            "provider promptSecret target is no longer active; cancelling"
        );
        send_prompt_secret_cancelled(ext.response_tx);
        return false;
    }
    let Some(agent) = app.agents.get_mut(&id) else {
        send_prompt_secret_cancelled(ext.response_tx);
        return false;
    };

    let Some(pending) = agent.pending_provider_login.as_ref() else {
        tracing::info!(
            provider = %request.provider,
            request_seq = request.request_seq,
            "provider promptSecret has no matching login request; cancelling"
        );
        send_prompt_secret_cancelled(ext.response_tx);
        return false;
    };
    let matches_pending = pending.provider == request.provider
        && pending.method == xai_acp_lib::ProviderAuthMethod::ApiKey
        && request
            .request_seq
            .is_none_or(|request_seq| request_seq == pending.request_seq);
    if !matches_pending {
        tracing::info!(
            provider = %request.provider,
            request_seq = request.request_seq,
            "stale provider promptSecret request; cancelling"
        );
        send_prompt_secret_cancelled(ext.response_tx);
        return false;
    }
    if pending.cancelled {
        agent.pending_provider_login = None;
        send_prompt_secret_cancelled(ext.response_tx);
        return true;
    }
    // Older shells may omit requestSeq. Once matched to the single active
    // attempt, normalize the local state to the exact sequence so submit/Esc
    // can release ownership and cancellation remains request-scoped.
    request.request_seq = Some(pending.request_seq);

    agent.cancel_provider_secret();
    // A stale picker/modal must not visually cover the secret state while the
    // secret state already owns all input.
    agent.active_modal = None;
    agent.provider_secret = Some(crate::app::provider_auth::ProviderSecretState::new(
        request,
        ext.response_tx,
    ));
    agent.last_active_at = Some(std::time::Instant::now());
    tracing::info!(
        provider = %agent
            .provider_secret
            .as_ref()
            .map(|state| state.request.provider.as_str())
            .unwrap_or("unknown"),
        target_active = is_active,
        "opened secure provider API-key input"
    );
    is_active
}

fn send_prompt_secret_cancelled(
    response_tx: tokio::sync::oneshot::Sender<xai_acp_lib::AcpResult<acp::ExtResponse>>,
) {
    let raw = serde_json::value::to_raw_value(&xai_acp_lib::PromptSecretResponse::Cancelled)
        .expect("serialize cancelled provider secret response");
    response_tx.send(Ok(acp::ExtResponse::new(raw.into()))).ok();
}

/// Handle `x.ai/ask_user_question` ext-method.
///
/// Parses the typed request, creates a `QuestionViewState` with the
/// `response_tx` stashed, and opens the question overlay. The pager does
/// NOT respond immediately — the response is sent later when the user
/// submits, cancels, or is replaced by another question.
///
/// If a question is already active, the old one is cancelled first
/// (`Cancelled` is sent on its stashed `response_tx`).
pub(crate) fn handle_ask_user_question(
    ext: xai_acp_lib::AcpArgs<acp::ExtRequest>,
    app: &mut AppView,
) -> bool {
    use crate::views::question_view::QuestionViewState;
    use xai_grok_tools::implementations::grok_build::ask_user_question::{
        AskUserQuestionExtRequest, AskUserQuestionExtResponse,
    };

    // Parse the typed request from the ext-method params.
    let ext_req: AskUserQuestionExtRequest = match serde_json::from_str(ext.request.params.get()) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "Failed to parse AskUserQuestionExtRequest");
            ext.response_tx
                .send(Err(acp::Error::new(-32602, format!("Invalid params: {e}"))))
                .ok();
            return false;
        }
    };

    // Route by the request's session id (like `session/update`), so a question
    // raised by a BACKGROUND session lands on its own view even when the user is
    // on the dashboard or another session — rather than failing because the
    // user hasn't entered the session yet.
    let Some(id) = interaction_target_agent(app, &ext_req.session_id) else {
        // No local view for this session. Do NOT send an error — that would FAIL
        // the tool (rendered red). Leave the reverse-request unanswered: the
        // agent keeps awaiting and the leader replays it when a client attaches
        // via `session/load`.
        tracing::info!(
            session_id = %ext_req.session_id,
            "ask_user_question for a session with no local view; parked for leader replay-on-attach"
        );
        drop(ext.response_tx);
        return false;
    };
    let is_active = is_matched_agent_active(app, id);
    let Some(agent) = app.agents.get_mut(&id) else {
        // `interaction_target_agent` only returns ids that exist; defensive.
        tracing::warn!("ask_user_question: agent {id:?} not found");
        drop(ext.response_tx);
        return false;
    };

    // If a question is already active, cancel it before replacing.
    if let Some(mut old_qv) = agent.question_view.take() {
        agent.record_question_pause(&old_qv);
        tracing::warn!(
            old_tool_call_id = %old_qv.tool_call_id,
            new_tool_call_id = %ext_req.tool_call_id,
            "Replacing active question - cancelling previous"
        );
        if let Some(old_tx) = old_qv.response_tx.take() {
            let cancelled = AskUserQuestionExtResponse::Cancelled;
            let raw = serde_json::value::to_raw_value(&cancelled)
                .expect("Cancelled serialization should not fail");
            old_tx.send(Ok(acp::ExtResponse::new(raw.into()))).ok();
        }
        agent.restore_card_prompt(old_qv.stashed_prompt);

        // Local question displaced by an ACP ask, so surface why it vanished.
        // Any directive it carried is dropped; the user re-issues the command after answering.
        if let Some(ref kind) = old_qv.local_kind {
            use crate::views::question_view::LocalQuestionKind;
            let cmd = match kind {
                LocalQuestionKind::Fork { .. } => "/fork",
                LocalQuestionKind::NewSession => "/new",
                LocalQuestionKind::CreditLimitUpsell { .. } => "credit-limit upsell",
                LocalQuestionKind::FreeUsageUpsell { .. } => "SuperGrok upsell",
                LocalQuestionKind::AgentTypeMismatch { .. } => "model switch",
                LocalQuestionKind::DoctorFix { .. } => "/doctor fix",
                LocalQuestionKind::DeleteCurrentSession => "/delete",
                LocalQuestionKind::Feedback => "/feedback",
            };
            let message = if matches!(kind, LocalQuestionKind::DoctorFix { .. }) {
                "/doctor fix was cancelled because another question opened.".to_owned()
            } else {
                format!("{cmd} cancelled because another question opened.")
            };
            agent.scrollback.push_block(RenderBlock::system(message));
        }
    }

    // Stash the composer so it comes back when this question closes.
    agent.question_view = Some(QuestionViewState::with_response_tx(
        ext_req.tool_call_id,
        ext_req.questions,
        agent.prompt.stash(),
        Some(ext.response_tx),
        ext_req.mode,
    ));

    // Clear prompt for question interaction.
    agent.prompt.set_text("");

    // Stamp the "last activity" anchor so the
    // dashboard's NeedsInput row reflects "time since this question
    // arrived" rather than the previous turn's end time.
    agent.last_active_at = Some(std::time::Instant::now());

    tracing::info!(
        mode = ?ext_req.mode,
        question_count = agent.question_view.as_ref().map(|q| q.questions.len()).unwrap_or(0),
        target_active = is_active,
        "Opened question view from ext_method"
    );

    // Only the currently-displayed view needs an immediate redraw; a question
    // parked on a background agent surfaces via the roster `NeedsInput` delta
    // and renders when the user switches to that session.
    is_active
}

/// Handle an `x.ai/exit_plan_mode` ext_method request.
///
/// Creates a `PlanApprovalViewState` overlay for interactive approval.
///
/// Follows the `handle_ask_user_question` pattern: parse → guard → cancel old
/// → stash prompt → create state → clear prompt → return true.
pub(super) fn handle_exit_plan_mode(
    ext: xai_acp_lib::AcpArgs<acp::ExtRequest>,
    app: &mut AppView,
) -> bool {
    use crate::views::plan_approval_view::{ExitPlanModeExtRequest, PlanApprovalViewState};

    // 1. Parse typed request from raw JSON params.
    let params: ExitPlanModeExtRequest = match serde_json::from_str(ext.request.params.get()) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to parse ExitPlanModeExtRequest: {e}");
            ext.response_tx
                .send(Err(acp::Error::new(
                    -32602,
                    format!("Invalid exit_plan_mode params: {e}"),
                )))
                .ok();
            return false;
        }
    };

    // 2. Route by the request's session id (like `session/update`), so a
    // plan-approval raised by a BACKGROUND session lands on its own view even
    // when the user isn't currently focused on it — rather than failing.
    let Some(id) = interaction_target_agent(app, &params.session_id) else {
        // No local view for this session. Do NOT error (that fails the tool):
        // leave the reverse-request unanswered and rely on the leader's
        // replay-on-attach.
        tracing::info!(
            session_id = %params.session_id,
            "exit_plan_mode for a session with no local view; parked for leader replay-on-attach"
        );
        drop(ext.response_tx);
        return false;
    };
    let is_active = is_matched_agent_active(app, id);
    let Some(agent) = app.agents.get_mut(&id) else {
        // `interaction_target_agent` only returns ids that exist; defensive.
        tracing::warn!("exit_plan_mode: agent {id:?} not found");
        drop(ext.response_tx);
        return false;
    };

    if let Some(mut old) = agent.plan_approval_view.take() {
        tracing::warn!(
            old_tool_call_id = %old.tool_call_id,
            new_tool_call_id = %params.tool_call_id,
            "Replacing active plan approval — dismissing previous"
        );
        old.send_stale_cancel();
        agent.plan_next_comment_id = old.next_comment_id;
        agent.prompt.restore(old.stashed_prompt);
        agent.line_viewer = None;
    }

    // Dismiss competing overlays so plan approval owns the screen.
    // - active_modal: draw returns before line_viewer (plan never paints);
    //   keys still route to the invisible plan viewer.
    // - block_viewer: draw returns on line_viewer (plan visible) but
    //   handle_scroll prefers block_viewer, so wheel hits the hidden Edit pane.
    agent.active_modal = None;
    agent.block_viewer = None;

    let source = plan_review_source_for_tool(&params.tool_call_id, agent);

    // If the user was mid-casual-comment when this new plan-approval
    // request arrived, restore the pre-comment prompt first so the
    // upcoming `stash()` captures the user's original text rather
    // than the in-progress comment draft. Also clears the now-stale
    // `casual_stashed_prompt` so it doesn't dangle into the next
    // casual entry.
    if let Some(stashed) = agent.casual_stashed_prompt.take() {
        agent.prompt.restore(stashed);
    }

    let stashed = agent.prompt.stash();
    let state = PlanApprovalViewState::with_source(params, source, stashed, ext.response_tx);

    agent.plan_comments.clear();
    agent.plan_next_comment_id = 0;

    if state.source == PlanReviewSource::Inline {
        agent.latest_inline_plan_content = state.plan_content.clone();
    } else {
        agent.latest_inline_plan_content = None;
    }
    agent.plan_approval_view = Some(state);
    agent.prompt.set_text("");

    agent.casual_commenting_range = None;
    agent.casual_editing_comment_id = None;

    agent.show_plan_preview_if_available();

    if agent.line_viewer.is_some() {
        if let Some(ref mut viewer) = agent.line_viewer {
            viewer.plan_mut().feedback_active = true;
        }
    } else if let Some(ref mut pav) = agent.plan_approval_view {
        pav.focus = crate::views::plan_approval_view::PlanApprovalFocus::Prompt;
    }

    tracing::info!(
        target_active = is_active,
        "Opened plan approval view from ext_method"
    );

    // Background-parked approval renders when the user switches to the session;
    // only the active view needs an immediate redraw.
    is_active
}

pub(super) fn plan_review_source_for_tool(
    tool_call_id: &str,
    agent: &AgentView,
) -> PlanReviewSource {
    agent
        .session
        .tracker
        .tool_title(tool_call_id)
        .filter(|title| *title == "CreatePlan" || *title == "Plan: Submit for approval")
        .map_or(PlanReviewSource::FileBacked, |_| PlanReviewSource::Inline)
}

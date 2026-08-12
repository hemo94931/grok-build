//! Tests for login, logout, account switching, and auth-code dispatchers.

use super::*;

#[test]
fn cta_mcps_loaded_needs_auth_opens_modal_and_seeds() {
    use crate::app::agent_view::CtaPhase;
    use crate::views::extensions_modal::{ExtensionsTab, TabDataState};
    use crate::views::mcps_modal::{McpSectionId, McpServerDisplayStatus, section_key};
    let mut app = test_app_with_agent();
    app.team_id = Some("team-uuid".into());
    let id = AgentId(0);
    app.agents.get_mut(&id).unwrap().plugin_cta.phase = CtaPhase::AwaitingMcps {
        name: "figma".into(),
    };
    let servers = vec![
        cta_mcp_server("grok_com_managed", None, McpServerDisplayStatus::Ready),
        cta_mcp_server("local-srv", None, McpServerDisplayStatus::Ready),
        cta_mcp_server("other-srv", Some("slack"), McpServerDisplayStatus::Ready),
        cta_mcp_server(
            "figma-srv",
            Some("figma"),
            McpServerDisplayStatus::NeedsAuth,
        ),
    ];
    let effects = dispatch(
        Action::TaskComplete(TaskResult::PluginCtaMcpsLoaded {
            agent_id: id,
            plugin_name: "figma".into(),
            result: Ok(servers),
        }),
        &mut app,
    );
    // Handoff complete: CTA settles to Hidden.
    assert_eq!(app.agents[&id].plugin_cta.phase, CtaPhase::Hidden);
    // Modal opened to the MCP Servers tab.
    let modal = app.agents[&id]
        .extensions_modal
        .as_ref()
        .expect("extensions modal should be open");
    assert_eq!(modal.active_tab, ExtensionsTab::McpServers);
    // Session team id seeded so the Managed subtitle deep link matches Ctrl+O.
    assert_eq!(modal.session_team_id.as_deref(), Some("team-uuid"));
    // MCP tab seeded directly from the read we already have (no flash).
    match &modal.mcps_data {
        TabDataState::Loaded(servers) => assert_eq!(servers.len(), 4),
        other => panic!("expected mcps_data Loaded, got {other:?}"),
    }
    // Managed + Local + other plugins collapsed; only target expanded.
    let collapsed = &modal.mcps_collapsed_sections;
    assert!(collapsed.contains(&section_key(&McpSectionId::Managed)));
    assert!(collapsed.contains(&section_key(&McpSectionId::Local)));
    assert!(collapsed.contains(&section_key(&McpSectionId::Plugin("slack".into()))));
    assert!(!collapsed.contains(&section_key(&McpSectionId::Plugin("figma".into()))));
    assert!(modal.mcps_section_collapse_initialized);
    // Emits the SAME full tab fetch-set as a manual open so no tab is stuck
    // Loading, plus the candidate refresh.
    assert_eq!(
        effects
            .iter()
            .filter(|e| matches!(e, Effect::FetchHooksList { .. }))
            .count(),
        1
    );
    assert_eq!(
        effects
            .iter()
            .filter(|e| matches!(e, Effect::FetchPluginsList { .. }))
            .count(),
        1
    );
    assert_eq!(
        effects
            .iter()
            .filter(|e| matches!(e, Effect::FetchMarketplaceList { .. }))
            .count(),
        1
    );
    assert_eq!(
        effects
            .iter()
            .filter(|e| matches!(e, Effect::FetchMcpsList { .. }))
            .count(),
        1
    );
    assert_eq!(
        effects
            .iter()
            .filter(|e| matches!(e, Effect::FetchSkillsList { .. }))
            .count(),
        1
    );
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::FetchPluginCtaCatalog { .. }))
    );
}

#[test]
fn cta_mcps_loaded_no_needs_auth_terminal_sets_installed() {
    use crate::app::agent_view::CtaPhase;
    use crate::views::mcps_modal::McpServerDisplayStatus;
    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let cta = &mut app.agents.get_mut(&id).unwrap().plugin_cta;
        cta.phase = CtaPhase::AwaitingMcps {
            name: "figma".into(),
        };
        cta.expects_mcp = true;
    }
    // Plugin server present and Ready (terminal, no auth) -> settle now.
    let servers = vec![cta_mcp_server(
        "figma-srv",
        Some("figma"),
        McpServerDisplayStatus::Ready,
    )];
    let effects = dispatch(
        Action::TaskComplete(TaskResult::PluginCtaMcpsLoaded {
            agent_id: id,
            plugin_name: "figma".into(),
            result: Ok(servers),
        }),
        &mut app,
    );
    assert_eq!(
        app.agents[&id].plugin_cta.phase,
        CtaPhase::Installed {
            name: "figma".into()
        }
    );
    assert!(app.agents[&id].extensions_modal.is_none());
    // No modal repopulation; settle emits the auto-dismiss timer + candidate
    // refresh, and never re-probes.
    assert!(
        !effects
            .iter()
            .any(|e| matches!(e, Effect::FetchMcpsList { .. }))
    );
    assert!(
        !effects
            .iter()
            .any(|e| matches!(e, Effect::RetryPluginCtaMcps { .. }))
    );
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::DismissCtaInstalled { .. }))
    );
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::FetchPluginCtaCatalog { .. }))
    );
}

#[test]
fn cta_mcps_loaded_later_needs_auth_opens_handoff() {
    use crate::app::agent_view::CtaPhase;
    use crate::views::mcps_modal::McpServerDisplayStatus;
    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let cta = &mut app.agents.get_mut(&id).unwrap().plugin_cta;
        cta.phase = CtaPhase::AwaitingMcps {
            name: "figma".into(),
        };
        cta.expects_mcp = true;
        // Several polls already elapsed before the server reached NeedsAuth.
        cta.mcp_attempt = 5;
    }
    let effects = dispatch(
        Action::TaskComplete(TaskResult::PluginCtaMcpsLoaded {
            agent_id: id,
            plugin_name: "figma".into(),
            result: Ok(vec![cta_mcp_server(
                "figma-srv",
                Some("figma"),
                McpServerDisplayStatus::NeedsAuth,
            )]),
        }),
        &mut app,
    );
    // NeedsAuth is terminal: hand off immediately even mid-poll.
    assert_eq!(app.agents[&id].plugin_cta.phase, CtaPhase::Hidden);
    assert!(app.agents[&id].extensions_modal.is_some());
    assert!(
        !effects
            .iter()
            .any(|e| matches!(e, Effect::RetryPluginCtaMcps { .. }))
    );
}

// ── agent-bound kinds (bash) ─────────

/// A bash command typed while a turn is RUNNING takes the
/// server-authoritative immediate path (Effect + optimistic echo, no local
/// queue entry).
#[test]
fn bash_while_running_is_server_authoritative() {
    let mut app = test_app_with_agent();
    let id = AgentId(0);
    app.agents.get_mut(&id).unwrap().session.state = AgentState::TurnRunning;

    let effects = dispatch(Action::SendBashCommand("ls -la".into()), &mut app);
    let pid = match &effects[0] {
        Effect::SendBashCommand {
            command, prompt_id, ..
        } => {
            assert_eq!(command, "ls -la");
            prompt_id.clone()
        }
        other => panic!("expected immediate SendBashCommand, got {other:?}"),
    };
    // Not in the local queue.
    assert_eq!(app.agents[&id].session.queue_len(), 0);
    // Optimistic echo present with kind="bash".
    let q = app
        .shared_prompt_queue("test-session")
        .expect("echo present");
    assert_eq!(q.len(), 1);
    assert_eq!(q[0].id, pid);
    assert_eq!(q[0].kind, "bash");
    assert_eq!(q[0].text, "ls -la");
}

#[test]
fn auth_complete_triggers_bundle_status_fetch() {
    let mut app = test_app();
    app.auth_state = AuthState::Authenticating {
        request_seq: 1,
        handle: None,
        auth_url: None,
        mode: AuthMode::Pending,
    };

    let effects = dispatch(
        Action::TaskComplete(TaskResult::AuthComplete {
            request_seq: 1,
            meta: None,
        }),
        &mut app,
    );

    assert!(matches!(app.auth_state, AuthState::Done));
    // Pager only refreshes the on-disk catalog snapshot; the actual
    // bundle download now runs inside the shell post-auth.
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::FetchBundleStatus))
    );
}

#[test]
fn auth_complete_with_deferred_load_also_fetches_status() {
    let mut app = test_app();
    app.auth_state = AuthState::Authenticating {
        request_seq: 1,
        handle: None,
        auth_url: None,
        mode: AuthMode::Pending,
    };
    app.deferred_startup.session =
        Some(crate::app::session_startup::DeferredSessionStartup::Load {
            session_id: "test-session".into(),
            session_cwd: None,
            chat_kind: false,
        });

    let effects = dispatch(
        Action::TaskComplete(TaskResult::AuthComplete {
            request_seq: 1,
            meta: None,
        }),
        &mut app,
    );

    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::FetchBundleStatus))
    );
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::LoadSession { .. }))
    );
    assert!(app.deferred_startup.session.is_none());
}

/// `/login` from the welcome screen (startup / logged-out) must NOT
/// stash a return view — the normal login-then-load flow is preserved.
#[test]
fn login_from_welcome_does_not_stash_return_view() {
    let mut app = test_app();
    assert_eq!(app.active_view, ActiveView::Welcome);

    dispatch(Action::Login, &mut app);

    assert_eq!(app.active_view, ActiveView::Welcome);
    assert_eq!(app.auth_return_view, None);
}

/// Compact-auth recovery: hold prompt across auto-compact 401, stash on
/// PromptResponse, resubmit on mid-session AuthComplete.
#[test]
fn e2e_compact_auth_failure_holds_prompt_and_resubmits_after_login() {
    use crate::app::acp_handler::apply_session_event_for_test;
    use crate::app::agent::{AgentState, InFlightPrompt};
    use crate::scrollback::EntryId;
    use crate::scrollback::block::RenderBlock;
    use xai_grok_shell::extensions::notification::{RetryState, SessionUpdate as XaiSessionUpdate};

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.session.state = AgentState::TurnRunning;
        agent.turn_started_at = Some(std::time::Instant::now());
        agent.session.session_id = Some(acp::SessionId::new("sess-compact-auth-e2e"));
        agent.session.current_prompt_id = Some("prompt-1".into());
        agent.session.in_flight_prompt = Some(InFlightPrompt {
            text: "please continue after login".into(),
            images: Vec::new(),
            scrollback_entry: EntryId::new(1),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });

        apply_session_event_for_test(
            &XaiSessionUpdate::AutoCompactStarted {
                tokens_used: 180_000,
                context_window: 200_000,
                percentage: 90,
                reason: "threshold".into(),
            },
            &mut agent.session,
            &mut agent.scrollback,
        );
        assert!(
            agent.session.in_flight_prompt.is_none(),
            "cancel rewind must still be blocked mid-compact"
        );
        assert_eq!(
            agent
                .session
                .compact_held_prompt
                .as_ref()
                .map(|p| p.text.as_str()),
            Some("please continue after login"),
            "must hold the prompt text for reauth auto-resubmit"
        );

        apply_session_event_for_test(
            &XaiSessionUpdate::AutoCompactFailed {
                error: "authentication problem — re-authenticate using /login and retry.".into(),
            },
            &mut agent.session,
            &mut agent.scrollback,
        );
        assert!(agent.session.compact_held_prompt.is_some());

        apply_session_event_for_test(
            &XaiSessionUpdate::RetryState(RetryState::Failed {
                error_type: "auth".into(),
                message: "Unauthorized (401): compaction failed".into(),
                provider_auth: None,
            }),
            &mut agent.session,
            &mut agent.scrollback,
        );
        let has_reauth = (0..agent.scrollback.len()).any(|i| {
            matches!(
                agent.scrollback.entry(i).map(|e| &e.block),
                Some(RenderBlock::SessionEvent(ev))
                    if matches!(ev.event, SessionEvent::ReAuthRequired)
            )
        });
        assert!(has_reauth, "RetryState auth must show ReAuthRequired");
    }

    dispatch(
        Action::TaskComplete(TaskResult::PromptResponse {
            agent_id: id,
            result: Err("Unauthorized (401)".to_string()),
            http_status: Some(401),
            prompt_id: Some("prompt-1".into()),
        }),
        &mut app,
    );
    assert_eq!(
        app.agents[&id]
            .reauth_stashed_prompt
            .as_ref()
            .map(|p| p.text.as_str()),
        Some("please continue after login"),
        "PromptResponse must stash the compact-held prompt for AuthComplete"
    );

    dispatch(Action::Login, &mut app);
    let seq = authenticating_seq(&app);
    let effects = dispatch(
        Action::TaskComplete(TaskResult::AuthComplete {
            request_seq: seq,
            meta: None,
        }),
        &mut app,
    );
    assert!(
        app.agents[&id].reauth_stashed_prompt.is_none(),
        "stash consumed on AuthComplete"
    );
    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::SendPrompt { .. } | Effect::SendPromptBlocks { .. }
        )),
        "AuthComplete must resubmit the prompt so compact runs again with valid auth, got: {effects:?}"
    );
}

#[test]
fn bare_provider_login_opens_metadata_driven_method_picker() {
    use crate::app::provider_auth::{ProviderAuthInfo, ProviderLoginIntent};
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    app.agents.get_mut(&AgentId(0)).unwrap().session.session_id =
        Some(acp::SessionId::new("provider-picker"));
    let effects = dispatch(Action::ProviderLogin(ProviderLoginIntent::Menu), &mut app);
    assert!(matches!(
        effects.as_slice(),
        [Effect::ProviderAuthInfo {
            agent_id: AgentId(0),
            intent: ProviderLoginIntent::Menu,
        }]
    ));

    let _ = dispatch(
        Action::TaskComplete(TaskResult::ProviderAuthInfoComplete {
            agent_id: AgentId(0),
            intent: ProviderLoginIntent::Menu,
            result: Ok(vec![
                ProviderAuthInfo {
                    id: "anthropic".into(),
                    display_name: "Anthropic".into(),
                    supported_methods: vec![ProviderAuthMethod::OAuth, ProviderAuthMethod::ApiKey],
                    authenticated: false,
                    credential_type: None,
                    model_count: 2,
                },
                ProviderAuthInfo {
                    id: "openai-codex".into(),
                    display_name: "OpenAI Codex".into(),
                    supported_methods: vec![ProviderAuthMethod::OAuth],
                    authenticated: false,
                    credential_type: None,
                    model_count: 1,
                },
            ]),
        }),
        &mut app,
    );
    let Some(crate::views::modal::ActiveModal::ArgPicker {
        command,
        original_items,
        ..
    }) = app.agents[&AgentId(0)].active_modal.as_ref()
    else {
        panic!("bare /login must open the method picker");
    };
    assert_eq!(command, "login");
    assert_eq!(original_items.len(), 4);
    assert_eq!(original_items[0].insert_text, "xai");
    assert!(
        original_items
            .iter()
            .any(|item| item.insert_text == "anthropic --api-key")
    );
    assert!(
        original_items
            .iter()
            .any(|item| item.insert_text == "openai-codex --oauth")
    );
}

#[test]
fn explicit_provider_method_is_validated_from_info_before_login_effect() {
    use crate::app::provider_auth::{ProviderAuthInfo, ProviderLoginIntent};
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    app.agents.get_mut(&AgentId(0)).unwrap().session.session_id =
        Some(acp::SessionId::new("provider-method"));
    let providers = vec![ProviderAuthInfo {
        id: "openai-codex".into(),
        display_name: "OpenAI Codex".into(),
        supported_methods: vec![ProviderAuthMethod::OAuth],
        authenticated: false,
        credential_type: None,
        model_count: 1,
    }];
    let effects = dispatch(
        Action::TaskComplete(TaskResult::ProviderAuthInfoComplete {
            agent_id: AgentId(0),
            intent: ProviderLoginIntent::Provider {
                provider: "openai-codex".into(),
                method: Some(ProviderAuthMethod::ApiKey),
            },
            result: Ok(providers.clone()),
        }),
        &mut app,
    );
    assert!(effects.is_empty());

    let effects = dispatch(
        Action::TaskComplete(TaskResult::ProviderAuthInfoComplete {
            agent_id: AgentId(0),
            intent: ProviderLoginIntent::Provider {
                provider: "openai-codex".into(),
                method: Some(ProviderAuthMethod::OAuth),
            },
            result: Ok(providers),
        }),
        &mut app,
    );
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::ProviderLogin {
            provider,
            method: ProviderAuthMethod::OAuth,
            ..
        } if provider == "openai-codex"
    )));
}

#[test]
fn stored_provider_failure_waits_for_prompt_stash_then_reopens_same_method() {
    use crate::app::acp_handler::apply_session_event_for_test;
    use crate::app::agent::{AgentState, InFlightPrompt};
    use crate::app::provider_auth::{PendingProviderReauth, ProviderLoginSuccess};
    use crate::scrollback::EntryId;
    use xai_acp_lib::{ProviderAuthMethod, ProviderAuthRemedy, ProviderAuthSource};
    use xai_grok_shell::extensions::notification::{RetryState, SessionUpdate as XaiSessionUpdate};

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let remedy = ProviderAuthRemedy {
        provider: "anthropic".into(),
        provider_display_name: "Anthropic".into(),
        method: ProviderAuthMethod::ApiKey,
        source: ProviderAuthSource::StoredApiKey,
    };
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.session.state = AgentState::TurnRunning;
        agent.turn_started_at = Some(std::time::Instant::now());
        agent.session.session_id = Some(acp::SessionId::new("sess-provider-key"));
        agent.session.current_prompt_id = Some("prompt-key".into());
        agent.session.in_flight_prompt = Some(InFlightPrompt {
            text: "retry with replacement key".into(),
            images: Vec::new(),
            scrollback_entry: EntryId::new(101),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });
        apply_session_event_for_test(
            &XaiSessionUpdate::RetryState(RetryState::Failed {
                error_type: "provider_auth:anthropic".into(),
                message: "Unauthorized (401)".into(),
                provider_auth: Some(remedy.clone()),
            }),
            &mut agent.session,
            &mut agent.scrollback,
        );
        agent.pending_provider_reauth = Some(PendingProviderReauth {
            prompt_id: Some("prompt-key".into()),
            remedy: remedy.clone(),
        });
        assert!(agent.reauth_stashed_prompt.is_none());
    }

    let effects = dispatch(
        Action::TaskComplete(TaskResult::PromptResponse {
            agent_id: id,
            result: Err("Unauthorized (401)".into()),
            http_status: Some(401),
            prompt_id: Some("prompt-key".into()),
        }),
        &mut app,
    );
    assert_eq!(
        app.agents[&id]
            .reauth_stashed_prompt
            .as_ref()
            .map(|prompt| prompt.text.as_str()),
        Some("retry with replacement key")
    );
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::ProviderLogin {
            provider,
            method: ProviderAuthMethod::ApiKey,
            ..
        } if provider == "anthropic"
    )));

    let auth_state_before = std::mem::discriminant(&app.auth_state);
    let login_method_before = app.login_method_id.clone();
    let effects = dispatch(
        Action::TaskComplete(TaskResult::ProviderLoginComplete {
            agent_id: id,
            provider: "anthropic".into(),
            display_name: "Anthropic".into(),
            method: ProviderAuthMethod::ApiKey,
            request_seq: 1,
            result: Ok(ProviderLoginSuccess {
                provider: "anthropic".into(),
                display_name: "Anthropic".into(),
                method: ProviderAuthMethod::ApiKey,
                message: "API key saved for Anthropic.".into(),
                catalog_refreshed: None,
            }),
        }),
        &mut app,
    );
    assert!(app.agents[&id].reauth_stashed_prompt.is_none());
    assert_eq!(std::mem::discriminant(&app.auth_state), auth_state_before);
    assert_eq!(app.login_method_id, login_method_before);
    assert!(!effects.iter().any(|effect| matches!(
        effect,
        Effect::FetchBilling { .. } | Effect::CheckSubscription { .. }
    )));
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::SendPrompt { .. } | Effect::SendPromptBlocks { .. }
    )));
}

#[test]
fn provider_reauth_cancel_or_failure_never_resends_stashed_prompt() {
    use crate::app::agent::InFlightPrompt;
    use crate::scrollback::EntryId;
    use xai_acp_lib::ProviderAuthMethod;

    for error in ["login cancelled", "provider login failed"] {
        let mut app = test_app_with_agent();
        let id = AgentId(0);
        let agent = app.agents.get_mut(&id).unwrap();
        agent.reauth_stashed_prompt = Some(InFlightPrompt {
            text: "must not resend".into(),
            images: Vec::new(),
            scrollback_entry: EntryId::new(105),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });
        agent.pending_provider_login = Some(crate::app::provider_auth::PendingProviderLogin {
            provider: "anthropic".into(),
            display_name: "Anthropic".into(),
            method: ProviderAuthMethod::ApiKey,
            request_seq: 1,
            owns_input: false,
            cancelled: false,
        });
        let auth_state_before = std::mem::discriminant(&app.auth_state);
        let effects = dispatch(
            Action::TaskComplete(TaskResult::ProviderLoginComplete {
                agent_id: id,
                provider: "anthropic".into(),
                display_name: "Anthropic".into(),
                method: ProviderAuthMethod::ApiKey,
                request_seq: 1,
                result: Err(error.into()),
            }),
            &mut app,
        );
        assert!(effects.is_empty());
        assert!(app.agents[&id].reauth_stashed_prompt.is_none());
        assert_eq!(std::mem::discriminant(&app.auth_state), auth_state_before);
    }
}

#[test]
fn cancelled_provider_login_ignores_a_racing_success() {
    use crate::app::agent::InFlightPrompt;
    use crate::app::provider_auth::PendingProviderLogin;
    use crate::scrollback::EntryId;
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let agent = app.agents.get_mut(&id).unwrap();
    agent.reauth_stashed_prompt = Some(InFlightPrompt {
        text: "must not resend after cancel".into(),
        images: Vec::new(),
        scrollback_entry: EntryId::new(106),
        combined_scrollback_entries: Vec::new(),
        chip_elements: Vec::new(),
    });
    agent.pending_provider_login = Some(PendingProviderLogin {
        provider: "anthropic".into(),
        display_name: "Anthropic".into(),
        method: ProviderAuthMethod::ApiKey,
        request_seq: 7,
        owns_input: false,
        cancelled: true,
    });

    let effects = dispatch(
        Action::TaskComplete(TaskResult::ProviderLoginComplete {
            agent_id: id,
            provider: "anthropic".into(),
            display_name: "Anthropic".into(),
            method: ProviderAuthMethod::ApiKey,
            request_seq: 7,
            result: Ok(crate::app::provider_auth::ProviderLoginSuccess {
                provider: "anthropic".into(),
                display_name: "Anthropic".into(),
                method: ProviderAuthMethod::ApiKey,
                message: "API key saved for Anthropic.".into(),
                catalog_refreshed: None,
            }),
        }),
        &mut app,
    );

    assert!(effects.is_empty());
    assert!(app.agents[&id].reauth_stashed_prompt.is_none());
    assert!(app.agents[&id].pending_provider_login.is_none());
}

#[test]
fn stored_oauth_failure_reopens_oauth_after_prompt_is_stashed() {
    use crate::app::agent::{AgentState, InFlightPrompt};
    use crate::app::provider_auth::PendingProviderReauth;
    use crate::scrollback::EntryId;
    use crate::scrollback::block::RenderBlock;
    use xai_acp_lib::{ProviderAuthMethod, ProviderAuthRemedy, ProviderAuthSource};

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let remedy = ProviderAuthRemedy {
        provider: "anthropic".into(),
        provider_display_name: "Anthropic".into(),
        method: ProviderAuthMethod::OAuth,
        source: ProviderAuthSource::StoredOAuth,
    };
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.session.state = AgentState::TurnRunning;
        agent.turn_started_at = Some(std::time::Instant::now());
        agent.session.session_id = Some(acp::SessionId::new("sess-provider-oauth"));
        agent.session.current_prompt_id = Some("prompt-oauth".into());
        agent.session.in_flight_prompt = Some(InFlightPrompt {
            text: "retry after OAuth".into(),
            images: Vec::new(),
            scrollback_entry: EntryId::new(104),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });
        agent.scrollback.push_block(RenderBlock::session_event(
            SessionEvent::ProviderReAuthRequired {
                provider: "anthropic".into(),
                remedy: Some(remedy.clone()),
            },
        ));
        agent.pending_provider_reauth = Some(PendingProviderReauth {
            prompt_id: Some("prompt-oauth".into()),
            remedy,
        });
    }

    let effects = dispatch(
        Action::TaskComplete(TaskResult::PromptResponse {
            agent_id: id,
            result: Err("Unauthorized (401)".into()),
            http_status: Some(401),
            prompt_id: Some("prompt-oauth".into()),
        }),
        &mut app,
    );
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::ProviderLogin {
            method: ProviderAuthMethod::OAuth,
            ..
        }
    )));
    assert_eq!(
        app.agents[&id]
            .reauth_stashed_prompt
            .as_ref()
            .map(|prompt| prompt.text.as_str()),
        Some("retry after OAuth")
    );
}

#[test]
fn provider_retry_notification_after_prompt_response_still_opens_same_method() {
    use crate::app::agent::{AgentState, InFlightPrompt};
    use crate::scrollback::EntryId;
    use xai_acp_lib::{ProviderAuthMethod, ProviderAuthRemedy, ProviderAuthSource};
    use xai_grok_shell::extensions::notification::{
        RetryState, SessionNotification, SessionUpdate as XaiSessionUpdate,
    };

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.session.state = AgentState::TurnRunning;
        agent.turn_started_at = Some(std::time::Instant::now());
        agent.session.session_id = Some(acp::SessionId::new("sess-provider-late"));
        agent.session.current_prompt_id = Some("prompt-late".into());
        agent.session.in_flight_prompt = Some(InFlightPrompt {
            text: "stash before remedy arrives".into(),
            images: Vec::new(),
            scrollback_entry: EntryId::new(103),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });
    }

    let effects = dispatch(
        Action::TaskComplete(TaskResult::PromptResponse {
            agent_id: id,
            result: Err("Unauthorized (401)".into()),
            http_status: Some(401),
            prompt_id: Some("prompt-late".into()),
        }),
        &mut app,
    );
    assert!(effects.is_empty());
    assert_eq!(
        app.agents[&id]
            .reauth_stashed_prompt
            .as_ref()
            .map(|prompt| prompt.text.as_str()),
        Some("stash before remedy arrives")
    );

    let payload = SessionNotification {
        session_id: acp::SessionId::new("sess-provider-late"),
        update: XaiSessionUpdate::RetryState(RetryState::Failed {
            error_type: "provider_auth:anthropic".into(),
            message: "Unauthorized (401)".into(),
            provider_auth: Some(ProviderAuthRemedy {
                provider: "anthropic".into(),
                provider_display_name: "Anthropic".into(),
                method: ProviderAuthMethod::ApiKey,
                source: ProviderAuthSource::StoredApiKey,
            }),
        }),
        meta: None,
    };
    let raw = serde_json::value::to_raw_value(&payload).unwrap();
    let notification =
        acp::ExtNotification::new("x.ai/session_notification", std::sync::Arc::from(raw));
    assert!(
        crate::app::acp_handler::handle_session_notification_for_test(&notification, &mut app,)
    );
    assert!(app.pending_effects.iter().any(|effect| matches!(
        effect,
        Effect::ProviderLogin {
            provider,
            method: ProviderAuthMethod::ApiKey,
            ..
        } if provider == "anthropic"
    )));
    assert!(app.agents[&id].pending_provider_reauth.is_none());
}

#[test]
fn replacing_api_key_login_cancels_old_reverse_request_and_sequence() {
    use crate::app::provider_auth::{PendingProviderLogin, ProviderSecretState};
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    let agent = app.agents.get_mut(&id).unwrap();
    agent.pending_provider_login = Some(PendingProviderLogin {
        provider: "anthropic".into(),
        display_name: "Anthropic".into(),
        method: ProviderAuthMethod::ApiKey,
        request_seq: 77,
        owns_input: true,
        cancelled: false,
    });
    agent.provider_secret = Some(ProviderSecretState::new(
        xai_acp_lib::PromptSecretRequest {
            provider: "anthropic".into(),
            provider_display_name: "Anthropic".into(),
            session_id: "test-session".into(),
            prompt: "Enter API key".into(),
            request_seq: Some(77),
        },
        response_tx,
    ));

    let effects = crate::app::dispatch::provider_auth::start_provider_login(
        &mut app,
        id,
        "openrouter".into(),
        "OpenRouter".into(),
        ProviderAuthMethod::ApiKey,
    );

    assert!(matches!(
        effects.as_slice(),
        [
            Effect::ProviderLoginCancel {
                provider: old_provider,
                request_seq: 77,
                ..
            },
            Effect::ProviderLogin {
                provider: new_provider,
                request_seq: 1,
                ..
            }
        ] if old_provider == "anthropic" && new_provider == "openrouter"
    ));
    let response = response_rx.blocking_recv().unwrap().unwrap();
    let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
    assert_eq!(json["outcome"], "cancelled");
    let pending = app.agents[&id].pending_provider_login.as_ref().unwrap();
    assert_eq!(pending.provider, "openrouter");
    assert_eq!(pending.request_seq, 1);
    assert!(pending.owns_input);
}

#[test]
fn api_key_login_can_be_cancelled_before_reverse_secret_request_arrives() {
    use crate::app::provider_auth::PendingProviderLogin;
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let agent = app.agents.get_mut(&id).unwrap();
    agent.prompt.set_text("keep this draft");
    agent.pending_provider_login = Some(PendingProviderLogin {
        provider: "anthropic".into(),
        display_name: "Anthropic".into(),
        method: ProviderAuthMethod::ApiKey,
        request_seq: 77,
        owns_input: true,
        cancelled: false,
    });
    let _ = app.handle_input(&Event::Paste("sk-typeahead-must-not-leak".into()));
    let _ = app.handle_input(&Event::Key(KeyEvent::new(
        KeyCode::Char('x'),
        KeyModifiers::NONE,
    )));
    assert_eq!(app.agents[&id].prompt.text(), "keep this draft");
    assert!(app.agents[&id].input_log.snapshot_entries().is_empty());

    let outcome = app.handle_input(&Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
    let crate::app::app_view::InputOutcome::Action(action @ Action::ProviderLoginCancel { .. }) =
        outcome
    else {
        panic!("Esc must produce a provider-scoped cancel action");
    };
    let effects = dispatch(action, &mut app);
    assert!(
        app.agents[&id]
            .pending_provider_login
            .as_ref()
            .is_some_and(|pending| pending.cancelled)
    );
    assert!(matches!(
        effects.as_slice(),
        [Effect::ProviderLoginCancel {
            agent_id: AgentId(0),
            provider,
            request_seq: 77,
        }] if provider == "anthropic"
    ));
}

#[test]
fn secret_escape_cancels_response_and_request_sequence() {
    use crate::app::provider_auth::{PendingProviderLogin, ProviderSecretState};
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let (tx, rx) = tokio::sync::oneshot::channel();
    let agent = app.agents.get_mut(&id).unwrap();
    agent.pending_provider_login = Some(PendingProviderLogin {
        provider: "anthropic".into(),
        display_name: "Anthropic".into(),
        method: ProviderAuthMethod::ApiKey,
        request_seq: 77,
        owns_input: true,
        cancelled: false,
    });
    agent.provider_secret = Some(ProviderSecretState::new(
        xai_acp_lib::PromptSecretRequest {
            provider: "anthropic".into(),
            provider_display_name: "Anthropic".into(),
            session_id: "s1".into(),
            prompt: "Enter API key".into(),
            request_seq: Some(77),
        },
        tx,
    ));

    let outcome = app.handle_input(&Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
    let crate::app::app_view::InputOutcome::Action(action) = outcome else {
        panic!("Esc must cancel both the reverse prompt and provider request");
    };
    let effects = dispatch(action, &mut app);
    assert!(app.agents[&id].provider_secret.is_none());
    assert!(
        app.agents[&id]
            .pending_provider_login
            .as_ref()
            .is_some_and(|pending| pending.cancelled)
    );
    assert!(matches!(
        effects.as_slice(),
        [Effect::ProviderLoginCancel {
            request_seq: 77,
            ..
        }]
    ));
    let response = rx.blocking_recv().unwrap().unwrap();
    let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
    assert_eq!(json["outcome"], "cancelled");
}

#[test]
fn stale_provider_login_result_cannot_clear_a_newer_attempt() {
    use crate::app::provider_auth::PendingProviderLogin;
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    app.agents.get_mut(&id).unwrap().pending_provider_login = Some(PendingProviderLogin {
        provider: "openrouter".into(),
        display_name: "OpenRouter".into(),
        method: ProviderAuthMethod::ApiKey,
        request_seq: 88,
        owns_input: true,
        cancelled: false,
    });
    let effects = dispatch(
        Action::TaskComplete(TaskResult::ProviderLoginComplete {
            agent_id: id,
            provider: "anthropic".into(),
            display_name: "Anthropic".into(),
            method: ProviderAuthMethod::ApiKey,
            request_seq: 77,
            result: Err("login cancelled".into()),
        }),
        &mut app,
    );
    assert!(effects.is_empty());
    assert_eq!(
        app.agents[&id]
            .pending_provider_login
            .as_ref()
            .map(|pending| pending.request_seq),
        Some(88)
    );
}

#[test]
fn pre_secret_cancel_action_still_cancels_if_reverse_request_wins_the_race() {
    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.agents.get_mut(&id).unwrap().provider_secret =
        Some(crate::app::provider_auth::ProviderSecretState::new(
            xai_acp_lib::PromptSecretRequest {
                provider: "anthropic".into(),
                provider_display_name: "Anthropic".into(),
                session_id: "s1".into(),
                prompt: "Enter API key".into(),
                request_seq: Some(77),
            },
            tx,
        ));

    let effects = dispatch(
        Action::ProviderLoginCancel {
            provider: "anthropic".into(),
            request_seq: 77,
        },
        &mut app,
    );
    assert!(app.agents[&id].provider_secret.is_none());
    assert!(matches!(
        effects.as_slice(),
        [Effect::ProviderLoginCancel {
            agent_id: AgentId(0),
            provider,
            request_seq: 77,
        }] if provider == "anthropic"
    ));
    let response = rx.blocking_recv().unwrap().unwrap();
    let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
    assert_eq!(json["outcome"], "cancelled");
}

#[test]
fn secure_provider_input_bypasses_the_input_flight_recorder() {
    use crate::app::provider_auth::ProviderSecretState;
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    app.agents.get_mut(&id).unwrap().provider_secret = Some(ProviderSecretState::new(
        xai_acp_lib::PromptSecretRequest {
            provider: "anthropic".into(),
            provider_display_name: "Anthropic".into(),
            session_id: "test-session".into(),
            prompt: "Enter API key".into(),
            request_seq: None,
        },
        response_tx,
    ));

    for ch in ['s', 'k', '-', 'x'] {
        let _ = app.handle_input(&Event::Key(KeyEvent::new(
            KeyCode::Char(ch),
            KeyModifiers::NONE,
        )));
    }
    assert!(app.agents[&id].input_log.snapshot_entries().is_empty());
    let _ = app.handle_input(&Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
    assert!(app.agents[&id].input_log.snapshot_entries().is_empty());
    let response = response_rx.blocking_recv().unwrap().unwrap();
    let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
    assert_eq!(json["outcome"], "cancelled");
}

#[test]
fn switching_views_cancels_secure_provider_input() {
    use crate::app::provider_auth::{PendingProviderLogin, ProviderSecretState};
    use xai_acp_lib::ProviderAuthMethod;

    let mut app = test_app_with_agent();
    let first = AgentId(0);
    let second = AgentId(1);
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    let first_agent = app.agents.get_mut(&first).unwrap();
    first_agent.pending_provider_login = Some(PendingProviderLogin {
        provider: "anthropic".into(),
        display_name: "Anthropic".into(),
        method: ProviderAuthMethod::ApiKey,
        request_seq: 77,
        owns_input: true,
        cancelled: false,
    });
    first_agent.provider_secret = Some(ProviderSecretState::new(
        xai_acp_lib::PromptSecretRequest {
            provider: "anthropic".into(),
            provider_display_name: "Anthropic".into(),
            session_id: "test-session".into(),
            prompt: "Enter API key".into(),
            request_seq: Some(77),
        },
        response_tx,
    ));
    let session = make_test_agent_session(&app, second, "second-session");
    app.agents
        .insert(second, AgentView::new(session, ScrollbackState::new()));

    switch_to_agent(&mut app, second, SwitchCause::Picker);

    assert!(app.agents[&first].provider_secret.is_none());
    assert!(
        app.agents[&first]
            .pending_provider_login
            .as_ref()
            .is_some_and(|pending| pending.cancelled)
    );
    assert!(app.pending_effects.iter().any(|effect| matches!(
        effect,
        Effect::ProviderLoginCancel {
            agent_id: AgentId(0),
            request_seq: 77,
            ..
        }
    )));
    let response = response_rx.blocking_recv().unwrap().unwrap();
    let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
    assert_eq!(json["outcome"], "cancelled");
}

#[test]
fn environment_provider_failure_never_opens_login_or_stashes_retry() {
    use crate::app::acp_handler::apply_session_event_for_test;
    use crate::app::agent::{AgentState, InFlightPrompt};
    use crate::scrollback::EntryId;
    use xai_acp_lib::{ProviderAuthMethod, ProviderAuthRemedy, ProviderAuthSource};
    use xai_grok_shell::extensions::notification::{RetryState, SessionUpdate as XaiSessionUpdate};

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.session.state = AgentState::TurnRunning;
        agent.turn_started_at = Some(std::time::Instant::now());
        agent.session.session_id = Some(acp::SessionId::new("sess-provider-env"));
        agent.session.current_prompt_id = Some("prompt-env".into());
        agent.session.in_flight_prompt = Some(InFlightPrompt {
            text: "do not stash".into(),
            images: Vec::new(),
            scrollback_entry: EntryId::new(102),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });
        apply_session_event_for_test(
            &XaiSessionUpdate::RetryState(RetryState::Failed {
                error_type: "provider_auth:anthropic".into(),
                message: "Unauthorized (401)".into(),
                provider_auth: Some(ProviderAuthRemedy {
                    provider: "anthropic".into(),
                    provider_display_name: "Anthropic".into(),
                    method: ProviderAuthMethod::ApiKey,
                    source: ProviderAuthSource::Environment {
                        variable: "ANTHROPIC_API_KEY".into(),
                    },
                }),
            }),
            &mut agent.session,
            &mut agent.scrollback,
        );
        assert!(agent.session.in_flight_prompt.is_none());
    }

    let effects = dispatch(
        Action::TaskComplete(TaskResult::PromptResponse {
            agent_id: id,
            result: Err("Unauthorized (401)".into()),
            http_status: Some(401),
            prompt_id: Some("prompt-env".into()),
        }),
        &mut app,
    );
    assert!(app.agents[&id].reauth_stashed_prompt.is_none());
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::ProviderLogin { .. }))
    );
}

/// Without compact_held, clearing in_flight on compact start leaves reauth empty.
#[test]
fn pre_fix_compact_start_without_hold_cannot_stash_for_reauth() {
    use crate::app::agent::AgentState;
    use crate::scrollback::block::RenderBlock;

    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.session.state = AgentState::TurnRunning;
        agent.turn_started_at = Some(std::time::Instant::now());
        agent.session.session_id = Some(acp::SessionId::new("sess-pre-fix"));
        agent.session.current_prompt_id = Some("p1".into());
        agent.session.in_flight_prompt = None;
        agent.session.compact_held_prompt = None;
        agent
            .scrollback
            .push_block(RenderBlock::session_event(SessionEvent::ReAuthRequired));
    }
    dispatch(
        Action::TaskComplete(TaskResult::PromptResponse {
            agent_id: id,
            result: Err("Unauthorized (401)".to_string()),
            http_status: Some(401),
            prompt_id: Some("p1".into()),
        }),
        &mut app,
    );
    assert!(
        app.agents[&id].reauth_stashed_prompt.is_none(),
        "without compact_held / in_flight, reauth cannot stash — the pre-fix bug"
    );
}

/// A second auth-failed turn with no rewindable prompt
/// (`in_flight_prompt == None`) must not clobber the stash from an
/// earlier 401.
#[test]
fn second_auth_failure_does_not_clobber_reauth_stash() {
    use crate::scrollback::block::RenderBlock;
    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.reauth_stashed_prompt = Some(crate::app::agent::InFlightPrompt {
            text: "first prompt".into(),
            images: Vec::new(),
            scrollback_entry: crate::scrollback::EntryId::new(0),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });
        agent
            .scrollback
            .push_block(RenderBlock::session_event(SessionEvent::ReAuthRequired));
        agent.session.state = AgentState::TurnRunning;
        agent.turn_started_at = Some(std::time::Instant::now());
        agent.session.in_flight_prompt = None;
    }

    dispatch(
        Action::TaskComplete(TaskResult::PromptResponse {
            agent_id: id,
            result: Err("Unauthorized (401)".to_string()),
            http_status: Some(401),
            prompt_id: None,
        }),
        &mut app,
    );

    assert_eq!(
        app.agents[&id]
            .reauth_stashed_prompt
            .as_ref()
            .map(|prompt| prompt.text.as_str()),
        Some("first prompt"),
        "a None in_flight_prompt must not wipe an earlier stash"
    );
}

/// Cancelling a mid-session re-auth drops the stashed prompt so it is
/// not silently resubmitted on a later, unrelated login.
#[test]
fn cancel_login_drops_reauth_stashed_prompt() {
    let mut app = test_app_with_agent();
    let id = AgentId(0);
    app.agents.get_mut(&id).unwrap().reauth_stashed_prompt =
        Some(crate::app::agent::InFlightPrompt {
            text: "stale".into(),
            images: Vec::new(),
            scrollback_entry: crate::scrollback::EntryId::new(0),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });

    dispatch(Action::Login, &mut app);
    dispatch(Action::CancelLogin, &mut app);

    assert!(
        app.agents[&id].reauth_stashed_prompt.is_none(),
        "cancelling re-auth must drop the stashed prompt"
    );
}

/// Cancelling a mid-session re-auth strips the stale `ReAuthRequired`
/// prompt from scrollback so a later `PromptResponse` cannot re-detect
/// it and re-stash the prompt for silent resubmission.
#[test]
fn cancel_login_strips_reauth_prompt_from_scrollback() {
    use crate::scrollback::block::RenderBlock;
    let mut app = test_app_with_agent();
    let id = AgentId(0);
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent.reauth_stashed_prompt = Some(crate::app::agent::InFlightPrompt {
            text: "stale".into(),
            images: Vec::new(),
            scrollback_entry: crate::scrollback::EntryId::new(0),
            combined_scrollback_entries: Vec::new(),
            chip_elements: Vec::new(),
        });
        agent
            .scrollback
            .push_block(RenderBlock::session_event(SessionEvent::ReAuthRequired));
    }

    dispatch(Action::Login, &mut app);
    dispatch(Action::CancelLogin, &mut app);

    let sb = &app.agents[&id].scrollback;
    let has_reauth = (0..sb.len()).any(|i| {
        matches!(
            sb.entry(i).map(|e| &e.block),
            Some(RenderBlock::SessionEvent(ev)) if matches!(ev.event, SessionEvent::ReAuthRequired)
        )
    });
    assert!(
        !has_reauth,
        "cancelling re-auth must strip the stale re-auth prompt from scrollback"
    );
}

/// Empty `auth_methods` (preferred_method pin unavailable) must not invent
/// `grok.com` or start an OIDC flow the agent did not advertise.
#[test]
fn login_with_empty_auth_methods_fails_closed() {
    let mut app = test_app_with_agent();
    app.auth_methods.clear();
    app.login_method_id = None;

    let effects = dispatch(Action::Login, &mut app);

    assert!(
        effects.is_empty(),
        "must not start Authenticate without an advertised method"
    );
    assert_eq!(
        app.active_view,
        ActiveView::Agent(AgentId(0)),
        "must stay on the session view"
    );
    assert!(
        matches!(
            &app.auth_state,
            AuthState::Pending { error: Some(msg) }
                if msg.contains("preferred_method=api_key")
        ),
        "must surface pin-unavailable error, got {:?}",
        app.auth_state
    );
    assert!(app.login_method_id.is_none());
}

/// Puts the app in `Authenticating` with a live task's abort handle installed
/// (as the event loop would), returning the task's JoinHandle and the seq.
/// Callers assert the task actually gets aborted (`unwrap_err().is_cancelled()`),
/// not merely that the handle slot was cleared.
fn install_live_auth_task(
    app: &mut AppView,
    rt: &tokio::runtime::Runtime,
) -> (tokio::task::JoinHandle<()>, u64) {
    dispatch(Action::Login, app);
    let task = rt.spawn(std::future::pending::<()>());
    match &mut app.auth_state {
        AuthState::Authenticating {
            handle,
            request_seq,
            ..
        } => {
            *handle = Some(task.abort_handle());
            (task, *request_seq)
        }
        other => panic!("expected Authenticating after Login, got {other:?}"),
    }
}

fn test_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
}

/// A second `/login` while already authenticating must abort the prior auth
/// task and bump the seq (single-flight: no stacked device-code mints).
#[test]
fn login_while_authenticating_aborts_prior_task() {
    let rt = test_runtime();
    let mut app = test_app_with_agent();
    let (prior_task, first_seq) = install_live_auth_task(&mut app, &rt);

    let effects = dispatch(Action::Login, &mut app);

    rt.block_on(async {
        assert!(
            prior_task.await.unwrap_err().is_cancelled(),
            "prior auth task must be aborted"
        );
    });
    match &app.auth_state {
        AuthState::Authenticating { request_seq, .. } => {
            assert!(
                *request_seq > first_seq,
                "re-login must bump request_seq for single-flight"
            );
        }
        other => panic!("expected Authenticating after re-Login, got {other:?}"),
    }
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::Authenticate { .. })),
        "re-login must emit a new Authenticate"
    );
}

/// A stale `AuthComplete` (from an attempt whose abort lost the race because
/// the task had already finished) must not complete the new attempt: the
/// request-seq guard is the only protection here.
#[test]
fn stale_auth_complete_after_relogin_is_ignored() {
    let mut app = test_app_with_agent();
    dispatch(Action::Login, &mut app);
    let first_seq = match &app.auth_state {
        AuthState::Authenticating { request_seq, .. } => *request_seq,
        other => panic!("expected Authenticating after Login, got {other:?}"),
    };
    dispatch(Action::Login, &mut app); // re-login bumps to seq2

    dispatch(
        Action::TaskComplete(TaskResult::AuthComplete {
            request_seq: first_seq,
            meta: None,
        }),
        &mut app,
    );

    match &app.auth_state {
        AuthState::Authenticating { request_seq, .. } => {
            assert!(
                *request_seq > first_seq,
                "stale AuthComplete must leave the new attempt authenticating"
            );
        }
        other => panic!("stale AuthComplete must be ignored, got {other:?}"),
    }
}

/// Switch-account while authenticating goes through the same single-flight
/// abort as `/login` (sibling entry point).
#[test]
fn switch_account_while_authenticating_aborts_prior_task() {
    let rt = test_runtime();
    let mut app = test_app_with_agent();
    let (prior_task, first_seq) = install_live_auth_task(&mut app, &rt);

    dispatch(Action::SwitchAccount, &mut app);

    rt.block_on(async {
        assert!(
            prior_task.await.unwrap_err().is_cancelled(),
            "prior auth task must be aborted on switch-account"
        );
    });
    match &app.auth_state {
        AuthState::Authenticating { request_seq, .. } => {
            assert!(*request_seq > first_seq, "switch must bump request_seq");
        }
        other => panic!("expected Authenticating after SwitchAccount, got {other:?}"),
    }
}

/// Cancelling a mid-session login aborts the in-flight auth task (not just
/// restores the view) so a retry cannot race a still-polling prior mint.
#[test]
fn cancel_login_aborts_prior_task() {
    let rt = test_runtime();
    let mut app = test_app_with_agent();
    // Login from a session view stashes `auth_return_view`, making CancelLogin live.
    let (prior_task, _) = install_live_auth_task(&mut app, &rt);

    dispatch(Action::CancelLogin, &mut app);

    rt.block_on(async {
        assert!(
            prior_task.await.unwrap_err().is_cancelled(),
            "cancel must abort the in-flight auth task"
        );
    });
}

/// Cancelling a mid-session login returns to the session rather than
/// quitting the app, and clears the stashed view + auth state.
#[test]
fn cancel_login_restores_view() {
    let mut app = test_app_with_agent();
    dispatch(Action::Login, &mut app);
    assert_eq!(app.active_view, ActiveView::Welcome);
    let prior_seq = match &app.auth_state {
        AuthState::Authenticating { request_seq, .. } => *request_seq,
        other => panic!("expected Authenticating after Login, got {other:?}"),
    };

    let effects = dispatch(Action::CancelLogin, &mut app);

    assert!(
        matches!(
            effects.as_slice(),
            [Effect::CancelAuth { request_seq }] if *request_seq == prior_seq
        ),
        "cancel must tell the shell to stop the in-flight auth poll for this attempt"
    );
    assert_eq!(app.active_view, ActiveView::Agent(AgentId(0)));
    assert_eq!(app.auth_return_view, None);
    assert!(matches!(app.auth_state, AuthState::Done));
}

/// `CancelLogin` outside a mid-session login is a no-op (must not move
/// off the welcome screen or panic).
#[test]
fn cancel_login_noop_without_stashed_view() {
    let mut app = test_app();
    let effects = dispatch(Action::CancelLogin, &mut app);
    assert!(effects.is_empty());
    assert_eq!(app.active_view, ActiveView::Welcome);
    assert_eq!(app.auth_return_view, None);
}

#[test]
fn auth_complete_extracts_show_resolved_model_from_meta() {
    let mut app = test_app();
    app.auth_state = AuthState::Authenticating {
        request_seq: 1,
        handle: None,
        auth_url: None,
        mode: AuthMode::Pending,
    };
    assert!(app.show_resolved_model);

    dispatch(
        Action::TaskComplete(TaskResult::AuthComplete {
            request_seq: 1,
            meta: Some(serde_json::json!({ "show_resolved_model": false })),
        }),
        &mut app,
    );

    assert!(!app.show_resolved_model);
}

#[test]
fn auth_complete_preserves_show_resolved_model_when_absent() {
    let mut app = test_app();
    app.show_resolved_model = false;
    app.auth_state = AuthState::Authenticating {
        request_seq: 1,
        handle: None,
        auth_url: None,
        mode: AuthMode::Pending,
    };

    dispatch(
        Action::TaskComplete(TaskResult::AuthComplete {
            request_seq: 1,
            meta: Some(serde_json::to_value(xai_grok_shell::auth::AuthMeta::default()).unwrap()),
        }),
        &mut app,
    );

    assert!(!app.show_resolved_model);
}

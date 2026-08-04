//! Responses server-compaction-specific support for [`SessionActor`].

use super::*;
use xai_grok_sampling_types::ResponsesCompactionMode;

pub(super) struct PreparedServerRequest {
    pub(super) snapshot: crate::session::responses_server_compaction::ResponsesRequestSnapshot,
    pub(super) client: xai_grok_sampler::SamplingClient,
    pub(super) request: xai_grok_sampler::ResponsesCompactRequest,
    pub(super) capability_key: crate::session::responses_server_compaction::CapabilityKey,
    pub(super) checkpoint_status: xai_chat_state::CheckpointReplayStatus,
}

/// Outcome of the checkpoint replay gate
/// (`ensure_checkpoint_replayable_for_request`).
#[derive(Debug)]
pub(crate) struct CheckpointGateOutcome {
    /// The gate ran a local continuity migration; the outer turn loop must
    /// rebuild and resubmit from the replacement history.
    pub resubmit: bool,
    /// Fully verified replay material for the one current checkpoint type.
    pub replay: Option<xai_grok_sampling_types::ValidatedResponsesReplay>,
}

impl CheckpointGateOutcome {
    fn proceed() -> Self {
        Self {
            resubmit: false,
            replay: None,
        }
    }

    fn resubmit() -> Self {
        Self {
            resubmit: true,
            replay: None,
        }
    }

    fn replayable(replay: xai_grok_sampling_types::ValidatedResponsesReplay) -> Self {
        Self {
            resubmit: false,
            replay: Some(replay),
        }
    }
}

#[derive(Debug)]
struct AuthManagerCompactResolver {
    manager: Arc<crate::auth::AuthManager>,
    principal_fingerprint: String,
}

impl xai_grok_sampler::CompactCredentialResolver for AuthManagerCompactResolver {
    fn current_credential(&self) -> Option<xai_grok_sampler::CompactCredential> {
        let auth = self.manager.current_wire_valid()?;
        (stable_auth_principal_fingerprint(&auth).as_deref()
            == Some(self.principal_fingerprint.as_str()))
        .then(|| {
            xai_grok_sampler::CompactCredential::bearer(
                auth.key,
                self.principal_fingerprint.clone(),
            )
        })
    }
}

fn stable_auth_principal_fingerprint(auth: &crate::auth::GrokAuth) -> Option<String> {
    let has_stable_id = [
        auth.principal_id.as_deref(),
        auth.team_id.as_deref(),
        auth.organization_id.as_deref(),
        Some(auth.user_id.as_str()).filter(|value| !value.is_empty()),
    ]
    .into_iter()
    .any(|value| value.is_some());
    if !has_stable_id {
        return None;
    }
    let value = serde_json::json!({
        "class": format!("{:?}", auth.auth_mode).to_ascii_lowercase(),
        "principal_id": auth.principal_id,
        "team_id": auth.team_id,
        "organization_id": auth.organization_id,
        "user_id": auth.user_id,
    });
    let bytes = xai_grok_sampling_types::canonical_json_bytes(&value).ok()?;
    use sha2::Digest as _;
    Some(format!("{:x}", sha2::Sha256::digest(bytes)))
}

fn keyed_principal_fingerprint(class: &str, key: &str) -> String {
    use sha2::Digest as _;
    let mut hash = sha2::Sha256::new();
    hash.update(class.as_bytes());
    hash.update([0]);
    hash.update(key.as_bytes());
    format!("{:x}", hash.finalize())
}

pub(super) fn responses_route_capabilities(
    model: &str,
    base_url: &str,
) -> crate::auth::providers::ProviderCapabilities {
    if let Some((provider, _)) = crate::auth::providers::parse_namespaced_model_id(model) {
        return crate::auth::providers::provider_descriptor(provider).capabilities;
    }
    if model.contains('/') || !crate::util::is_xai_api_url(base_url) {
        return crate::auth::providers::ProviderCapabilities::default();
    }
    crate::auth::providers::ProviderCapabilities {
        supports_remote_compaction: true,
        accepts_responses_checkpoint: true,
    }
}

impl SessionActor {
    pub(crate) fn compact_credential(
        &self,
        config: &xai_grok_sampler::SamplerConfig,
    ) -> Option<(xai_grok_sampler::RequestCredentialSnapshot, String)> {
        let key = config.api_key.as_deref().filter(|key| !key.is_empty())?;
        let auth = self
            .auth_manager
            .as_ref()
            .and_then(|manager| manager.current_wire_valid());
        let provider_route =
            crate::auth::providers::parse_namespaced_model_id(&config.model).is_some();
        let uses_session_auth = !provider_route
            && (auth.as_ref().is_some_and(|auth| auth.key == key)
                || (config.bearer_resolver.is_some() && auth.is_some()));
        let principal = if uses_session_auth {
            stable_auth_principal_fingerprint(auth.as_ref()?)?
        } else if let Some(deployment_id) = config.deployment_id.as_deref() {
            keyed_principal_fingerprint("deployment", deployment_id)
        } else {
            keyed_principal_fingerprint(
                match config.auth_scheme {
                    xai_grok_sampler::AuthScheme::Bearer => "bearer_key",
                    xai_grok_sampler::AuthScheme::XApiKey => "x_api_key",
                },
                key,
            )
        };
        let credential = match config.auth_scheme {
            xai_grok_sampler::AuthScheme::Bearer => {
                let snapshot = xai_grok_sampler::RequestCredentialSnapshot::bearer(
                    key.to_string(),
                    principal.clone(),
                );
                if uses_session_auth {
                    snapshot.with_resolver(Arc::new(AuthManagerCompactResolver {
                        manager: self.auth_manager.as_ref()?.clone(),
                        principal_fingerprint: principal.clone(),
                    }))
                } else {
                    snapshot
                }
            }
            xai_grok_sampler::AuthScheme::XApiKey => {
                xai_grok_sampler::RequestCredentialSnapshot::x_api_key(
                    key.to_string(),
                    principal.clone(),
                )
            }
        };
        Some((credential, principal))
    }

    pub(crate) fn portable_history_for_request(
        &self,
        items: &[ConversationItem],
    ) -> std::io::Result<Vec<ConversationItem>> {
        // Fail closed on any checkpoint that is not the unique item at
        // index 0: a wrapper buried mid-history must never flow into
        // request construction (recap budgeting, auxiliary preprocessing
        // or a POST body).
        let checkpoint_count = items
            .iter()
            .filter(|item| item.is_responses_checkpoint())
            .count();
        if checkpoint_count > 1
            || (checkpoint_count == 1
                && !items
                    .first()
                    .is_some_and(|item| item.is_responses_checkpoint()))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "a responses compaction checkpoint must be the unique item at index 0",
            ));
        }
        match items.first() {
            Some(ConversationItem::ResponsesCompactionCheckpoint(wrapper)) => {
                let session_dir = crate::session::persistence::session_dir(&self.session_info);
                let mut portable =
                    crate::session::storage::responses_compaction::read_checkpoint_for_wrapper(
                        &session_dir,
                        wrapper,
                    )?
                    .portable_history;
                portable.extend_from_slice(&items[1..]);
                Ok(portable)
            }
            _ => Ok(items.to_vec()),
        }
    }

    pub(super) async fn prepare_server_request(
        &self,
        user_context: Option<&str>,
        normal_request: Option<&ConversationRequest>,
        trigger: xai_grok_telemetry::events::CompactionTrigger,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<Option<PreparedServerRequest>, acp::Error> {
        let sampling = self
            .chat_state_handle
            .get_sampling_config()
            .await
            .ok_or_else(|| acp::Error::internal_error().data("missing sampling config"))?;
        if sampling.api_backend != ApiBackend::Responses
            || !responses_route_capabilities(&sampling.model, &sampling.base_url)
                .supports_remote_compaction
        {
            return Ok(None);
        }
        let full_config = self.reconstruct_full_config().await;
        if full_config.api_backend != ApiBackend::Responses
            || !responses_route_capabilities(&full_config.model, &full_config.base_url)
                .supports_remote_compaction
        {
            return Ok(None);
        }

        let mut request = match normal_request {
            Some(request) => request.clone(),
            None => {
                let tool_definitions = self.prepare_tool_definitions().await;
                let tools = if let Some(ref override_tools) = self.forked_tool_override {
                    override_tools.clone()
                } else {
                    self.turn_base_tool_specs(&tool_definitions)
                };
                let request_id = format!("compact-{}", uuid::Uuid::now_v7());
                let mut request = self
                    .chat_state_handle
                    .build_request(
                        tools,
                        None,
                        false,
                        None,
                        self.session_info.id.to_string(),
                        request_id,
                    )
                    .await
                    .ok_or_else(|| {
                        acp::Error::internal_error().data("chat-state actor unavailable")
                    })?;
                request.x_grok_session_id = Some(self.session_info.id.to_string());
                request.x_grok_turn_idx =
                    Some(self.chat_state_handle.get_prompt_index().await.to_string());
                request.x_grok_agent_id = Some(xai_grok_telemetry::id::agent_id());
                request.hosted_tools = self.hosted_tools_for_turn();
                request
            }
        };
        let request_history_revision = request.history_revision.ok_or_else(|| {
            acp::Error::internal_error().data("responses_compaction_missing_request_revision")
        })?;

        full_config.apply_conversation_defaults_to(&mut request);
        request.parallel_tool_calls = Some(true);
        let routing_model = request
            .model
            .clone()
            .unwrap_or_else(|| full_config.model.clone());
        let cache_routing = self.cache_routing_for_config(&full_config, &routing_model);
        request.prompt_cache_key = cache_routing
            .as_ref()
            .map(|routing| routing.prompt_cache_key());
        let wire_instructions = self.current_wire_instructions(&request).await;
        request.instructions = (!wire_instructions.is_empty()).then_some(wire_instructions);

        let client = xai_grok_sampler::SamplingClient::new(full_config.clone())
            .map_err(|error| self.to_acp_error(error))?;
        let Some((credential, principal)) = self.compact_credential(&full_config) else {
            return Ok(None);
        };
        let endpoint_fingerprint = client.responses_compact_endpoint_fingerprint();
        let provider_id = "xai".to_string();
        let trusted_envelope = self.current_trusted_envelope(&request).await?;
        let live_wrapper = request.items.first().and_then(|item| match item {
            ConversationItem::ResponsesCompactionCheckpoint(wrapper) => Some(wrapper.clone()),
            _ => None,
        });
        let prior_checkpoint_id = live_wrapper
            .as_ref()
            .map(|wrapper| wrapper.checkpoint_id.clone());
        let identity = xai_grok_sampling_types::CheckpointIdentity {
            provider_id: provider_id.clone(),
            api: "responses".into(),
            endpoint_fingerprint: endpoint_fingerprint.clone(),
            model: routing_model.clone(),
            auth_principal_fingerprint: principal.clone(),
            contract_version: xai_grok_sampling_types::RESPONSES_COMPACTION_CONTRACT.into(),
            prompt_envelope_fingerprint: trusted_envelope.envelope_fingerprint.clone(),
            base_instructions_sha256: trusted_envelope.base_instructions_sha256.clone(),
            prior_checkpoint_id,
            cache_route_fingerprint: Some(xai_grok_sampling_types::cache_route_fingerprint(
                &provider_id,
                &xai_grok_sampling_types::normalize_base_url_for_routing(&full_config.base_url),
                &xai_grok_sampling_types::model_cache_family(&routing_model),
                &principal,
            )),
        };
        let binding_identity = live_wrapper.as_ref().map_or_else(
            || identity.clone(),
            |wrapper| {
                crate::session::responses_server_compaction::current_identity_for_recompact_binding(
                    &identity, wrapper,
                )
            },
        );
        let bound = self
            .chat_state_handle
            .bind_request_identity_at_revision(binding_identity, request_history_revision)
            .await
            .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?;
        let (binding, state) = match bound {
            xai_chat_state::RequestIdentityBindResult::Bound {
                binding,
                compaction_snapshot,
            } => (binding, *compaction_snapshot),
            xai_chat_state::RequestIdentityBindResult::StaleHistory { .. } => {
                return Err(acp::Error::internal_error().data("responses_compaction_stale_request"));
            }
        };

        let (resolved, portable_history) = if let Some(wrapper) = live_wrapper.as_ref() {
            if !matches!(
                binding.checkpoint_status,
                xai_chat_state::CheckpointReplayStatus::Replayable
            ) {
                return Ok(None);
            }
            let session_dir = crate::session::persistence::session_dir(&self.session_info);
            let sidecar =
                crate::session::storage::responses_compaction::read_checkpoint_for_wrapper(
                    &session_dir,
                    wrapper,
                )
                .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
            let replay = xai_grok_sampling_types::ValidatedResponsesReplay::verify(
                wrapper,
                &sidecar.replay_material,
                &sidecar.portable_history,
                &trusted_envelope,
                &request.items[1..],
                request_history_revision,
                binding.request_identity_generation,
            )
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
            let resolved =
                xai_grok_sampling_types::ResolvedCompactRequest::from_validated_recompact(
                    &replay,
                    &request,
                    user_context,
                )
                .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
            let mut portable_history = sidecar.portable_history;
            portable_history.extend_from_slice(&request.items[1..]);
            (resolved, portable_history)
        } else {
            if !matches!(
                binding.checkpoint_status,
                xai_chat_state::CheckpointReplayStatus::NoCheckpoint
            ) {
                return Ok(None);
            }
            let resolved =
                xai_grok_sampling_types::ResolvedCompactRequest::try_normal(&request, user_context)
                    .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
            (resolved, request.items.clone())
        };

        let final_body = resolved.body().clone();
        let compact_request =
            xai_grok_sampler::ResponsesCompactRequest::from_resolved(&resolved)
                .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        let request_bytes = compact_request.to_bounded_bytes().map_err(|error| {
            let reason =
                if error.failure() == xai_grok_sampler::ResponsesCompactFailure::RequestTooLarge {
                    "responses_compaction_request_too_large"
                } else {
                    "responses_compaction_request_build_failed"
                };
            acp::Error::internal_error().data(reason)
        })?;
        let semantic_envelope =
            crate::session::responses_server_compaction::resolved_prompt_envelope(&final_body)
                .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        let semantic_envelope_tokens =
            crate::session::responses_server_compaction::prompt_envelope_token_estimate(
                &final_body,
            )
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        let input = final_body
            .get("input")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let string_field = |name: &str| {
            final_body
                .get(name)
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        };
        let capability_key = crate::session::responses_server_compaction::CapabilityKey {
            endpoint_fingerprint,
            model: identity.model.clone(),
            auth_principal_fingerprint: principal,
            contract_version: identity.contract_version.clone(),
        };
        Ok(Some(PreparedServerRequest {
            snapshot: crate::session::responses_server_compaction::ResponsesRequestSnapshot {
                chat_revision: state.history_revision,
                request_identity_generation: binding.request_identity_generation,
                prompt_index: state.prompt_index,
                pre_compaction_tokens: state.total_tokens,
                final_request: final_body.clone(),
                credential,
                model: identity.model.clone(),
                input,
                portable_history,
                instructions: string_field("instructions"),
                prompt_cache_key: string_field("prompt_cache_key"),
                prompt_cache_options: final_body.get("prompt_cache_options").cloned(),
                prompt_cache_retention: string_field("prompt_cache_retention"),
                service_tier: string_field("service_tier"),
                semantic_envelope,
                semantic_envelope_tokens,
                identity,
                trusted_envelope,
                request_bytes,
                trigger,
                mode: self.responses_mode_metadata(),
                user_context: user_context.map(str::to_owned),
                cancellation,
            },
            client,
            request: compact_request,
            capability_key,
            checkpoint_status: binding.checkpoint_status,
        }))
    }

    /// Resolve the current checkpoint compatibility envelope. Base
    /// instructions and the canonical non-transcript envelope participate in
    /// compatibility; current memory is reflected only in the diagnostic wire
    /// hash and may change without invalidating the checkpoint.
    async fn current_trusted_envelope(
        &self,
        request: &ConversationRequest,
    ) -> Result<xai_grok_sampling_types::TrustedPromptEnvelope, acp::Error> {
        let base_from_items: Vec<String> = request
            .items
            .iter()
            .filter_map(|item| match item {
                ConversationItem::System(system)
                    if system.source == xai_grok_sampling_types::SystemSource::BaseInstructions =>
                {
                    let content = system.content.trim();
                    (!content.is_empty()).then(|| content.to_string())
                }
                _ => None,
            })
            .collect();
        let base_instructions = if !base_from_items.is_empty() {
            base_from_items.join(xai_grok_sampling_types::INSTRUCTIONS_MEMORY_SEPARATOR)
        } else {
            let session_dir = crate::session::persistence::session_dir(&self.session_info);
            super::super::load_system_prompt_from_dir(&session_dir)
                .map(|base| base.trim().to_string())
                .unwrap_or_default()
        };
        let rendered = self.current_wire_instructions(request).await;
        Ok(xai_grok_sampling_types::TrustedPromptEnvelope {
            base_instructions_sha256: xai_grok_sampling_types::base_instructions_sha256(
                &base_instructions,
            ),
            memory_revision: None,
            envelope_fingerprint: xai_grok_sampling_types::canonical_envelope_fingerprint(request),
            wire_prompt_sha256: xai_grok_sampling_types::wire_prompt_sha256(&rendered),
        })
    }

    async fn run_checkpoint_builtin_migration(
        self: &Arc<Self>,
        request: &ConversationRequest,
        full_config: &xai_grok_sampler::SamplerConfig,
        reason: &'static str,
    ) -> Result<CheckpointGateOutcome, acp::Error> {
        self.maybe_pre_compaction_flush(
            self.chat_state_handle.get_total_tokens().await,
            full_config.context_window,
            "continuity_migration",
        )
        .await;
        self.run_compact_inner(
            None,
            None,
            xai_grok_telemetry::events::CompactionTrigger::Manual,
            CompactionStrategy::BuiltinMigration(reason),
            Some(request.clone()),
            None,
            false,
            0,
        )
        .await?;
        Ok(CheckpointGateOutcome::resubmit())
    }

    pub(crate) async fn ensure_checkpoint_replayable_for_request(
        self: &Arc<Self>,
        request: &ConversationRequest,
        full_config: &xai_grok_sampler::SamplerConfig,
    ) -> Result<CheckpointGateOutcome, acp::Error> {
        let checkpoint_count = request
            .items
            .iter()
            .filter(|item| item.is_responses_checkpoint())
            .count();
        let Some(ConversationItem::ResponsesCompactionCheckpoint(wrapper)) = request.items.first()
        else {
            if checkpoint_count != 0 {
                return Err(acp::Error::internal_error()
                    .data("responses_compaction_invalid_checkpoint_layout"));
            }
            return Ok(CheckpointGateOutcome::proceed());
        };
        if checkpoint_count != 1 {
            return Err(
                acp::Error::internal_error().data("responses_compaction_invalid_checkpoint_layout")
            );
        }
        let model = request.model.as_deref().unwrap_or(&full_config.model);
        if !responses_route_capabilities(model, &full_config.base_url).accepts_responses_checkpoint
        {
            return self
                .run_checkpoint_builtin_migration(request, full_config, "unsupported_route")
                .await;
        }
        if full_config.api_backend != ApiBackend::Responses {
            return self
                .run_checkpoint_builtin_migration(request, full_config, "non_responses")
                .await;
        }

        let request_history_revision = request.history_revision.ok_or_else(|| {
            acp::Error::internal_error().data("responses_compaction_missing_request_revision")
        })?;
        let client = xai_grok_sampler::SamplingClient::new(full_config.clone())
            .map_err(|error| self.to_acp_error(error))?;
        let Some((_, principal)) = self.compact_credential(full_config) else {
            return self
                .run_checkpoint_builtin_migration(request, full_config, "continuity_mismatch")
                .await;
        };
        let current_envelope = self.current_trusted_envelope(request).await?;
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| full_config.model.clone());
        let provider_id = "xai".to_string();
        let identity = xai_grok_sampling_types::CheckpointIdentity {
            provider_id: provider_id.clone(),
            api: "responses".into(),
            endpoint_fingerprint: client.responses_compact_endpoint_fingerprint(),
            model: model.clone(),
            auth_principal_fingerprint: principal.clone(),
            contract_version: xai_grok_sampling_types::RESPONSES_COMPACTION_CONTRACT.into(),
            prompt_envelope_fingerprint: current_envelope.envelope_fingerprint.clone(),
            base_instructions_sha256: current_envelope.base_instructions_sha256.clone(),
            prior_checkpoint_id: wrapper.prior_checkpoint_id.clone(),
            cache_route_fingerprint: Some(xai_grok_sampling_types::cache_route_fingerprint(
                &provider_id,
                &xai_grok_sampling_types::normalize_base_url_for_routing(&full_config.base_url),
                &xai_grok_sampling_types::model_cache_family(&model),
                &principal,
            )),
        };
        let binding = match self
            .chat_state_handle
            .bind_request_identity_at_revision(identity, request_history_revision)
            .await
            .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?
        {
            xai_chat_state::RequestIdentityBindResult::Bound { binding, .. } => binding,
            xai_chat_state::RequestIdentityBindResult::StaleHistory { .. } => {
                return Ok(CheckpointGateOutcome::resubmit());
            }
        };
        if !matches!(
            binding.checkpoint_status,
            xai_chat_state::CheckpointReplayStatus::Replayable
        ) {
            return self
                .run_checkpoint_builtin_migration(request, full_config, "continuity_mismatch")
                .await;
        }

        let session_dir = crate::session::persistence::session_dir(&self.session_info);
        let sidecar = crate::session::storage::responses_compaction::read_checkpoint_for_wrapper(
            &session_dir,
            wrapper,
        )
        .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        let replay = xai_grok_sampling_types::ValidatedResponsesReplay::verify(
            wrapper,
            &sidecar.replay_material,
            &sidecar.portable_history,
            &current_envelope,
            &request.items[1..],
            request_history_revision,
            binding.request_identity_generation,
        )
        .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        Ok(CheckpointGateOutcome::replayable(replay))
    }

    fn responses_mode_metadata(&self) -> ResponsesCompactionMode {
        ResponsesCompactionMode {
            name: self.compaction.compaction_mode.to_string(),
            detail: self
                .compaction
                .compaction_mode
                .segment_detail()
                .map(|detail| serde_json::Value::String(detail.to_string())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn log_strategy_attempt(
        &self,
        compaction_id: &str,
        supersedes_compaction_id: Option<&str>,
        prepared: Option<&PreparedServerRequest>,
        strategy: &'static str,
        eligible: bool,
        skip_reason: Option<&'static str>,
        attempts: u8,
        outcome: &'static str,
        failure: Option<&'static str>,
        status: Option<u16>,
        latency_ms: u64,
        response_bytes: u64,
        output_items: u64,
        capability_cache_hit: bool,
        cas_outcome: &'static str,
        checkpoint_bytes: Option<u64>,
        tokens_after: Option<u64>,
        token_seed_source: Option<&'static str>,
        commit_failure_step: Option<&'static str>,
    ) {
        let fallback_model = self
            .agent
            .borrow()
            .compaction_policy()
            .compact_model
            .clone();
        xai_grok_telemetry::session_ctx::log_event(
            xai_grok_telemetry::events::CompactionStrategyAttempt {
                compaction_id: compaction_id.to_string(),
                supersedes_compaction_id: supersedes_compaction_id.map(str::to_owned),
                strategy,
                eligible,
                skip_reason,
                attempts,
                outcome,
                failure,
                status,
                latency_ms,
                request_bytes: prepared
                    .map(|prepared| prepared.snapshot.request_bytes.len() as u64)
                    .unwrap_or(0),
                response_bytes,
                output_items,
                capability_cache_hit,
                fallback_model,
                fallback_latency_ms: None,
                prefire_consumed: strategy == "builtin_fallback",
                prefire_wasted: strategy == "server" && outcome == "committed",
                prefire_stale: false,
                history_revision: prepared
                    .map(|prepared| prepared.snapshot.chat_revision)
                    .unwrap_or(0),
                request_identity_generation: prepared
                    .map(|prepared| prepared.snapshot.request_identity_generation)
                    .unwrap_or(0),
                cas_outcome,
                checkpoint_schema: (outcome == "committed").then_some(match strategy {
                    "server" => 3,
                    _ => 1,
                }),
                checkpoint_bytes,
                restore_outcome: None,
                marker_repaired: false,
                tokens_before: prepared
                    .map(|prepared| prepared.snapshot.pre_compaction_tokens)
                    .unwrap_or(0),
                tokens_after,
                token_seed_source,
                commit_failure_step,
            },
        );
    }

    pub(super) async fn notify_compaction_fallback(
        &self,
        strategy_started_notified: &mut bool,
        reason: crate::session::responses_server_compaction::ServerCompactionFailureReason,
    ) {
        if !*strategy_started_notified {
            *strategy_started_notified = true;
            self.send_xai_notification(
                crate::extensions::notification::SessionUpdate::CompactionFallbackStarted {
                    reason: reason.as_str().to_string(),
                },
            )
            .await;
        }
    }

    pub(super) async fn notify_compaction_migration(
        &self,
        strategy_started_notified: &mut bool,
        reason: &str,
    ) {
        if !*strategy_started_notified {
            *strategy_started_notified = true;
            self.send_xai_notification(
                crate::extensions::notification::SessionUpdate::CompactionMigrationStarted {
                    reason: reason.to_string(),
                },
            )
            .await;
        }
    }

    pub(super) async fn persist_server_sidecar(
        &self,
        relative_path: String,
        checkpoint: crate::session::storage::responses_compaction::CompactionCheckpointFile,
    ) -> Result<(), acp::Error> {
        let (respond_to, response) = tokio::sync::oneshot::channel();
        self.notifications
            .persistence_tx
            .send(PersistenceMsg::ResponsesCompactionCheckpoint {
                relative_path,
                checkpoint,
                respond_to,
            })
            .map_err(|_| {
                acp::Error::internal_error().data("checkpoint persistence channel unavailable")
            })?;
        response
            .await
            .map_err(|_| {
                acp::Error::internal_error().data("checkpoint persistence acknowledgement lost")
            })?
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))
    }

    pub(super) async fn stage_server_segment(
        &self,
        staging: crate::session::storage::responses_compaction::ResponsesCompactionSegmentStaging,
    ) -> Result<(), acp::Error> {
        let (respond_to, response) = tokio::sync::oneshot::channel();
        self.notifications
            .persistence_tx
            .send(PersistenceMsg::ResponsesCompactionSegmentStage {
                staging,
                respond_to,
            })
            .map_err(|_| {
                acp::Error::internal_error().data("segment staging channel unavailable")
            })?;
        response
            .await
            .map_err(|_| acp::Error::internal_error().data("segment staging acknowledgement lost"))?
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))
    }

    pub(super) async fn publish_server_segment(
        &self,
        checkpoint_id: String,
        operation_id: String,
        wrapper_digest: String,
    ) -> Result<(), acp::Error> {
        let (respond_to, response) = tokio::sync::oneshot::channel();
        self.notifications
            .persistence_tx
            .send(PersistenceMsg::ResponsesCompactionSegmentPublish {
                checkpoint_id,
                operation_id,
                wrapper_digest,
                respond_to,
            })
            .map_err(|_| {
                acp::Error::internal_error().data("segment publication channel unavailable")
            })?;
        response
            .await
            .map_err(|_| {
                acp::Error::internal_error().data("segment publication acknowledgement lost")
            })?
            .map(|_| ())
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))
    }

    /// Stage D4: per-session checkpoint quota pressure, rate-limited to once
    /// per session per hour (telemetry + tracing only). The caller has
    /// already fallen back to builtin compaction for this attempt; active
    /// recovery data is never deleted to make room.
    pub(super) fn notify_checkpoint_quota_pressure(&self) {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0);
        let last = self
            .compaction
            .quota_pressure_notified_at
            .load(std::sync::atomic::Ordering::Relaxed);
        if now_secs.saturating_sub(last) < 60 * 60 {
            return;
        }
        if self
            .compaction
            .quota_pressure_notified_at
            .compare_exchange(
                last,
                now_secs,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            return;
        }
        let session_dir = crate::session::persistence::session_dir(&self.session_info);
        let quota_bytes = crate::session::compaction_gc::session_checkpoint_quota_bytes();
        let session_checkpoint_bytes =
            crate::session::compaction_gc::session_checkpoint_bytes(&session_dir).unwrap_or(0);
        tracing::warn!(
            session_checkpoint_bytes,
            quota_bytes,
            "session checkpoint quota exceeded; new remote checkpoints fall back to builtin until GC frees space"
        );
        xai_grok_telemetry::session_ctx::log_event(
            xai_grok_telemetry::events::CompactionQuotaPressure {
                session_checkpoint_bytes,
                quota_bytes,
            },
        );
    }

    /// Stage D4: best-effort detached GC of this session's compaction
    /// sidecars/segment staging, spawned after a committed server compaction.
    /// Runs the fail-closed full reachability scan (live wrapper + markers +
    /// recovery records + journal + published segments); errors are only
    /// logged.
    pub(super) fn spawn_compaction_gc(&self) {
        let session_dir = crate::session::persistence::session_dir(&self.session_info);
        let _ = tokio::spawn(async move {
            let result = crate::session::compaction_gc::gc_session_compaction_artifacts(
                &session_dir,
                crate::session::compaction_gc::GcOptions::default(),
            )
            .await;
            let report = match result {
                Ok(report) => report,
                Err(error) => {
                    tracing::warn!(
                        session_dir = %session_dir.display(),
                        error = %error,
                        "compaction GC skipped (fail-closed full scan)"
                    );
                    return;
                }
            };
            if report.deleted_bytes > 0 {
                tracing::info!(
                    session_dir = %session_dir.display(),
                    deleted_bytes = report.deleted_bytes,
                    files_deleted = report.files_deleted,
                    orphan_bytes = report.orphan_bytes,
                    "compaction GC deleted orphaned checkpoint artifacts"
                );
            }
            let quota_bytes = crate::session::compaction_gc::session_checkpoint_quota_bytes();
            let session_checkpoint_bytes =
                crate::session::compaction_gc::session_checkpoint_bytes(&session_dir)
                    .unwrap_or(report.scanned_bytes);
            let quota_exceeded =
                crate::session::compaction_gc::quota_exceeded(&session_dir, quota_bytes)
                    .unwrap_or(false);
            xai_grok_telemetry::session_ctx::log_event(
                xai_grok_telemetry::events::CompactionGcOutcome {
                    session_checkpoint_bytes,
                    orphan_bytes: report.orphan_bytes,
                    deleted_bytes: report.deleted_bytes,
                    retained_orphan_bytes: report.retained_orphan_bytes,
                    files_scanned: report.files_scanned,
                    files_deleted: report.files_deleted,
                    quota_exceeded,
                    dry_run: false,
                },
            );
        });
    }

    pub(crate) async fn persist_server_marker(
        &self,
        wrapper: &xai_grok_sampling_types::ServerResponsesCheckpoint,
        replacement: &[ConversationItem],
        journal_operation_id: &str,
    ) -> Result<(), acp::Error> {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        use crate::session::persistence::DurableAppendError;

        let marker = crate::session::storage::responses_compaction::marker_for_wrapper(wrapper);
        let repairs =
            crate::session::storage::responses_compaction::tail_journal_repairs_for_replacement(
                journal_operation_id,
                replacement,
            )
            .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
        let updates = std::iter::once(XaiSessionUpdate::CompactionCheckpoint(Box::new(marker)))
            .chain(repairs.into_iter().flat_map(|(prepared, committed)| {
                [
                    XaiSessionUpdate::ConversationAppendPrepared(Box::new(prepared)),
                    XaiSessionUpdate::ConversationAppendCommitted(committed),
                ]
            }));
        for update in updates {
            match self.persist_xai_update_durable(update).await {
                Ok(()) => {}
                Err(DurableAppendError::Committed(error)) => {
                    tracing::warn!(%error, "Responses recovery record committed with bookkeeping error");
                }
                Err(error) => {
                    return Err(acp::Error::internal_error().data(format!(
                        "Responses compaction committed history but failed to persist its recovery journal: {error}"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_compaction_route_capabilities_fail_closed() {
        let xai = responses_route_capabilities("grok-4", "https://api.x.ai/v1");
        assert!(xai.supports_remote_compaction);
        assert!(xai.accepts_responses_checkpoint);

        for provider in crate::auth::providers::ProviderId::ALL {
            let route =
                responses_route_capabilities(&format!("{provider}/model"), "https://api.x.ai/v1");
            assert!(!route.supports_remote_compaction, "{provider}");
            assert!(!route.accepts_responses_checkpoint, "{provider}");
        }

        for (model, base_url) in [
            ("unknown/model", "https://api.x.ai/v1"),
            ("custom", "https://example.com/v1"),
        ] {
            let route = responses_route_capabilities(model, base_url);
            assert!(!route.supports_remote_compaction, "{model} {base_url}");
            assert!(!route.accepts_responses_checkpoint, "{model} {base_url}");
        }
    }
}

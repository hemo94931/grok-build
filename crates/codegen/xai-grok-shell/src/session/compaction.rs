//! Compaction methods for `SessionActor`.
//!
//! This module contains all compaction-related methods: manual `/compact`,
//! auto-compact threshold checks, inline auto-compact with auto-continue,
//! error-recovery compaction, preflight overflow detection, and checkpoint
//! persistence. These methods form a second `impl SessionActor` block that
//! lives alongside the primary one in `acp_session.rs`.
#[path = "compaction/responses.rs"]
mod responses;

pub(crate) use responses::CheckpointGateOutcome;
use responses::PreparedServerRequest;

use super::SessionActor;
use super::is_project_instructions;
use crate::remote::DEFAULT_CONTEXT_WINDOW;
use crate::session::compaction_config::{
    AsyncCompactionCache, SUPPRESS_AUTH, SUPPRESS_NONE, SUPPRESS_STICKY, SUPPRESS_TURN,
    SUPPRESS_UNTIL_SUCCESS,
};
use crate::session::helpers::CompactionStateContext;
use crate::session::helpers::compaction_context::CompactionInputs;
use crate::session::helpers::compaction_context::to_system_reminder;
use crate::session::helpers::session_compact::{
    CompactOutput, CompactionOutcome, build_compaction_chat_history,
    build_two_pass_compaction_prompt, generate_session_compact, is_context_length_error,
};
use crate::session::persistence::PersistenceMsg;
use crate::session::two_pass::{
    TWO_PASS_DEFAULT_SPLIT_FRACTION, build_two_pass_pass1_history, build_two_pass_pass2_history,
    note_for_two_pass_pass2, split_conversation_for_two_pass,
};
use agent_client_protocol as acp;
use std::sync::Arc;
use xai_chat_state::compaction_utils::{
    CompactedHistoryInput, CompactionAttempt, build_compacted_history, is_degenerate_summary,
    prepare_conversation_for_verbatim_summarization, sanitize_compacted_history,
    validate_compacted_history,
};
use xai_grok_sampling_types::{ApiBackend, ConversationItem, ConversationRequest};
/// Default percentage points below the auto-compact threshold at which prefire
/// (background pass-1) starts, giving pass-1 runway to finish before the limit.
/// Override with `GROK_PREFIRE_LEAD_PERCENT`.
const DEFAULT_PREFIRE_LEAD_PERCENT: u64 = 10;
fn prefire_lead_percent() -> u64 {
    std::env::var("GROK_PREFIRE_LEAD_PERCENT")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_PREFIRE_LEAD_PERCENT)
}
/// Cheap fingerprint of a conversation prefix for prefire NOTE₁ validity. A
/// mismatch means the prefix changed (edit / rewind / branch) since pass-1, so
/// the cached NOTE₁ no longer summarizes the current prefix and must be dropped.
fn fingerprint_prefix(items: &[ConversationItem]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    items.len().hash(&mut h);
    for it in items {
        let tag: u8 = match it {
            ConversationItem::System(_) => 0,
            ConversationItem::User(_) => 1,
            ConversationItem::Assistant(_) => 2,
            ConversationItem::ToolResult(_) => 3,
            ConversationItem::BackendToolCall(_) => 4,
            ConversationItem::Reasoning(_) => 5,
            ConversationItem::ResponsesCompactionCheckpoint(checkpoint) => {
                checkpoint.checkpoint_id.hash(&mut h);
                checkpoint.portable_history_sha256.hash(&mut h);
                6
            }
        };
        tag.hash(&mut h);
        it.text_content().hash(&mut h);
    }
    h.finish()
}

/// Prefire accepts normal history or one checkpoint at index zero. The latter
/// is expanded through [`SessionActor::portable_history_for_request`] before
/// splitting or fingerprinting; misplaced and duplicate wrappers fail closed.
fn prefire_layout_allows(items: &[ConversationItem]) -> bool {
    let checkpoint_count = items
        .iter()
        .filter(|item| item.is_responses_checkpoint())
        .count();
    checkpoint_count == 0
        || (checkpoint_count == 1
            && items
                .first()
                .is_some_and(ConversationItem::is_responses_checkpoint))
}

/// Return the committed token total only after both stages of remote shrink
/// validation have passed: the token seed exists because seed validation
/// succeeded, and adding the retained typed tail still shrinks the history.
/// A caller may discard speculative prefire state only on `Some`.
fn verified_remote_shrink_for_prefire_discard(
    validated_token_seed: Option<u64>,
    retained_tail_tokens: u64,
    pre_compaction_tokens: u64,
) -> Option<u64> {
    let committed_total = validated_token_seed?.saturating_add(retained_tail_tokens);
    (pre_compaction_tokens == 0 || committed_total < pre_compaction_tokens)
        .then_some(committed_total)
}
/// Outcome of a background prefire pass-1 run, recorded on the
/// `session.prefire_pass1` span as `compaction_prefire_outcome`.
/// [`PrefireOutcome::as_str`] values are stable telemetry keys
/// (telemetry/dashboards key off them) — don't rename the strings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrefireOutcome {
    Cached,
    Disabled,
    DebugFailPass1,
    TooSmall,
    EmptySplit,
    SampleFailed,
    EmptyNote1,
}
impl PrefireOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cached => "cached",
            Self::Disabled => "disabled",
            Self::DebugFailPass1 => "debug_fail_pass1",
            Self::TooSmall => "too_small",
            Self::EmptySplit => "empty_split",
            Self::SampleFailed => "sample_failed",
            Self::EmptyNote1 => "empty_note1",
        }
    }
}
/// Telemetry from one prefire pass-1 run; recorded onto the
/// `session.prefire_pass1` span by [`SessionActor::run_prefire_pass1`].
/// `None` fields = the run exited before that stage.
struct PrefirePass1Run {
    outcome: PrefireOutcome,
    prefix_len: Option<usize>,
    prefix_est_tokens: Option<u64>,
    pass1_latency_ms: Option<u64>,
    note1_chars: Option<usize>,
}
impl From<PrefireOutcome> for PrefirePass1Run {
    /// A run that exited before splitting/sampling — outcome only.
    fn from(outcome: PrefireOutcome) -> Self {
        Self {
            outcome,
            prefix_len: None,
            prefix_est_tokens: None,
            pass1_latency_ms: None,
            note1_chars: None,
        }
    }
}
#[cfg(test)]
mod two_pass_prefire_helper_tests {
    use super::{
        fingerprint_prefix, prefire_layout_allows, prefire_lead_percent,
        verified_remote_shrink_for_prefire_discard,
    };
    use xai_grok_sampler::ResponsesCompactResponse;
    use xai_grok_sampling_types::{
        CheckpointIdentity, ConversationItem, RESPONSES_COMPACTION_CONTRACT,
        ResponsesCompactionMode, ServerResponsesCheckpoint, TokenSeedSource,
    };

    fn checkpoint_item() -> ConversationItem {
        ConversationItem::ResponsesCompactionCheckpoint(Box::new(ServerResponsesCheckpoint {
            checkpoint_id: "checkpoint-current".into(),
            operation_id: "operation-current".into(),
            prompt_index: 2,
            created_at: chrono::Utc::now(),
            auto_continue: false,
            mode: ResponsesCompactionMode {
                name: "default".into(),
                detail: None,
            },
            branch_id: "branch-current".into(),
            identity: CheckpointIdentity {
                provider_id: "xai".into(),
                api: "responses".into(),
                endpoint_fingerprint: "endpoint".into(),
                model: "grok-test".into(),
                auth_principal_fingerprint: "principal".into(),
                contract_version: RESPONSES_COMPACTION_CONTRACT.into(),
                prompt_envelope_fingerprint: "envelope".into(),
                base_instructions_sha256: "base".into(),
                prior_checkpoint_id: None,
                cache_route_fingerprint: None,
            },
            output: vec![serde_json::json!({
                "type": "compaction",
                "encrypted_content": "opaque"
            })],
            portable_history_path: "compaction_checkpoints/checkpoint-current.json".into(),
            portable_history_sha256: "portable-digest".into(),
            portable_history_bytes: 10,
            checkpoint_token_seed: 25,
            token_seed_source: TokenSeedSource::UsageOutputTokens,
            server_output_item_count: 1,
            prior_checkpoint_id: None,
            memory_revision: None,
        }))
    }

    #[test]
    fn fingerprint_stable_for_same_prefix() {
        let items = vec![
            ConversationItem::system("sys"),
            ConversationItem::user("hello"),
            ConversationItem::assistant("hi"),
        ];
        assert_eq!(fingerprint_prefix(&items), fingerprint_prefix(&items));
    }

    #[test]
    fn fingerprint_changes_when_prefix_content_changes() {
        let base = vec![
            ConversationItem::system("sys"),
            ConversationItem::user("hello"),
        ];
        let edited = vec![
            ConversationItem::system("sys"),
            ConversationItem::user("HELLO there"), // a real edit/rewind of the prefix
        ];
        assert_ne!(
            fingerprint_prefix(&base),
            fingerprint_prefix(&edited),
            "a changed prefix must invalidate the cached NOTE1 fingerprint"
        );
    }

    #[test]
    fn fingerprint_changes_with_length() {
        let short = vec![ConversationItem::user("a")];
        let long = vec![
            ConversationItem::user("a"),
            ConversationItem::assistant("b"),
        ];
        assert_ne!(fingerprint_prefix(&short), fingerprint_prefix(&long));
    }

    #[test]
    fn prefire_layout_accepts_normal_and_unique_leading_checkpoint_only() {
        assert!(prefire_layout_allows(&[
            ConversationItem::user("normal"),
            ConversationItem::assistant("history"),
        ]));

        let checkpoint = checkpoint_item();
        assert!(prefire_layout_allows(&[
            checkpoint.clone(),
            ConversationItem::user("typed tail"),
        ]));
        assert!(!prefire_layout_allows(&[
            ConversationItem::user("misplaced"),
            checkpoint.clone(),
        ]));
        assert!(!prefire_layout_allows(&[checkpoint.clone(), checkpoint,]));
    }

    #[test]
    fn prefire_discard_requires_seed_and_committed_total_to_shrink() {
        let response = ResponsesCompactResponse {
            output: vec![serde_json::json!({
                "type": "compaction",
                "encrypted_content": "opaque"
            })],
            usage_output_tokens: Some(95),
            usage_total_tokens: None,
            response_bytes: 32,
            attempts: 1,
        };
        let rejected_seed =
            crate::session::responses_server_compaction::server_checkpoint_token_seed(
                &response, 5, 100,
            )
            .ok()
            .map(|(seed, _)| seed);
        assert_eq!(rejected_seed, None, "DidNotShrink cannot discard prefire");
        assert_eq!(
            verified_remote_shrink_for_prefire_discard(rejected_seed, 0, 100),
            None
        );
        assert_eq!(
            verified_remote_shrink_for_prefire_discard(Some(80), 20, 100),
            None,
            "a retained tail that erases the shrink must keep prefire"
        );
        assert_eq!(
            verified_remote_shrink_for_prefire_discard(Some(80), 10, 100),
            Some(90)
        );
    }

    #[test]
    fn prefire_lead_percent_defaults_to_10() {
        unsafe { std::env::remove_var("GROK_PREFIRE_LEAD_PERCENT") };
        assert_eq!(prefire_lead_percent(), 10);
    }
}
impl SessionActor {
    /// Two-pass active for this session: flag resolved on at build AND not an
    /// agent that keeps its single short self-summary.
    pub(crate) fn two_pass_active(&self) -> bool {
        let agent = self.agent.borrow();
        agent.compaction_policy().two_pass_enabled
    }
    async fn prepare_builtin_compaction_sampling(
        &self,
    ) -> Result<
        (
            xai_grok_sampler::SamplerConfig,
            xai_grok_sampler::SamplingClient,
        ),
        acp::Error,
    > {
        self.refresh_token_if_expired().await;
        let current = self.reconstruct_full_config().await;
        let compact_model = self
            .agent
            .borrow()
            .compaction_policy()
            .compact_model
            .clone();
        let mut config = compact_model
            .as_deref()
            .filter(|model| *model != current.model)
            .and_then(|model| self.models_manager.sampling_config_for_model_id(model))
            .unwrap_or_else(|| current.clone());
        config.origin_client = current.origin_client.clone();
        config.client_identifier = current.client_identifier.clone();
        config.attribution_callback = current.attribution_callback.clone();
        config.header_injector = current.header_injector.clone();
        if config.auth_scheme == xai_grok_sampler::AuthScheme::Bearer
            && config.api_key == current.api_key
        {
            config.bearer_resolver = current.bearer_resolver.clone();
        }
        config.compactions_remaining = None;
        config.compaction_at_tokens = None;
        config.doom_loop_recovery = None;
        let client = xai_grok_sampler::SamplingClient::new(config.clone())
            .map_err(|error| self.to_acp_error(error))?;
        Ok((config, client))
    }

    /// Run one summarization sample over a fully-built two-pass history (the
    /// prompt is already embedded, so this bypasses the single-pass sampler and
    /// calls `generate_session_compact` directly). Returns `None` on any error
    /// so callers fall back to single-pass.
    ///
    /// Agent `RefCell` borrows are only taken for synchronous snapshots (never
    /// held across `.await`). Prefire is `spawn_local` on the same LocalSet as
    /// the turn loop; a long-lived borrow would race with turn/compact/cancel
    /// and panic on double-borrow.
    async fn two_pass_sample(&self, history: Vec<ConversationItem>) -> Option<CompactOutput> {
        let (sampling_config, client) = match self.prepare_builtin_compaction_sampling().await {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!(error = %e, "two_pass: failed to prepare sampling client");
                return None;
            }
        };
        let tool_defs = self.prepare_tool_definitions().await;
        let tools = self.turn_base_tool_specs(&tool_defs);
        let wall_clock_budget_secs = self
            .agent
            .borrow()
            .compaction_policy()
            .wall_clock_budget_secs;
        let hosted_tools = self.hosted_tools_for_turn();
        let (cancel, _cancel_scope) = self.compaction.cancel.enter();
        match generate_session_compact(
            history,
            tools,
            hosted_tools,
            client,
            self.session_info.id.clone(),
            &sampling_config,
            self.inference_idle_timeout,
            wall_clock_budget_secs,
            self.compaction.tool_choice,
            &cancel,
        )
        .await
        {
            Ok(out) => Some(out),
            Err(e) => {
                tracing::warn!(error = ?e, "two_pass: summarization sample failed");
                None
            }
        }
    }
    /// Per-turn prefire decision: usage has reached `threshold - lead` (so there
    /// is still runway before the hard auto-compact line at `threshold`).
    pub(crate) async fn should_prefire_two_pass(&self) -> bool {
        let conversation = self.chat_state_handle.get_conversation().await;
        if !prefire_layout_allows(&conversation) {
            return false;
        }
        let sampling_cfg = self.chat_state_handle.get_sampling_config().await;
        let Some(cw) = sampling_cfg.as_ref().map(|c| c.context_window.get()) else {
            return false;
        };
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        let threshold = self.compaction.threshold_percent.get() as u64;
        let start_pct = threshold.saturating_sub(prefire_lead_percent());
        xai_token_estimation::exceeds_threshold(estimated_total, cw, start_pct as u8)
    }
    /// Background pass-1: summarize the ~95% prefix → NOTE₁ and cache it for a
    /// later pass-2 apply. Always releases the in-flight guard. Spawned via
    /// `spawn_local` from the turn loop; reads a conversation snapshot and does
    /// not mutate session state. The span makes speculative pass-1 spend
    /// measurable (hit rate, wasted input tokens) ahead of the fleet-wide ramp.
    #[tracing::instrument(
        name = "session.prefire_pass1",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            compaction_prefire_outcome = tracing::field::Empty,
            compaction_pass1_latency_ms = tracing::field::Empty,
            compaction_prefire_prefix_len = tracing::field::Empty,
            compaction_prefire_prefix_est_tokens = tracing::field::Empty,
            compaction_prefire_note1_chars = tracing::field::Empty,
        )
    )]
    pub(crate) async fn run_prefire_pass1(self: &Arc<Self>) {
        struct InFlightGuard<'a>(&'a crate::session::compaction_config::PrefireState);
        impl Drop for InFlightGuard<'_> {
            fn drop(&mut self) {
                self.0.finish();
            }
        }
        let _guard = InFlightGuard(&self.compaction.prefire);
        let run = self.run_prefire_pass1_inner().await;
        let span = tracing::Span::current();
        span.record("compaction_prefire_outcome", run.outcome.as_str());
        if let Some(v) = run.prefix_len {
            span.record("compaction_prefire_prefix_len", v as i64);
        }
        if let Some(v) = run.prefix_est_tokens {
            span.record("compaction_prefire_prefix_est_tokens", v as i64);
        }
        if let Some(v) = run.pass1_latency_ms {
            span.record("compaction_pass1_latency_ms", v as i64);
        }
        if let Some(v) = run.note1_chars {
            span.record("compaction_prefire_note1_chars", v as i64);
        }
    }
    async fn run_prefire_pass1_inner(self: &Arc<Self>) -> PrefirePass1Run {
        if !self.two_pass_active() {
            return PrefireOutcome::Disabled.into();
        }
        if std::env::var("GROK_DEBUG_TWO_PASS_FAIL_PASS1")
            .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
        {
            tracing::info!(
                target: "two_pass",
                "two_pass: DEBUG GROK_DEBUG_TWO_PASS_FAIL_PASS1 — prefire pass1 produces no cache"
            );
            return PrefireOutcome::DebugFailPass1.into();
        }
        let conversation = self.chat_state_handle.get_conversation().await;
        // A live server checkpoint must be expanded through the same
        // checkpoint-aware resolver as every other reader; otherwise the
        // wrapper would leak into the pass-1 sampling request.
        let conversation = match self.portable_history_for_request(&conversation) {
            Ok(conversation) => conversation,
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "two_pass: prefire skipped — checkpoint history unavailable"
                );
                return PrefireOutcome::SampleFailed.into();
            }
        };
        if conversation.len() < 4 {
            return PrefireOutcome::TooSmall.into();
        }
        let split = split_conversation_for_two_pass(&conversation, TWO_PASS_DEFAULT_SPLIT_FRACTION);
        if split.prefix.is_empty() || split.tail.is_empty() {
            return PrefireOutcome::EmptySplit.into();
        }
        let sampling_cfg = self.chat_state_handle.get_sampling_config().await;
        let strips = sampling_cfg
            .as_ref()
            .map(|c| c.api_backend == ApiBackend::Messages)
            .unwrap_or(false);
        let current_model = sampling_cfg
            .as_ref()
            .map(|config| config.model.clone())
            .unwrap_or_default();
        let model_slug = self
            .agent
            .borrow()
            .compaction_policy()
            .compact_model
            .clone()
            .unwrap_or(current_model);
        let prefix_prepared =
            prepare_conversation_for_verbatim_summarization(split.prefix.to_vec(), strips);
        let prefix_est_tokens = prefix_prepared
            .iter()
            .map(xai_chat_state::estimate_item_tokens)
            .sum::<u64>();
        let prompt = build_two_pass_compaction_prompt(None);
        let pass1_history = build_two_pass_pass1_history(&prefix_prepared, &prompt);
        let started = std::time::Instant::now();
        let cancellation = super::tasks_cancel::current_turn_cancellation();
        let out = tokio::select! {
            biased;
            _ = cancellation.cancelled() => None,
            out = self.two_pass_sample(pass1_history) => out,
        };
        if cancellation.is_cancelled() {
            return PrefireOutcome::SampleFailed.into();
        }
        let pass1_latency_ms = started.elapsed().as_millis() as u64;
        let attempted = |outcome: PrefireOutcome, note1_chars: Option<usize>| PrefirePass1Run {
            outcome,
            prefix_len: Some(split.split_idx),
            prefix_est_tokens: Some(prefix_est_tokens),
            pass1_latency_ms: Some(pass1_latency_ms),
            note1_chars,
        };
        let Some(out) = out else {
            return attempted(PrefireOutcome::SampleFailed, None);
        };
        let note1 = note_for_two_pass_pass2(&out.content);
        if note1.trim().is_empty() {
            return attempted(PrefireOutcome::EmptyNote1, None);
        }
        let note1_chars = note1.chars().count();
        let cache = AsyncCompactionCache {
            note1,
            prefix_len: split.split_idx,
            fingerprint: fingerprint_prefix(&conversation[..split.split_idx]),
            model_slug,
            pass1_latency_ms,
        };
        tracing::info!(
            target: "two_pass",
            prefix_len = cache.prefix_len,
            pass1_latency_ms = cache.pass1_latency_ms,
            "two_pass: prefire pass1 cached NOTE1"
        );
        self.compaction.prefire.store(cache);
        attempted(PrefireOutcome::Cached, Some(note1_chars))
    }
    /// Pass-2 apply: if a valid cached NOTE₁ exists for the current conversation,
    /// summarize (NOTE₁ + recent tail + special prompt) → final summary and
    /// return its `CompactOutput`. `None` → caller runs the single-pass path.
    ///
    /// **telemetry / `session.compact_inner` latency:** the returned `CompactOutput`
    /// stream timings are what land on `compaction_ttft_ms` /
    /// `compaction_stream_ms`. Those reflect **user-visible sync wait only**:
    /// - background pass-1 that already finished before compact is *not*
    ///   included (prefire hid that cost);
    /// - if pass-1 is still in flight we **do** add that await into
    ///   `ttft_ms` (time until first token of the final summary), because the
    ///   user is blocked on it;
    /// - `stream_ms` / `delta_count` / `itl_max_ms` are always pass-2 only
    ///   (the only sample that streams the successor-visible summary).
    async fn try_two_pass_pass2_apply(
        &self,
        user_context: Option<&str>,
        strips_reasoning: bool,
    ) -> Option<CompactOutput> {
        if !self.two_pass_active() {
            return None;
        }
        let cancellation = super::tasks_cancel::current_turn_cancellation();
        let mut prefire_waited_ms = 0u64;
        if let Some(mut handle) = self.compaction.prefire.take_handle() {
            let was_in_flight = self.compaction.prefire.is_in_flight();
            let waited = std::time::Instant::now();
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    handle.abort();
                    let _ = handle.await;
                    self.compaction.prefire.clear();
                    self.compaction.prefire.finish();
                    return None;
                }
                _ = &mut handle => {}
            }
            if was_in_flight {
                prefire_waited_ms = waited.elapsed().as_millis() as u64;
                tracing::Span::current()
                    .record("compaction_prefire_waited_ms", prefire_waited_ms as i64);
                tracing::info!(
                    target: "two_pass",
                    wait_ms = prefire_waited_ms,
                    "two_pass: waited for in-flight prefire pass1 before pass2"
                );
            }
        }
        if cancellation.is_cancelled() {
            return None;
        }
        let cache = self.compaction.prefire.take()?;
        let live = self.chat_state_handle.get_conversation().await;
        // Match prefire's exact provider-visible source. A checkpoint wrapper
        // is local metadata and must be expanded before length/fingerprint
        // validation and before building the builtin pass-2 request.
        let live = self.portable_history_for_request(&live).ok()?;
        let current_model = self
            .chat_state_handle
            .get_sampling_config()
            .await
            .map(|config| config.model)
            .unwrap_or_default();
        let model_slug = self
            .agent
            .borrow()
            .compaction_policy()
            .compact_model
            .clone()
            .unwrap_or(current_model);
        if cache.prefix_len == 0
            || cache.prefix_len > live.len()
            || cache.model_slug != model_slug
            || fingerprint_prefix(&live[..cache.prefix_len]) != cache.fingerprint
        {
            tracing::Span::current().record("compaction_prefire_stale", true);
            tracing::info!(
                target: "two_pass",
                "two_pass: cached NOTE1 stale or model changed; falling back to single-pass"
            );
            return None;
        }
        let prefix = &live[..cache.prefix_len];
        let tail = &live[cache.prefix_len..];
        let prepared_tail =
            prepare_conversation_for_verbatim_summarization(tail.to_vec(), strips_reasoning);
        let prompt = build_two_pass_compaction_prompt(user_context);
        let pass2_history =
            build_two_pass_pass2_history(prefix, &prepared_tail, &cache.note1, &prompt);
        let started = std::time::Instant::now();
        let mut out = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return None,
            out = self.two_pass_sample(pass2_history) => out?,
        };
        if cancellation.is_cancelled() {
            return None;
        }
        if is_degenerate_summary(&out.content) {
            tracing::Span::current().record("compaction_prefire_stale", true);
            tracing::info!(
                target: "two_pass",
                "two_pass: pass2 summary empty/degenerate; falling back to single-pass"
            );
            return None;
        }
        let pass2_latency_ms = started.elapsed().as_millis() as u64;
        if prefire_waited_ms > 0 {
            out.ttft_ms = Some(out.ttft_ms.unwrap_or(0).saturating_add(prefire_waited_ms));
        }
        let span = tracing::Span::current();
        span.record("compaction_two_pass_used", true);
        span.record("compaction_prefire_hit", true);
        span.record("compaction_pass2_latency_ms", pass2_latency_ms as i64);
        tracing::info!(
            target: "two_pass",
            prefix_len = cache.prefix_len,
            tail_len = tail.len(),
            prefire_waited_ms,
            pass2_latency_ms,
            pass1_bg_latency_ms = cache.pass1_latency_ms,
            "two_pass: pass2 applied cached NOTE1 (prefire hit)"
        );
        Some(out)
    }
}
/// Trigger info for auto-compact decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompactionStrategy {
    ServerFirst,
    BuiltinMigration(&'static str),
}

enum CompactionAttemptOutcome {
    Committed,
    Superseded {
        compaction_id: String,
        strategy_started_notified: bool,
    },
}

/// Trigger info for auto-compact decisions.
pub(crate) struct AutoCompactTriggerInfo {
    pub tokens_used: u64,
    pub context_window: u64,
    pub percentage: u8,
}
/// Why auto-compaction was suppressed after a deterministic failure.
/// [`SuppressReason::as_str`] is a stable telemetry value (BQ/OTLP/dashboards key
/// off it) — don't rename the strings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SuppressReason {
    CreditBlock,
    Size,
    Auth,
    Schema,
    Other,
}
impl SuppressReason {
    fn as_str(self) -> &'static str {
        match self {
            SuppressReason::CreditBlock => "credit_block",
            SuppressReason::Size => "size",
            SuppressReason::Auth => "auth",
            SuppressReason::Schema => "schema",
            SuppressReason::Other => "other",
        }
    }
    /// Suppression scope for this reason:
    /// - `size | schema` → [`SUPPRESS_STICKY`]: cleared only on a context-budget change.
    /// - `credit_block` → [`SUPPRESS_UNTIL_SUCCESS`]: wait for a model `200`.
    /// - `auth` → [`SUPPRESS_AUTH`]: clear on login/token refresh (not 200 — over-window deadlock).
    /// - `other` → [`SUPPRESS_TURN`]: optimistic per-turn retry.
    fn suppress_state(self) -> u8 {
        match self {
            SuppressReason::Size | SuppressReason::Schema => SUPPRESS_STICKY,
            SuppressReason::CreditBlock => SUPPRESS_UNTIL_SUCCESS,
            SuppressReason::Auth => SUPPRESS_AUTH,
            SuppressReason::Other => SUPPRESS_TURN,
        }
    }
}
/// Splice the preserved prefix (`conversation[0..prefix_len]`) onto the compacted
/// suffix, dropping the suffix's leading System and — if the prefix already has an
/// AGENTS.md item — its re-injected AGENTS.md too (else the model sees it twice).
/// Returns `Err(compacted_history)` unchanged when `prefix_len` is 0 or out of range.
fn preserve_inherited_prefix(
    conversation: &[ConversationItem],
    compacted_history: Vec<ConversationItem>,
    prefix_len: usize,
) -> Result<Vec<ConversationItem>, Vec<ConversationItem>> {
    if prefix_len == 0 || prefix_len > conversation.len() {
        return Err(compacted_history);
    }
    let inherited = &conversation[..prefix_len];
    let drop_reinjected_agents_md = inherited.iter().any(is_project_instructions);
    let mut preserved = inherited.to_vec();
    let child_items = compacted_history
        .into_iter()
        .skip_while(|i| matches!(i, ConversationItem::System(_)))
        .filter(|i| !(drop_reinjected_agents_md && is_project_instructions(i)));
    preserved.extend(child_items);
    Ok(preserved)
}
/// Project the token count a re-pinned (preserved) history would reseed to, so the
/// release decision compares against the same threshold the auto-compact trigger
/// applies next turn. This only APPROXIMATES the compaction reseed
/// (`xai-chat-state` `replace_conversation`, the authority): it matches the reseed's
/// round-and-cap but divides by the current conversation estimate, not the reseed's
/// frozen `estimate_at_last_response`. The conversation only grows, so the current
/// estimate is >= that frozen value; this therefore under-estimates the reseed (a
/// lower bound) and can lean toward preserve. That never re-loops: the post-replace
/// `exceeds_threshold` check on the real reseeded total still sets sticky Size
/// suppression if a preserve leaves the fork over budget.
fn project_preserved_reseed_tokens(
    preserved_estimate: u64,
    tokens_before: u64,
    full_conv_estimate: u64,
) -> u64 {
    let ratio = tokens_before as f64 / full_conv_estimate.max(1) as f64;
    ((preserved_estimate as f64 * ratio).round() as u64).min(tokens_before)
}
impl SessionActor {
    /// Path to the raw `updates.jsonl` transcript if it exists, else `None`.
    /// `pub(crate)` so the `Transcript`-mode dispatch in `compaction_segments`
    /// and transcript-location pointers can both reuse it.
    ///
    /// The `path.exists()` guard keeps the pointer safe when a session (e.g. a
    /// nested sub-agent) never wrote one -- the hint is simply omitted rather
    /// than dangling.
    pub(crate) fn get_transcript_path(&self) -> Option<String> {
        let path =
            crate::session::persistence::session_dir(&self.session_info).join("updates.jsonl");
        if path.exists() {
            Some(path.to_string_lossy().into_owned())
        } else {
            None
        }
    }
    /// Increment the compaction counter and launch a pre-compaction memory flush.
    ///
    /// The counter is incremented before the flush check so the once-per-cycle
    /// guard does not suppress the first eligible flush.
    async fn maybe_pre_compaction_flush(
        self: &Arc<Self>,
        total_tokens: u64,
        context_window: u64,
        trigger: &'static str,
    ) {
        let compaction_count = self
            .compaction
            .count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if !self.agent.borrow().compaction_policy().memory_flush_enabled {
            return;
        }
        let last_flush = self
            .memory
            .last_flush_compaction
            .load(std::sync::atomic::Ordering::Relaxed);
        if crate::session::helpers::memory_flush::should_flush(
            total_tokens,
            context_window,
            self.compaction.threshold_percent.get(),
            &self.memory.flush_config,
            last_flush,
            compaction_count,
        ) {
            let snapshot = match self.snapshot_memory_flush_state().await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    tracing::warn!(%error, "memory flush skipped: checkpoint history unavailable");
                    return;
                }
            };
            tokio::task::spawn_local({
                let session = self.clone();
                async move {
                    if session.run_memory_flush(trigger, Some(snapshot)).await {
                        session
                            .memory
                            .last_flush_compaction
                            .store(compaction_count, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    }
    /// Tag the current `session.compact` span with `mode` (and `detail`, for
    /// `segments`) — the A/B variant key for grouping outcomes in telemetry.
    fn record_compaction_variant(&self) {
        let mode = self.compaction.compaction_mode;
        let span = tracing::Span::current();
        span.record("mode", tracing::field::display(mode));
        if let Some(detail) = mode.segment_detail() {
            span.record("detail", tracing::field::display(detail));
        }
    }
    /// Runs the compact operation over here which compresses the current conversation
    /// and helps with saving the context for the model
    #[tracing::instrument(
        name = "session.compact",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            trigger = "manual",
            mode = tracing::field::Empty,
            detail = tracing::field::Empty,
            pre_tokens = tracing::field::Empty,
            post_tokens = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    pub(crate) async fn run_compact(
        self: &Arc<Self>,
        user_context: Option<String>,
    ) -> Result<(), acp::Error> {
        let (_cancel, _cancel_scope) = self.compaction.cancel.enter();
        self.record_compaction_variant();
        let total_tokens = self.chat_state_handle.get_total_tokens().await;
        tracing::Span::current().record("pre_tokens", total_tokens as i64);
        let sampling_config = self.chat_state_handle.get_sampling_config().await;
        let context_window = sampling_config
            .as_ref()
            .map(|c| c.context_window.get())
            .unwrap_or(DEFAULT_CONTEXT_WINDOW);
        self.maybe_pre_compaction_flush(total_tokens, context_window, "pre_compaction")
            .await;
        if let Err(e) = self
            .run_compact_inner(
                user_context,
                None,
                xai_grok_telemetry::events::CompactionTrigger::Manual,
                CompactionStrategy::ServerFirst,
                None,
                None,
                false,
                0,
            )
            .await
        {
            let span = tracing::Span::current();
            span.record("success", false);
            span.record("error", e.to_string().as_str());
            return Err(e);
        }
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        let tokens_after = self.chat_state_handle.get_total_tokens().await;
        let span = tracing::Span::current();
        span.record("post_tokens", tokens_after as i64);
        span.record("success", true);
        self.send_xai_notification(XaiSessionUpdate::AutoCompactCompleted {
            tokens_before: Some(total_tokens),
            tokens_after,
            elapsed_ms: None,
            summary_preview: None,
        })
        .await;
        Ok(())
    }
    async fn notify_compact_cancelled(&self, auto_trigger: bool) {
        if auto_trigger {
            use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
            self.send_xai_notification(XaiSessionUpdate::AutoCompactCancelled {
                reason: crate::extensions::notification::AutoCompactCancelReason::UserCancelled,
            })
            .await;
        }
    }

    async fn emit_compact_cancelled(&self, auto_trigger: bool) -> Result<(), acp::Error> {
        self.notify_compact_cancelled(auto_trigger).await;
        Err(crate::session::helpers::session_compact::CompactFailure::cancelled_error())
    }
    /// Suppress AUTO compaction after a deterministic failure. Scope depends on
    /// the reason (see [`SuppressReason::suppress_state`]): size/schema sticky,
    /// credit until 200, auth until credentials recover, other clears next turn.
    /// Telemetry + one notification per transition; manual `/compact` exempt.
    async fn suppress_auto_compaction(
        &self,
        reason: SuppressReason,
        estimated_tokens: u64,
        context_window: u64,
    ) {
        let new_state = reason.suppress_state();
        if self
            .compaction
            .auto_compact_suppressed
            .compare_exchange(
                SUPPRESS_NONE,
                new_state,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            tracing::warn!(
                suppress_reason = reason.as_str(),
                estimated_tokens,
                context_window,
                "auto-compaction suppressed after deterministic compaction failure"
            );
            xai_grok_telemetry::session_ctx::log_event(
                xai_grok_telemetry::events::AutoCompactSuppressed {
                    reason: reason.as_str(),
                    estimated_tokens,
                    context_window,
                },
            );
            let message = match reason {
                SuppressReason::CreditBlock => {
                    "out of credits or over your spending limit. Add credits and retry."
                }
                SuppressReason::Auth => {
                    "authentication problem — re-authenticate using /login and retry."
                }
                SuppressReason::Size => "this conversation is too large to compact.",
                SuppressReason::Schema => "this conversation can't be summarized.",
                SuppressReason::Other => {
                    "it'll retry on the next turn, or start a new session using /new."
                }
            };
            self.send_xai_notification(
                crate::extensions::notification::SessionUpdate::AutoCompactFailed {
                    error: message.to_string(),
                },
            )
            .await;
        }
    }
    pub(crate) fn is_compaction_cancelled(error: &acp::Error) -> bool {
        error
            .data
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .is_some_and(|message| {
                message == "responses_compaction_cancelled"
                    || message
                        .contains(crate::session::helpers::session_compact::COMPACT_CANCELLED_MSG)
            })
    }

    /// Map a deterministic failure's error text to a fixed, content-free
    /// [`SuppressReason`] (drives telemetry + sticky-vs-per-turn scope).
    fn classify_suppress_reason(error_msg: &str) -> SuppressReason {
        let m = error_msg.to_ascii_lowercase();
        if m.contains("spending-limit")
            || m.contains("spending limit")
            || m.contains("out of credits")
            || m.contains("usage balance exhausted")
            || m.contains("usage limit reached")
        {
            SuppressReason::CreditBlock
        } else if is_context_length_error(&m) {
            SuppressReason::Size
        } else if m.contains("status 401") || m.contains("unauthorized") {
            SuppressReason::Auth
        } else if m.contains("invalid_request_error") {
            SuppressReason::Schema
        } else {
            SuppressReason::Other
        }
    }
    /// ACP error payload string (plain string or `{message, ...}`).
    fn acp_error_message(err: &acp::Error) -> String {
        match err.data.as_ref() {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(obj) => obj
                .get("message")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| obj.to_string()),
            None => err.message.clone(),
        }
    }
    /// Auth/401 compact failure — abort for reauth resubmit; don't sample oversized.
    pub(crate) fn is_auth_compact_error(err: &acp::Error) -> bool {
        matches!(
            Self::classify_suppress_reason(&Self::acp_error_message(err)),
            SuppressReason::Auth
        )
    }
    /// Terminal auth compact failure: emit RetryState auth (reauth stash) + auth_required.
    /// Separate from `AutoCompactFailed` (user-facing); this aborts the turn.
    pub(crate) async fn surface_compact_auth_failure(&self, err: acp::Error) -> acp::Error {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        let detailed = Self::acp_error_message(&err);
        let message = if detailed.to_ascii_lowercase().contains("unauthorized") {
            detailed
        } else {
            format!(
                "Unauthorized (401): compaction failed — re-authenticate with /login \
                 and retry. ({detailed})"
            )
        };
        tracing::warn!(
            session_id = %self.session_info.id.0,
            error = %message,
            "auto-compact auth failure: aborting turn for re-auth"
        );
        xai_grok_telemetry::unified_log::warn(
            "auto-compact auth failure: aborting turn for re-auth",
            Some(self.session_info.id.0.as_ref()),
            Some(serde_json::json!({
                "message": crate::util::truncate(&message, 300),
            })),
        );
        self.send_xai_notification(XaiSessionUpdate::RetryState(
            crate::extensions::notification::RetryState::Failed {
                error_type: "auth".to_string(),
                message: message.clone(),
            },
        ))
        .await;
        acp::Error::auth_required().data(crate::sampling::error::terminal_error_data(
            message,
            Some(401),
            xai_grok_sampler::SamplingErrorKind::Auth,
        ))
    }
    /// Clear [`SUPPRESS_AUTH`] on login/token refresh (credit suppress waits for a 200).
    pub(crate) fn clear_auth_compact_suppression(&self) {
        let _ = self.compaction.auto_compact_suppressed.compare_exchange(
            SUPPRESS_AUTH,
            SUPPRESS_NONE,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    /// Credit or auth suppress — a model switch cannot clear these.
    fn is_account_state_suppressed(&self) -> bool {
        matches!(
            self.compaction
                .auto_compact_suppressed
                .load(std::sync::atomic::Ordering::Relaxed),
            SUPPRESS_UNTIL_SUCCESS | SUPPRESS_AUTH
        )
    }
    /// Choose the post-compaction history for a forked session: re-pin the inherited
    /// prefix, or release it (fall back to the self-contained summary the summarizer
    /// already built from the whole conversation) when re-pinning would leave the fork
    /// at/over the auto-compact threshold. On release, sets the sticky flag and records
    /// the release span field (this runs within the `run_compact_inner` span).
    ///
    /// This runtime release compensates for a verbatim mirror-fork that pinned its whole
    /// parent transcript; bounding the inherited prefix at fork admission is the
    /// structural alternative that would remove this path.
    async fn resolve_forked_compacted_history(
        &self,
        compacted_history: Vec<ConversationItem>,
        prefix_len: usize,
        tokens_before: u64,
        context_window: u64,
    ) -> Vec<ConversationItem> {
        let full_conv = self.chat_state_handle.get_conversation().await;
        let compacted_len = compacted_history.len();
        let release_candidate = compacted_history.clone();
        match preserve_inherited_prefix(&full_conv, compacted_history, prefix_len) {
            Ok(preserved) => {
                let projected_preserved = project_preserved_reseed_tokens(
                    xai_chat_state::estimate_conversation_tokens(&preserved),
                    tokens_before,
                    xai_chat_state::estimate_conversation_tokens(&full_conv),
                );
                if xai_token_estimation::exceeds_threshold(
                    projected_preserved,
                    context_window,
                    self.compaction.threshold_percent.get(),
                ) {
                    self.compaction
                        .prefix_released
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::Span::current().record("compaction_prefix_released", true);
                    tracing::info!(
                        session_id = %self.session_info.id.0,
                        prefix_len,
                        projected_preserved,
                        "compaction: releasing inherited prefix under pressure"
                    );
                    release_candidate
                } else {
                    tracing::info!(
                        session_id = %self.session_info.id.0,
                        prefix_len,
                        compacted_len,
                        "Preserving inherited prefix across compaction"
                    );
                    preserved
                }
            }
            Err(original) => {
                tracing::warn!(
                    session_id = %self.session_info.id.0,
                    prefix_len,
                    conversation_len = full_conv.len(),
                    "Inherited prefix invalid, using compacted history as-is"
                );
                original
            }
        }
    }
    async fn discard_prefire(&self) {
        if let Some(handle) = self.compaction.prefire.take_handle() {
            handle.abort();
            let _ = handle.await;
        }
        self.compaction.prefire.clear();
        self.compaction.prefire.finish();
    }

    async fn commit_compaction_replacement(
        &self,
        operation_id: String,
        history_revision: u64,
        request_identity_generation: u64,
        replacement: Vec<ConversationItem>,
        committed_total_tokens: u64,
    ) -> Result<bool, acp::Error> {
        // Stage-D3 observability: a commit that installs a Responses
        // checkpoint marks the next usage record as `post_compact_first`.
        let installs_checkpoint = replacement
            .first()
            .is_some_and(|item| item.is_responses_checkpoint());
        let result = self
            .chat_state_handle
            .commit_compaction(xai_chat_state::CommitCompaction {
                operation_id,
                expected_history_revision: history_revision,
                expected_request_identity_generation: request_identity_generation,
                replacement,
                committed_total_tokens,
            })
            .await
            .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?;
        match result {
            xai_chat_state::CommitCompactionResult::Committed { .. } => {
                // 2 = first post-compact usage pending; 0 = the commit
                // removed the checkpoint (builtin / migration).
                self.post_compact_usage_state.store(
                    if installs_checkpoint { 2 } else { 0 },
                    std::sync::atomic::Ordering::Relaxed,
                );
                Ok(true)
            }
            xai_chat_state::CommitCompactionResult::Superseded { .. } => Ok(false),
            xai_chat_state::CommitCompactionResult::PersistenceFailed(error) => {
                Err(acp::Error::internal_error().data(error.to_string()))
            }
        }
    }

    async fn finish_committed_compaction(
        &self,
        new_len: usize,
        context_window: u64,
        compact_source: &str,
    ) -> u64 {
        if self.startup_hints.inherited_prefix_len.is_some() {
            let post_replace_tokens = self.chat_state_handle.get_total_tokens().await;
            if xai_token_estimation::exceeds_threshold(
                post_replace_tokens,
                context_window,
                self.compaction.threshold_percent.get(),
            ) {
                self.compaction
                    .auto_compact_suppressed
                    .store(SUPPRESS_STICKY, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    session_id = %self.session_info.id.0,
                    post_replace_tokens,
                    context_window,
                    "compaction: released history still over threshold; suppressing AUTO to avoid a re-loop"
                );
            } else {
                self.compaction
                    .auto_compact_suppressed
                    .store(SUPPRESS_NONE, std::sync::atomic::Ordering::Relaxed);
            }
        } else {
            self.compaction
                .auto_compact_suppressed
                .store(SUPPRESS_NONE, std::sync::atomic::Ordering::Relaxed);
        }
        self.last_idle_flush_conversation_len
            .store(new_len, std::sync::atomic::Ordering::Relaxed);
        self.memory
            .context_injected
            .store(false, std::sync::atomic::Ordering::Relaxed);
        if self.memory.is_enabled() {
            tracing::info!(target: xai_grok_telemetry::memory_log::TARGET, "MEMORY_COMPACT: post-compaction reset, next turn re-checks injection (search only if no block persisted)");
        }
        let _ = self
            .notifications
            .persistence_tx
            .send(PersistenceMsg::PlanState(
                crate::tools::todo::TodoState::default(),
            ));
        self.agent
            .borrow()
            .tool_bridge()
            .on_agents_md_compaction()
            .await;
        self.agent
            .borrow()
            .tool_bridge()
            .on_skill_discovery_compaction()
            .await;
        self.persist_announcement_state().await;
        self.plan_mode.lock().reset_after_compaction();
        self.persist_plan_mode_state();
        self.dispatch_hook(
            xai_grok_hooks::event::HookEventName::PostCompact,
            xai_grok_hooks::event::HookPayload::PostCompact {
                source: compact_source.into(),
            },
            None,
            None,
        )
        .await;
        self.chat_state_handle.get_total_tokens().await
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_compact_inner(
        &self,
        user_context: Option<String>,
        auto_continue: Option<crate::extensions::notification::AutoContinueInfo>,
        trigger: xai_grok_telemetry::events::CompactionTrigger,
        strategy: CompactionStrategy,
        mut normal_request: Option<ConversationRequest>,
        mut supersedes_compaction_id: Option<String>,
        mut strategy_started_notified: bool,
        mut supersede_attempt: u8,
    ) -> Result<(), acp::Error> {
        loop {
            match self
                .run_compact_attempt(
                    user_context.clone(),
                    auto_continue.clone(),
                    trigger,
                    strategy,
                    normal_request.take(),
                    supersedes_compaction_id.clone(),
                    strategy_started_notified,
                    supersede_attempt,
                )
                .await?
            {
                CompactionAttemptOutcome::Committed => return Ok(()),
                CompactionAttemptOutcome::Superseded {
                    compaction_id,
                    strategy_started_notified: notified,
                } => {
                    supersedes_compaction_id = Some(compaction_id);
                    strategy_started_notified = notified;
                    supersede_attempt = supersede_attempt.saturating_add(1);
                }
            }
        }
    }

    /// Inner implementation of compaction that supports an optional `auto_continue`
    /// payload for the checkpoint.
    #[tracing::instrument(
        name = "session.compact_inner",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            compaction_tokens_before = tracing::field::Empty,
            compaction_tokens_after = tracing::field::Empty,
            compaction_summary_chars = tracing::field::Empty,
            compaction_degenerate_rejections = tracing::field::Empty,
            compaction_input_overflow_rejections = tracing::field::Empty,
            compaction_deterministic_rejections = tracing::field::Empty,
            compaction_transient_rejections = tracing::field::Empty,
            compaction_attempts = tracing::field::Empty,
            compaction_trigger = tracing::field::Empty,
            compaction_trigger_pct = tracing::field::Empty,
            compaction_threshold_pct = tracing::field::Empty,
            compaction_outcome = tracing::field::Empty,
            compaction_stop_reason = tracing::field::Empty,
            compaction_ttft_ms = tracing::field::Empty,
            compaction_stream_ms = tracing::field::Empty,
            compaction_delta_count = tracing::field::Empty,
            compaction_itl_max_ms = tracing::field::Empty,
            compaction_two_pass_used = tracing::field::Empty,
            compaction_prefire_hit = tracing::field::Empty,
            compaction_pass2_latency_ms = tracing::field::Empty,
            compaction_prefire_waited_ms = tracing::field::Empty,
            compaction_prefire_stale = tracing::field::Empty,
            compaction_prefix_released = tracing::field::Empty,
        )
    )]
    async fn run_compact_attempt(
        &self,
        user_context: Option<String>,
        auto_continue: Option<crate::extensions::notification::AutoContinueInfo>,
        trigger: xai_grok_telemetry::events::CompactionTrigger,
        strategy: CompactionStrategy,
        normal_request: Option<ConversationRequest>,
        supersedes_compaction_id: Option<String>,
        mut strategy_started_notified: bool,
        supersede_attempt: u8,
    ) -> Result<CompactionAttemptOutcome, acp::Error> {
        let (cancel, _cancel_scope) = self.compaction.cancel.enter();
        let tokens_before = self.chat_state_handle.get_total_tokens().await;
        tracing::Span::current().record("compaction_tokens_before", tokens_before as i64);
        if supersede_attempt == 0 {
            self.signals_handle().record_compaction(tokens_before);
        }
        let trigger_str = match trigger {
            xai_grok_telemetry::events::CompactionTrigger::Manual => "manual",
            xai_grok_telemetry::events::CompactionTrigger::Auto => "auto",
        };
        let auto_trigger = matches!(trigger, xai_grok_telemetry::events::CompactionTrigger::Auto);
        let sampling_config = self.chat_state_handle.get_sampling_config().await;
        let context_window = sampling_config
            .as_ref()
            .map(|c| c.context_window.get())
            .unwrap_or(DEFAULT_CONTEXT_WINDOW);
        {
            let span = tracing::Span::current();
            let trigger_pct = if context_window == 0 {
                0
            } else {
                ((tokens_before as f64 / context_window as f64) * 100.0).round() as i64
            };
            span.record("compaction_trigger_pct", trigger_pct);
            span.record(
                "compaction_threshold_pct",
                self.compaction.threshold_percent.get() as i64,
            );
            span.record("compaction_trigger", trigger_str);
        }
        let summary_strips_reasoning = sampling_config
            .as_ref()
            .map(|c| c.api_backend == ApiBackend::Messages)
            .unwrap_or(false);
        let model_id = sampling_config.map(|c| c.model).unwrap_or_default();
        let compaction = xai_grok_telemetry::events::CompactionScope::begin(
            trigger,
            tokens_before,
            context_window,
            model_id.clone(),
            user_context.is_some(),
        );
        let compact_source = trigger_str;
        if supersede_attempt == 0 {
            self.dispatch_hook(
                xai_grok_hooks::event::HookEventName::PreCompact,
                xai_grok_hooks::event::HookPayload::PreCompact {
                    source: compact_source.into(),
                },
                None,
                None,
            )
            .await;
        }
        let max_retries = 3u32;
        let retry_delay_secs = 3u64;
        let cancellation = cancel.clone();
        let preparation = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                self.notify_compact_cancelled(auto_trigger).await;
                compaction.complete(tokens_before);
                return Err(acp::Error::internal_error().data("responses_compaction_cancelled"));
            }
            result = self.prepare_server_request(
                user_context.as_deref(),
                normal_request.as_ref(),
                trigger,
                cancellation.clone(),
            ) => result,
        };
        let (prepared_server, server_preparation_failure) = match preparation {
            Ok(prepared) => (prepared, None),
            Err(error)
                if error.data.as_ref().and_then(serde_json::Value::as_str)
                    == Some("responses_compaction_stale_request") =>
            {
                self.log_strategy_attempt(
                    &compaction.compaction_id,
                    supersedes_compaction_id.as_deref(),
                    None,
                    "server",
                    true,
                    None,
                    0,
                    "superseded",
                    None,
                    None,
                    0,
                    0,
                    0,
                    false,
                    "superseded",
                    None,
                    None,
                    None,
                    Some("request_snapshot"),
                );
                let current_tokens = self.chat_state_handle.get_total_tokens().await;
                let superseded_compaction_id = compaction.compaction_id.clone();
                compaction.complete(current_tokens);
                return Ok(CompactionAttemptOutcome::Superseded {
                    compaction_id: superseded_compaction_id,
                    strategy_started_notified,
                });
            }
            Err(error) => {
                let reason = if error.data.as_ref().and_then(serde_json::Value::as_str)
                    == Some("responses_compaction_request_too_large")
                {
                    crate::session::responses_server_compaction::ServerCompactionFailureReason::RequestTooLarge
                } else {
                    crate::session::responses_server_compaction::ServerCompactionFailureReason::InvalidResponse
                };
                (None, Some(reason))
            }
        };
        let actor_snapshot = if let Some(prepared) = prepared_server.as_ref() {
            xai_chat_state::ChatCompactionSnapshot {
                history_revision: prepared.snapshot.chat_revision,
                request_identity_generation: prepared.snapshot.request_identity_generation,
                prompt_index: prepared.snapshot.prompt_index,
                total_tokens: prepared.snapshot.pre_compaction_tokens,
                conversation: prepared.snapshot.portable_history.clone(),
                sampling_config: self
                    .chat_state_handle
                    .get_sampling_config()
                    .await
                    .ok_or_else(|| acp::Error::internal_error().data("missing sampling config"))?,
                bound_request_identity: Some(prepared.snapshot.identity.clone()),
            }
        } else {
            let mut snapshot = self
                .chat_state_handle
                .get_compaction_snapshot()
                .await
                .ok_or_else(|| acp::Error::internal_error().data("chat-state actor unavailable"))?;
            snapshot.conversation = self
                .portable_history_for_request(&snapshot.conversation)
                .map_err(|error| acp::Error::internal_error().data(error.to_string()))?;
            snapshot
        };
        let expected_history_revision = actor_snapshot.history_revision;
        let expected_identity_generation = actor_snapshot.request_identity_generation;
        let prompt_index_at_compaction = actor_snapshot.prompt_index;
        let full_conversation = actor_snapshot.conversation.clone();
        let conv_len = full_conversation.len();
        let system_message = full_conversation
            .iter()
            .find(|item| matches!(item, ConversationItem::System(_)))
            .cloned();
        let segment_messages = if self.compaction.compaction_mode.writes_segments() {
            xai_chat_state::compaction_utils::prepare_conversation_for_segment(
                full_conversation.clone(),
            )
        } else {
            Vec::new()
        };

        let migration_reason = match strategy {
            CompactionStrategy::BuiltinMigration(reason) => Some(reason),
            CompactionStrategy::ServerFirst => {
                prepared_server
                    .as_ref()
                    .and_then(|prepared| match prepared.checkpoint_status {
                        xai_chat_state::CheckpointReplayStatus::MigrationRequired
                        | xai_chat_state::CheckpointReplayStatus::InvalidCheckpoint => {
                            Some("continuity_mismatch")
                        }
                        _ => None,
                    })
            }
        };
        if let Some(reason) = migration_reason {
            self.discard_prefire().await;
            self.notify_compaction_migration(&mut strategy_started_notified, reason)
                .await;
        }

        // Remote compaction has one eligibility rule: server-first strategy,
        // the agent policy enabled, the Responses backend, no continuity
        // migration, and no checkpoint quota pressure.
        let server_candidate = matches!(strategy, CompactionStrategy::ServerFirst)
            && self.agent.borrow().compaction_policy().server_compaction
            && actor_snapshot.sampling_config.api_backend == ApiBackend::Responses
            && migration_reason.is_none();
        let session_dir = crate::session::persistence::session_dir(&self.session_info);
        let quota_bytes = crate::session::compaction_gc::session_checkpoint_quota_bytes();
        let mut quota_pressure = server_candidate
            && crate::session::compaction_gc::quota_exceeded(&session_dir, quota_bytes)
                .unwrap_or(false);
        if quota_pressure {
            // A failed/cancelled pre-CAS attempt can leave an orphan. Run the
            // fail-closed collector before enforcing quota so old orphans can
            // never permanently prevent the next successful remote compact.
            match crate::session::compaction_gc::gc_session_compaction_artifacts(
                &session_dir,
                crate::session::compaction_gc::GcOptions::default(),
            )
            .await
            {
                Ok(_) => {
                    quota_pressure =
                        crate::session::compaction_gc::quota_exceeded(&session_dir, quota_bytes)
                            .unwrap_or(true);
                }
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "pre-compaction checkpoint GC failed closed under quota pressure"
                    );
                }
            }
        }
        if quota_pressure {
            self.notify_checkpoint_quota_pressure();
        }
        let server_enabled = server_candidate && !quota_pressure;
        let server_preparation_fallback = server_enabled
            .then_some(server_preparation_failure)
            .flatten();
        if let Some(reason) = server_preparation_fallback {
            self.log_strategy_attempt(
                &compaction.compaction_id,
                supersedes_compaction_id.as_deref(),
                None,
                "server",
                true,
                None,
                0,
                "fallback_started",
                Some(reason.as_str()),
                None,
                0,
                0,
                0,
                false,
                "not_started",
                None,
                None,
                None,
                Some("request_build"),
            );
            self.notify_compaction_fallback(&mut strategy_started_notified, reason)
                .await;
        }
        let uses_builtin_fallback = server_preparation_fallback.is_some()
            || (server_candidate && prepared_server.is_some());

        if server_enabled && let Some(prepared) = prepared_server.as_ref() {
            let cache_hit =
                crate::session::responses_server_compaction::process_cache_is_unsupported(
                    &prepared.capability_key,
                );
            let mut server_latency_ms = 0;
            let server_response = if cache_hit {
                self.log_strategy_attempt(
                    &compaction.compaction_id,
                    supersedes_compaction_id.as_deref(),
                    Some(prepared),
                    "server",
                    true,
                    None,
                    0,
                    "fallback_started",
                    Some("unsupported"),
                    None,
                    0,
                    0,
                    0,
                    true,
                    "not_started",
                    None,
                    None,
                    None,
                    None,
                );
                self.notify_compaction_fallback(
                    &mut strategy_started_notified,
                    crate::session::responses_server_compaction::ServerCompactionFailureReason::Unsupported,
                )
                .await;
                None
            } else {
                let server_started = std::time::Instant::now();
                match prepared
                    .client
                    .compact_responses(
                        &prepared.request,
                        &prepared.snapshot.credential,
                        &prepared.snapshot.cancellation,
                    )
                    .await
                {
                    Ok(response) => {
                        server_latency_ms = server_started.elapsed().as_millis() as u64;
                        Some(response)
                    }
                    Err(error)
                        if error.failure()
                            == xai_grok_sampler::ResponsesCompactFailure::Cancelled =>
                    {
                        self.log_strategy_attempt(
                            &compaction.compaction_id,
                            supersedes_compaction_id.as_deref(),
                            Some(prepared),
                            "server",
                            true,
                            None,
                            error.attempts(),
                            "cancelled",
                            None,
                            error.status(),
                            server_started.elapsed().as_millis() as u64,
                            0,
                            0,
                            false,
                            "not_started",
                            None,
                            None,
                            None,
                            None,
                        );
                        self.notify_compact_cancelled(auto_trigger).await;
                        return Err(
                            acp::Error::internal_error().data("responses_compaction_cancelled")
                        );
                    }
                    Err(error) => {
                        if error.status().is_some_and(
                            crate::session::responses_server_compaction::NegativeCapabilityCache::status_is_unsupported,
                        ) {
                            crate::session::responses_server_compaction::process_cache_record_unsupported(
                                prepared.capability_key.clone(),
                            );
                        }
                        let reason =
                            crate::session::responses_server_compaction::classify_compact_failure(
                                error.failure(),
                                error.status(),
                                error.error_code(),
                            )
                            .expect("non-cancel compact failures always map to fallback");
                        self.log_strategy_attempt(
                            &compaction.compaction_id,
                            supersedes_compaction_id.as_deref(),
                            Some(prepared),
                            "server",
                            true,
                            None,
                            error.attempts(),
                            "fallback_started",
                            Some(reason.as_str()),
                            error.status(),
                            server_started.elapsed().as_millis() as u64,
                            0,
                            0,
                            false,
                            "not_started",
                            None,
                            None,
                            None,
                            None,
                        );
                        self.notify_compaction_fallback(&mut strategy_started_notified, reason)
                            .await;
                        None
                    }
                }
            };

            if let Some(response) = server_response {
                let server_attempts = response.attempts;
                let server_response_bytes = response.response_bytes as u64;
                let server_output_items = response.output.len() as u64;
                match crate::session::responses_server_compaction::server_checkpoint_token_seed(
                    &response,
                    prepared.snapshot.semantic_envelope_tokens,
                    prepared.snapshot.pre_compaction_tokens,
                ) {
                    Ok((token_seed, token_seed_source)) => {
                        let token_seed_source_name = match token_seed_source {
                            xai_grok_sampling_types::TokenSeedSource::UsageOutputTokens => {
                                "usage_output_tokens"
                            }
                            xai_grok_sampling_types::TokenSeedSource::EstimatedCanonicalOutput => {
                                "estimated_canonical_output"
                            }
                        };
                        let checkpoint_id = uuid::Uuid::now_v7().to_string();
                        let operation_id = uuid::Uuid::now_v7().to_string();
                        let branch_id = uuid::Uuid::now_v7().to_string();
                        let relative_path = format!("compaction_checkpoints/{checkpoint_id}.json");
                        let mode_tail = self
                            .transcript_hint()
                            .map(ConversationItem::system_reminder)
                            .into_iter()
                            .collect::<Vec<_>>();
                        let portable_history = &prepared.snapshot.portable_history;
                        let portable_digest =
                            crate::session::storage::responses_compaction::portable_history_digest(
                                portable_history,
                            )
                            .map_err(|error| {
                                acp::Error::internal_error().data(error.to_string())
                            })?;
                        let provisional =
                            crate::session::responses_server_compaction::build_server_successor(
                                &checkpoint_id,
                                &operation_id,
                                prepared.snapshot.prompt_index,
                                auto_continue.is_some(),
                                prepared.snapshot.mode.clone(),
                                &branch_id,
                                prepared.snapshot.identity.clone(),
                                response.output,
                                &relative_path,
                                portable_digest,
                                token_seed,
                                token_seed_source,
                                prepared.snapshot.identity.prior_checkpoint_id.clone(),
                                prepared.snapshot.trusted_envelope.memory_revision,
                                mode_tail.clone(),
                            );
                        let ConversationItem::ResponsesCompactionCheckpoint(wrapper) =
                            provisional[0].clone()
                        else {
                            unreachable!("server successor starts with a checkpoint wrapper")
                        };
                        let original_user_info = prepared
                            .snapshot
                            .portable_history
                            .iter()
                            .find_map(|item| match item {
                                ConversationItem::User(user) => {
                                    user.content.iter().find_map(|part| match part {
                                        xai_grok_sampling_types::ContentPart::Text { text } => {
                                            Some(text.to_string())
                                        }
                                        _ => None,
                                    })
                                }
                                _ => None,
                            });
                        let replay_material =
                            xai_grok_sampling_types::CheckpointReplayMaterial::try_new(
                                &wrapper,
                                prepared.snapshot.trusted_envelope.clone(),
                                portable_history,
                            )
                            .map_err(|error| {
                                acp::Error::internal_error().data(error.to_string())
                            })?;
                        let sidecar =
                            crate::session::storage::responses_compaction::CompactionCheckpointFile::new(
                                *wrapper,
                                replay_material,
                                portable_history.clone(),
                                original_user_info,
                                Vec::new(),
                            )
                            .map_err(|error| {
                                acp::Error::internal_error().data(error.to_string())
                            })?;
                        let segment_staging = self
                            .compaction
                            .compaction_mode
                            .segment_detail()
                            .map(|detail| {
                                crate::session::storage::responses_compaction::ResponsesCompactionSegmentStaging::new(
                                    checkpoint_id.clone(),
                                    operation_id.clone(),
                                    branch_id.clone(),
                                    sidecar.wrapper.wrapper_digest(),
                                    portable_history.clone(),
                                    "Server Responses checkpoint",
                                    detail,
                                    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                                )
                            })
                            .transpose()
                            .map_err(|error| {
                                acp::Error::internal_error().data(error.to_string())
                            })?;
                        let mut replacement =
                            vec![ConversationItem::ResponsesCompactionCheckpoint(Box::new(
                                sidecar.wrapper.clone(),
                            ))];
                        replacement.extend(mode_tail);
                        let committed_total_tokens = verified_remote_shrink_for_prefire_discard(
                            Some(token_seed),
                            xai_chat_state::estimate_conversation_tokens(&replacement[1..]),
                            prepared.snapshot.pre_compaction_tokens,
                        );

                        // Keep speculative pass-1 available until the remote
                        // result has passed both token-seed validation and the
                        // final committed-total shrink check. Invalid server
                        // output must still be usable by builtin pass 2.
                        if committed_total_tokens.is_none() {
                            self.log_strategy_attempt(
                                &compaction.compaction_id,
                                supersedes_compaction_id.as_deref(),
                                Some(prepared),
                                "server",
                                true,
                                None,
                                server_attempts,
                                "fallback_started",
                                Some("invalid_response"),
                                Some(200),
                                server_latency_ms,
                                server_response_bytes,
                                server_output_items,
                                false,
                                "not_started",
                                None,
                                None,
                                Some(token_seed_source_name),
                                None,
                            );
                            self.notify_compaction_fallback(
                                &mut strategy_started_notified,
                                crate::session::responses_server_compaction::ServerCompactionFailureReason::InvalidResponse,
                            )
                            .await;
                        } else {
                            let committed_total_tokens = committed_total_tokens
                                .expect("checked remote shrink carries its committed token total");
                            if prepared.snapshot.cancellation.is_cancelled() {
                                self.log_strategy_attempt(
                                    &compaction.compaction_id,
                                    supersedes_compaction_id.as_deref(),
                                    Some(prepared),
                                    "server",
                                    true,
                                    None,
                                    server_attempts,
                                    "cancelled",
                                    None,
                                    Some(200),
                                    server_latency_ms,
                                    server_response_bytes,
                                    server_output_items,
                                    false,
                                    "not_started",
                                    None,
                                    None,
                                    Some(token_seed_source_name),
                                    None,
                                );
                                self.notify_compact_cancelled(auto_trigger).await;
                                compaction
                                    .complete(self.chat_state_handle.get_total_tokens().await);
                                return Err(acp::Error::internal_error()
                                    .data("responses_compaction_cancelled"));
                            }

                            let checkpoint_bytes = sidecar.wrapper.portable_history_bytes;
                            if let Err(error) = self
                                .persist_server_sidecar(relative_path, sidecar.clone())
                                .await
                            {
                                self.log_strategy_attempt(
                                    &compaction.compaction_id,
                                    supersedes_compaction_id.as_deref(),
                                    Some(prepared),
                                    "server",
                                    true,
                                    None,
                                    server_attempts,
                                    "failed",
                                    None,
                                    Some(200),
                                    server_latency_ms,
                                    server_response_bytes,
                                    server_output_items,
                                    false,
                                    "not_started",
                                    Some(checkpoint_bytes),
                                    None,
                                    Some(token_seed_source_name),
                                    Some("sidecar"),
                                );
                                return Err(error);
                            }
                            if let Some(staging) = segment_staging.clone()
                                && let Err(error) = self.stage_server_segment(staging).await
                            {
                                self.log_strategy_attempt(
                                    &compaction.compaction_id,
                                    supersedes_compaction_id.as_deref(),
                                    Some(prepared),
                                    "server",
                                    true,
                                    None,
                                    server_attempts,
                                    "failed",
                                    None,
                                    Some(200),
                                    server_latency_ms,
                                    server_response_bytes,
                                    server_output_items,
                                    false,
                                    "not_started",
                                    Some(checkpoint_bytes),
                                    None,
                                    Some(token_seed_source_name),
                                    Some("segment_staging"),
                                );
                                return Err(error);
                            }
                            if prepared.snapshot.cancellation.is_cancelled() {
                                self.log_strategy_attempt(
                                    &compaction.compaction_id,
                                    supersedes_compaction_id.as_deref(),
                                    Some(prepared),
                                    "server",
                                    true,
                                    None,
                                    server_attempts,
                                    "cancelled",
                                    None,
                                    Some(200),
                                    server_latency_ms,
                                    server_response_bytes,
                                    server_output_items,
                                    false,
                                    "not_started",
                                    Some(checkpoint_bytes),
                                    None,
                                    Some(token_seed_source_name),
                                    Some("precommit"),
                                );
                                self.notify_compact_cancelled(auto_trigger).await;
                                compaction
                                    .complete(self.chat_state_handle.get_total_tokens().await);
                                return Err(acp::Error::internal_error()
                                    .data("responses_compaction_cancelled"));
                            }
                            let committed = match self
                                .commit_compaction_replacement(
                                    operation_id,
                                    prepared.snapshot.chat_revision,
                                    prepared.snapshot.request_identity_generation,
                                    replacement.clone(),
                                    committed_total_tokens,
                                )
                                .await
                            {
                                Ok(committed) => committed,
                                Err(error) => {
                                    self.log_strategy_attempt(
                                        &compaction.compaction_id,
                                        supersedes_compaction_id.as_deref(),
                                        Some(prepared),
                                        "server",
                                        true,
                                        None,
                                        server_attempts,
                                        "failed",
                                        None,
                                        Some(200),
                                        server_latency_ms,
                                        server_response_bytes,
                                        server_output_items,
                                        false,
                                        "failed",
                                        Some(checkpoint_bytes),
                                        None,
                                        Some(token_seed_source_name),
                                        Some("history"),
                                    );
                                    return Err(error);
                                }
                            };
                            if !committed {
                                self.log_strategy_attempt(
                                    &compaction.compaction_id,
                                    supersedes_compaction_id.as_deref(),
                                    Some(prepared),
                                    "server",
                                    true,
                                    None,
                                    server_attempts,
                                    "superseded",
                                    None,
                                    Some(200),
                                    server_latency_ms,
                                    server_response_bytes,
                                    server_output_items,
                                    false,
                                    "superseded",
                                    Some(checkpoint_bytes),
                                    None,
                                    Some(token_seed_source_name),
                                    None,
                                );
                                let current_tokens =
                                    self.chat_state_handle.get_total_tokens().await;
                                let superseded_compaction_id = compaction.compaction_id.clone();
                                compaction.complete(current_tokens);
                                return Ok(CompactionAttemptOutcome::Superseded {
                                    compaction_id: superseded_compaction_id,
                                    strategy_started_notified,
                                });
                            }

                            // The speculative builtin NOTE1 remains available
                            // through every pre-CAS persistence/cancellation/
                            // supersession failure. Only an installed remote
                            // successor makes it obsolete.
                            self.discard_prefire().await;
                            self.chat_state_handle
                                .record_compaction_at(prepared.snapshot.prompt_index);
                            self.compaction
                                .prefix_released
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                            self.persist_server_marker(
                                &sidecar.wrapper,
                                &replacement,
                                &sidecar.wrapper.operation_id,
                            )
                            .await?;
                            let segment_publish_failed = if segment_staging.is_some() {
                                self.publish_server_segment(
                                    checkpoint_id,
                                    sidecar.wrapper.operation_id.clone(),
                                    sidecar.wrapper.wrapper_digest(),
                                )
                                .await
                                .err()
                            } else {
                                None
                            };
                            if let Some(error) = segment_publish_failed.as_ref() {
                                tracing::warn!(
                                    error = %error,
                                    "committed Responses compaction segment will be repaired on resume"
                                );
                            }
                            let tokens_after = self
                                .finish_committed_compaction(
                                    replacement.len(),
                                    context_window,
                                    compact_source,
                                )
                                .await;
                            self.log_strategy_attempt(
                                &compaction.compaction_id,
                                supersedes_compaction_id.as_deref(),
                                Some(prepared),
                                "server",
                                true,
                                None,
                                server_attempts,
                                "committed",
                                None,
                                Some(200),
                                server_latency_ms,
                                server_response_bytes,
                                server_output_items,
                                false,
                                "committed",
                                Some(checkpoint_bytes),
                                Some(tokens_after),
                                Some(token_seed_source_name),
                                segment_publish_failed.as_ref().map(|_| "segment_publish"),
                            );
                            let span = tracing::Span::current();
                            span.record("compaction_tokens_after", tokens_after as i64);
                            span.record("compaction_attempts", 1_i64);
                            span.record("compaction_outcome", "server_success");
                            compaction.complete(tokens_after);
                            self.spawn_compaction_gc();
                            return Ok(CompactionAttemptOutcome::Committed);
                        }
                    }
                    Err(_) => {
                        self.log_strategy_attempt(
                            &compaction.compaction_id,
                            supersedes_compaction_id.as_deref(),
                            Some(prepared),
                            "server",
                            true,
                            None,
                            server_attempts,
                            "fallback_started",
                            Some("invalid_response"),
                            Some(200),
                            server_latency_ms,
                            server_response_bytes,
                            server_output_items,
                            false,
                            "not_started",
                            None,
                            None,
                            None,
                            None,
                        );
                        self.notify_compaction_fallback(
                            &mut strategy_started_notified,
                            crate::session::responses_server_compaction::ServerCompactionFailureReason::InvalidResponse,
                        )
                        .await;
                    }
                }
            }
        }
        if let Some(reason) = migration_reason {
            self.log_strategy_attempt(
                &compaction.compaction_id,
                supersedes_compaction_id.as_deref(),
                prepared_server.as_ref(),
                "builtin_migration",
                false,
                Some(reason),
                0,
                "started",
                None,
                None,
                0,
                0,
                0,
                false,
                "not_started",
                None,
                None,
                None,
                None,
            );
        } else if (!server_enabled || prepared_server.is_none())
            && server_preparation_fallback.is_none()
        {
            let skip_reason = if !self.agent.borrow().compaction_policy().server_compaction {
                "feature_off"
            } else if actor_snapshot.sampling_config.api_backend != ApiBackend::Responses {
                "non_responses"
            } else if quota_pressure {
                "quota_exceeded"
            } else {
                "unstable_principal"
            };
            self.log_strategy_attempt(
                &compaction.compaction_id,
                supersedes_compaction_id.as_deref(),
                prepared_server.as_ref(),
                "builtin_direct",
                false,
                Some(skip_reason),
                0,
                "started",
                None,
                None,
                0,
                0,
                0,
                false,
                "not_started",
                None,
                None,
                None,
                None,
            );
        }
        if cancellation.is_cancelled() {
            self.notify_compact_cancelled(auto_trigger).await;
            compaction.complete(self.chat_state_handle.get_total_tokens().await);
            return Err(acp::Error::internal_error().data("responses_compaction_cancelled"));
        }
        let builtin_started = std::time::Instant::now();
        const SUMMARY_BUDGET_RESERVE_TOKENS: u64 = 32_768;
        let verbatim_input_enabled = self.compaction.verbatim_input;
        let simplified_messages = if verbatim_input_enabled {
            xai_chat_state::compaction_utils::prepare_conversation_for_verbatim_summarization(
                full_conversation.clone(),
                summary_strips_reasoning,
            )
        } else {
            xai_chat_state::compaction_utils::prepare_conversation_for_summarization(
                full_conversation.clone(),
            )
        };
        if conv_len == 0 {
            tracing::error!(
                session_id = %self.session_info.id.0,
                "Compaction failed: conversation is empty (ChatStateActor may have died)"
            );
            return Err(
                acp::Error::internal_error().data("Compaction failed: conversation is empty")
            );
        }
        let system_message = match system_message {
            Some(msg) => msg,
            None => {
                tracing::error!(
                    session_id = %self.session_info.id.0,
                    conversation_len = conv_len,
                    "Compaction failed: no system message in conversation history"
                );
                return Err(acp::Error::internal_error()
                    .data("Compaction failed: no system message in conversation history"));
            }
        };
        if simplified_messages.is_empty() {
            tracing::error!(
                session_id = %self.session_info.id.0,
                conversation_len = conv_len,
                "Compaction failed: simplified conversation is empty"
            );
            return Err(acp::Error::internal_error()
                .data("Compaction failed: simplified conversation is empty"));
        }
        if !simplified_messages
            .iter()
            .any(|msg| matches!(msg, ConversationItem::System(_)))
        {
            tracing::error!(
                session_id = %self.session_info.id.0,
                conversation_len = conv_len,
                simplified_len = simplified_messages.len(),
                "Compaction failed: no system message in simplified conversation"
            );
            return Err(acp::Error::internal_error()
                .data("Compaction failed: no system message in simplified conversation"));
        }
        let (sampling_config, sampling_client) = self.prepare_builtin_compaction_sampling().await?;
        let backend_search_active = self.backend_search_active();
        let effective_tool_defs: Vec<xai_grok_sampling_types::ToolDefinition> = self
            .prepare_tool_definitions()
            .await
            .into_iter()
            .filter(|td| !backend_search_active || td.function.name != "web_search")
            .collect();
        let compaction_tool_tokens =
            xai_chat_state::estimate_tool_definitions_tokens(&effective_tool_defs);
        let compaction_tools: Vec<xai_grok_sampling_types::ToolSpec> = effective_tool_defs
            .into_iter()
            .map(xai_grok_sampling_types::ToolSpec::from)
            .collect();
        let compaction_hosted_tools: Vec<xai_grok_sampling_types::HostedTool> =
            self.hosted_tools_for_turn();
        tracing::info!(
            num_tools = compaction_tools.len(),
            tool_tokens = compaction_tool_tokens,
            "Running compact with model '{}' (user model: '{}')",
            &sampling_config.model,
            &sampling_config.model
        );
        let mut last_error: Option<acp::Error> = None;
        let mut last_failure_outcome = CompactionOutcome::Failed;
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum InputStage {
            Verbatim,
            VerbatimFitted,
            Lossy,
        }
        impl InputStage {
            fn as_str(self) -> &'static str {
                match self {
                    Self::Verbatim => "verbatim",
                    Self::VerbatimFitted => "verbatim_fitted",
                    Self::Lossy => "lossy",
                }
            }
        }
        let mut input_stage = if verbatim_input_enabled {
            InputStage::Verbatim
        } else {
            InputStage::Lossy
        };
        let use_short_prompt = false;
        let started_at = chrono::Utc::now().to_rfc3339();
        let estimated_input_tokens =
            xai_chat_state::estimate_conversation_tokens(&simplified_messages);
        let wall_clock_budget_secs = self
            .agent
            .borrow()
            .compaction_policy()
            .wall_clock_budget_secs;
        let sampler = crate::session::helpers::full_replace_compaction::ShellCompactionSampler::new(
            use_short_prompt,
            user_context.clone(),
            compaction_tools.clone(),
            compaction_hosted_tools.clone(),
            sampling_client,
            self.session_info.id.clone(),
            sampling_config.clone(),
            self.inference_idle_timeout,
            wall_clock_budget_secs,
            self.compaction.tool_choice,
            cancel.clone(),
        );
        let observer =
            crate::session::helpers::full_replace_compaction::ShellFullReplaceObserver::new(
                trigger,
                context_window,
                compaction.compaction_id.clone(),
                self.session_info.id.0.to_string(),
                estimated_input_tokens,
                retry_delay_secs,
            );
        let fr_config = xai_grok_compaction::FullReplaceConfig {
            max_attempts: max_retries,
            retry_delay_secs,
            sampling_timeout_secs: 0,
        };
        let mut request_turns = simplified_messages.clone();
        let mut input_overflow_rejections: u32 = 0;
        let two_pass_output = self
            .try_two_pass_pass2_apply(user_context.as_deref(), summary_strips_reasoning)
            .await;
        // `None` normally means a stale/missing prefire and permits the
        // single-pass fallback. Cancellation is different: pass-2 already
        // observed it, so do not launch a fresh paid request without a
        // cancellation token.
        if cancellation.is_cancelled() {
            self.notify_compact_cancelled(auto_trigger).await;
            compaction.complete(self.chat_state_handle.get_total_tokens().await);
            return Err(acp::Error::internal_error().data("responses_compaction_cancelled"));
        }
        let mut compact_summary: Option<String> =
            two_pass_output.as_ref().map(|o| o.content.clone());
        while compact_summary.is_none() {
            match xai_grok_compaction::sample_full_replace_summary(
                &sampler,
                &request_turns,
                user_context.as_deref(),
                &fr_config,
                &observer,
            )
            .await
            {
                Ok(summary) => {
                    compact_summary = Some(summary.summary);
                    break;
                }
                Err(xai_grok_compaction::FullReplaceError::NothingToCompact) => {
                    last_error = Some(
                        acp::Error::internal_error().data("compact failed: nothing to compact"),
                    );
                    break;
                }
                Err(xai_grok_compaction::FullReplaceError::EmptyResponse) => {
                    last_failure_outcome = if observer.degenerate_seen() {
                        CompactionOutcome::Degenerate
                    } else {
                        CompactionOutcome::Transient
                    };
                    last_error = Some(acp::Error::internal_error().data(
                        observer.last_error_message().unwrap_or_else(|| {
                            "compact failed: model returned empty response".to_string()
                        }),
                    ));
                    break;
                }
                Err(xai_grok_compaction::FullReplaceError::Sampler {
                    message,
                    deterministic,
                    context_overflow,
                }) => {
                    if cancel.is_cancelled()
                        || message.contains(
                            crate::session::helpers::session_compact::COMPACT_CANCELLED_MSG,
                        )
                    {
                        self.notify_compact_cancelled(auto_trigger).await;
                        return Err(
                            crate::session::helpers::session_compact::CompactFailure::cancelled_error(),
                        );
                    }
                    if context_overflow {
                        let next_stage = match input_stage {
                            InputStage::Verbatim => Some(InputStage::VerbatimFitted),
                            InputStage::VerbatimFitted => Some(InputStage::Lossy),
                            InputStage::Lossy => None,
                        };
                        if let Some(stage) = next_stage {
                            input_overflow_rejections += 1;
                            xai_grok_telemetry::session_ctx::log_event(
                                xai_grok_telemetry::events::CompactionRetryDegraded {
                                    trigger,
                                    reason: "input_overflow",
                                    from_stage: Some(input_stage.as_str()),
                                    to_stage: Some(stage.as_str()),
                                    summary_chars: None,
                                    attempt: observer.attempt_count(),
                                    context_window,
                                    compaction_id: compaction.compaction_id.clone(),
                                },
                            );
                            tracing::warn!(
                                session_id = %self.session_info.id.0,
                                ?stage,
                                error = %message,
                                "Compaction input overflowed deterministically; stepping down the input ladder to avoid an incompactable state"
                            );
                            let conv = full_conversation.clone();
                            request_turns = match stage {
                                InputStage::VerbatimFitted => {
                                    let budget = context_window
                                        .saturating_sub(SUMMARY_BUDGET_RESERVE_TOKENS)
                                        .saturating_sub(compaction_tool_tokens);
                                    let verbatim = xai_chat_state::compaction_utils::prepare_conversation_for_verbatim_summarization(
                                        conv,
                                        summary_strips_reasoning,
                                    );
                                    xai_chat_state::compaction_utils::fit_conversation_to_budget(
                                        verbatim, budget,
                                    )
                                }
                                InputStage::Lossy => {
                                    let lossy_budget = (context_window.saturating_mul(7) / 10)
                                        .saturating_sub(compaction_tool_tokens);
                                    xai_chat_state::compaction_utils::fit_conversation_to_budget(
                                        xai_chat_state::compaction_utils::prepare_conversation_for_summarization(
                                            conv,
                                        ),
                                        lossy_budget,
                                    )
                                }
                                InputStage::Verbatim => {
                                    unreachable!("ladder only steps forward")
                                }
                            };
                            input_stage = stage;
                            continue;
                        }
                        last_failure_outcome = CompactionOutcome::Deterministic;
                        if auto_trigger {
                            self.suppress_auto_compaction(
                                SuppressReason::Size,
                                estimated_input_tokens,
                                context_window,
                            )
                            .await;
                        }
                        last_error = Some(acp::Error::internal_error().data(message));
                        break;
                    }
                    if deterministic {
                        last_failure_outcome = CompactionOutcome::Deterministic;
                        if auto_trigger {
                            let reason = Self::classify_suppress_reason(&message);
                            self.suppress_auto_compaction(
                                reason,
                                estimated_input_tokens,
                                context_window,
                            )
                            .await;
                        }
                        last_error = Some(acp::Error::internal_error().data(message));
                        break;
                    }
                    last_failure_outcome = CompactionOutcome::Transient;
                    last_error = Some(acp::Error::internal_error().data(message));
                    break;
                }
            }
        }
        let telemetry = observer.into_telemetry();
        if two_pass_output.is_none() {
            let request_chat_history = build_compaction_chat_history(
                request_turns,
                user_context.as_deref(),
                use_short_prompt,
            );
            self.persist_compaction_request_artifact(
                request_chat_history,
                compaction_tools,
                user_context.as_deref(),
                use_short_prompt,
                &sampling_config.model,
                trigger,
                compact_summary
                    .as_deref()
                    .or(telemetry.last_rejected_summary.as_deref()),
                last_error.as_ref(),
                telemetry.attempts,
                telemetry.attempt_details,
                started_at,
            );
        }
        let compact_output = match compact_summary {
            Some(_) => match two_pass_output {
                Some(tp) => tp,
                None => sampler
                    .take_last_success()
                    .expect("a successful full-replace sample stashes its CompactOutput"),
            },
            None => {
                let span = tracing::Span::current();
                span.record("compaction_attempts", telemetry.attempts as i64);
                span.record(
                    "compaction_degenerate_rejections",
                    telemetry.degenerate_rejections as i64,
                );
                span.record(
                    "compaction_input_overflow_rejections",
                    input_overflow_rejections as i64,
                );
                span.record(
                    "compaction_deterministic_rejections",
                    telemetry.deterministic_rejections as i64,
                );
                span.record(
                    "compaction_transient_rejections",
                    telemetry.transient_rejections as i64,
                );
                span.record("compaction_outcome", last_failure_outcome.as_str());
                return Err(last_error.unwrap_or_else(|| {
                    acp::Error::internal_error().data("compaction failed: unknown error")
                }));
            }
        };
        let generate_session_compact = compact_output.content.clone();
        let user_message_prefix = self.build_user_message_prefix().await;
        let conversation = full_conversation.clone();
        let (discovered_agents_md, all_skills_for_compaction, _agent_edited_paths, state_context) =
            if use_short_prompt {
                let empty_edited: std::collections::BTreeSet<String> = Default::default();
                let ctx =
                    CompactionStateContext::build(&conversation, CompactionInputs::default()).await;
                (Vec::<std::path::PathBuf>::new(), vec![], empty_edited, ctx)
            } else {
                let agents_md: Vec<std::path::PathBuf> = self
                    .agent
                    .borrow()
                    .tool_bridge()
                    .agents_md_reminded_paths()
                    .await
                    .into_iter()
                    .collect();
                let skills = self.slash_skills_for_resolve().await;
                let edited_paths = self.chat_state_handle.get_agent_edited_paths().await;
                let ctx = {
                    let bridge_tasks = self
                        .agent
                        .borrow()
                        .tool_bridge()
                        .list_background_tasks()
                        .await;
                    let pending_tasks: Vec<_> =
                        bridge_tasks.into_iter().filter(|t| !t.completed).collect();
                    let (execute_tool_name, monitor_tool_name) = if pending_tasks.is_empty() {
                        (None, None)
                    } else {
                        let agent_ref = self.agent.borrow();
                        let bridge = agent_ref.tool_bridge();
                        let empty = serde_json::json!({});
                        let execute = bridge
                            .render_prompt("${{ tools.by_kind.execute }}", &empty)
                            .await
                            .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                        let monitor = bridge
                            .render_prompt("${{ tools.by_kind.monitor }}", &empty)
                            .await
                            .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                        (execute, monitor)
                    };
                    let running_tasks: Vec<_> = pending_tasks
                        .into_iter()
                        .map(|t| {
                            let tool_name = match t.kind {
                                xai_grok_tools::computer::types::TaskKind::Monitor => {
                                    monitor_tool_name.clone()
                                }
                                xai_grok_tools::computer::types::TaskKind::Bash => {
                                    execute_tool_name.clone()
                                }
                            };
                            CompactionStateContext::task_summary(
                                t.task_id, t.command, "running", tool_name,
                            )
                        })
                        .collect();
                    let running_subagents = if let Some(ref event_tx) =
                        self.tool_context.subagent_event_tx
                    {
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        use xai_grok_tools::implementations::grok_build::task::types::{
                            SubagentEvent, SubagentListActiveRequest,
                        };
                        let _ =
                            event_tx.send(SubagentEvent::ListActive(SubagentListActiveRequest {
                                parent_session_id: self.session_id_string(),
                                respond_to: tx,
                            }));
                        rx.await
                        .unwrap_or_default()
                        .into_iter()
                        .map(|s| crate::session::helpers::compaction_context::RunningSubagentSummary {
                            subagent_id: s.subagent_id,
                            subagent_type: s.subagent_type,
                            description: s.description,
                            elapsed_ms: s.elapsed_ms,
                        })
                        .collect()
                    } else {
                        vec![]
                    };
                    let connected_mcp_servers = {
                        use crate::session::helpers::compaction_context::CompactionServerSummary;
                        use xai_grok_tools::implementations::search_tool::{
                            sanitize_description, truncate_description,
                        };
                        self.connected_server_summaries()
                            .into_iter()
                            .map(|s| {
                                let desc = s
                                    .description
                                    .map(|d| truncate_description(&sanitize_description(&d)))
                                    .filter(|d| !d.is_empty());
                                CompactionServerSummary {
                                    name: s.name,
                                    tool_count: s.tool_count,
                                    description: desc,
                                }
                            })
                            .collect()
                    };
                    let todos = {
                        use crate::session::helpers::compaction_context::{
                            TodoSummary, TodoSummaryStatus,
                        };
                        use crate::tools::todo::{TodoState, TodoStatus};
                        use xai_grok_tools::types::resources::State;
                        let bridge = self.agent.borrow().tool_bridge().clone();
                        bridge
                            .read_resource::<State<TodoState>>()
                            .await
                            .map(|s| {
                                s.0.todo_items_with_ids()
                                    .map(|(id, item)| TodoSummary {
                                        id: id.clone(),
                                        content: item.content.clone(),
                                        status: match item.status {
                                            TodoStatus::Pending => TodoSummaryStatus::Pending,
                                            TodoStatus::InProgress => TodoSummaryStatus::InProgress,
                                            TodoStatus::Completed => TodoSummaryStatus::Completed,
                                            TodoStatus::Cancelled => TodoSummaryStatus::Cancelled,
                                        },
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    };
                    CompactionStateContext::build(
                        &conversation,
                        CompactionInputs {
                            running_tasks,
                            running_subagents,
                            agent_edited_paths: edited_paths.clone(),
                            connected_mcp_servers,
                            todos,
                            ..Default::default()
                        },
                    )
                    .await
                };
                (agents_md, skills, edited_paths, ctx)
            };
        use crate::session::helpers::compaction_context::SubagentToolNames;
        let subagent_tool_names: Option<SubagentToolNames> =
            if use_short_prompt || state_context.running_subagents.is_empty() {
                None
            } else {
                let agent_ref = self.agent.borrow();
                let bridge = agent_ref.tool_bridge();
                let empty = serde_json::json!({});
                let poll_name = bridge
                    .render_prompt("${{ tools.by_kind.background_task_action }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                let cancel_name = bridge
                    .render_prompt("${{ tools.by_kind.kill_task_action }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                match (poll_name, cancel_name) {
                    (Some(poll), Some(cancel)) => Some(SubagentToolNames { poll, cancel }),
                    (poll, cancel) => {
                        tracing::warn!(
                            session_id = %self.session_info.id.0,
                            poll_resolved = poll.is_some(),
                            cancel_resolved = cancel.is_some(),
                            "could not resolve subagent tool names, \
                             omitting subagent reminder from compacted conversation"
                        );
                        None
                    }
                }
            };
        use crate::session::helpers::compaction_context::McpToolNames;
        let mcp_tool_names: Option<McpToolNames> =
            if use_short_prompt || state_context.connected_mcp_servers.is_empty() {
                None
            } else {
                let agent_ref = self.agent.borrow();
                let bridge = agent_ref.tool_bridge();
                let empty = serde_json::json!({});
                let search_name = bridge
                    .render_prompt("${{ tools.by_kind.search_tool }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                let call_name = bridge
                    .render_prompt("${{ tools.by_kind.use_tool }}", &empty)
                    .await
                    .filter(|s| !s.is_empty() && !s.contains("by_kind"));
                match (search_name, call_name) {
                    (Some(search), Some(call)) => Some(McpToolNames { search, call }),
                    _ => None,
                }
            };
        let memory_backend_impl = {
            let g = self.memory.storage.borrow();
            g.as_ref()
                .zip(self.memory.backend_params.as_ref())
                .map(|(storage, params)| {
                    crate::session::memory::MemoryBackendImpl::from_session_params(
                        storage.clone(),
                        &crate::session::memory::MemoryBackendParams {
                            search_source: "compaction_recovery",
                            ..params.clone()
                        },
                    )
                })
        };
        let memory_opt_out = false;
        let memory_ref: Option<&dyn xai_grok_tools::types::memory_backend::MemoryBackend> =
            if memory_opt_out {
                None
            } else {
                memory_backend_impl
                    .as_ref()
                    .map(|b| b as &dyn xai_grok_tools::types::memory_backend::MemoryBackend)
            };
        let suppress_state_reminder = false;
        let system_reminder = if suppress_state_reminder {
            None
        } else {
            to_system_reminder(
                &state_context,
                &discovered_agents_md,
                &all_skills_for_compaction,
                memory_ref,
                subagent_tool_names.as_ref(),
                mcp_tool_names.as_ref(),
            )
            .await
        };
        let system_reminder = {
            let plan_path = {
                let guard = self.plan_mode.lock();
                guard
                    .is_active()
                    .then(|| guard.plan_file_path().to_path_buf())
            };
            if let Some(plan_path) = plan_path {
                let plan_has_content =
                    crate::session::plan_mode::plan_file_has_content(&plan_path).await;
                let template = crate::session::plan_mode::plan_mode_reminder_full_template();
                let wrapper = self.reminder_wrapper_tag();
                let rendered = self
                    .render_plan_template(template, &plan_path, plan_has_content)
                    .await;
                match (system_reminder, rendered) {
                    (Some(mut existing), Some(plan_section)) => {
                        if let Some(pos) = existing.rfind("</system-reminder>") {
                            existing.insert_str(pos, &format!("\n\n{}\n", plan_section));
                        } else {
                            existing.push_str("\n\n");
                            existing.push_str(&plan_section);
                        }
                        Some(existing)
                    }
                    (None, Some(plan_section)) => Some(format!(
                        "<{tag}>\n{body}\n</{tag}>",
                        tag = wrapper,
                        body = plan_section,
                    )),
                    (existing, None) => {
                        tracing::warn!(
                            session_id = %self.session_info.id.0,
                            "compaction: plan mode active but template render failed"
                        );
                        existing
                    }
                }
            } else {
                system_reminder
            }
        };
        if let Some(ref recovery_backend) = memory_backend_impl {
            let n = recovery_backend
                .search_counter
                .load(std::sync::atomic::Ordering::Relaxed);
            if n > 0 {
                self.memory
                    .compaction_recovery_count
                    .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(
                    target: xai_grok_telemetry::memory_log::TARGET,
                    count = n,
                    "MEMORY_COMPACTION_RECOVERY: {} search(es) performed",
                    n,
                );
            }
        }
        let agents_md_reminder = self.agent.borrow().agents_md_user_reminder();
        let compaction_context = state_context.for_compaction();
        let compaction_state_context: &CompactionStateContext = &compaction_context;
        let transcript_hint = self.transcript_hint();
        let summary_count = self
            .compaction
            .count
            .load(std::sync::atomic::Ordering::Relaxed);
        let raw_compacted = build_compacted_history(CompactedHistoryInput {
            system_message: system_message.clone(),
            user_message_prefix: user_message_prefix.clone(),
            agents_md_reminder: agents_md_reminder.clone(),
            state_context: compaction_state_context,
            compaction_summary: generate_session_compact.clone(),
            system_reminder: system_reminder.clone(),
            summary_before_recent: use_short_prompt,
            transcript_hint: transcript_hint.clone(),
            summary_count,
        });
        let sanitize_result = sanitize_compacted_history(raw_compacted);
        let compacted_history = if sanitize_result.stripped_tool_call_ids.is_empty() {
            sanitize_result.items
        } else {
            tracing::warn!(
                session_id = %self.session_info.id,
                stripped_count = sanitize_result.stripped_tool_call_ids.len(),
                stripped_ids = ?sanitize_result.stripped_tool_call_ids,
                "compaction: stripped orphaned ToolResults from compacted history"
            );
            sanitize_result.items
        };
        let remaining_violations = validate_compacted_history(&compacted_history);
        let compacted_history = if remaining_violations.is_empty() {
            compacted_history
        } else {
            tracing::error!(
                session_id = %self.session_info.id,
                violation_count = remaining_violations.len(),
                violation_ids = ?remaining_violations,
                "compaction: sanitized history still has invalid ToolResults -- \
                 falling back to minimal compacted history (no recent_messages)"
            );
            build_compacted_history(CompactedHistoryInput {
                system_message,
                user_message_prefix,
                agents_md_reminder,
                state_context: &state_context.for_compaction(),
                compaction_summary: generate_session_compact.clone(),
                system_reminder,
                summary_before_recent: use_short_prompt,
                transcript_hint,
                summary_count,
            })
        };
        let original_user_info = full_conversation.iter().find_map(|item| match item {
            ConversationItem::User(parts) => parts.content.iter().find_map(|part| match part {
                xai_grok_sampling_types::ContentPart::Text { text } => Some(text.to_string()),
                _ => None,
            }),
            _ => None,
        });
        if cancel.is_cancelled() {
            let tokens = self.chat_state_handle.get_total_tokens().await;
            let result = self.emit_compact_cancelled(auto_trigger).await;
            compaction.complete(tokens);
            return match result {
                Err(error) => Err(error),
                Ok(()) => unreachable!("emit_compact_cancelled always returns Err"),
            };
        }
        self.persist_compaction_segment(&segment_messages, &generate_session_compact);
        let checkpoint_marker = self.persist_compaction_checkpoint_file(
            &compacted_history,
            prompt_index_at_compaction,
            auto_continue.clone(),
            original_user_info,
        );
        let prefix_len = if self
            .compaction
            .prefix_released
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0
        } else {
            self.startup_hints.inherited_prefix_len.unwrap_or(0)
        };
        let compacted_history = if prefix_len == 0 {
            compacted_history
        } else {
            self.resolve_forked_compacted_history(
                compacted_history,
                prefix_len,
                tokens_before,
                context_window,
            )
            .await
        };
        let new_len = compacted_history.len();
        let committed_total_tokens =
            xai_chat_state::estimate_conversation_tokens(&compacted_history);
        if cancel.is_cancelled() {
            let tokens = self.chat_state_handle.get_total_tokens().await;
            let result = self.emit_compact_cancelled(auto_trigger).await;
            compaction.complete(tokens);
            return match result {
                Err(error) => Err(error),
                Ok(()) => unreachable!("emit_compact_cancelled always returns Err"),
            };
        }
        let operation_id = uuid::Uuid::now_v7().to_string();
        if !self
            .commit_compaction_replacement(
                operation_id,
                expected_history_revision,
                expected_identity_generation,
                compacted_history,
                committed_total_tokens,
            )
            .await?
        {
            self.log_strategy_attempt(
                &compaction.compaction_id,
                supersedes_compaction_id.as_deref(),
                prepared_server.as_ref(),
                if migration_reason.is_some() {
                    "builtin_migration"
                } else if uses_builtin_fallback {
                    "builtin_fallback"
                } else {
                    "builtin_direct"
                },
                false,
                migration_reason,
                telemetry.attempts.min(u32::from(u8::MAX)) as u8,
                "superseded",
                None,
                None,
                builtin_started.elapsed().as_millis() as u64,
                0,
                0,
                false,
                "superseded",
                None,
                None,
                None,
                None,
            );
            let current_tokens = self.chat_state_handle.get_total_tokens().await;
            let superseded_compaction_id = compaction.compaction_id.clone();
            compaction.complete(current_tokens);
            return Ok(CompactionAttemptOutcome::Superseded {
                compaction_id: superseded_compaction_id,
                strategy_started_notified,
            });
        }
        self.chat_state_handle
            .record_compaction_at(prompt_index_at_compaction);
        self.persist_xai_update_only(
            crate::extensions::notification::SessionUpdate::CompactionCheckpoint(Box::new(
                checkpoint_marker,
            )),
        );
        let (respond_to, response) = tokio::sync::oneshot::channel();
        if self
            .notifications
            .persistence_tx
            .send(PersistenceMsg::FlushAndAck { respond_to })
            .is_ok()
        {
            let _ = response.await;
        }
        let tokens_after = self
            .finish_committed_compaction(new_len, context_window, compact_source)
            .await;
        {
            let span = tracing::Span::current();
            span.record("compaction_tokens_after", tokens_after as i64);
            span.record(
                "compaction_summary_chars",
                compact_output.content.chars().count() as i64,
            );
            span.record("compaction_attempts", telemetry.attempts as i64);
            span.record(
                "compaction_degenerate_rejections",
                telemetry.degenerate_rejections as i64,
            );
            span.record(
                "compaction_input_overflow_rejections",
                input_overflow_rejections as i64,
            );
            span.record(
                "compaction_deterministic_rejections",
                telemetry.deterministic_rejections as i64,
            );
            span.record(
                "compaction_transient_rejections",
                telemetry.transient_rejections as i64,
            );
            let stop_reason = compact_output.stop_reason.as_deref().unwrap_or("stop");
            span.record("compaction_stop_reason", stop_reason);
            let outcome = if compact_output.truncated {
                CompactionOutcome::Truncated
            } else {
                CompactionOutcome::Success
            };
            span.record("compaction_outcome", outcome.as_str());
            span.record("compaction_delta_count", compact_output.delta_count as i64);
            if let Some(ms) = compact_output.ttft_ms {
                span.record("compaction_ttft_ms", ms as i64);
            }
            if let Some(ms) = compact_output.stream_ms {
                span.record("compaction_stream_ms", ms as i64);
            }
            if let Some(ms) = compact_output.itl_max_ms {
                span.record("compaction_itl_max_ms", ms as i64);
            }
        }
        let builtin_strategy = if migration_reason.is_some() {
            "builtin_migration"
        } else if uses_builtin_fallback {
            "builtin_fallback"
        } else {
            "builtin_direct"
        };
        self.log_strategy_attempt(
            &compaction.compaction_id,
            supersedes_compaction_id.as_deref(),
            prepared_server.as_ref(),
            builtin_strategy,
            false,
            migration_reason,
            telemetry.attempts.min(u32::from(u8::MAX)) as u8,
            "committed",
            None,
            None,
            builtin_started.elapsed().as_millis() as u64,
            0,
            0,
            false,
            "committed",
            None,
            Some(tokens_after),
            None,
            None,
        );
        compaction.complete(tokens_after);
        Ok(CompactionAttemptOutcome::Committed)
    }
    /// Check if auto-compact should be triggered based on context window usage.
    /// Returns Some(AutoCompactTriggerInfo) if threshold is reached, None otherwise.
    pub(crate) fn should_auto_compact(
        &self,
        total_tokens: u64,
        context_window: std::num::NonZeroU64,
    ) -> Option<AutoCompactTriggerInfo> {
        let cw = context_window.get();
        if xai_token_estimation::exceeds_threshold(
            total_tokens,
            cw,
            self.compaction.threshold_percent.get(),
        ) {
            let percentage = xai_token_estimation::usage_percentage_u8(total_tokens, cw);
            Some(AutoCompactTriggerInfo {
                tokens_used: total_tokens,
                context_window: cw,
                percentage,
            })
        } else {
            None
        }
    }
    /// Returns true if the error response indicates tokens exceed the
    /// model's context window. Inspects only the model-metadata
    /// portion of the [`SamplingErrorInfo`] (the `context_window`
    /// field) against the session's tracked token estimate.
    ///
    /// Called from `handle_sampling_failure` with the
    /// `SamplingErrorInfo` the sampler hands back.
    pub(crate) async fn should_compact_on_error(
        &self,
        err: &xai_grok_sampler::SamplingErrorInfo,
    ) -> bool {
        if self
            .compaction
            .auto_compact_suppressed
            .load(std::sync::atomic::Ordering::Relaxed)
            != SUPPRESS_NONE
        {
            return false;
        }
        let Some(ref metadata) = err.model_metadata else {
            return false;
        };
        let Some(context_window) = metadata.context_window else {
            return false;
        };
        if context_window == 0 {
            return false;
        }
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        estimated_total > context_window
    }
    /// Pre-sampling compaction check. Uses `get_estimated_total_tokens()`
    /// (exact prior count + byte-estimate of items since last response) so
    /// tool results are accounted for. Returns `None` when `is_flushing`.
    pub(crate) async fn check_auto_compact_needed(&self) -> Option<AutoCompactTriggerInfo> {
        if self
            .memory
            .is_flushing
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return None;
        }
        let sampling_cfg = self.chat_state_handle.get_sampling_config().await;
        let context_window = sampling_cfg.as_ref().map(|c| c.context_window)?;
        let cw = context_window.get();
        let model = sampling_cfg
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_default();
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        self.signals_handle()
            .update_context_usage(estimated_total, cw);
        if self
            .compaction
            .auto_compact_suppressed
            .load(std::sync::atomic::Ordering::Relaxed)
            != SUPPRESS_NONE
        {
            return None;
        }
        if self
            .compaction
            .force_compact
            .compare_exchange(
                true,
                false,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            let percentage = xai_token_estimation::usage_percentage_u8(estimated_total, cw);
            tracing::info!(
                "Forced auto-compact trigger (debug): model={model}, \
                 {percentage}% full ({estimated_total}/{cw} tokens)",
            );
            return Some(AutoCompactTriggerInfo {
                tokens_used: estimated_total,
                context_window: cw,
                percentage,
            });
        }
        if let Some(trigger_info) = self.should_auto_compact(estimated_total, context_window) {
            tracing::info!(
                "Pre-sampling auto-compact trigger: model={model}, \
                 {}% full ({}/{} tokens)",
                trigger_info.percentage,
                trigger_info.tokens_used,
                trigger_info.context_window,
            );
            return Some(trigger_info);
        }
        None
    }
    /// Returns `Some` when tool call outputs have pushed the estimated token
    /// count past the context window, indicating pre-emptive compaction is needed.
    pub(crate) async fn check_preflight_overflow(&self) -> Option<AutoCompactTriggerInfo> {
        if self
            .compaction
            .auto_compact_suppressed
            .load(std::sync::atomic::Ordering::Relaxed)
            != SUPPRESS_NONE
        {
            return None;
        }
        let estimated_total = self.chat_state_handle.get_estimated_total_tokens().await;
        let cfg = self.chat_state_handle.get_sampling_config().await?;
        let cw = cfg.context_window.get();
        if estimated_total <= cw {
            return None;
        }
        let overflow = estimated_total.saturating_sub(cw);
        let percentage = xai_token_estimation::usage_percentage_u8(estimated_total, cw);
        tracing::warn!(
            estimated_total,
            context_window = cw,
            overflow,
            model = %cfg.model,
            "CONTEXT_OVERFLOW_PREFLIGHT: estimated tokens exceed context window \
             after tool call outputs"
        );
        Some(AutoCompactTriggerInfo {
            tokens_used: estimated_total,
            context_window: cw,
            percentage,
        })
    }
    /// On model change: clear sticky/other suppress and compact if the window shrank.
    /// Leaves credit/auth suppress (a switch can't fix those) and short-circuits.
    /// Auth compact failures abort the turn (same as pre-sampling/preflight).
    pub(crate) async fn maybe_compact_on_model_switch(self: &Arc<Self>) -> Result<(), acp::Error> {
        self.refresh_token_if_expired().await;
        let Some(prev) = self.compaction.previous_model.take() else {
            return Ok(());
        };
        let Some(cfg) = self.chat_state_handle.get_sampling_config().await else {
            return Ok(());
        };
        if cfg.model == prev.model_slug {
            return Ok(());
        }
        if self.is_account_state_suppressed() {
            return Ok(());
        }
        self.compaction
            .auto_compact_suppressed
            .store(SUPPRESS_NONE, std::sync::atomic::Ordering::Relaxed);
        if prev.context_window <= cfg.context_window.get() {
            return Ok(());
        }
        let total_tokens = self.chat_state_handle.get_estimated_total_tokens().await;
        let Some(trigger_info) = self.should_auto_compact(total_tokens, cfg.context_window) else {
            return Ok(());
        };
        tracing::info!(
            "Proactive model-switch compact: {} ({}) -> {} ({}), {}% full",
            prev.model_slug,
            prev.context_window,
            cfg.model,
            cfg.context_window.get(),
            trigger_info.percentage,
        );
        // Defer to the next exact-request pre-provider stage so the downshift
        // compaction snapshots the final tools/prompt/auth envelope.
        self.compaction
            .force_compact
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    /// Record the current model for model-switch detection on the next turn.
    pub(crate) async fn record_turn_model(&self) {
        if let Some(cfg) = self.chat_state_handle.get_sampling_config().await {
            self.compaction.previous_model.set(Some(
                crate::session::compaction_config::PreviousModelInfo {
                    model_slug: cfg.model.clone(),
                    context_window: cfg.context_window.get(),
                },
            ));
        }
    }
    /// Compact without auto-continue. The outer turn loop rebuilds and retries.
    pub(crate) async fn run_compact_only(
        self: &Arc<Self>,
        trigger_info: AutoCompactTriggerInfo,
    ) -> Result<(), acp::Error> {
        self.run_compact_only_with_request(trigger_info, None).await
    }

    /// Same operation when the turn already froze its exact normal request.
    /// Emits telemetry (`auto_compact_fired`) and UI notifications automatically.
    #[tracing::instrument(
        name = "session.compact",
        skip_all,
        fields(
            session_id = %self.session_info.id.0,
            trigger = "auto",
            mode = tracing::field::Empty,
            detail = tracing::field::Empty,
            pre_tokens = tracing::field::Empty,
            post_tokens = tracing::field::Empty,
            success = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    pub(crate) async fn run_compact_only_with_request(
        self: &Arc<Self>,
        trigger_info: AutoCompactTriggerInfo,
        normal_request: Option<ConversationRequest>,
    ) -> Result<(), acp::Error> {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        let (_cancel, _cancel_scope) = self.compaction.cancel.enter();
        self.record_compaction_variant();
        let tokens_before = self.chat_state_handle.get_total_tokens().await;
        tracing::Span::current().record("pre_tokens", tokens_before as i64);
        xai_grok_telemetry::session_ctx::log_event(xai_grok_telemetry::events::AutoCompactFired {
            tokens_before: trigger_info.tokens_used,
            percentage: trigger_info.percentage,
        });
        self.send_xai_notification(XaiSessionUpdate::AutoCompactStarted {
            tokens_used: trigger_info.tokens_used,
            context_window: trigger_info.context_window,
            percentage: trigger_info.percentage,
            reason: format!("Context window {}% full", trigger_info.percentage),
        })
        .await;
        self.maybe_pre_compaction_flush(
            trigger_info.tokens_used,
            trigger_info.context_window,
            "pre_compact_on_error",
        )
        .await;
        let compact_start = std::time::Instant::now();
        let result = self
            .run_compact_inner(
                None,
                None,
                xai_grok_telemetry::events::CompactionTrigger::Auto,
                CompactionStrategy::ServerFirst,
                normal_request,
                None,
                false,
                0,
            )
            .await;
        let elapsed_ms = compact_start.elapsed().as_millis() as i64;
        match result {
            Ok(()) => {
                let tokens_after = self.chat_state_handle.get_total_tokens().await;
                let span = tracing::Span::current();
                span.record("post_tokens", tokens_after as i64);
                span.record("success", true);
                self.send_xai_notification(XaiSessionUpdate::AutoCompactCompleted {
                    tokens_before: Some(trigger_info.tokens_used),
                    tokens_after,
                    elapsed_ms: Some(elapsed_ms),
                    summary_preview: None,
                })
                .await;
                Ok(())
            }
            Err(e) => {
                let span = tracing::Span::current();
                span.record("success", false);
                span.record("error", e.to_string().as_str());
                let cancelled = Self::is_compaction_cancelled(&e)
                    || self.compaction.cancel.is_cancelled()
                    || e.data
                        .as_ref()
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|message| {
                            message.contains(
                                crate::session::helpers::session_compact::COMPACT_CANCELLED_MSG,
                            )
                        });
                if cancelled {
                    return Err(e);
                }
                if self
                    .compaction
                    .auto_compact_suppressed
                    .load(std::sync::atomic::Ordering::Relaxed)
                    == SUPPRESS_NONE
                {
                    self.send_xai_notification(XaiSessionUpdate::AutoCompactFailed {
                        error: String::new(),
                    })
                    .await;
                }
                Err(e)
            }
        }
    }
    /// Persist a compaction request artifact for offline prompt iteration.
    ///
    /// Writes `{session_dir}/compaction_requests/{request_id}.json` containing
    /// the exact ConversationItem list sent to the compaction model plus the
    /// summary (or final error) it produced. The file rides on
    /// the post-turn session archive to cloud storage via the existing per-turn upload
    /// pipeline — no separate upload path is needed.
    ///
    /// `created_at` is taken from the caller-supplied `started_at` (captured
    /// before the retry loop) rather than `Utc::now()` here, so transient
    /// retries don't skew the timestamp away from when the call actually
    /// started.
    ///
    /// Best-effort: send-failures are logged at `warn` and never surfaced to
    /// the user, because the artifact is purely for offline analysis.
    #[allow(clippy::too_many_arguments)]
    fn persist_compaction_request_artifact(
        &self,
        chat_history: Vec<ConversationItem>,
        tools: Vec<xai_grok_sampling_types::ToolSpec>,
        user_context: Option<&str>,
        use_short_prompt: bool,
        model: &str,
        trigger: xai_grok_telemetry::events::CompactionTrigger,
        summary: Option<&str>,
        error: Option<&acp::Error>,
        attempts: u32,
        attempt_details: Vec<CompactionAttempt>,
        started_at: String,
    ) {
        use crate::extensions::notification::CompactionRequestFile;
        let request_id = uuid::Uuid::new_v4().to_string();
        let trigger_str = match trigger {
            xai_grok_telemetry::events::CompactionTrigger::Manual => "manual",
            xai_grok_telemetry::events::CompactionTrigger::Auto => "auto",
        };
        let prompt_variant = if use_short_prompt {
            "short"
        } else {
            "detailed"
        };
        let error_str = error.map(|e| {
            e.data
                .as_ref()
                .and_then(|d| d.as_str())
                .unwrap_or("<no error data>")
                .to_owned()
        });
        let artifact = CompactionRequestFile {
            schema_version: 2,
            request_id,
            created_at: started_at,
            trigger: trigger_str.to_owned(),
            prompt_variant: prompt_variant.to_owned(),
            model: model.to_owned(),
            user_context: user_context.map(str::to_owned),
            chat_history,
            tools,
            summary: summary.map(str::to_owned),
            error: error_str,
            attempts,
            attempt_details,
        };
        if self
            .notifications
            .persistence_tx
            .send(PersistenceMsg::CompactionRequest(artifact))
            .is_err()
        {
            tracing::warn!(
                session_id = %self.session_info.id.0,
                "Failed to send compaction request artifact to persistence channel"
            );
        }
    }
    /// Queue the builtin checkpoint file before history commit. The caller
    /// appends the returned marker only after the acknowledged CAS commits.
    fn persist_compaction_checkpoint_file(
        &self,
        compacted_history: &[ConversationItem],
        prompt_index_at_compaction: usize,
        auto_continue: Option<crate::extensions::notification::AutoContinueInfo>,
        original_user_info: Option<String>,
    ) -> crate::extensions::notification::CompactionCheckpointInfo {
        use crate::extensions::notification::{
            CompactionCheckpointFile, CompactionCheckpointInfo, CompactionCheckpointKind,
        };
        let checkpoint_id = uuid::Uuid::new_v4().to_string();
        let checkpoint_file = format!("compaction_checkpoints/{checkpoint_id}.json");
        let created_at = chrono::Utc::now().to_rfc3339();
        let file_data = CompactionCheckpointFile {
            kind: CompactionCheckpointKind::Builtin,
            checkpoint_id: checkpoint_id.clone(),
            prompt_index_at_compaction,
            compacted_history: compacted_history.to_vec(),
            created_at: created_at.clone(),
            original_user_info,
            reread_file_paths: vec![],
        };
        if self
            .notifications
            .persistence_tx
            .send(PersistenceMsg::CompactionCheckpoint(file_data))
            .is_err()
        {
            tracing::warn!("Failed to send compaction checkpoint file to persistence channel");
        }
        let info = CompactionCheckpointInfo {
            kind: CompactionCheckpointKind::Builtin,
            checkpoint_id,
            prompt_index_at_compaction,
            checkpoint_file,
            auto_continue,
            operation_id: None,
            branch_id: None,
            portable_history_sha256: None,
            responses_mode: None,
            responses_auto_continue: None,
            wrapper_digest: None,
            prior_checkpoint_id: None,
            created_at,
        };
        tracing::info!(
            prompt_index_at_compaction,
            "Queued compaction checkpoint file"
        );
        info
    }
}
#[cfg(test)]
mod inline_auto_compact_flow_tests {
    use super::super::support::*;
    use super::super::*;
    use super::{AutoCompactTriggerInfo, SuppressReason};
    use crate::session::acp_session::McpReminderMode;
    use crate::terminal::AsyncTerminalRunner;
    use crate::terminal::runner::{TerminalError, TerminalRunRequest, TerminalRunResult};
    use std::sync::OnceLock;
    use tokio::sync::mpsc;
    use xai_grok_paths::AbsPathBuf;
    use xai_grok_workspace::file_system::MockFs;
    use xai_grok_workspace::permission::PermissionHandle;
    #[derive(Debug)]
    struct DummyTerminal;
    #[async_trait::async_trait]
    impl AsyncTerminalRunner for DummyTerminal {
        async fn run(
            &self,
            _request: TerminalRunRequest,
        ) -> Result<TerminalRunResult, TerminalError> {
            Err(TerminalError::Other("dummy terminal".into()))
        }
    }
    /// Create a minimal SessionActor for testing auto-compact logic.
    async fn create_test_actor(
        total_tokens: u64,
        context_window: u64,
        threshold_percent: u8,
        gateway_tx: mpsc::UnboundedSender<xai_acp_lib::AcpClientMessage>,
        persistence_tx: mpsc::UnboundedSender<PersistenceMsg>,
    ) -> SessionActor {
        let cwd = AbsPathBuf::new(std::path::PathBuf::from("/tmp")).unwrap();
        let fs = Arc::new(MockFs::new(cwd.to_path_buf()));
        let terminal = Arc::new(DummyTerminal {});
        let (hunk_tx, _hunk_rx) = tokio::sync::mpsc::unbounded_channel();
        let hunk_tracker_handle = xai_hunk_tracker::HunkTrackerActor::spawn(
            "test-auto-compact".to_string(),
            cwd.to_path_buf(),
            hunk_tx,
            xai_hunk_tracker::TrackingMode::AgentOnly,
            tokio_util::sync::CancellationToken::new(),
        );
        let tool_context =
            ToolContext::new(cwd.clone(), None, None, fs, terminal, hunk_tracker_handle);
        let state = TokioMutex::new(State {
            running_task: None,
            pending_inputs: VecDeque::new(),
            combine_edit_holds: std::collections::HashSet::new(),
            pending_notifications: Vec::new(),
            notifications_suppressed: false,
            rewindable: false,
            nudges_used_this_session: 0,
        });
        let (chat_event_tx, _chat_event_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, _event_rx) =
            tokio::sync::mpsc::unbounded_channel::<crate::session::replay_events::SessionEvent>();
        let chat_state_handle = xai_chat_state::ChatStateActor::spawn(
            vec![],
            xai_grok_sampling_types::SamplingConfig {
                base_url: "http://localhost".to_string(),
                model: "test".to_string(),
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                api_backend: Default::default(),
                extra_headers: Default::default(),
                query_params: Default::default(),
                env_http_headers: Default::default(),
                context_window: std::num::NonZeroU64::new(context_window)
                    .expect("test context_window must be non-zero"),
                reasoning_effort: None,
                stream_tool_calls: None,
            },
            Box::new(xai_chat_state::NullChatPersistence),
            chat_event_tx,
            tokio_util::sync::CancellationToken::new(),
        );
        chat_state_handle.record_token_usage(total_tokens);
        SessionActor {
            unattributed_background_usage: std::sync::atomic::AtomicBool::new(false),
            session_info: SessionInfo {
                id: acp::SessionId::new("test-auto-compact"),
                cwd: cwd.as_str().to_string(),
            },
            auth_method_id: test_auth_method_id("test-auth"),
            model_auth_memo: std::cell::RefCell::new(None),
            attribution_callback: None,
            auth_manager: None,
            is_chat_kind: false,
            state,
            notifications: NotificationSender {
                gateway: GatewaySender::new(gateway_tx),
                gateway_enabled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
                persistence_tx,
            },
            permissions: PermissionHandle::allow_all(),
            tool_context,
            deny_read_globs: Vec::new(),
            mcp_state: Arc::new(TokioMutex::new(McpState::new(vec![]))),
            mcp_strategy: McpInitStrategy::Blocking,
            chat_state_handle,
            current_prompt_id: std::sync::Arc::new(std::sync::Mutex::new(None)),
            pending_interactions: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            telemetry_enabled: false,
            supports_backend_search: std::cell::Cell::new(false),
            tool_overrides: std::cell::RefCell::new(None),
            resolved_tool_overrides: std::sync::Arc::new(arc_swap::ArcSwapOption::empty()),
            compactions_remaining: std::cell::Cell::new(None),
            compaction_at_tokens: std::cell::Cell::new(None),
            doom_loop_recovery: None,
            doom_loop_turn_tally: Default::default(),
            file_state_tracker: Arc::new(FileStateTracker::new()),
            rewind_pending_prompt: std::sync::Mutex::new(None),
            startup_hints: StartupHints::default(),
            forked_tool_override: None,
            compaction: crate::session::compaction_config::CompactionConfig {
                threshold_percent: std::cell::Cell::new(threshold_percent),
                force_compact: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                context_window_override: None,
                count: std::sync::atomic::AtomicU64::new(0),
                auto_compact_suppressed: std::sync::atomic::AtomicU8::new(0),
                quota_pressure_notified_at: std::sync::atomic::AtomicI64::new(0),
                previous_model: std::cell::Cell::new(None),
                compaction_mode: xai_chat_state::CompactionMode::Transcript,
                verbatim_input: true,
                tool_choice: crate::util::config::CompactionToolChoice::Auto,
                prefire: crate::session::compaction_config::PrefireState::default(),
                prefix_released: std::sync::atomic::AtomicBool::new(false),
                cancel: Default::default(),
            },
            memory: crate::session::memory_state::SessionMemory {
                flush_config: crate::config::MemoryFlushConfig::default(),
                is_flushing: std::sync::atomic::AtomicBool::new(false),
                last_flush_compaction: std::sync::atomic::AtomicU64::new(0),
                storage: std::cell::RefCell::new(None),
                save_on_end: true,
                backend_params: None,
                initial_injection_config: Default::default(),
                context_injected: std::sync::atomic::AtomicBool::new(false),
                flush_count: std::sync::atomic::AtomicU64::new(0),
                last_flush_content: std::cell::RefCell::new(None),
                flush_success_count: std::sync::atomic::AtomicU64::new(0),
                flush_error_count: std::sync::atomic::AtomicU64::new(0),
                search_counter: std::cell::RefCell::new(None),
                injection_count: std::sync::atomic::AtomicU64::new(0),
                compaction_recovery_count: std::sync::atomic::AtomicU64::new(0),
                chunks_added: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                dream_config: Default::default(),
                dream_count: std::sync::atomic::AtomicU64::new(0),
                dream_success_count: std::sync::atomic::AtomicU64::new(0),
                dream_error_count: std::sync::atomic::AtomicU64::new(0),
            },
            session_start: std::time::Instant::now(),
            inference_idle_timeout: std::time::Duration::from_secs(300),
            max_retries: 3,
            max_turns: None,
            pending_interjections: InterjectionBuffer::new(),
            pending_skill_reminders: Mutex::new(Vec::new()),
            idle_flush_timeout: None,
            dream_check_timeout: None,
            last_idle_flush_conversation_len: std::sync::atomic::AtomicUsize::new(0),
            event_tx,
            buffering_settings: None,
            client_identifier: None,
            origin_client: None,
            feedback_manager: Arc::new(FeedbackManager::local_only("test-session")),
            upload_queue: Arc::new(OnceLock::new()),
            sync_loop_cancel: None,
            agent: std::cell::RefCell::new(test_agent_default().await),
            last_reported_branch: std::sync::Arc::new(parking_lot::Mutex::new(None)),
            git_head_enabled: false,
            models_manager: Default::default(),
            display_cwd: std::sync::OnceLock::new(),
            active_agent_type: parking_lot::Mutex::new(None),
            queue_exit_reminder_on_approved_exit: Arc::new(std::sync::atomic::AtomicBool::new(
                false,
            )),
            active_skill: parking_lot::Mutex::new(None),
            current_prompt_mode: Arc::new(parking_lot::Mutex::new(PromptMode::Agent)),
            turn_start_prompt_mode: parking_lot::Mutex::new(PromptMode::Agent),
            turn_prompt_mode: Arc::new(parking_lot::Mutex::new(PromptMode::Agent)),
            plan_mode: Arc::new(parking_lot::Mutex::new(
                crate::session::plan_mode::PlanModeTracker::new(std::path::PathBuf::from(
                    "/tmp/test-session",
                )),
            )),
            goal_enabled: false,
            background_workflows_enabled: false,
            goal_harness_enabled: std::sync::atomic::AtomicBool::new(false),
            goal_harness_availability_reconciled: std::sync::atomic::AtomicBool::new(false),
            goal_tracker: Arc::new(parking_lot::Mutex::new(
                crate::session::goal_tracker::GoalTracker::new(std::path::PathBuf::from(
                    "/tmp/test-session",
                )),
            )),
            goal_turn_task_ids: parking_lot::Mutex::new(std::collections::HashSet::new()),
            goal_continuation_streak: std::sync::atomic::AtomicU32::new(0),
            goal_blocked_streak: std::sync::atomic::AtomicU32::new(0),
            goal_update_rx: std::cell::RefCell::new(None),
            goal_update_tx: tokio::sync::mpsc::unbounded_channel().0,
            workflow_manager: crate::session::workflow::manager::WorkflowManager::test_bundle().0,
            workflow_launch_tx: tokio::sync::mpsc::unbounded_channel().0,
            goal_classifier_enabled: false,
            goal_planner_enabled: false,
            goal_summary_enabled: false,
            goal_verifier_skeptic_count: 1,
            goal_role_models: Default::default(),
            goal_use_current_model_only: false,
            goal_classifier_max_runs:
                crate::session::goal_classifier::GOAL_CLASSIFIER_MAX_RUNS_DEFAULT,
            goal_strategist_every: 5,
            goal_reverify_after: crate::session::acp_session::GOAL_REVERIFY_AFTER_DEFAULT,
            goal_plan_reconciled: std::sync::atomic::AtomicBool::new(false),
            pending_classifier_completions: parking_lot::Mutex::new(
                std::collections::VecDeque::new(),
            ),
            goal_classifier_in_flight: std::sync::atomic::AtomicBool::new(false),
            managed_mcp_handle: Default::default(),
            managed_mcp_expires_at: std::sync::Mutex::new(None),
            initial_client_mcp_servers: vec![],
            tool_metadata_snapshot: Arc::new(std::sync::Mutex::new(Default::default())),
            mcp_announced_servers: parking_lot::Mutex::new(std::collections::HashMap::new()),
            mcp_reminder_mode: McpReminderMode::Delta,
            mcp_reminder_dirty: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            mcp_connecting_reminder_injected: std::cell::Cell::new(false),
            mcp_handshakes_done: Arc::new(tokio::sync::Notify::new()),
            user_input_generation: std::sync::atomic::AtomicU64::new(0),
            laziness_debug_log: None,
            deferred_prefix: TaskSlot::new(),
            extension_registry: xai_agent_lifecycle::LocalExtensionRegistry::default(),
            last_announced_local_date: std::cell::Cell::new(chrono::Local::now().date_naive()),
            prefix_carries_fallback_date: std::cell::Cell::new(false),
            last_search_prompt_index: std::sync::atomic::AtomicI64::new(-1),
            last_api_request_at: std::sync::atomic::AtomicI64::new(0),
            hook_registry: std::cell::RefCell::new(None),
            client_hooks: Default::default(),
            hook_resolved_workspace_root: String::new(),
            vcs_kind: xai_grok_workspace::session::git::VcsKind::Git,
            hook_load_errors: std::cell::RefCell::new(Vec::new()),
            plugin_registry: std::cell::RefCell::new(None),
            plugin_registry_handle: None,
            events: crate::session::events::EventTracker::new(std::path::Path::new("/tmp")),
            observability_bridge: noop_observability_bridge(),
            current_turn_number: std::cell::Cell::new(0),
            last_recap_main_turn: std::cell::Cell::new(0),
            recap_in_flight: std::cell::Cell::new(false),
            recap_epoch: std::cell::Cell::new(0),
            session_turn_active: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            streaming_turn_capture: parking_lot::Mutex::new(
                crate::session::acp_session::StreamingTurnCapture::default(),
            ),
            turn_stream_drained: parking_lot::Mutex::new(None),
            post_compact_usage_state: std::sync::atomic::AtomicU8::new(0),
            sampler_handle: xai_grok_sampler::SamplerHandle::noop(),
            rebuild_spec: crate::session::agent_rebuild::test_rebuild_spec_default(),
            image_description_model: crate::test_support::TEST_MODEL.to_owned(),
            image_describe_cache: Arc::new(
                crate::session::image_describe::ImageDescribeCache::new(),
            ),
            subagent_token_records: parking_lot::Mutex::new(std::collections::HashMap::new()),
            workspace_ops: xai_grok_workspace::WorkspaceOps::for_test(),
            trace_config_template: std::cell::RefCell::new(None),
        }
    }
    /// Test check_auto_compact_needed uses state values.
    #[tokio::test(flavor = "current_thread")]
    async fn test_check_auto_compact_needed_uses_state() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) =
                    mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(90_000, 100_000, 85, gateway_tx, persistence_tx).await;
                let result = actor.check_auto_compact_needed().await;
                assert!(result.is_some(), "Should trigger at 90%");
                let info = result.unwrap();
                assert_eq!(info.percentage, 90);
            })
            .await;
    }
    /// Test that overriding context_window on the sampling config changes
    /// auto-compact behavior. Forked sessions must use the new model's
    /// context window, not the source session's. Without this, auto-compact
    /// fires at the wrong threshold.
    #[tokio::test(flavor = "current_thread")]
    async fn test_context_window_override_affects_auto_compact() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) =
                    mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(86_000, 100_000, 85, gateway_tx, persistence_tx).await;
                let result = actor.check_auto_compact_needed().await;
                assert!(result.is_some(), "Should trigger at 86% of 100K window");
                if let Some(mut cfg) = actor.chat_state_handle.get_sampling_config().await {
                    cfg.model = "larger-model".to_string();
                    cfg.context_window = std::num::NonZeroU64::new(200_000).unwrap();
                    actor.chat_state_handle.update_sampling_config(cfg);
                }
                let result = actor.check_auto_compact_needed().await;
                assert!(
                    result.is_none(),
                    "Should NOT trigger at 43% of 200K window after context_window override"
                );
            })
            .await;
    }
    /// Test the reverse direction: overriding to a smaller context window
    /// should make auto-compact trigger sooner.
    #[tokio::test(flavor = "current_thread")]
    async fn test_context_window_override_to_smaller_triggers_compact() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) =
                    mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(86_000, 200_000, 85, gateway_tx, persistence_tx).await;
                let result = actor.check_auto_compact_needed().await;
                assert!(result.is_none(), "Should NOT trigger at 43% of 200K window");
                if let Some(mut cfg) = actor.chat_state_handle.get_sampling_config().await {
                    cfg.model = "smaller-model".to_string();
                    cfg.context_window = std::num::NonZeroU64::new(100_000).unwrap();
                    actor.chat_state_handle.update_sampling_config(cfg);
                }
                let result = actor.check_auto_compact_needed().await;
                assert!(
                    result.is_some(),
                    "Should trigger at 86% of 100K window after context_window override"
                );
            })
            .await;
    }
    /// Suppression gates both AUTO paths; the reset scope depends on the reason:
    /// `other` clears next turn, `credit_block` holds until a successful model call,
    /// `size` is sticky until a full reset (success / rewind / model switch).
    #[tokio::test(flavor = "current_thread")]
    async fn suppression_gates_and_reset_is_reason_scoped() {
        use crate::session::compaction_config::{
            SUPPRESS_NONE, SUPPRESS_TURN, SUPPRESS_UNTIL_SUCCESS,
        };
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor =
                    create_test_actor(214_000, 200_000, 85, gateway_tx, persistence_tx).await;
                let err = api_error_with_context_window(200_000);
                assert!(actor.check_auto_compact_needed().await.is_some());
                assert!(actor.should_compact_on_error(&err).await);
                actor
                    .suppress_auto_compaction(SuppressReason::Other, 1_000, 200_000)
                    .await;
                assert!(actor.check_auto_compact_needed().await.is_none());
                assert!(!actor.should_compact_on_error(&err).await);
                let _ = actor.compaction.auto_compact_suppressed.compare_exchange(
                    SUPPRESS_TURN,
                    SUPPRESS_NONE,
                    Relaxed,
                    Relaxed,
                );
                assert!(actor.check_auto_compact_needed().await.is_some());
                actor
                    .suppress_auto_compaction(SuppressReason::CreditBlock, 1_000, 200_000)
                    .await;
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_UNTIL_SUCCESS
                );
                assert!(actor.check_auto_compact_needed().await.is_none());
                assert!(!actor.should_compact_on_error(&err).await);
                let _ = actor.compaction.auto_compact_suppressed.compare_exchange(
                    SUPPRESS_TURN,
                    SUPPRESS_NONE,
                    Relaxed,
                    Relaxed,
                );
                assert!(
                    actor.check_auto_compact_needed().await.is_none(),
                    "credit-block suppression must survive the per-turn reset"
                );
                let _ = actor.compaction.auto_compact_suppressed.compare_exchange(
                    SUPPRESS_UNTIL_SUCCESS,
                    SUPPRESS_NONE,
                    Relaxed,
                    Relaxed,
                );
                assert!(actor.check_auto_compact_needed().await.is_some());
                actor
                    .suppress_auto_compaction(SuppressReason::Size, 1_000, 200_000)
                    .await;
                assert!(actor.check_auto_compact_needed().await.is_none());
                let _ = actor.compaction.auto_compact_suppressed.compare_exchange(
                    SUPPRESS_TURN,
                    SUPPRESS_NONE,
                    Relaxed,
                    Relaxed,
                );
                assert!(
                    actor.check_auto_compact_needed().await.is_none(),
                    "sticky suppression must survive the per-turn reset"
                );
                actor
                    .compaction
                    .auto_compact_suppressed
                    .store(SUPPRESS_NONE, Relaxed);
                assert!(actor.check_auto_compact_needed().await.is_some());
            })
            .await;
    }
    /// A model switch clears suppression the switch (or the fresh budget-driven
    /// trigger) can resolve — sticky size/schema and a stale per-turn `other` — so
    /// the gates re-evaluate against the new window. Account-state credit/auth is
    /// covered by `model_switch_keeps_account_state_suppression`.
    #[tokio::test(flavor = "current_thread")]
    async fn model_switch_clears_sticky_suppression() {
        use crate::session::compaction_config::{PreviousModelInfo, SUPPRESS_NONE};
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor = Arc::new(
                    create_test_actor(50_000, 200_000, 85, gateway_tx, persistence_tx).await,
                );
                for reason in [SuppressReason::Size, SuppressReason::Other] {
                    actor.suppress_auto_compaction(reason, 1_000, 200_000).await;
                    assert_ne!(
                        actor.compaction.auto_compact_suppressed.load(Relaxed),
                        SUPPRESS_NONE,
                        "{reason:?} should set suppression"
                    );
                    actor.compaction.previous_model.set(Some(PreviousModelInfo {
                        model_slug: "old-small-model".to_string(),
                        context_window: 100_000,
                    }));
                    actor
                        .maybe_compact_on_model_switch()
                        .await
                        .expect("non-auth model-switch path must not abort");
                    assert_eq!(
                        actor.compaction.auto_compact_suppressed.load(Relaxed),
                        SUPPRESS_NONE,
                        "model switch must clear {reason:?} suppression so the gates re-evaluate"
                    );
                }
            })
            .await;
    }
    /// Model switch must not clear credit/auth suppress or compact under it.
    #[tokio::test(flavor = "current_thread")]
    async fn model_switch_keeps_account_state_suppression() {
        use crate::session::compaction_config::{
            PreviousModelInfo, SUPPRESS_AUTH, SUPPRESS_UNTIL_SUCCESS,
        };
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor = Arc::new(
                    create_test_actor(214_000, 200_000, 85, gateway_tx, persistence_tx).await,
                );
                for (reason, expected) in [
                    (SuppressReason::CreditBlock, SUPPRESS_UNTIL_SUCCESS),
                    (SuppressReason::Auth, SUPPRESS_AUTH),
                ] {
                    actor.suppress_auto_compaction(reason, 1_000, 200_000).await;
                    assert_eq!(
                        actor.compaction.auto_compact_suppressed.load(Relaxed),
                        expected,
                        "{reason:?} suppress state"
                    );
                    actor.compaction.previous_model.set(Some(PreviousModelInfo {
                        model_slug: "old-big-model".to_string(),
                        context_window: 400_000,
                    }));
                    actor
                        .maybe_compact_on_model_switch()
                        .await
                        .expect("suppressed model-switch path must not abort");
                    assert_eq!(
                        actor.compaction.auto_compact_suppressed.load(Relaxed),
                        expected,
                        "model switch must NOT clear {reason:?} suppression"
                    );
                    actor
                        .compaction
                        .auto_compact_suppressed
                        .store(crate::session::compaction_config::SUPPRESS_NONE, Relaxed);
                }
            })
            .await;
    }
    /// Auth suppress clears on credential recovery, not on a model 200.
    #[tokio::test(flavor = "current_thread")]
    async fn auth_suppress_clears_on_credential_recovery() {
        use crate::session::compaction_config::{SUPPRESS_AUTH, SUPPRESS_NONE};
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor =
                    create_test_actor(180_000, 200_000, 85, gateway_tx, persistence_tx).await;
                actor
                    .suppress_auto_compaction(SuppressReason::Auth, 1_000, 200_000)
                    .await;
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_AUTH
                );
                assert!(actor.check_auto_compact_needed().await.is_none());
                actor.clear_auth_compact_suppression();
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_NONE
                );
                assert!(actor.check_auto_compact_needed().await.is_some());
            })
            .await;
    }
    /// Auth recovery must not clear credit suppress.
    #[tokio::test(flavor = "current_thread")]
    async fn clear_auth_suppress_leaves_credit_suppress() {
        use crate::session::compaction_config::SUPPRESS_UNTIL_SUCCESS;
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor =
                    create_test_actor(180_000, 200_000, 85, gateway_tx, persistence_tx).await;
                actor
                    .suppress_auto_compaction(SuppressReason::CreditBlock, 1_000, 200_000)
                    .await;
                actor.clear_auth_compact_suppression();
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_UNTIL_SUCCESS,
                    "credential recovery must not clear a credit-block suppress"
                );
            })
            .await;
    }
    /// After /login, clearing auth suppress must re-arm pre-sampling compact
    /// before the next sample (ordering that prepare_sampler-after-gate broke).
    #[tokio::test(flavor = "current_thread")]
    async fn clear_auth_suppress_rearms_pre_sampling_compact_gate() {
        use crate::session::compaction_config::SUPPRESS_AUTH;
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor =
                    create_test_actor(180_000, 200_000, 85, gateway_tx, persistence_tx).await;
                actor
                    .suppress_auto_compaction(SuppressReason::Auth, 1_000, 200_000)
                    .await;
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_AUTH
                );
                assert!(
                    actor.check_auto_compact_needed().await.is_none(),
                    "auth suppress must block pre-sampling compact"
                );
                actor.clear_auth_compact_suppression();
                assert!(
                    actor.check_auto_compact_needed().await.is_some(),
                    "after credential recovery, pre-sampling compact must re-arm"
                );
            })
            .await;
    }
    #[test]
    fn is_auth_compact_error_classifies_401_messages() {
        let auth = acp::Error::internal_error()
            .data("compact failed: API error (status 401 Unauthorized)");
        assert!(SessionActor::is_auth_compact_error(&auth));
        let credit = acp::Error::internal_error().data("compact failed: out of credits");
        assert!(!SessionActor::is_auth_compact_error(&credit));
        let size = acp::Error::internal_error()
            .data("compact failed: The prompt is too long for this model's context window.");
        assert!(!SessionActor::is_auth_compact_error(&size));
    }
    #[tokio::test(flavor = "current_thread")]
    async fn surface_compact_auth_failure_emits_reauthable_retry_state() {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        use crate::session::storage::SessionUpdate;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, mut persistence_rx) = mpsc::unbounded_channel();
                let actor =
                    create_test_actor(10_000, 200_000, 85, gateway_tx, persistence_tx).await;
                let err = acp::Error::internal_error()
                    .data("compact failed: API error (status 401 Unauthorized)");
                let out = actor.surface_compact_auth_failure(err).await;
                assert_eq!(out.code, acp::Error::auth_required().code);
                let mut saw_retry_auth = false;
                while let Ok(msg) = persistence_rx.try_recv() {
                    if let PersistenceMsg::Update(SessionUpdate::Xai(notif)) = msg
                        && let XaiSessionUpdate::RetryState(
                            crate::extensions::notification::RetryState::Failed {
                                error_type,
                                message,
                            },
                        ) = &notif.update
                    {
                        assert_eq!(error_type, "auth");
                        assert!(
                            message.contains("Unauthorized (401)") || message.contains("401"),
                            "message={message}"
                        );
                        saw_retry_auth = true;
                    }
                }
                assert!(
                    saw_retry_auth,
                    "expected RetryState::Failed auth notification"
                );
            })
            .await;
    }
    /// The per-turn suppression notification is tailored to the failure reason.
    #[tokio::test(flavor = "current_thread")]
    async fn suppression_notification_is_reason_specific() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                async fn notification_for(reason: SuppressReason) -> String {
                    let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                    let (persistence_tx, mut persistence_rx) = mpsc::unbounded_channel();
                    let actor = create_test_actor(
                            10_000,
                            200_000,
                            85,
                            gateway_tx,
                            persistence_tx,
                        )
                        .await;
                    actor.suppress_auto_compaction(reason, 1_000, 200_000).await;
                    let mut text = None;
                    while let Ok(msg) = persistence_rx.try_recv() {
                        if let PersistenceMsg::Update(
                            crate::session::storage::SessionUpdate::Xai(notif),
                        ) = msg
                            && let crate::extensions::notification::SessionUpdate::AutoCompactFailed {
                                error,
                            } = &notif.update
                        {
                            text = Some(error.clone());
                        }
                    }
                    text.expect("expected an AutoCompactFailed notification")
                }
                let credit = notification_for(SuppressReason::CreditBlock).await;
                assert!(credit.contains("spending limit"), "credit_block: {credit}");
                let auth = notification_for(SuppressReason::Auth).await;
                assert!(auth.contains("/login"), "auth: {auth}");
                let size = notification_for(SuppressReason::Size).await;
                assert!(size.contains("too large to compact"), "size: {size}");
                let schema = notification_for(SuppressReason::Schema).await;
                assert!(schema.contains("can't be summarized"), "schema: {schema}");
                let other = notification_for(SuppressReason::Other).await;
                assert!(other.contains("/new"), "other: {other}");
            })
            .await;
    }
    /// Mock LLM endpoint answering every request with a deterministic 400.
    async fn spawn_deterministic_400_server() -> String {
        spawn_status_body_server(
            400,
            r#"{"error":{"type":"invalid_request_error","message":"bad schema"}}"#,
        )
        .await
    }
    /// Mock LLM that answers every request with 401.
    async fn spawn_deterministic_401_server() -> String {
        spawn_status_body_server(
            401,
            r#"{"error":{"type":"authentication_error","message":"Unauthorized (401)"}}"#,
        )
        .await
    }
    async fn spawn_status_body_server(status: u16, body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let status_line = match status {
            400 => "400 Bad Request",
            401 => "401 Unauthorized",
            other => panic!("add status line for {other}"),
        };
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let resp = format!(
                        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len(),
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }
    /// 401 auto-compact: SUPPRESS_AUTH + reauthable RetryState (abort for /login).
    #[tokio::test(flavor = "current_thread")]
    async fn e2e_auto_compact_401_suppresses_auth_and_surfaces_reauth() {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        use crate::session::compaction_config::SUPPRESS_AUTH;
        use crate::session::storage::SessionUpdate;
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, mut persistence_rx) = mpsc::unbounded_channel();
                let actor = Arc::new(
                    create_test_actor(180_000, 200_000, 85, gateway_tx, persistence_tx).await,
                );
                let base_url = spawn_deterministic_401_server().await;
                let mut cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                cfg.base_url = base_url;
                actor.chat_state_handle.update_sampling_config(cfg);
                actor.chat_state_handle.replace_conversation(vec![
                    ConversationItem::system("sys"),
                    ConversationItem::user("hello"),
                    ConversationItem::assistant("hi"),
                    ConversationItem::user("compact me"),
                ]);
                let err = actor
                    .run_compact_only(AutoCompactTriggerInfo {
                        tokens_used: 180_000,
                        context_window: 200_000,
                        percentage: 90,
                    })
                    .await
                    .expect_err("401 mock must fail auto-compact");
                assert!(
                    SessionActor::is_auth_compact_error(&err),
                    "401 compact failure must classify as auth: {err:?}"
                );
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_AUTH,
                    "auth compact failure must use SUPPRESS_AUTH (cleared on re-login)"
                );
                let surfaced = actor.surface_compact_auth_failure(err).await;
                assert_eq!(surfaced.code, acp::Error::auth_required().code);
                let mut saw_retry_auth = false;
                let mut saw_auto_failed = false;
                while let Ok(msg) = persistence_rx.try_recv() {
                    if let PersistenceMsg::Update(SessionUpdate::Xai(notif)) = msg {
                        match &notif.update {
                            XaiSessionUpdate::RetryState(
                                crate::extensions::notification::RetryState::Failed {
                                    error_type,
                                    message,
                                },
                            ) => {
                                assert_eq!(error_type, "auth");
                                assert!(
                                    message.contains("Unauthorized") || message.contains("401"),
                                    "message={message}"
                                );
                                saw_retry_auth = true;
                            }
                            XaiSessionUpdate::AutoCompactFailed { error } => {
                                assert!(
                                    error.contains("/login") || error.contains("authentication"),
                                    "auto-failed={error}"
                                );
                                saw_auto_failed = true;
                            }
                            _ => {}
                        }
                    }
                }
                assert!(saw_auto_failed, "expected AutoCompactFailed notification");
                assert!(
                    saw_retry_auth,
                    "expected RetryState::Failed auth so pager can stash + reauth"
                );
                actor.clear_auth_compact_suppression();
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    crate::session::compaction_config::SUPPRESS_NONE
                );
            })
            .await;
    }
    /// Model-switch compact 401 must surface reauth (same path as pre-sampling).
    #[tokio::test(flavor = "current_thread")]
    async fn e2e_model_switch_compact_401_surfaces_reauth() {
        use crate::extensions::notification::SessionUpdate as XaiSessionUpdate;
        use crate::session::compaction_config::{PreviousModelInfo, SUPPRESS_AUTH};
        use crate::session::storage::SessionUpdate;
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, mut persistence_rx) = mpsc::unbounded_channel();
                let actor = Arc::new(
                    create_test_actor(214_000, 200_000, 85, gateway_tx, persistence_tx).await,
                );
                let base_url = spawn_deterministic_401_server().await;
                let mut cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                cfg.base_url = base_url;
                actor.chat_state_handle.update_sampling_config(cfg);
                actor.chat_state_handle.replace_conversation(vec![
                    ConversationItem::system("sys"),
                    ConversationItem::user("hello"),
                    ConversationItem::assistant("hi"),
                    ConversationItem::user("compact me"),
                ]);
                actor.chat_state_handle.record_token_usage(214_000);
                actor.compaction.previous_model.set(Some(PreviousModelInfo {
                    model_slug: "old-big-model".to_string(),
                    context_window: 400_000,
                }));
                let err = actor
                    .maybe_compact_on_model_switch()
                    .await
                    .expect_err("model-switch 401 compact must abort for reauth");
                assert_eq!(err.code, acp::Error::auth_required().code);
                assert!(
                    SessionActor::is_auth_compact_error(&err)
                        || err.message.to_ascii_lowercase().contains("unauthorized")
                        || format!("{err:?}").contains("401"),
                    "surfaced error should be reauthable auth: {err:?}"
                );
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_AUTH,
                    "auth compact failure must use SUPPRESS_AUTH"
                );
                let mut saw_retry_auth = false;
                while let Ok(msg) = persistence_rx.try_recv() {
                    if let PersistenceMsg::Update(SessionUpdate::Xai(notif)) = msg
                        && let XaiSessionUpdate::RetryState(
                            crate::extensions::notification::RetryState::Failed {
                                error_type,
                                message,
                            },
                        ) = &notif.update
                    {
                        assert_eq!(error_type, "auth");
                        assert!(
                            message.contains("Unauthorized") || message.contains("401"),
                            "message={message}"
                        );
                        saw_retry_auth = true;
                    }
                }
                assert!(
                    saw_retry_auth,
                    "expected RetryState::Failed auth so pager can stash + reauth"
                );
            })
            .await;
    }
    /// Non-auth model-switch compact failures stay log-only (turn continues).
    #[tokio::test(flavor = "current_thread")]
    async fn e2e_model_switch_compact_non_auth_failure_does_not_abort() {
        use crate::session::compaction_config::{PreviousModelInfo, SUPPRESS_NONE};
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor = Arc::new(
                    create_test_actor(214_000, 200_000, 85, gateway_tx, persistence_tx).await,
                );
                let base_url = spawn_deterministic_400_server().await;
                let mut cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                cfg.base_url = base_url;
                actor.chat_state_handle.update_sampling_config(cfg);
                actor.chat_state_handle.replace_conversation(vec![
                    ConversationItem::system("sys"),
                    ConversationItem::user("hello"),
                ]);
                actor.chat_state_handle.record_token_usage(214_000);
                actor.compaction.previous_model.set(Some(PreviousModelInfo {
                    model_slug: "old-big-model".to_string(),
                    context_window: 400_000,
                }));
                actor
                    .maybe_compact_on_model_switch()
                    .await
                    .expect("non-auth model-switch compact failure must not abort the turn");
                assert_ne!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_NONE,
                    "schema/other compact failure must suppress after attempt"
                );
            })
            .await;
    }
    /// After clearing auth suppress, a shrink switch can re-evaluate and compact.
    #[tokio::test(flavor = "current_thread")]
    async fn clear_auth_suppress_allows_model_switch_compact_reeval() {
        use crate::session::compaction_config::{PreviousModelInfo, SUPPRESS_AUTH, SUPPRESS_NONE};
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor = Arc::new(
                    create_test_actor(214_000, 200_000, 85, gateway_tx, persistence_tx).await,
                );
                actor
                    .suppress_auto_compaction(SuppressReason::Auth, 1_000, 200_000)
                    .await;
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_AUTH
                );
                actor.compaction.previous_model.set(Some(PreviousModelInfo {
                    model_slug: "old-big-model".to_string(),
                    context_window: 400_000,
                }));
                actor
                    .maybe_compact_on_model_switch()
                    .await
                    .expect("suppressed switch must not abort");
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_AUTH
                );
                actor.clear_auth_compact_suppression();
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_NONE
                );
                actor.compaction.previous_model.set(Some(PreviousModelInfo {
                    model_slug: "old-big-model".to_string(),
                    context_window: 400_000,
                }));
                let base_url = spawn_deterministic_400_server().await;
                let mut cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                cfg.base_url = base_url;
                actor.chat_state_handle.update_sampling_config(cfg);
                actor.chat_state_handle.replace_conversation(vec![
                    ConversationItem::system("sys"),
                    ConversationItem::user("hello"),
                ]);
                actor.chat_state_handle.record_token_usage(214_000);
                actor
                    .maybe_compact_on_model_switch()
                    .await
                    .expect("post-clear switch compact re-eval must not abort on non-auth");
                assert_ne!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_NONE,
                    "post-clear switch must re-evaluate and attempt compact"
                );
            })
            .await;
    }
    /// A deterministic failure suppresses auto-compaction only on the AUTO
    /// path — never for a bare manual `/compact`.
    #[tokio::test(flavor = "current_thread")]
    async fn bare_manual_compact_failure_does_not_suppress_auto() {
        use crate::session::compaction_config::SUPPRESS_NONE;
        use std::sync::atomic::Ordering::Relaxed;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let actor = Arc::new(
                    create_test_actor(50_000, 200_000, 85, gateway_tx, persistence_tx).await,
                );
                let base_url = spawn_deterministic_400_server().await;
                let mut cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                cfg.base_url = base_url;
                actor.chat_state_handle.update_sampling_config(cfg);
                actor.chat_state_handle.replace_conversation(vec![
                    ConversationItem::system("sys"),
                    ConversationItem::user("hello"),
                ]);
                let result = actor.run_compact(None).await;
                assert!(result.is_err(), "mock 400 must fail the compaction");
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_NONE,
                    "manual /compact (even without args) must never set auto-compact suppression"
                );
                let result = actor
                    .run_compact_only(AutoCompactTriggerInfo {
                        tokens_used: 180_000,
                        context_window: 200_000,
                        percentage: 90,
                    })
                    .await;
                assert!(result.is_err(), "mock 400 must fail the compaction");
                assert_ne!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_NONE,
                    "the same deterministic failure on the AUTO path must suppress"
                );
            })
            .await;
    }
    /// A forked session whose whole-transcript inherited prefix alone exceeds
    /// the auto-compact threshold releases the prefix on compaction (so the
    /// conversation can actually shrink below the threshold) and keeps the
    /// release sticky across further compactions (no unbounded compaction loop).
    #[tokio::test(flavor = "current_thread")]
    async fn forked_prefix_released_under_pressure_and_stays_released() {
        use crate::session::compaction_config::SUPPRESS_NONE;
        use std::sync::atomic::Ordering::Relaxed;
        use xai_grok_test_support::MockInferenceServer;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel();
                let filler = "x".repeat(8_000);
                let mut conv = vec![ConversationItem::system("small system prompt")];
                for i in 0..9 {
                    conv.push(ConversationItem::user(format!("u{i} {filler}")));
                    conv.push(ConversationItem::assistant(format!("a{i} {filler}")));
                }
                conv.push(ConversationItem::user("final query"));
                let prefix_len = conv.len();
                let mut actor = create_test_actor(0, 40_000, 80, gateway_tx, persistence_tx).await;
                actor.startup_hints.inherited_prefix_len = Some(prefix_len);
                let actor = Arc::new(actor);
                let server = MockInferenceServer::start().await.unwrap();
                server.set_response("Summary of prior work. ".repeat(30));
                let mut cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                cfg.base_url = server.url();
                actor.chat_state_handle.update_sampling_config(cfg);
                actor.chat_state_handle.replace_conversation(conv);
                let threshold_tokens = 40_000u64 * 80 / 100;
                let before = actor.chat_state_handle.get_total_tokens().await;
                assert!(
                    before > threshold_tokens,
                    "seed must exceed threshold: {before} <= {threshold_tokens}"
                );
                let result = actor.run_compact(None).await;
                assert!(result.is_ok(), "compaction should succeed: {result:?}");
                assert!(
                    actor.compaction.prefix_released.load(Relaxed),
                    "prefix must be released under pressure"
                );
                let after = actor.chat_state_handle.get_total_tokens().await;
                assert!(
                    after < threshold_tokens,
                    "released history must drop below threshold: {after} >= {threshold_tokens}"
                );
                assert!(
                    actor.chat_state_handle.get_conversation_len().await < prefix_len,
                    "conversation must shrink below the pinned prefix floor"
                );
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_NONE,
                    "a shrunk conversation must not suppress AUTO"
                );
                let result = actor.run_compact(None).await;
                assert!(
                    result.is_ok(),
                    "second compaction should succeed: {result:?}"
                );
                assert!(
                    actor.compaction.prefix_released.load(Relaxed),
                    "release must stay sticky across compactions"
                );
                let after2 = actor.chat_state_handle.get_total_tokens().await;
                assert!(
                    after2 < threshold_tokens,
                    "sticky release must keep the session under threshold: {after2}"
                );
            })
            .await;
    }
    /// When even the released (summarized) history still exceeds the threshold
    /// -- the pathological case where the system prompt alone is over budget --
    /// a forked session sets sticky suppression (WITHOUT a user-facing failure
    /// event) instead of clearing it, so AUTO is not immediately re-armed while the
    /// compaction itself still reports success.
    #[tokio::test(flavor = "current_thread")]
    async fn forked_release_still_over_threshold_suppresses_auto() {
        use crate::session::compaction_config::SUPPRESS_STICKY;
        use std::sync::atomic::Ordering::Relaxed;
        use xai_grok_test_support::MockInferenceServer;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel();
                let (persistence_tx, mut persistence_rx) = mpsc::unbounded_channel();
                let huge_system = "s".repeat(150_000);
                let conv = vec![
                    ConversationItem::system(huge_system),
                    ConversationItem::user("q"),
                    ConversationItem::assistant("a"),
                    ConversationItem::user("final query"),
                ];
                let prefix_len = conv.len();
                let mut actor = create_test_actor(0, 40_000, 80, gateway_tx, persistence_tx).await;
                actor.startup_hints.inherited_prefix_len = Some(prefix_len);
                let actor = Arc::new(actor);
                let server = MockInferenceServer::start().await.unwrap();
                server.set_response("Summary. ".repeat(70));
                let mut cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                cfg.base_url = server.url();
                actor.chat_state_handle.update_sampling_config(cfg);
                actor.chat_state_handle.replace_conversation(conv);
                let threshold_tokens = 40_000u64 * 80 / 100;
                let before = actor.chat_state_handle.get_total_tokens().await;
                assert!(
                    before > threshold_tokens,
                    "seed must exceed threshold: {before}"
                );
                let result = actor.run_compact(None).await;
                assert!(result.is_ok(), "compaction should succeed: {result:?}");
                assert!(
                    actor.compaction.prefix_released.load(Relaxed),
                    "prefix must be released under pressure"
                );
                assert_eq!(
                    actor.compaction.auto_compact_suppressed.load(Relaxed),
                    SUPPRESS_STICKY,
                    "an over-threshold released history must set sticky suppression"
                );
                let mut saw_failure = false;
                while let Ok(msg) = persistence_rx.try_recv() {
                    if let PersistenceMsg::Update(crate::session::storage::SessionUpdate::Xai(
                        notif,
                    )) = msg
                        && matches!(
                            &notif.update,
                            crate::extensions::notification::SessionUpdate::AutoCompactFailed { .. }
                        )
                    {
                        saw_failure = true;
                    }
                }
                assert!(
                    !saw_failure,
                    "successful compaction must not emit AutoCompactFailed"
                );
            })
            .await;
    }
    /// `classify_suppress_reason` maps each deterministic-failure shape to its
    /// fixed [`SuppressReason`].
    #[test]
    fn classify_suppress_reason_maps_error_text() {
        let classify = SessionActor::classify_suppress_reason;
        assert_eq!(
            classify("caller does not have permission … spending-limit reached"),
            SuppressReason::CreditBlock
        );
        assert_eq!(
            classify("you have run out of credits"),
            SuppressReason::CreditBlock
        );
        assert_eq!(
            classify("API error (status 402 Payment Required): Grok Build usage balance exhausted"),
            SuppressReason::CreditBlock
        );
        assert_eq!(
            classify("Grok Build usage limit reached"),
            SuppressReason::CreditBlock
        );
        assert_eq!(
            classify("This model's maximum prompt length is 500000"),
            SuppressReason::Size
        );
        assert_eq!(
            classify("compact failed: The prompt is too long for this model's context window."),
            SuppressReason::Size
        );
        assert_eq!(
            classify("provider error: context_length_exceeded"),
            SuppressReason::Size
        );
        assert_eq!(
            classify("API error (status 401 Unauthorized)"),
            SuppressReason::Auth
        );
        assert_eq!(
            classify("provider returned invalid_request_error: messages.3"),
            SuppressReason::Schema
        );
        assert_eq!(
            classify("upstream 500 internal error"),
            SuppressReason::Other
        );
    }

    #[test]
    fn compaction_cancel_classification_accepts_server_and_builtin_errors() {
        let server = acp::Error::internal_error().data("responses_compaction_cancelled");
        assert!(SessionActor::is_compaction_cancelled(&server));

        let builtin = crate::session::helpers::session_compact::CompactFailure::cancelled_error();
        assert!(SessionActor::is_compaction_cancelled(&builtin));

        let unrelated = acp::Error::internal_error().data("compaction failed");
        assert!(!SessionActor::is_compaction_cancelled(&unrelated));
    }

    /// `SuppressReason::as_str` is the stable telemetry wire value — BQ/OTLP and
    /// dashboards key off these exact strings. Lock them so a rename can't break monitoring.
    #[test]
    fn suppress_reason_as_str_is_stable() {
        assert_eq!(SuppressReason::CreditBlock.as_str(), "credit_block");
        assert_eq!(SuppressReason::Size.as_str(), "size");
        assert_eq!(SuppressReason::Auth.as_str(), "auth");
        assert_eq!(SuppressReason::Schema.as_str(), "schema");
        assert_eq!(SuppressReason::Other.as_str(), "other");
    }
    mod preserve_prefix {
        use super::super::preserve_inherited_prefix;
        use super::super::project_preserved_reseed_tokens;
        use xai_grok_sampling_types::conversation::ConversationItem;
        #[test]
        fn splices_inherited_with_compacted_suffix() {
            let conversation = vec![
                ConversationItem::system("sys"),
                ConversationItem::user("parent q1"),
                ConversationItem::assistant("parent a1"),
                ConversationItem::user("child q1"),
            ];
            let compacted = vec![
                ConversationItem::system("sys"),
                ConversationItem::user("summary"),
            ];
            let items = preserve_inherited_prefix(&conversation, compacted, 3).expect("Ok");
            assert_eq!(items.len(), 4);
            assert!(matches!(items[0], ConversationItem::System(_)));
        }
        /// Invariant: a head-only prefix lets compaction shrink the conversation;
        /// a whole-transcript prefix does not (that pinned floor is the loop).
        #[test]
        fn head_only_shrinks_full_transcript_does_not() {
            let mut conversation = vec![ConversationItem::system("sys")];
            for i in 0..8 {
                conversation.push(ConversationItem::user(format!("u{i}")));
                conversation.push(ConversationItem::assistant(format!("a{i}")));
            }
            let compacted = vec![
                ConversationItem::system("sys"),
                ConversationItem::assistant("summary"),
            ];
            let fixed = preserve_inherited_prefix(&conversation, compacted.clone(), 1).expect("Ok");
            assert!(fixed.len() < conversation.len(), "head-only shrinks");
            let buggy = preserve_inherited_prefix(&conversation, compacted, conversation.len())
                .expect("Ok");
            assert!(
                buggy.len() >= conversation.len(),
                "full prefix never shrinks"
            );
        }
        /// The reseed projection calibrates the bytes/4 estimate to real tokens
        /// (ratio != 1) and caps at the pre-compaction total, so the release
        /// decision reflects what the trigger applies next turn.
        #[test]
        fn project_preserved_reseed_tokens_calibrates_and_caps() {
            assert_eq!(
                project_preserved_reseed_tokens(30_000, 100_000, 50_000),
                60_000
            );
            assert_eq!(
                project_preserved_reseed_tokens(40_000, 70_000, 35_000),
                70_000
            );
            assert_eq!(
                project_preserved_reseed_tokens(20_000, 40_000, 40_000),
                20_000
            );
            assert_eq!(project_preserved_reseed_tokens(10, 5, 0), 5);
        }
        /// Both prefix and re-injected suffix may carry AGENTS.md; the splice must
        /// leave exactly one (else the model sees project instructions twice).
        #[test]
        fn does_not_duplicate_agents_md() {
            let conversation = vec![
                ConversationItem::system("sys"),
                ConversationItem::project_instructions("AGENTS.md"),
                ConversationItem::user("work"),
            ];
            let compacted = vec![
                ConversationItem::system("sys"),
                ConversationItem::project_instructions("AGENTS.md"),
                ConversationItem::user("summary"),
            ];
            let items = preserve_inherited_prefix(&conversation, compacted, 2).expect("Ok");
            let pi = items
                .iter()
                .filter(|i| super::super::is_project_instructions(i))
                .count();
            assert_eq!(pi, 1, "exactly one project-instructions item, not two");
        }
        #[test]
        fn keeps_reinjected_agents_md_when_prefix_lacks_it() {
            let conversation = vec![
                ConversationItem::system("sys"),
                ConversationItem::user("work"),
            ];
            let compacted = vec![
                ConversationItem::system("sys"),
                ConversationItem::project_instructions("AGENTS.md"),
                ConversationItem::user("summary"),
            ];
            let items = preserve_inherited_prefix(&conversation, compacted, 1).expect("Ok");
            let pi = items
                .iter()
                .filter(|i| super::super::is_project_instructions(i))
                .count();
            assert_eq!(
                pi, 1,
                "re-injected AGENTS.md preserved when prefix lacks one"
            );
        }
    }
    #[allow(clippy::field_reassign_with_default)]
    async fn create_test_actor_with_memory(
        total_tokens: u64,
        context_window: u64,
        threshold_percent: u8,
        gateway_tx: mpsc::UnboundedSender<xai_acp_lib::AcpClientMessage>,
        persistence_tx: mpsc::UnboundedSender<PersistenceMsg>,
        memory_config: Option<crate::config::MemoryConfig>,
    ) -> SessionActor {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd_path = tmp.path().to_path_buf();
        let memory_storage = memory_config
            .as_ref()
            .filter(|mc| mc.enabled)
            .map(|_| crate::session::memory::MemoryStorage::new(&cwd_path, None));
        std::mem::forget(tmp);
        let memory_initial_injection_config = memory_config
            .as_ref()
            .map_or_else(Default::default, |mc| mc.initial_injection.clone());
        let mut actor = create_test_actor(
            total_tokens,
            context_window,
            threshold_percent,
            gateway_tx,
            persistence_tx,
        )
        .await;
        actor.memory = crate::session::memory_state::SessionMemory {
            flush_config: memory_config
                .as_ref()
                .map_or_else(Default::default, |mc| mc.flush.clone()),
            is_flushing: std::sync::atomic::AtomicBool::new(false),
            last_flush_compaction: std::sync::atomic::AtomicU64::new(0),
            storage: std::cell::RefCell::new(memory_storage),
            save_on_end: true,
            backend_params: None,
            initial_injection_config: memory_initial_injection_config,
            context_injected: std::sync::atomic::AtomicBool::new(false),
            flush_count: std::sync::atomic::AtomicU64::new(0),
            last_flush_content: std::cell::RefCell::new(None),
            flush_success_count: std::sync::atomic::AtomicU64::new(0),
            flush_error_count: std::sync::atomic::AtomicU64::new(0),
            search_counter: std::cell::RefCell::new(None),
            injection_count: std::sync::atomic::AtomicU64::new(0),
            compaction_recovery_count: std::sync::atomic::AtomicU64::new(0),
            chunks_added: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            dream_config: Default::default(),
            dream_count: std::sync::atomic::AtomicU64::new(0),
            dream_success_count: std::sync::atomic::AtomicU64::new(0),
            dream_error_count: std::sync::atomic::AtomicU64::new(0),
        };
        actor.idle_flush_timeout = memory_config
            .as_ref()
            .and_then(|mc| mc.flush.idle_timeout_secs)
            .map(std::time::Duration::from_secs);
        actor.dream_check_timeout = memory_config
            .as_ref()
            .filter(|mc| mc.dream.enabled)
            .and_then(|mc| mc.dream.check_interval_secs)
            .filter(|&s| s > 0)
            .map(std::time::Duration::from_secs);
        actor
    }
    /// Verify that `last_idle_flush_conversation_len` is reset after
    /// compaction shrinks the conversation. Without this reset the
    /// interval flush guard (`current_len > last_len`) stays false
    /// because the compacted conversation is shorter than the stored
    /// pre-compaction length.
    #[tokio::test(flavor = "current_thread")]
    #[allow(clippy::field_reassign_with_default)]
    async fn test_idle_flush_conversation_len_reset_after_compaction() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _) = mpsc::unbounded_channel();
                let (persistence_tx, _) = mpsc::unbounded_channel();
                let mut config = crate::config::MemoryConfig::default();
                config.enabled = true;
                config.flush.idle_timeout_secs = Some(60);
                let actor = create_test_actor_with_memory(
                    50_000,
                    100_000,
                    85,
                    gateway_tx,
                    persistence_tx,
                    Some(config),
                )
                .await;
                for _ in 0..80 {
                    actor
                        .chat_state_handle
                        .push_user_message(ConversationItem::user("hello".to_string()));
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                actor
                    .last_idle_flush_conversation_len
                    .store(80, std::sync::atomic::Ordering::Relaxed);
                {
                    let current_len = actor.chat_state_handle.get_conversation_len().await;
                    let last_len = actor
                        .last_idle_flush_conversation_len
                        .load(std::sync::atomic::Ordering::Relaxed);
                    assert_eq!(current_len, 80);
                    assert!(
                        current_len <= last_len,
                        "guard should block: no new messages"
                    );
                }
                {
                    let compacted = vec![ConversationItem::user("compacted summary".to_string())];
                    let new_len = compacted.len();
                    actor
                        .chat_state_handle
                        .replace_conversation_for_compaction(compacted);
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    actor
                        .last_idle_flush_conversation_len
                        .store(new_len, std::sync::atomic::Ordering::Relaxed);
                }
                {
                    actor
                        .chat_state_handle
                        .push_user_message(ConversationItem::user("new message".to_string()));
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    let current_len = actor.chat_state_handle.get_conversation_len().await;
                    let last_len = actor
                        .last_idle_flush_conversation_len
                        .load(std::sync::atomic::Ordering::Relaxed);
                    assert_eq!(current_len, 2, "summary + new message");
                    assert_eq!(last_len, 1, "reset to post-compaction length");
                    assert!(
                        current_len > last_len,
                        "guard should allow flush after compaction + new message"
                    );
                }
            })
            .await;
    }
    fn api_error_with_context_window(context_window: u64) -> xai_grok_sampler::SamplingErrorInfo {
        xai_grok_sampler::SamplingErrorInfo {
            kind: xai_grok_sampler::SamplingErrorKind::Api,
            status_code: Some(400),
            message: "prompt is too long".to_string(),
            is_retryable: false,
            retry_after_secs: None,
            should_retry: None,
            model_metadata: Some(crate::sampling::ResponseModelMetadata {
                context_window: Some(context_window),
                max_completion_tokens: None,
                models_etag: None,
            }),
            empty_response_context: None,
            doom_loop_triggers: None,
            doom_loop_aborted_at_chunk: None,
            credential: xai_grok_sampling_types::SentCredential::Unknown,
        }
    }
    /// Primary scenario: remote settings shrinks the context window mid-session.
    /// The shell's last-known token count (214K) exceeds the new limit (200K) —
    /// should_compact_on_error must return true so the session can recover.
    #[tokio::test(flavor = "current_thread")]
    async fn test_compact_on_error_triggers_when_tokens_exceed_new_window() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _) = mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(214_000, 1_000_000, 85, gateway_tx, persistence_tx).await;
                let err = api_error_with_context_window(200_000);
                assert!(actor.should_compact_on_error(&err).await);
            })
            .await;
    }
    /// When tracked tokens are within the new limit, the error was not a context
    /// overflow — do not compact.
    #[tokio::test(flavor = "current_thread")]
    async fn test_compact_on_error_no_trigger_when_tokens_within_new_window() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _) = mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(150_000, 1_000_000, 85, gateway_tx, persistence_tx).await;
                let err = api_error_with_context_window(200_000);
                assert!(!actor.should_compact_on_error(&err).await);
            })
            .await;
    }
    /// If the proxy hasn't been updated yet, model_metadata is None — must be
    /// a no-op for backwards compatibility.
    #[tokio::test(flavor = "current_thread")]
    async fn test_compact_on_error_noop_without_model_metadata() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _) = mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(500_000, 200_000, 85, gateway_tx, persistence_tx).await;
                let err = xai_grok_sampler::SamplingErrorInfo {
                    kind: xai_grok_sampler::SamplingErrorKind::Api,
                    status_code: Some(400),
                    message: "prompt is too long".to_string(),
                    is_retryable: false,
                    retry_after_secs: None,
                    should_retry: None,
                    model_metadata: None,
                    empty_response_context: None,
                    doom_loop_triggers: None,
                    doom_loop_aborted_at_chunk: None,
                    credential: xai_grok_sampling_types::SentCredential::Unknown,
                };
                assert!(!actor.should_compact_on_error(&err).await);
            })
            .await;
    }
    /// Pre-sampling check uses estimated tokens (includes tool-result delta).
    #[tokio::test(flavor = "current_thread")]
    async fn test_pre_sampling_uses_estimated_tokens() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _) = mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(80_000, 100_000, 85, gateway_tx, persistence_tx).await;
                let result = actor.check_auto_compact_needed().await;
                assert!(result.is_none(), "80% should not trigger at 85% threshold");
                actor.chat_state_handle.record_token_usage(90_000);
                let result = actor.check_auto_compact_needed().await;
                assert!(result.is_some(), "90% should trigger");
                assert_eq!(result.unwrap().percentage, 90);
            })
            .await;
    }
    /// Model-switch compaction fires when switching to a smaller context window.
    #[tokio::test(flavor = "current_thread")]
    async fn test_model_switch_compaction_triggers_on_downgrade() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _) = mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _) = mpsc::unbounded_channel::<PersistenceMsg>();
                let actor =
                    create_test_actor(86_000, 100_000, 85, gateway_tx, persistence_tx).await;
                actor.compaction.previous_model.set(Some(
                    crate::session::compaction_config::PreviousModelInfo {
                        model_slug: "large-model".to_string(),
                        context_window: 200_000,
                    },
                ));
                let prev = actor.compaction.previous_model.take();
                assert!(prev.is_some());
                let prev = prev.unwrap();
                assert_eq!(prev.context_window, 200_000);
                let cfg = actor.chat_state_handle.get_sampling_config().await.unwrap();
                assert!(prev.context_window > cfg.context_window.get());
                let total = actor.chat_state_handle.get_estimated_total_tokens().await;
                let trigger = actor.should_auto_compact(total, cfg.context_window);
                assert!(trigger.is_some(), "86% > 85% threshold, should trigger");
                actor.compaction.previous_model.set(Some(
                    crate::session::compaction_config::PreviousModelInfo {
                        model_slug: "small-model".to_string(),
                        context_window: 50_000,
                    },
                ));
                let prev = actor.compaction.previous_model.take().unwrap();
                assert!(prev.context_window <= cfg.context_window.get());
            })
            .await;
    }
    #[tokio::test(flavor = "current_thread")]
    async fn get_transcript_path_returns_some_when_file_exists() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (gateway_tx, _gateway_rx) =
                    mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
                let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
                let mut actor =
                    create_test_actor(50_000, 200_000, 85, gateway_tx, persistence_tx).await;
                actor.compaction.compaction_mode = xai_chat_state::CompactionMode::Transcript;
                let session_dir = crate::session::persistence::session_dir(&actor.session_info);
                std::fs::create_dir_all(&session_dir).unwrap();
                let updates_path = session_dir.join("updates.jsonl");
                std::fs::write(&updates_path, "{}\n").unwrap();
                let result = actor.get_transcript_path();
                assert!(result.is_some(), "file exists → Some");
                assert!(
                    result.as_ref().unwrap().ends_with("updates.jsonl"),
                    "path should end with updates.jsonl, got: {:?}",
                    result,
                );
                let hint = actor.transcript_hint().expect("transcript hint present");
                assert!(hint.contains("read the full transcript"));
                assert!(hint.ends_with("updates.jsonl"));
                actor.compaction.compaction_mode = xai_chat_state::CompactionMode::Summary;
                assert!(actor.transcript_hint().is_none());
                let _ = std::fs::remove_file(&updates_path);
                let _ = std::fs::remove_dir_all(&session_dir);
            })
            .await;
    }
}

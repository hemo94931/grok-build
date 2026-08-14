//! ConversationRequest assembly — image compaction, pruning, repair, memory injection.

use xai_grok_sampling_types::{ConversationItem, ConversationRequest, ToolSpec, TraceContext};

use super::ChatStateActor;
use crate::events::ChatStateEvent;
use crate::image_budget::{ImageBudgetOutcome, apply_image_budget};
use crate::types::PruningConfig;

/// Placeholder inserted when a tool result is hard-cleared.
///
/// `pub(super)` so that `mutations.rs` can use the same string when it
/// hard-clears tool results in the retained in-memory conversation.
pub(super) const HARD_CLEAR_PLACEHOLDER: &str = "[Tool result omitted — too old]";

/// Separator inserted between head and tail in soft-trimmed results.
const SOFT_TRIM_SEPARATOR: &str = "\n\n[…trimmed…]\n\n";

impl ChatStateActor {
    /// Build a `ConversationRequest` from the current actor state.
    ///
    /// 1. Evict oldest inline images when the inline-image bytes near 50 MB
    /// 2. Prune old tool results if over 50% context utilization
    /// 3. Optionally persist the memory reminder into actor state
    /// 4. Inject memory reminder into the request clone (if needed)
    /// 5. Assemble and return the `ConversationRequest`
    ///
    /// # Repair invariant
    ///
    /// The `BuildConversationRequest` command handler calls
    /// `ensure_conversation_integrity()` on the actor's own conversation
    /// **before** this function runs. The clone therefore starts from an
    /// already-repaired state, so there is no need to run
    /// `dedup_duplicate_tool_results` / `repair_dangling_tool_calls` on the
    /// clone — those would be O(n) no-ops.
    pub(super) async fn build_conversation_request(
        &mut self,
        tool_definitions: Vec<ToolSpec>,
        memory_reminder: Option<String>,
        persist_memory_reminder: bool,
        trace: Option<Box<dyn TraceContext>>,
        conv_id: String,
        req_id: String,
    ) -> ConversationRequest {
        let needs_prune = should_prune(
            self.state.total_tokens,
            self.state.sampling_config.context_window,
        );
        let mut memory_reminder = memory_reminder;
        if let Some(reminder) = memory_reminder.as_deref()
            && persist_memory_reminder
        {
            if self
                .state
                .conversation
                .first()
                .is_some_and(|item| item.is_responses_checkpoint())
            {
                // A new reminder is a strict typed-tail append. An existing
                // dedicated reminder is durably replaced before memory changes.
                let _ = self.persist_checkpoint_memory_reminder(reminder).await;
                // Never send a reminder that failed its persistence boundary.
                memory_reminder = None;
            } else {
                // A live in-place inject can prepend a `System` item, shifting indices
                // under an active capture; snapshot + rebase like the other mutators.
                self.snapshot_turn_slice();
                let before_tokens =
                    super::state::estimate_conversation_tokens(&self.state.conversation);
                let injected = inject_memory_reminder(&mut self.state.conversation, reminder);
                if injected {
                    let after_tokens =
                        super::state::estimate_conversation_tokens(&self.state.conversation);
                    if after_tokens >= before_tokens {
                        self.state.estimated_tokens_since_model = self
                            .state
                            .estimated_tokens_since_model
                            .saturating_add(after_tokens - before_tokens);
                    } else {
                        self.state.estimated_tokens_since_model = self
                            .state
                            .estimated_tokens_since_model
                            .saturating_sub(before_tokens - after_tokens);
                    }
                    self.persistence.replace_history(&self.state.conversation);
                    self.state.bump_history_revision();
                    memory_reminder = None;
                }
                self.rebase_turn_capture_offset();
            }
        }
        let budgeted = apply_image_budget(self.state.conversation.clone());
        let ImageBudgetOutcome {
            body_bytes,
            body_bytes_after,
            inline_images,
            needs_image_compaction,
            evicted,
        } = budgeted.outcome;
        let mut items = budgeted.items;
        if inline_images > 0 {
            self.send_event(ChatStateEvent::ImageBudget {
                body_bytes,
                trigger_bytes: crate::image_budget::IMAGE_COMPACT_TRIGGER_BYTES,
                reclaim_target_bytes: crate::image_budget::IMAGE_COMPACT_RECLAIM_TARGET_BYTES,
                inline_images,
                needs_image_compaction,
                evicted,
                body_bytes_after,
            });
        }
        if needs_prune {
            prune_conversation(&mut items, &self.pruning_config);
        }
        if let Some(reminder) = memory_reminder {
            inject_memory_reminder(&mut items, &reminder);
        }

        // Step 4: Assemble request
        ConversationRequest {
            history_revision: Some(self.state.history_revision),
            items,
            tools: tool_definitions,
            hosted_tools: vec![],
            tool_choice: None,
            model: Some(self.state.sampling_config.model.clone()),
            temperature: self.state.sampling_config.temperature,
            max_output_tokens: self.state.sampling_config.max_completion_tokens,
            top_p: self.state.sampling_config.top_p,
            x_grok_conv_id: Some(conv_id),
            x_grok_req_id: Some(req_id),
            x_grok_session_id: None,
            x_grok_turn_idx: None,
            x_grok_agent_id: None,
            x_grok_deployment_id: None,
            x_grok_user_id: None,
            trace,
            instructions: None,
            prompt_cache_key: None,
            prompt_cache_options: None,
            prompt_cache_retention: None,
            service_tier: None,
            // The main agent loop supports parallel tool execution. Keep this
            // explicit on every request so normal, compact, and post-compact
            // envelope fingerprints bind the same semantics.
            parallel_tool_calls: Some(true),
            reasoning_effort: self.state.sampling_config.reasoning_effort,
            json_schema: None,
        }
    }

    async fn persist_checkpoint_memory_reminder(&mut self, reminder: &str) -> bool {
        // An existing memory item is either the dedicated `MemoryContext`
        // item or a legacy unmarked System still carrying the block; both
        // are replaced durably through the atomic history boundary.
        let existing = self.state.conversation.iter().skip(1).any(|item| {
            matches!(
                item,
                ConversationItem::System(system)
                    if system.source == SystemSource::MemoryContext
                        || system.content.contains(MEMORY_CONTEXT_OPEN_TAG)
            )
        });
        if !existing {
            // A new reminder is a strict typed-tail append of a *marked*
            // item — never an unmarked trailing System.
            let item = ConversationItem::memory_context(reminder);
            if self.persist_append(&item).await {
                self.apply_pushed_message(item);
                return true;
            }
            return false;
        }

        let mut replacement = self.state.conversation.clone();
        if !inject_memory_reminder(&mut replacement, reminder) {
            return true;
        }
        let checkpoint_id = match replacement.first() {
            Some(item) => match item.as_responses_checkpoint() {
                Some(checkpoint) => checkpoint.checkpoint_id.clone(),
                None => return false,
            },
            None => return false,
        };
        let operation_id = format!(
            "memory-{checkpoint_id}-{}",
            self.state.history_revision.saturating_add(1)
        );
        let persisted = self
            .persistence
            .replace_history_and_ack(&operation_id, &replacement)
            .await;
        match persisted {
            Ok(Ok(())) => {}
            Ok(Err(crate::persistence::HistoryReplaceError::Committed(error))) => {
                tracing::warn!(%error, "checkpoint memory reminder committed but acknowledgement was lost");
            }
            Ok(Err(error)) => {
                tracing::error!(%error, "checkpoint memory reminder was not committed");
                return false;
            }
            Err(_) => {
                tracing::error!("checkpoint memory reminder acknowledgement was dropped");
                return false;
            }
        }

        self.snapshot_turn_slice();
        let before_tokens = super::state::estimate_conversation_tokens(&self.state.conversation);
        let after_tokens = super::state::estimate_conversation_tokens(&replacement);
        if after_tokens >= before_tokens {
            self.state.estimated_tokens_since_model = self
                .state
                .estimated_tokens_since_model
                .saturating_add(after_tokens - before_tokens);
        } else {
            self.state.estimated_tokens_since_model = self
                .state
                .estimated_tokens_since_model
                .saturating_sub(before_tokens - after_tokens);
        }
        self.state.conversation = replacement;
        self.state.bump_history_revision();
        self.rebase_turn_capture_offset();
        true
    }
}

// ============================================================================
// Pruning (standalone functions, no actor state needed)
// ============================================================================

/// Check whether pruning should run based on context utilization.
///
/// Returns `true` when `total_tokens` exceeds 50% of `context_window`.
pub(crate) fn should_prune(total_tokens: u64, context_window: std::num::NonZeroU64) -> bool {
    total_tokens > context_window.get() / 2
}

/// Prune old, large tool results from the conversation in place.
///
/// Turn age is estimated by walking backward through the conversation and
/// counting `User` items to determine which "turn" each tool result belongs to.
pub(crate) fn prune_conversation(conversation: &mut [ConversationItem], config: &PruningConfig) {
    if !config.enabled {
        return;
    }

    let mut turn_from_end: usize = 0;
    let mut seen_first_user = false;

    for i in (0..conversation.len()).rev() {
        if matches!(&conversation[i], ConversationItem::User(_)) {
            if seen_first_user {
                turn_from_end += 1;
            }
            seen_first_user = true;
            continue;
        }

        let ConversationItem::ToolResult(tool_result) = &mut conversation[i] else {
            continue;
        };

        // Never prune recent turns.
        if turn_from_end < config.keep_last_n_turns {
            continue;
        }

        // Hard clear: very old tool results → replace entirely.
        if turn_from_end >= config.hard_clear_age_turns {
            if tool_result.content.as_ref() != HARD_CLEAR_PLACEHOLDER {
                tool_result.content = std::sync::Arc::<str>::from(HARD_CLEAR_PLACEHOLDER);
            }
            continue;
        }

        // Soft trim: large tool results → keep head + tail.
        let content_len = tool_result.content.chars().count();
        if content_len > config.soft_trim_threshold {
            let head = safe_char_slice(&tool_result.content, 0, config.soft_trim_head);
            let tail = safe_char_slice_tail(&tool_result.content, config.soft_trim_tail);
            tool_result.content =
                std::sync::Arc::<str>::from(format!("{head}{SOFT_TRIM_SEPARATOR}{tail}"));
        }
    }
}

// ============================================================================
// Memory reminder injection
// ============================================================================

use crate::types::{MEMORY_CONTEXT_CLOSE_TAG, MEMORY_CONTEXT_OPEN_TAG};
use xai_grok_sampling_types::SystemSource;

/// Upsert a memory reminder as a dedicated [`SystemSource::MemoryContext`]
/// item.
///
/// Rules (stage D1c):
///
/// * base instructions and memory context are separate items;
/// * a memory update only replaces the `MemoryContext` item — the base
///   System string is never concatenated with `<memory-context>` again;
/// * a legacy combined leading System (memory block embedded in the base
///   string) is split only when the block is unambiguous (exactly one
///   well-formed open/close pair); malformed combinations are left for
///   migration;
/// * with a live checkpoint the memory item lives in the typed tail and is
///   always marked — never an unmarked trailing System.
///
/// Returns `true` when the conversation was changed.
pub(super) fn inject_memory_reminder(items: &mut Vec<ConversationItem>, reminder: &str) -> bool {
    let reminder = reminder.trim();
    if reminder.is_empty() {
        return false;
    }

    // 1. Replace the dedicated memory item.
    if let Some(system) = items.iter_mut().find_map(|item| match item {
        ConversationItem::System(system) if system.source == SystemSource::MemoryContext => {
            Some(system)
        }
        _ => None,
    }) {
        if system.content.as_ref() == reminder {
            return false;
        }
        system.content = std::sync::Arc::<str>::from(reminder);
        return true;
    }

    // 2. Legacy combined form: an unmarked System still carrying the memory
    //    block inside its string. Split only when unambiguous.
    let legacy_index = items.iter().position(|item| match item {
        ConversationItem::System(system)
            if system.source != SystemSource::MemoryContext
                && system.content.contains(MEMORY_CONTEXT_OPEN_TAG) =>
        {
            true
        }
        _ => false,
    });
    if let Some(index) = legacy_index {
        let split = match &items[index] {
            ConversationItem::System(system) => split_legacy_memory_block(&system.content),
            _ => None,
        };
        if let Some((base_part, _old_memory)) = split {
            if base_part.is_empty() {
                // The item was pure memory: convert it in place.
                if let Some(ConversationItem::System(system)) = items.get_mut(index) {
                    system.content = std::sync::Arc::<str>::from(reminder);
                    system.source = SystemSource::MemoryContext;
                }
            } else {
                if let Some(ConversationItem::System(system)) = items.get_mut(index) {
                    system.content = std::sync::Arc::<str>::from(base_part);
                }
                items.insert(index + 1, ConversationItem::memory_context(reminder));
            }
            return true;
        }
        // Ambiguous legacy block: leave the item untouched (migration
        // normalizes it) and fall through to inserting a dedicated item.
    }

    // 3. Insert after the base head: after the checkpoint wrapper or the
    //    leading base System, else at the front.
    let insert_at = match items.first() {
        Some(item) if item.is_responses_checkpoint() => 1,
        Some(ConversationItem::System(system)) if system.source != SystemSource::MemoryContext => 1,
        _ => 0,
    };
    items.insert(insert_at, ConversationItem::memory_context(reminder));
    true
}

/// Split a legacy combined system prompt into `(base, memory_block)` when
/// the memory block is unambiguous: exactly one well-formed open/close pair
/// and no further memory tags anywhere else.
fn split_legacy_memory_block(content: &str) -> Option<(String, String)> {
    let open = content.find(MEMORY_CONTEXT_OPEN_TAG)?;
    let close_rel = content[open..].find(MEMORY_CONTEXT_CLOSE_TAG)?;
    let close = open + close_rel;
    let after = close + MEMORY_CONTEXT_CLOSE_TAG.len();
    if content[after..].contains(MEMORY_CONTEXT_OPEN_TAG) {
        return None;
    }
    let before = content[..open].trim();
    let rest = content[after..].trim();
    let base = match (before.is_empty(), rest.is_empty()) {
        (true, true) => String::new(),
        (false, true) => before.to_string(),
        (true, false) => rest.to_string(),
        (false, false) => format!("{before}\n\n{rest}"),
    };
    let memory = content[open..after].trim().to_string();
    Some((base, memory))
}

// ============================================================================
// String helpers
// ============================================================================

fn safe_char_slice(s: &str, start: usize, count: usize) -> String {
    s.chars().skip(start).take(count).collect()
}

fn safe_char_slice_tail(s: &str, count: usize) -> String {
    let total = s.chars().count();
    if count >= total {
        return s.to_string();
    }
    s.chars().skip(total - count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_prune_gating() {
        use std::num::NonZeroU64;
        let cw = NonZeroU64::new(10000).unwrap();
        assert!(!should_prune(1000, cw)); // 10%
        assert!(should_prune(6000, cw)); // 60%
        assert!(!should_prune(5000, cw)); // 50% exact (> not >=)
    }

    #[test]
    fn prune_disabled_is_noop() {
        let mut conv = vec![ConversationItem::tool_result("c1", "x".repeat(10_000))];
        let config = PruningConfig {
            enabled: false,
            ..Default::default()
        };
        prune_conversation(&mut conv, &config);
        if let ConversationItem::ToolResult(ref tr) = conv[0] {
            assert_eq!(tr.content.len(), 10_000);
        }
    }

    #[test]
    fn inject_memory_creates_dedicated_item_after_base() {
        let mut items = vec![
            ConversationItem::base_instructions("You are helpful."),
            ConversationItem::user("hi"),
        ];
        inject_memory_reminder(&mut items, "Remember: user likes rust");
        // Base instructions are never concatenated with the memory block.
        assert_eq!(
            items[0].text_content(),
            "You are helpful.",
            "base instructions stay untouched"
        );
        assert_eq!(items.len(), 3);
        let ConversationItem::System(memory) = &items[1] else {
            panic!("expected memory system item");
        };
        assert_eq!(memory.source, SystemSource::MemoryContext);
        assert_eq!(memory.content.as_ref(), "Remember: user likes rust");
    }

    #[test]
    fn inject_memory_replaces_only_the_memory_item() {
        let mut items = vec![
            ConversationItem::base_instructions("You are helpful."),
            ConversationItem::memory_context("old memory"),
            ConversationItem::user("hi"),
        ];
        assert!(inject_memory_reminder(&mut items, "new memory"));
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].text_content(), "You are helpful.");
        assert_eq!(items[1].text_content(), "new memory");
        assert_eq!(items[1].system_source(), Some(SystemSource::MemoryContext));
        // Idempotent: same reminder is a no-op.
        assert!(!inject_memory_reminder(&mut items, "new memory"));
    }

    #[test]
    fn inject_memory_splits_unambiguous_legacy_combined_system() {
        let mut items = vec![
            ConversationItem::system("You are helpful.\n\n<memory-context>old</memory-context>"),
            ConversationItem::user("hi"),
        ];
        assert!(inject_memory_reminder(
            &mut items,
            "<memory-context>new</memory-context>"
        ));
        assert_eq!(items.len(), 3);
        // Legacy item shrinks to the base part; the memory block moves into
        // its own marked item right after it.
        assert_eq!(items[0].text_content(), "You are helpful.");
        assert_eq!(
            items[0].system_source(),
            Some(SystemSource::LegacyUnclassified)
        );
        assert_eq!(items[1].system_source(), Some(SystemSource::MemoryContext));
        assert_eq!(
            items[1].text_content(),
            "<memory-context>new</memory-context>"
        );
    }

    #[test]
    fn inject_memory_converts_pure_legacy_memory_item_in_place() {
        let mut items = vec![
            ConversationItem::system("<memory-context>old</memory-context>"),
            ConversationItem::user("hi"),
        ];
        assert!(inject_memory_reminder(
            &mut items,
            "<memory-context>new</memory-context>"
        ));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].system_source(), Some(SystemSource::MemoryContext));
        assert_eq!(
            items[0].text_content(),
            "<memory-context>new</memory-context>"
        );
    }

    #[test]
    fn inject_memory_leaves_ambiguous_legacy_block_for_migration() {
        let mut items = vec![ConversationItem::system(
            "base <memory-context>one</memory-context> mid <memory-context>two</memory-context>",
        )];
        assert!(inject_memory_reminder(
            &mut items,
            "<memory-context>new</memory-context>"
        ));
        // The ambiguous legacy item is untouched; a dedicated item is added.
        assert!(items[0].text_content().contains("one"));
        assert_eq!(items[1].system_source(), Some(SystemSource::MemoryContext));
    }

    #[test]
    fn inject_memory_prepends_when_no_system() {
        let mut items = vec![ConversationItem::user("hi")];
        inject_memory_reminder(&mut items, "Remember: user likes rust");
        assert_eq!(items.len(), 2);
        assert!(matches!(&items[0], ConversationItem::System(_)));
        assert_eq!(items[0].system_source(), Some(SystemSource::MemoryContext));
    }
}

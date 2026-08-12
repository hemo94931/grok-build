use std::sync::{OnceLock, RwLock};

use crate::slash::command::ArgItem;

fn provider_items_cache() -> &'static RwLock<Vec<ArgItem>> {
    static ITEMS: OnceLock<RwLock<Vec<ArgItem>>> = OnceLock::new();
    ITEMS.get_or_init(|| RwLock::new(default_provider_items()))
}

/// Provider-name completion may stay a thin local helper; supported methods
/// never live here and are replaced from `x.ai/providerAuth/info` metadata.
fn default_provider_items() -> Vec<ArgItem> {
    [
        ("xAI", "xAI x.ai xai", "xai", "Grok and xAI models"),
        (
            "Anthropic",
            "Anthropic anthropic",
            "anthropic",
            "Anthropic models",
        ),
        (
            "OpenAI Codex",
            "OpenAI Codex openai-codex codex",
            "openai-codex",
            "OpenAI Codex models",
        ),
        (
            "GitHub Copilot",
            "GitHub Copilot github-copilot copilot",
            "github-copilot",
            "GitHub Copilot models",
        ),
        (
            "OpenRouter",
            "OpenRouter openrouter",
            "openrouter",
            "OpenRouter models",
        ),
        (
            "Kimi Coding",
            "Kimi Coding kimi-coding kimi",
            "kimi-coding",
            "Kimi Coding models",
        ),
        ("Radius", "Radius radius", "radius", "Radius models"),
        (
            "DeepSeek",
            "DeepSeek deepseek",
            "deepseek",
            "DeepSeek models",
        ),
    ]
    .into_iter()
    .map(|(display, match_text, insert_text, description)| ArgItem {
        display: display.to_owned(),
        match_text: match_text.to_owned(),
        insert_text: insert_text.to_owned(),
        description: description.to_owned(),
    })
    .collect()
}

pub(crate) fn replace_provider_items(items: Vec<ArgItem>) {
    let mut merged = default_provider_items();
    for item in items {
        if let Some(existing) = merged
            .iter_mut()
            .find(|existing| existing.insert_text == item.insert_text)
        {
            *existing = item;
        } else {
            merged.push(item);
        }
    }
    *provider_items_cache()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = merged;
}

pub(super) fn provider_items() -> Vec<ArgItem> {
    provider_items_cache()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Normalize xAI aliases, metadata display names, or a provider id-like value.
/// Method support is deliberately not encoded here; the shell registry metadata
/// returned by `x.ai/providerAuth/info` is authoritative for validation.
pub(super) fn normalize_known_provider(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.chars().any(char::is_control) {
        return None;
    }
    if value.eq_ignore_ascii_case("xai") || value.eq_ignore_ascii_case("x.ai") {
        return Some("xai".to_owned());
    }
    provider_items()
        .into_iter()
        .find(|item| {
            item.insert_text.eq_ignore_ascii_case(value) || item.display.eq_ignore_ascii_case(value)
        })
        .map(|item| item.insert_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_names_and_ids_normalize_without_a_method_matrix() {
        assert_eq!(
            normalize_known_provider("OpenAI Codex").as_deref(),
            Some("openai-codex")
        );
        assert_eq!(normalize_known_provider("x.ai").as_deref(), Some("xai"));
        assert_eq!(
            normalize_known_provider("deepseek").as_deref(),
            Some("deepseek")
        );
        assert_eq!(normalize_known_provider("bad/value"), None);
    }
}

use crate::slash::command::ArgItem;

const PROVIDERS: [(&str, &str); 7] = [
    ("xai", "xAI"),
    ("anthropic", "Anthropic"),
    ("openai-codex", "OpenAI Codex"),
    ("github-copilot", "GitHub Copilot"),
    ("openrouter", "OpenRouter"),
    ("kimi-coding", "Kimi Coding"),
    ("radius", "Radius"),
];

pub(super) fn provider_items() -> Vec<ArgItem> {
    PROVIDERS
        .into_iter()
        .map(|(id, name)| ArgItem {
            display: name.to_owned(),
            match_text: format!("{name} {id}"),
            insert_text: id.to_owned(),
            description: if id == "xai" {
                "Grok and xAI models".to_owned()
            } else {
                format!("Models authenticated through {name}")
            },
        })
        .collect()
}

pub(super) fn normalize_provider(value: &str) -> Option<&'static str> {
    let value = value.trim();
    PROVIDERS
        .into_iter()
        .find(|(id, name)| {
            id.eq_ignore_ascii_case(value)
                || name.eq_ignore_ascii_case(value)
                || (*id == "xai" && value.eq_ignore_ascii_case("x.ai"))
        })
        .map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_names_and_ids_resolve() {
        assert_eq!(normalize_provider("OpenAI Codex"), Some("openai-codex"));
        assert_eq!(normalize_provider("x.ai"), Some("xai"));
        assert_eq!(normalize_provider("unknown"), None);
        assert_eq!(provider_items().len(), 7);
    }
}

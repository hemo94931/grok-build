//! Canonical Responses envelope and stable cache routing (Plan 阶段 6/7).
//!
//! normal / compact / post-compact / recompact requests must all derive
//! from one [`CanonicalResponsesContext`] so the provider sees the same
//! envelope everywhere. Prompt-cache affinity comes from a persistent
//! per-session `logical_cache_namespace_id` (independent of the typed-tail
//! `branch_id`) routed through
//! `prompt_cache_key = hash(provider + normalized deployment + model
//! family + namespace)`. Auxiliary requests derive isolated namespaces so
//! their cache hits never pollute the main-session affinity.

use serde::{Deserialize, Serialize};

use super::responses::canonical_value_digest;

/// Field separator for cache-key hash inputs: a control byte that can
/// never appear inside any component, preventing concatenation ambiguity.
const KEY_HASH_SEPARATOR: &str = "\u{1f}";

/// The single context every Responses request kind derives from.
///
/// `prompt_cache_key` is routing, not compatibility: it never enters the
/// checkpoint compatibility identity, only the separate
/// `cache_route_fingerprint`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalResponsesContext {
    pub model: String,
    pub base_instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_context: Option<String>,
    pub tools: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<serde_json::Value>,
    /// Explicit — never defaulted at the endpoint. The compact endpoint
    /// allowlist and the normal body both read this exact value.
    pub parallel_tool_calls: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_options: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

impl CanonicalResponsesContext {
    /// Wire instructions: base + fixed separator + current memory, each at
    /// most once. Used identically by normal, compact and replay bodies.
    pub fn instructions(&self) -> String {
        let mut parts = vec![self.base_instructions.clone()];
        if let Some(memory) = self
            .memory_context
            .as_ref()
            .map(|memory| memory.trim())
            .filter(|memory| !memory.is_empty())
        {
            parts.push(memory.to_string());
        }
        parts.join(super::INSTRUCTIONS_MEMORY_SEPARATOR)
    }

    /// Canonical fingerprint of the compatibility-relevant envelope:
    /// everything except the routing-only `prompt_cache_key`. Items never
    /// participate (they are transcript, not envelope).
    pub fn envelope_fingerprint(&self) -> std::io::Result<String> {
        #[derive(Serialize)]
        struct EnvelopeView<'a> {
            model: &'a str,
            base_instructions: &'a str,
            memory_context: Option<&'a str>,
            tools: &'a serde_json::Value,
            tool_choice: Option<&'a serde_json::Value>,
            reasoning: Option<&'a serde_json::Value>,
            text: Option<&'a serde_json::Value>,
            parallel_tool_calls: bool,
            prompt_cache_options: Option<&'a serde_json::Value>,
            prompt_cache_retention: Option<&'a str>,
            service_tier: Option<&'a str>,
        }
        let view = EnvelopeView {
            model: &self.model,
            base_instructions: &self.base_instructions,
            memory_context: self.memory_context.as_deref(),
            tools: &self.tools,
            tool_choice: self.tool_choice.as_ref(),
            reasoning: self.reasoning.as_ref(),
            text: self.text.as_ref(),
            parallel_tool_calls: self.parallel_tool_calls,
            prompt_cache_options: self.prompt_cache_options.as_ref(),
            prompt_cache_retention: self.prompt_cache_retention.as_deref(),
            service_tier: self.service_tier.as_deref(),
        };
        let value = serde_json::to_value(view).map_err(std::io::Error::other)?;
        canonical_value_digest(&value).map_err(std::io::Error::other)
    }
}

/// Normalize a base URL for cache routing: lowercase scheme + authority,
/// default ports elided, **no path** — `/responses` and
/// `/responses/compact` on the same deployment MUST share a cache route.
pub fn normalize_base_url_for_routing(base_url: &str) -> String {
    let trimmed = base_url.trim();
    let (scheme, rest) = match trimmed.split_once("://") {
        Some((scheme, rest)) => (scheme.to_ascii_lowercase(), rest),
        None => ("https".to_string(), trimmed),
    };
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(rest)
        .to_ascii_lowercase();
    let authority = authority
        .strip_suffix(":443")
        .filter(|_| scheme == "https")
        .or_else(|| authority.strip_suffix(":80").filter(|_| scheme == "http"))
        .unwrap_or(&authority)
        .to_string();
    format!("{scheme}://{authority}")
}

/// The cache family a model belongs to. Providers route caches per model
/// family, not per exact checkpoint version; today the family is the
/// exact model id (a future mapping may collapse versioned aliases).
pub fn model_cache_family(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

/// Stable route fingerprint: provider + normalized deployment + model
/// family + auth principal. Deliberately excludes the concrete endpoint
/// path so compact and post-compact share the route.
pub fn cache_route_fingerprint(
    provider_id: &str,
    normalized_base_url: &str,
    model_cache_family: &str,
    auth_principal_fingerprint: &str,
) -> String {
    let joined = [
        provider_id,
        normalized_base_url,
        model_cache_family,
        auth_principal_fingerprint,
    ]
    .join(KEY_HASH_SEPARATOR);
    let digest = canonical_value_digest(&serde_json::Value::String(joined))
        .unwrap_or_else(|_| "unstable".to_string());
    format!("cache-route:{digest}")
}

/// Main-session prompt cache key: route fingerprint + persistent logical
/// namespace. The `prompt_cache_key` field is capped by providers, so the
/// key is a digest, never the raw components.
pub fn prompt_cache_key_for_namespace(cache_route_fingerprint: &str, namespace_id: &str) -> String {
    let joined = [cache_route_fingerprint, namespace_id].join(KEY_HASH_SEPARATOR);
    let digest = canonical_value_digest(&serde_json::Value::String(joined))
        .unwrap_or_else(|_| "unstable".to_string());
    format!("grok:{digest}")
}

/// Isolated auxiliary namespace: stable per (session namespace, aux kind)
/// but never equal to the main-session key. Aux cache hits must not count
/// toward the main-session cache-affinity SLO.
pub fn aux_cache_namespace(namespace_id: &str, auxiliary_kind: &str) -> String {
    let joined = [namespace_id, auxiliary_kind].join(KEY_HASH_SEPARATOR);
    let digest = canonical_value_digest(&serde_json::Value::String(joined))
        .unwrap_or_else(|_| "unstable".to_string());
    format!("grok-aux:{digest}")
}

/// Generate a fresh logical cache namespace id. Persisted once per
/// session/branch; fork/mirror/subagent allocate new ones explicitly.
pub fn new_logical_cache_namespace_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context_fixture() -> CanonicalResponsesContext {
        CanonicalResponsesContext {
            model: "grok-test".into(),
            base_instructions: "base".into(),
            memory_context: Some("memory".into()),
            tools: serde_json::json!([]),
            tool_choice: None,
            reasoning: None,
            text: None,
            parallel_tool_calls: true,
            prompt_cache_key: Some("key-a".into()),
            prompt_cache_options: None,
            prompt_cache_retention: None,
            service_tier: None,
        }
    }

    #[test]
    fn instructions_place_base_then_memory_once() {
        let ctx = context_fixture();
        assert_eq!(
            ctx.instructions(),
            format!(
                "base{}memory",
                crate::conversation::INSTRUCTIONS_MEMORY_SEPARATOR
            )
        );
        let mut no_memory = ctx.clone();
        no_memory.memory_context = None;
        assert_eq!(no_memory.instructions(), "base");
        let mut blank_memory = ctx.clone();
        blank_memory.memory_context = Some("  ".into());
        assert_eq!(blank_memory.instructions(), "base");
    }

    #[test]
    fn envelope_fingerprint_excludes_cache_key() {
        let ctx = context_fixture();
        let fp = ctx.envelope_fingerprint().unwrap();
        let mut other = ctx.clone();
        other.prompt_cache_key = Some("key-b".into());
        assert_eq!(other.envelope_fingerprint().unwrap(), fp);
        let mut different_model = ctx.clone();
        different_model.model = "grok-other".into();
        assert_ne!(different_model.envelope_fingerprint().unwrap(), fp);
        let mut different_parallel = ctx.clone();
        different_parallel.parallel_tool_calls = false;
        assert_ne!(different_parallel.envelope_fingerprint().unwrap(), fp);
    }

    #[test]
    fn base_url_normalization_strips_paths_and_default_ports() {
        assert_eq!(
            normalize_base_url_for_routing("https://api.x.ai/v1/responses"),
            "https://api.x.ai"
        );
        assert_eq!(
            normalize_base_url_for_routing("HTTPS://API.X.AI:443/responses/compact"),
            "https://api.x.ai"
        );
        assert_eq!(
            normalize_base_url_for_routing("http://localhost:8080/v1"),
            "http://localhost:8080"
        );
        assert_eq!(
            normalize_base_url_for_routing("api.example.com/"),
            "https://api.example.com"
        );
    }

    #[test]
    fn route_fingerprint_ignores_endpoint_path() {
        let a = cache_route_fingerprint(
            "xai",
            &normalize_base_url_for_routing("https://api.x.ai/v1/responses"),
            &model_cache_family("grok-4"),
            "principal",
        );
        let b = cache_route_fingerprint(
            "xai",
            &normalize_base_url_for_routing("https://api.x.ai/v1/responses/compact"),
            &model_cache_family("grok-4"),
            "principal",
        );
        assert_eq!(a, b, "compact and post-compact must share a cache route");
        let other_principal = cache_route_fingerprint(
            "xai",
            "https://api.x.ai",
            "grok-4",
            "other-principal",
        );
        assert_ne!(a, other_principal);
    }

    #[test]
    fn namespace_keys_are_stable_and_isolated() {
        let route = cache_route_fingerprint("xai", "https://api.x.ai", "grok-4", "p");
        let main_a = prompt_cache_key_for_namespace(&route, "ns-1");
        assert_eq!(main_a, prompt_cache_key_for_namespace(&route, "ns-1"));
        assert_ne!(main_a, prompt_cache_key_for_namespace(&route, "ns-2"));
        let aux = aux_cache_namespace("ns-1", "recap");
        assert_eq!(aux, aux_cache_namespace("ns-1", "recap"));
        assert_ne!(aux, aux_cache_namespace("ns-1", "side_question"));
        assert_ne!(aux, main_a, "aux keys never collide with the main key");
        assert!(main_a.starts_with("grok:"));
        assert!(aux.starts_with("grok-aux:"));
    }
}

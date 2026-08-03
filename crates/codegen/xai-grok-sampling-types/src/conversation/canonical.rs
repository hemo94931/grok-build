//! Stable Responses cache routing.
//!
//! Prompt-cache affinity comes from a persistent
//! per-session `logical_cache_namespace_id` (independent of the typed-tail
//! `branch_id`) routed through
//! `prompt_cache_key = hash(provider + normalized deployment + model
//! family + namespace)`. Auxiliary requests derive isolated namespaces so
//! their cache hits never pollute the main-session affinity.

use super::responses::canonical_value_digest;

/// Field separator for cache-key hash inputs: a control byte that can
/// never appear inside any component, preventing concatenation ambiguity.
const KEY_HASH_SEPARATOR: &str = "\u{1f}";

/// Normalize a base URL for cache routing: lowercase scheme + authority,
/// default ports elided, deployment/account path components preserved
/// (path-addressed deployments on one host must NOT collapse onto one
/// cache route), with only the concrete `/responses` and
/// `/responses/compact` endpoint suffixes stripped so compact and
/// post-compact share a route.
pub fn normalize_base_url_for_routing(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    let (scheme, rest) = match trimmed.split_once("://") {
        Some((scheme, rest)) => (scheme.to_ascii_lowercase(), rest),
        None => ("https".to_string(), trimmed),
    };
    let mut parts = rest.splitn(2, '/');
    let authority = parts
        .next()
        .unwrap_or(rest)
        .split(['?', '#'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let authority = authority
        .strip_suffix(":443")
        .filter(|_| scheme == "https")
        .or_else(|| authority.strip_suffix(":80").filter(|_| scheme == "http"))
        .unwrap_or(&authority)
        .to_string();
    let path = parts.next().unwrap_or_default().trim_end_matches('/');
    // Strip only the concrete endpoint suffixes; keep deployment/account
    // prefixes (e.g. `/v1`, `/deployments/prod`) in the route.
    let path = path
        .strip_suffix("responses/compact")
        .or_else(|| path.strip_suffix("responses"))
        .unwrap_or(path)
        .trim_end_matches('/');
    if path.is_empty() {
        format!("{scheme}://{authority}")
    } else {
        format!("{scheme}://{authority}/{path}")
    }
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

const PROVIDER_PROMPT_CACHE_KEY_MAX_LEN: usize = 64;

fn prefixed_prompt_cache_key(prefix: &str, digest: &str) -> String {
    debug_assert!(prefix.len() <= PROVIDER_PROMPT_CACHE_KEY_MAX_LEN);
    let digest = digest
        .chars()
        .take(PROVIDER_PROMPT_CACHE_KEY_MAX_LEN.saturating_sub(prefix.len()))
        .collect::<String>();
    format!("{prefix}{digest}")
}

/// Main-session prompt cache key: route fingerprint + persistent logical
/// namespace. The `prompt_cache_key` field is capped by providers, so the
/// key is a digest, never the raw components.
pub fn prompt_cache_key_for_namespace(cache_route_fingerprint: &str, namespace_id: &str) -> String {
    let joined = [cache_route_fingerprint, namespace_id].join(KEY_HASH_SEPARATOR);
    let digest = canonical_value_digest(&serde_json::Value::String(joined))
        .unwrap_or_else(|_| "unstable".to_string());
    prefixed_prompt_cache_key("grok:", &digest)
}

/// Isolated auxiliary namespace: stable per (session namespace, aux kind)
/// but never equal to the main-session key. Aux cache hits must not count
/// toward the main-session cache-affinity SLO.
pub fn aux_cache_namespace(namespace_id: &str, auxiliary_kind: &str) -> String {
    let joined = [namespace_id, auxiliary_kind].join(KEY_HASH_SEPARATOR);
    let digest = canonical_value_digest(&serde_json::Value::String(joined))
        .unwrap_or_else(|_| "unstable".to_string());
    prefixed_prompt_cache_key("grok-aux:", &digest)
}

/// Generate a fresh logical cache namespace id. Persisted once per
/// session/branch; fork/mirror/subagent allocate new ones explicitly.
pub fn new_logical_cache_namespace_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_normalization_strips_paths_and_default_ports() {
        // Concrete endpoint suffixes collapse onto the deployment route.
        assert_eq!(
            normalize_base_url_for_routing("https://api.x.ai/v1/responses"),
            "https://api.x.ai/v1"
        );
        assert_eq!(
            normalize_base_url_for_routing("HTTPS://API.X.AI:443/responses/compact"),
            "https://api.x.ai"
        );
        assert_eq!(
            normalize_base_url_for_routing("http://localhost:8080/v1"),
            "http://localhost:8080/v1"
        );
        assert_eq!(
            normalize_base_url_for_routing("api.example.com/"),
            "https://api.example.com"
        );
        // Path-addressed deployments stay distinct cache routes.
        assert_ne!(
            normalize_base_url_for_routing("https://proxy.example.com/accounts/a"),
            normalize_base_url_for_routing("https://proxy.example.com/accounts/b")
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
        let other_principal =
            cache_route_fingerprint("xai", "https://api.x.ai", "grok-4", "other-principal");
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
        assert_eq!(main_a.len(), 64, "provider-visible keys must fit the cap");
        assert!(aux.starts_with("grok-aux:"));
        assert_eq!(aux.len(), 64, "auxiliary keys must fit the same cap");
    }
}

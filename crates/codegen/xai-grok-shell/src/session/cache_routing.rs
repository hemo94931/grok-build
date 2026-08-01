//! Persistent logical cache namespace and route fingerprint (Plan 阶段 7:
//! 稳定缓存路由).
//!
//! The provider-visible `prompt_cache_key` must stay identical across
//! normal / `/responses` / `/responses/compact` / post-compact requests of
//! one logical session branch, independent of the typed-tail `branch_id`.
//! That requires a *persisted* `logical_cache_namespace_id` (not derived
//! from the current tail) combined with a route fingerprint that is
//! normalized over provider + deployment + model family + auth principal
//! and deliberately excludes the concrete endpoint path.
//!
//! Forks, mirrors and subagent sessions must allocate a fresh namespace —
//! they never inherit this file (see `copy_session_data_sync`'s explicit
//! file allowlist and `with_explicit_session_dir` subagent dirs), so a new
//! session dir auto-creates one on first use.

use std::io;
use std::path::Path;

use xai_grok_sampling_types::{
    aux_cache_namespace, cache_route_fingerprint, model_cache_family,
    new_logical_cache_namespace_id, normalize_base_url_for_routing,
    prompt_cache_key_for_namespace,
};

/// File name of the persisted logical cache namespace inside a session dir.
pub(crate) const CACHE_NAMESPACE_FILE: &str = "cache_namespace";

/// Sanity cap for a persisted namespace id (uuids are 36 chars; anything
/// far longer is corrupt).
const MAX_NAMESPACE_LEN: usize = 128;

/// Stable cache routing for one session dir: the persisted logical cache
/// namespace id plus the derived route fingerprint.
///
/// `route` is intentionally re-derived on every `load_or_create` call (it
/// is a pure function of the current provider/base URL/model/principal);
/// only `namespace_id` is durable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionCacheRouting {
    namespace_id: String,
    route: String,
    provider_id: String,
}

impl SessionCacheRouting {
    /// The provider-visible `prompt_cache_key` for main-session requests:
    /// route fingerprint + persisted logical namespace.
    pub(crate) fn prompt_cache_key(&self) -> String {
        prompt_cache_key_for_namespace(&self.route, &self.namespace_id)
    }

    /// The provider id this route was derived for (`xai` /
    /// `openai_compatible`). Informational (telemetry) only.
    pub(crate) fn provider_id(&self) -> &str {
        &self.provider_id
    }

    /// The provider-visible `prompt_cache_key` for an auxiliary request
    /// kind (recap, side_question, ...): route fingerprint + a stable
    /// isolated namespace derived from the session namespace. Aux keys
    /// never equal the main key, so aux cache hits cannot pollute the
    /// main-session cache-affinity SLO.
    pub(crate) fn aux_prompt_cache_key(&self, kind: &str) -> String {
        prompt_cache_key_for_namespace(&self.route, &aux_cache_namespace(&self.namespace_id, kind))
    }

    #[allow(dead_code)] // consumed by the stage-D4 V2 writer
    pub(crate) fn route_fingerprint(&self) -> &str {
        &self.route
    }
}

/// Load the persisted logical cache namespace for `session_dir`, or create
/// it on first use. The route fingerprint is derived from the current
/// route components (they are never persisted — a model/base-url/principal
/// change must reroute, not reuse).
///
/// Corruption tolerance: an unreadable or corrupt/empty namespace file is
/// regenerated in place (a warn is logged because regenerating loses cache
/// affinity with every key minted under the old namespace). Concurrent
/// first-use is tolerated via a read-after-write re-read: whichever
/// namespace landed on disk wins, so concurrent creators converge on the
/// same file content.
pub(crate) fn load_or_create(
    session_dir: &Path,
    provider_id: &str,
    base_url: &str,
    model: &str,
    principal: &str,
) -> io::Result<SessionCacheRouting> {
    let path = session_dir.join(CACHE_NAMESPACE_FILE);
    let route = cache_route_fingerprint(
        provider_id,
        &normalize_base_url_for_routing(base_url),
        &model_cache_family(model),
        principal,
    );
    let (existing, file_existed) = match std::fs::read_to_string(&path) {
        Ok(contents) => (parse_namespace(&contents), true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (None, false),
        Err(error) => return Err(error),
    };
    if let Some(namespace_id) = existing {
        return Ok(SessionCacheRouting {
            namespace_id,
            route,
            provider_id: provider_id.to_string(),
        });
    }
    if file_existed {
        // A corrupt/empty file means the namespace id was lost: the next
        // minted key diverges from every key minted under the old id.
        tracing::warn!(
            path = %path.display(),
            "cache_namespace file corrupt or empty; regenerating (loses prompt-cache affinity)"
        );
    }
    std::fs::create_dir_all(session_dir)?;
    let namespace_id = new_logical_cache_namespace_id();
    std::fs::write(&path, format!("{namespace_id}\n"))?;
    // Tolerate concurrent creation: another creator may have written the
    // file between our read and write. Re-read; the on-disk value wins so
    // all processes sharing this session dir converge on one namespace.
    let namespace_id = std::fs::read_to_string(&path)
        .ok()
        .and_then(|contents| parse_namespace(&contents))
        .unwrap_or(namespace_id);
    Ok(SessionCacheRouting {
        namespace_id,
        route,
        provider_id: provider_id.to_string(),
    })
}

/// Accept a persisted namespace id: single-line, non-empty, bounded length.
/// Anything else (multi-line junk, whitespace, absurd length) is corrupt.
fn parse_namespace(contents: &str) -> Option<String> {
    let trimmed = contents.trim();
    if trimmed.is_empty()
        || trimmed.len() > MAX_NAMESPACE_LEN
        || trimmed.contains(['\n', '\r', '\0'])
    {
        return None;
    }
    Some(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROVIDER: &str = "xai";
    const BASE_URL: &str = "https://api.x.ai/v1/responses";
    const MODEL: &str = "grok-4";
    const PRINCIPAL: &str = "principal-a";

    fn load(dir: &Path) -> io::Result<SessionCacheRouting> {
        load_or_create(dir, PROVIDER, BASE_URL, MODEL, PRINCIPAL)
    }

    #[test]
    fn persists_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let first = load(dir.path()).unwrap();
        assert!(first.namespace_id.len() == 36, "uuid v4 namespace");
        let second = load(dir.path()).unwrap();
        assert_eq!(first, second, "reload must reuse the persisted namespace");
        assert_eq!(
            first.prompt_cache_key(),
            second.prompt_cache_key(),
            "main cache key must be stable across reload"
        );
        assert_eq!(
            first.aux_prompt_cache_key("recap"),
            second.aux_prompt_cache_key("recap"),
            "aux cache key must be stable across reload"
        );
        assert!(dir.path().join(CACHE_NAMESPACE_FILE).is_file());
    }

    #[test]
    fn regenerates_on_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        let original = load(dir.path()).unwrap();
        // Multi-line junk is corrupt; an empty file is corrupt too.
        for corrupt in ["not-a-namespace\nsecond line\n", "   \n", ""] {
            std::fs::write(dir.path().join(CACHE_NAMESPACE_FILE), corrupt).unwrap();
            let regenerated = load(dir.path()).unwrap();
            assert_ne!(
                regenerated.namespace_id, original.namespace_id,
                "corrupt file ({corrupt:?}) must regenerate a fresh namespace"
            );
            // The regenerated value is now persisted and stable.
            let reloaded = load(dir.path()).unwrap();
            assert_eq!(regenerated, reloaded);
        }
    }

    #[test]
    fn aux_keys_are_isolated_from_main_and_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let routing = load(dir.path()).unwrap();
        let main = routing.prompt_cache_key();
        let recap = routing.aux_prompt_cache_key("recap");
        let side_question = routing.aux_prompt_cache_key("side_question");
        assert_ne!(main, recap, "aux key must never equal the main key");
        assert_ne!(recap, side_question, "aux kinds must be isolated");
        assert_eq!(
            routing.aux_prompt_cache_key("recap"),
            recap,
            "aux key must be stable for the same kind"
        );
        assert!(main.starts_with("grok:"));
        assert!(recap.starts_with("grok:"));
        assert_ne!(main, recap, "isolation is by hashed namespace, not prefix");
    }

    #[test]
    fn route_is_stable_across_endpoint_paths() {
        let dir = tempfile::tempdir().unwrap();
        let normal = load_or_create(
            dir.path(),
            PROVIDER,
            "https://api.x.ai/v1/responses",
            MODEL,
            PRINCIPAL,
        )
        .unwrap();
        let compact = load_or_create(
            dir.path(),
            PROVIDER,
            "https://api.x.ai/v1/responses/compact",
            MODEL,
            PRINCIPAL,
        )
        .unwrap();
        assert_eq!(
            normal.route, compact.route,
            "compact and post-compact must share one cache route"
        );
        assert_eq!(
            normal.prompt_cache_key(),
            compact.prompt_cache_key(),
            "same namespace + route ⇒ same prompt cache key"
        );
        // Different deployment / model / principal reroute.
        let other_deployment = load_or_create(
            dir.path(),
            PROVIDER,
            "https://other.example.com/v1/responses",
            MODEL,
            PRINCIPAL,
        )
        .unwrap();
        assert_ne!(normal.route, other_deployment.route);
        let other_model = load_or_create(
            dir.path(),
            PROVIDER,
            BASE_URL,
            "grok-5",
            PRINCIPAL,
        )
        .unwrap();
        assert_ne!(normal.route, other_model.route);
        let other_principal =
            load_or_create(dir.path(), PROVIDER, BASE_URL, MODEL, "principal-b").unwrap();
        assert_ne!(normal.route, other_principal.route);
    }

    #[test]
    fn pre_existing_namespace_is_adopted() {
        // Simulates a concurrent creator that already won: a valid id on
        // disk must be adopted verbatim, never regenerated.
        let dir = tempfile::tempdir().unwrap();
        let fixed = "11111111-2222-4333-8444-555555555555";
        std::fs::write(dir.path().join(CACHE_NAMESPACE_FILE), format!("{fixed}\n")).unwrap();
        let routing = load(dir.path()).unwrap();
        assert_eq!(routing.namespace_id, fixed);
    }
}

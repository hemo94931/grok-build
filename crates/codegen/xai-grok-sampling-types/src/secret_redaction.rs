//! Best-effort redaction for untrusted upstream text that may echo credentials.
//!
//! This lives in the sampling-types layer so HTTP parsing can sanitize before
//! an error reaches tracing, ACP payloads, persistence, or user-visible text.

const REDACTED: &str = "[REDACTED]";

/// Replace the exact request credential before applying best-effort shape
/// redaction. Request-scoped knowledge is the only reliable way to remove an
/// arbitrary unlabeled echo, including a two-byte credential.
pub fn redact_known_credential(raw: &str, credential: Option<&str>) -> String {
    redact_known_credentials(raw, credential.into_iter())
}

/// Replace every exact request-scoped secret before shape redaction.
///
/// This is used when a request can carry credentials in more than one place,
/// such as an auth header plus configured query parameters. Empty values are
/// ignored and replacement order is longest-first so an overlapping shorter
/// secret cannot leave a suffix of a longer one behind.
pub fn redact_known_credentials<'a>(
    raw: &str,
    credentials: impl IntoIterator<Item = &'a str>,
) -> String {
    let mut credentials = credentials
        .into_iter()
        .filter(|credential| !credential.is_empty())
        .flat_map(|credential| {
            let prefix = credential.chars().take(8).collect::<String>();
            let suffix = credential
                .chars()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>();
            [credential.to_owned(), prefix, suffix]
        })
        .collect::<Vec<_>>();
    credentials.sort_unstable_by_key(|credential| std::cmp::Reverse(credential.len()));
    credentials.dedup();

    let exact = credentials
        .into_iter()
        .fold(raw.to_owned(), |text, credential| {
            text.replace(&credential, REDACTED)
        });
    redact_credential_shaped_text(&exact)
}

/// Replace labeled credentials and common provider-token formats with a
/// constant marker. The returned string never reveals the secret's length.
pub fn redact_credential_shaped_text(raw: &str) -> String {
    static LABELED_SECRET: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static PREFIXED_SECRET: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let labeled = LABELED_SECRET.get_or_init(|| {
        regex::Regex::new(
            r#"(?i)((?:(?:bearer|basic)\s+|authorization[\"']?\s*[:=]\s*[\"']?(?:(?:bearer|basic)\s+)?|(?:api[_ -]?key|access[_ -]?token|secret|password)[\"']?\s*[:=]\s*[\"']?))([^\s,;\"'}]+)"#,
        )
        .expect("credential labeled-secret regex")
    });
    let prefixed = PREFIXED_SECRET.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b(?:x-xai-token-auth[a-z0-9._-]*|xai[-_][a-z0-9._-]{6,}|(?:sk[-_]|gh[pousr]_|github_pat_|gsk_|or-|rk-|pk-)[a-z0-9._-]{6,})",
        )
        .expect("credential prefixed-secret regex")
    });
    let redacted = labeled.replace_all(raw, "${1}[REDACTED]");
    prefixed
        .replace_all(&redacted, |capture: &regex::Captures<'_>| {
            let matched = capture.get(0).expect("whole regex match").as_str();
            if matched.eq_ignore_ascii_case("x-xai-token-auth")
                || matched.eq_ignore_ascii_case("xai-grok-cli")
            {
                matched.to_owned()
            } else {
                REDACTED.to_owned()
            }
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_labeled_prefixed_and_short_credentials_without_fragments() {
        for secret in [
            "sentinel-prefix-raw-value-sentinel-suffix",
            "sk-ant-oat-sentinel-prefix-middle-sentinel-suffix",
            "xy",
        ] {
            let raw = format!("authorization: Bearer {secret}; api_key={secret}");
            let redacted = redact_credential_shaped_text(&raw);
            assert_secret_absent(secret, &redacted);
        }
        assert!(
            redact_credential_shaped_text("failed for sk-ant-oat-abcdef123456").contains(REDACTED)
        );
    }

    #[test]
    fn exact_request_credential_redacts_unlabeled_short_and_query_echoes() {
        for (secret, raw) in [
            (
                "sentinel-prefix-raw-value-sentinel-suffix",
                "upstream rejected sentinel-prefix-raw-value-sentinel-suffix",
            ),
            ("xy", "bad credential: xy"),
            (
                "abcdefghijklmnop",
                "URL https://host/v1?key=abcdefghijklmnop; key sent was abcdefghijklmnop",
            ),
        ] {
            let redacted = redact_known_credential(raw, Some(secret));
            assert_secret_absent(secret, &redacted);
        }
    }

    #[test]
    fn exact_request_credential_redacts_standalone_edge_fragments() {
        let secret = "sentinel-prefix-raw-value-sentinel-suffix";
        let raw = format!(
            "upstream retained only prefix={} and suffix={}",
            &secret[..8],
            &secret[secret.len() - 8..]
        );
        let redacted = redact_known_credential(&raw, Some(secret));
        assert!(!redacted.contains(&secret[..8]));
        assert!(!redacted.contains(&secret[secret.len() - 8..]));
    }

    #[test]
    fn exact_request_credentials_redact_auth_and_unlabeled_query_secrets() {
        let auth = "header-secret-long";
        let query = "xy";
        let raw = format!("auth {auth}; upstream URL https://host/v1?tenant={query}");
        let redacted = redact_known_credentials(&raw, [auth, query]);
        assert_secret_absent(auth, &redacted);
        assert_secret_absent(query, &redacted);
    }

    #[test]
    fn xai_secret_shapes_redact_without_corrupting_fixed_identifiers() {
        let identifiers = redact_credential_shaped_text("X-XAI-Token-Auth: xai-grok-cli");
        assert_eq!(identifiers, "X-XAI-Token-Auth: xai-grok-cli");

        for secret in [
            "xai-secret-value",
            "xai_grok_cli_secret",
            "x-xai-token-auth-secret",
        ] {
            let redacted = redact_credential_shaped_text(&format!("failed for {secret}"));
            assert_secret_absent(secret, &redacted);
        }
    }

    fn assert_secret_absent(secret: &str, redacted: &str) {
        for fragment in [
            secret,
            &secret[..secret.len().min(8)],
            &secret[secret.len().saturating_sub(secret.len().min(8))..],
        ] {
            assert!(
                !redacted.contains(fragment),
                "leaked {fragment:?}: {redacted}"
            );
        }
    }
}

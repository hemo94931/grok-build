//! Provider-auth ACP extension protocol types.
//!
//! The API-key entry flow uses a dedicated reverse ACP method instead of the
//! generic ask-user-question tool so secrets never travel through ordinary
//! prompts, annotations, notifications, or telemetry surfaces.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use zeroize::{Zeroize, Zeroizing};

/// Reverse ACP method used by a provider-auth login flow to ask the client for
/// an API key/secret.
pub const PROMPT_SECRET_METHOD: &str = "x.ai/providerAuth/promptSecret";

/// Client-capability key advertised in ACP initialize metadata.
pub const PROMPT_SECRET_CAPABILITY: &str = "x.ai/providerAuth.promptSecret";

/// Redact credential-shaped values before provider-auth errors reach tracing,
/// ACP error data, truncation, or user-visible scrollback.
pub fn redact_provider_auth_error(raw: &str) -> String {
    static LABELED_SECRET: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static PREFIXED_SECRET: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let labeled = LABELED_SECRET.get_or_init(|| {
        regex::Regex::new(
            r#"(?i)((?:bearer\s+|authorization[\"']?\s*[:=]\s*[\"']?(?:(?:bearer|basic)\s+)?|(?:api[_ -]?key|access[_ -]?token|secret|password)[\"']?\s*[:=]\s*[\"']?))([^\s,;\"'}]+)"#,
        )
        .expect("provider auth labeled-secret regex")
    });
    let prefixed = PREFIXED_SECRET.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b(?:sk[-_]|gh[pousr]_|github_pat_|gsk_|or-|rk-|pk-|xai[-_])[a-z0-9._-]{6,}",
        )
        .expect("provider auth prefixed-secret regex")
    });
    let redacted = labeled.replace_all(raw, "${1}[REDACTED]");
    prefixed.replace_all(&redacted, "[REDACTED]").into_owned()
}

/// Provider credential method carried by the provider-auth ACP protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderAuthMethod {
    #[serde(rename = "oauth")]
    OAuth,
    #[serde(rename = "api_key")]
    ApiKey,
}

impl ProviderAuthMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OAuth => "oauth",
            Self::ApiKey => "api_key",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::OAuth => "OAuth",
            Self::ApiKey => "API key",
        }
    }
}

impl fmt::Display for ProviderAuthMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Credential source that produced a provider-auth failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderAuthSource {
    #[serde(rename = "stored_oauth")]
    StoredOAuth,
    #[serde(rename = "stored_api_key")]
    StoredApiKey,
    Environment {
        variable: String,
    },
    Model {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        environment_variable: Option<String>,
    },
}

/// Structured, secret-free recovery metadata for a provider credential error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderAuthRemedy {
    pub provider: String,
    pub provider_display_name: String,
    pub method: ProviderAuthMethod,
    pub source: ProviderAuthSource,
}

impl ProviderAuthRemedy {
    /// Stored credentials can be repaired by immediately reopening the same
    /// provider login method. Environment/model BYOK needs guidance instead.
    pub const fn automatic_login_method(&self) -> Option<ProviderAuthMethod> {
        match self.source {
            ProviderAuthSource::StoredOAuth => Some(ProviderAuthMethod::OAuth),
            ProviderAuthSource::StoredApiKey => Some(ProviderAuthMethod::ApiKey),
            ProviderAuthSource::Environment { .. } | ProviderAuthSource::Model { .. } => None,
        }
    }
}

/// Metadata-only request body for [`PROMPT_SECRET_METHOD`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptSecretRequest {
    /// Provider id from the provider-auth registry (for example `anthropic`).
    pub provider: String,
    /// Human-readable provider name for UI copy.
    pub provider_display_name: String,
    /// ACP session id associated with the login flow.
    pub session_id: String,
    /// Prompt label/heading. Metadata only; never contains a user secret.
    pub prompt: String,
    /// Optional monotonic request sequence from the provider-auth login flow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_seq: Option<u64>,
}

/// Mutable secret input buffer whose diagnostics never expose bytes or length.
/// A fixed upper-bound allocation prevents `String` growth from leaving old
/// secret prefixes behind in freed allocator blocks.
pub struct RedactedSecretBuffer(Zeroizing<String>);

const SECRET_BUFFER_CAPACITY: usize = 16 * 1024;

impl Default for RedactedSecretBuffer {
    fn default() -> Self {
        Self(Zeroizing::new(String::with_capacity(
            SECRET_BUFFER_CAPACITY,
        )))
    }
}

impl RedactedSecretBuffer {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn push(&mut self, ch: char) {
        if self.0.len().saturating_add(ch.len_utf8()) <= SECRET_BUFFER_CAPACITY {
            self.0.push(ch);
        }
    }

    pub fn push_str(&mut self, value: &str) {
        let remaining = SECRET_BUFFER_CAPACITY.saturating_sub(self.0.len());
        let end = value
            .char_indices()
            .map(|(index, ch)| index + ch.len_utf8())
            .take_while(|end| *end <= remaining)
            .last()
            .unwrap_or(0);
        self.0.push_str(&value[..end]);
    }

    pub fn pop(&mut self) -> bool {
        let Some(ch) = self.0.chars().next_back() else {
            return false;
        };
        let new_len = self.0.len() - ch.len_utf8();
        // SAFETY: `new_len` is the boundary immediately before the final UTF-8
        // scalar. We wipe exactly that scalar's bytes and then truncate them,
        // leaving the retained prefix valid UTF-8.
        unsafe {
            let bytes = self.0.as_mut_vec();
            bytes[new_len..].zeroize();
            bytes.truncate(new_len);
        }
        true
    }

    pub fn clear(&mut self) {
        self.0.zeroize();
    }

    /// Move a non-blank, outer-whitespace-trimmed value into the wire wrapper.
    /// When trimming is needed, the retained value is copied once into its
    /// zeroizing wire owner before the complete input allocation is wiped.
    pub fn take_trimmed(&mut self) -> Option<RedactedSecret> {
        let trimmed = self.0.trim();
        if trimmed.is_empty() {
            self.0.zeroize();
            return None;
        }
        if trimmed.len() == self.0.len() {
            return Some(RedactedSecret(Zeroizing::new(std::mem::take(&mut *self.0))));
        }
        let secret = Zeroizing::new(trimmed.to_owned());
        self.0.zeroize();
        Some(RedactedSecret(secret))
    }
}

impl fmt::Debug for RedactedSecretBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RedactedSecretBuffer([REDACTED])")
    }
}

/// Secret-bearing value whose diagnostics are always redacted.
pub struct RedactedSecret(Zeroizing<String>);

impl RedactedSecret {
    pub fn new(secret: impl Into<String>) -> Self {
        Self(Zeroizing::new(secret.into()))
    }

    /// Consume the wrapper and return a zeroizing string for the caller to use
    /// at the storage boundary without cloning into ordinary prompt/trace state.
    pub fn into_zeroizing_string(self) -> Zeroizing<String> {
        self.0
    }
}

impl fmt::Debug for RedactedSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RedactedSecret([REDACTED])")
    }
}

impl Serialize for RedactedSecret {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.0.as_str())
    }
}

impl<'de> Deserialize<'de> for RedactedSecret {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self::new)
    }
}

/// Response body for [`PROMPT_SECRET_METHOD`].
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PromptSecretResponse {
    Accepted { secret: RedactedSecret },
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_auth_error_redaction_covers_labels_bearers_and_provider_prefixes() {
        let raw = "api_key=plain-value Authorization: Bearer plain-bearer-value \
            {\"access_token\":\"json-plain-value\"} github_pat_abcdef123456";
        let redacted = redact_provider_auth_error(raw);
        for secret in [
            "plain-value",
            "plain-bearer-value",
            "json-plain-value",
            "github_pat_abcdef123456",
        ] {
            assert!(!redacted.contains(secret));
        }
        assert!(redacted.matches("[REDACTED]").count() >= 3);
    }

    #[test]
    fn provider_auth_method_and_remedy_have_stable_wire_shapes() {
        let remedy = ProviderAuthRemedy {
            provider: "anthropic".to_string(),
            provider_display_name: "Anthropic".to_string(),
            method: ProviderAuthMethod::ApiKey,
            source: ProviderAuthSource::Environment {
                variable: "ANTHROPIC_API_KEY".to_string(),
            },
        };

        let json = serde_json::to_value(&remedy).unwrap();
        assert_eq!(json["method"], "api_key");
        assert_eq!(json["source"]["type"], "environment");
        assert_eq!(json["source"]["variable"], "ANTHROPIC_API_KEY");
        let round_trip: ProviderAuthRemedy = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, remedy);
        assert_eq!(round_trip.automatic_login_method(), None);

        let stored = ProviderAuthRemedy {
            source: ProviderAuthSource::StoredOAuth,
            method: ProviderAuthMethod::OAuth,
            ..remedy
        };
        assert_eq!(
            stored.automatic_login_method(),
            Some(ProviderAuthMethod::OAuth)
        );
        let stored_json = serde_json::to_value(&stored).unwrap();
        assert_eq!(stored_json["method"], "oauth");
        assert_eq!(stored_json["source"]["type"], "stored_oauth");
    }

    #[test]
    fn prompt_secret_request_is_metadata_only() {
        let req = PromptSecretRequest {
            provider: "anthropic".to_string(),
            provider_display_name: "Anthropic".to_string(),
            session_id: "s1".to_string(),
            prompt: "Enter Anthropic API key".to_string(),
            request_seq: Some(7),
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["provider"], "anthropic");
        assert_eq!(json["providerDisplayName"], "Anthropic");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["requestSeq"], 7);
        assert!(json.get("secret").is_none());
        assert!(json.get("answer").is_none());
    }

    #[test]
    fn redacted_secret_buffer_mutates_without_debug_or_length_disclosure() {
        let mut buffer = RedactedSecretBuffer::default();
        buffer.push_str("  sk-buffer-secret  ");
        buffer.pop();
        buffer.push(' ');
        let debug = format!("{buffer:?}");
        assert_eq!(debug, "RedactedSecretBuffer([REDACTED])");
        assert!(!debug.contains("sk-buffer-secret"));
        let secret = buffer.take_trimmed().unwrap().into_zeroizing_string();
        assert_eq!(secret.as_str(), "sk-buffer-secret");
        assert!(buffer.is_empty());

        buffer.push_str(" \t\r\n ");
        assert!(buffer.take_trimmed().is_none());
        assert!(buffer.is_empty());
    }

    #[test]
    fn redacted_secret_buffer_never_reallocates_secret_prefixes() {
        let mut buffer = RedactedSecretBuffer::default();
        let initial_ptr = buffer.0.as_ptr();
        let initial_capacity = buffer.0.capacity();
        buffer.push_str(&"s".repeat(SECRET_BUFFER_CAPACITY * 2));
        assert_eq!(buffer.0.len(), SECRET_BUFFER_CAPACITY);
        assert_eq!(buffer.0.capacity(), initial_capacity);
        assert_eq!(buffer.0.as_ptr(), initial_ptr);
        buffer.push('x');
        assert_eq!(buffer.0.len(), SECRET_BUFFER_CAPACITY);
    }

    #[test]
    fn redacted_secret_buffer_wipes_removed_bytes() {
        let mut buffer = RedactedSecretBuffer::default();
        buffer.push_str("secret");
        let ptr = buffer.0.as_ptr();
        let original_len = buffer.0.len();
        assert!(buffer.pop());
        // SAFETY: the allocation remains owned by `buffer`; the byte removed
        // from the final UTF-8 scalar was initialized before `pop` and must be
        // overwritten rather than merely moved beyond String::len().
        let removed = unsafe { std::slice::from_raw_parts(ptr.add(original_len - 1), 1) };
        assert_eq!(removed, &[0]);

        let remaining_len = buffer.0.len();
        buffer.clear();
        // SAFETY: same live allocation, inspecting the initialized range that
        // `clear` is required to wipe before setting the logical length to 0.
        let cleared = unsafe { std::slice::from_raw_parts(ptr, remaining_len) };
        assert!(cleared.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn redacted_secret_debug_never_exposes_value_and_consumes_to_zeroizing() {
        let secret = RedactedSecret::new("sk-test-secret");
        let debug = format!("{secret:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("sk-test-secret"));

        let extracted = secret.into_zeroizing_string();
        assert_eq!(extracted.as_str(), "sk-test-secret");
    }

    #[test]
    fn prompt_secret_response_serializes_wire_secret_but_debug_redacts() {
        let response = PromptSecretResponse::Accepted {
            secret: RedactedSecret::new("sk-response-secret"),
        };

        let debug = format!("{response:?}");
        assert!(!debug.contains("sk-response-secret"));
        assert!(debug.contains("[REDACTED]"));

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["outcome"], "accepted");
        assert_eq!(json["secret"], "sk-response-secret");
    }
}

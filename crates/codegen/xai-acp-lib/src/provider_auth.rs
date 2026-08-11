//! Provider-auth ACP extension protocol types.
//!
//! The API-key entry flow uses a dedicated reverse ACP method instead of the
//! generic ask-user-question tool so secrets never travel through ordinary
//! prompts, annotations, notifications, or telemetry surfaces.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use zeroize::Zeroizing;

/// Reverse ACP method used by a provider-auth login flow to ask the client for
/// an API key/secret.
pub const PROMPT_SECRET_METHOD: &str = "x.ai/providerAuth/promptSecret";

/// Client-capability key advertised in ACP initialize metadata.
pub const PROMPT_SECRET_CAPABILITY: &str = "x.ai/providerAuth.promptSecret";

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

//! Core sampler types.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use xai_grok_sampling_types::{ConversationRequest, ResolvedResponsesRequest};

/// What the sampler actor should send for one request.
///
/// `Normal` is a typed conversation request; conversation defaults are
/// applied per attempt and the body is rebuilt inside the client, exactly
/// like historical behavior. `ResolvedResponses` carries an opaque,
/// already-validated frozen body: every retry reuses the identical body,
/// correlation headers and continuity metadata (plan-once, body-frozen
/// retry), and the attempt never re-reads caller state.
#[derive(Clone, Debug)]
pub enum SamplingDispatch {
    Normal(Box<ConversationRequest>),
    ResolvedResponses(Arc<ResolvedResponsesRequest>),
}

impl From<ConversationRequest> for SamplingDispatch {
    fn from(request: ConversationRequest) -> Self {
        Self::Normal(Box::new(request))
    }
}

/// Unique identifier for a sampling request.
///
/// Wraps a `String` so callers can pass an externally-assigned ID
/// (e.g., a session-assigned UUID) or generate a fresh random one via
/// [`RequestId::random`].
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RequestId(String);

impl RequestId {
    /// Generate a fresh random request ID backed by a UUIDv4.
    pub fn random() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Borrow the underlying string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for RequestId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for RequestId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_string_roundtrips() {
        let id: RequestId = String::from("abc-123").into();
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn from_str_roundtrips() {
        let id: RequestId = "xyz-789".into();
        assert_eq!(id.as_str(), "xyz-789");
    }

    #[test]
    fn display_matches_inner_string() {
        let id: RequestId = "display-me".into();
        assert_eq!(format!("{id}"), "display-me");
    }

    #[test]
    fn random_produces_unique_values() {
        let a = RequestId::random();
        let b = RequestId::random();
        assert_ne!(a, b, "two random IDs must differ");
        // UUIDv4 strings are 36 characters (8-4-4-4-12 hex with hyphens).
        assert_eq!(a.as_str().len(), 36);
    }
}

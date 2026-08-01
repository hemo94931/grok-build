//! Digest-stability proofs for `SystemItem::source`.
//!
//! The `SystemSource` field must never change the canonical bytes of
//! historical items: old JSON without a `source` field deserializes as
//! `LegacyUnclassified` and re-serializes to byte-identical canonical JSON,
//! so `portable_history_digest`, wrapper/marker comparisons and append
//! idempotency checks are unaffected.

use super::*;

/// A historical V1 system item as stored on disk (no `source` field).
const LEGACY_SYSTEM_JSON: &str = r#"{"type":"system","content":"base instructions"}"#;
/// The same item in canonical (key-sorted) form, as hashed by
/// `portable_history_digest`.
const LEGACY_SYSTEM_CANONICAL: &str =
    r#"{"content":"base instructions","type":"system"}"#;

#[test]
fn legacy_system_json_roundtrips_byte_identical() {
    let item: ConversationItem = serde_json::from_str(LEGACY_SYSTEM_JSON).unwrap();
    let ConversationItem::System(system) = &item else {
        panic!("expected system item");
    };
    assert_eq!(system.source, SystemSource::LegacyUnclassified);

    // Plain JSON round-trip is byte-identical…
    let serialized = serde_json::to_string(&item).unwrap();
    assert_eq!(serialized, LEGACY_SYSTEM_JSON);

    // …and the canonical encoding (used by portable_history_digest) matches.
    let canonical = canonical_json_bytes(&serde_json::to_value(&item).unwrap()).unwrap();
    assert_eq!(canonical, LEGACY_SYSTEM_CANONICAL.as_bytes());
}

#[test]
fn legacy_portable_history_digest_is_stable() {
    // A minimal historical portable history: system + user + assistant.
    let legacy_json = r#"[
        {"type":"system","content":"base instructions"},
        {"type":"user","content":[{"type":"text","text":"hello"}]},
        {"type":"assistant","content":"hi there"}
    ]"#;
    let history: Vec<ConversationItem> = serde_json::from_str(legacy_json).unwrap();

    let canonical_of = |items: &[ConversationItem]| {
        canonical_json_bytes(&serde_json::to_value(items).unwrap()).unwrap()
    };
    let first = canonical_of(&history);
    // Serialize and re-parse: the canonical bytes must not drift — this is
    // exactly what `portable_history_digest` hashes.
    let serialized = serde_json::to_string(&history).unwrap();
    let reparsed: Vec<ConversationItem> = serde_json::from_str(&serialized).unwrap();
    assert_eq!(first, canonical_of(&reparsed));

    // And the canonical bytes equal the historical ones verbatim.
    let expected = canonical_json_bytes(
        &serde_json::from_str::<serde_json::Value>(legacy_json).unwrap(),
    )
    .unwrap();
    assert_eq!(first, expected);
}

#[test]
fn explicit_sources_serialize_and_roundtrip() {
    for (item, source, fragment) in [
        (
            ConversationItem::base_instructions("base"),
            SystemSource::BaseInstructions,
            r#""source":"base_instructions""#,
        ),
        (
            ConversationItem::memory_context("<memory-context>x</memory-context>"),
            SystemSource::MemoryContext,
            r#""source":"memory_context""#,
        ),
        (
            ConversationItem::runtime_system("note"),
            SystemSource::Runtime,
            r#""source":"runtime""#,
        ),
    ] {
        let json = serde_json::to_string(&item).unwrap();
        assert!(
            json.contains(fragment),
            "expected {fragment} in {json}"
        );
        let parsed: ConversationItem = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.system_source(), Some(source));
    }
}

#[test]
fn default_source_is_never_serialized() {
    let json = serde_json::to_string(&ConversationItem::system("s")).unwrap();
    assert!(!json.contains("source"), "default source leaked into {json}");
}

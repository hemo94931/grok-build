use xai_grok_telemetry::events::{CompactionStrategyAttempt, TelemetryEvent};

#[test]
fn compaction_strategy_event_is_metadata_only() {
    assert_eq!(
        <CompactionStrategyAttempt as TelemetryEvent>::NAME,
        "compaction_strategy_attempt"
    );
    let value = serde_json::to_value(CompactionStrategyAttempt {
        compaction_id: "compaction".into(),
        supersedes_compaction_id: Some("previous-compaction".into()),
        strategy: "server",
        eligible: true,
        skip_reason: None,
        attempts: 1,
        outcome: "committed",
        failure: None,
        status: Some(200),
        latency_ms: 12,
        request_bytes: 100,
        response_bytes: 80,
        output_items: 1,
        capability_cache_hit: false,
        fallback_model: None,
        fallback_latency_ms: None,
        prefire_consumed: false,
        prefire_wasted: true,
        prefire_stale: false,
        history_revision: 7,
        request_identity_generation: 3,
        cas_outcome: "committed",
        checkpoint_schema: Some(2),
        checkpoint_bytes: Some(64),
        restore_outcome: None,
        marker_repaired: false,
        tokens_before: 1000,
        tokens_after: Some(100),
        token_seed_source: Some("usage_output_tokens"),
        commit_failure_step: None,
    })
    .unwrap();

    assert_eq!(value["supersedes_compaction_id"], "previous-compaction");

    for forbidden in [
        "identity",
        "digest",
        "url",
        "query",
        "header",
        "credential",
        "request_body",
        "response_body",
        "output",
    ] {
        assert!(
            value.get(forbidden).is_none(),
            "forbidden field: {forbidden}"
        );
    }
}

//! Live end-to-end verification of the compaction_trigger remote compaction
//! contract against the **real** ChatGPT Codex backend.
//!
//! Unlike the mock-based suites, this test drives an in-process agent with the
//! machine's real provider credentials (`~/.grok/providers.json`,
//! openai-codex OAuth), sends a large prompt, runs `/compact`, and asserts the
//! on-disk checkpoint artifacts (hard evidence) plus a successful follow-up
//! turn over the compacted context (replay acceptance).
//!
//! Gated twice: `#[ignore]`d by default AND requires `GROK_LIVE_CODEX_E2E=1`.
//!
//! ```bash
//! GROK_LIVE_CODEX_E2E=1 cargo test -p xai-grok-shell \
//!     --test responses_compaction_live_e2e -- --ignored --nocapture
//! ```

mod acp_harness;

use std::time::Duration;

use acp_harness::{AutoApproveClient, connect_noauth, prompt_turn};
use agent_client_protocol::{self as acp, Agent as _};
use serde_json::json;

const LIVE_MODEL: &str = "openai-codex/gpt-5.6-luna";
const TURN_TIMEOUT: Duration = Duration::from_secs(300);

/// The shrink check needs the conversation to clearly exceed the ~9.5k-token
/// prompt envelope; ~120 KB of text ≈ 30k tokens (codex-backend-quirks.md).
const MARKER: &str = "ZXQ-LIVE-E2E-MARKER: the lighthouse keeper keeps three brass keys";

fn big_prompt() -> String {
    let filler =
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima ".repeat(600);
    format!(
        "Remember this identifier verbatim: {MARKER}. \
         Acknowledge it, then answer with one short sentence.\n\n{filler}"
    )
}

fn session_compaction_dir(
    cwd: &std::path::Path,
    session_id: &acp::SessionId,
) -> std::path::PathBuf {
    let encoded = cwd.to_string_lossy().replace('/', "%2F");
    let home = std::env::var("GROK_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").expect("HOME")).join(".grok")
        });
    home.join("sessions")
        .join(encoded)
        .join(session_id.0.to_string())
}

/// Requires real openai-codex OAuth credentials and network access.
#[test]
#[ignore = "live backend test: run explicitly with GROK_LIVE_CODEX_E2E=1"]
fn live_codex_remote_compaction_trigger_e2e() {
    if std::env::var("GROK_LIVE_CODEX_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set GROK_LIVE_CODEX_E2E=1 to run against the live backend");
        return;
    }
    let _ = rustls::crypto::ring::default_provider().install_default();
    // SAFETY: single-test binary; no other threads read env yet.
    unsafe {
        std::env::set_var("GROK_TELEMETRY_ENABLED", "false");
        std::env::set_var("GROK_FEEDBACK_ENABLED", "false");
        std::env::set_var("GROK_TRACE_UPLOAD", "false");
        std::env::set_var("GROK_TURN_SUMMARY", "false");
        // Headless ACP auth gate only recognizes xAI-side credentials; the
        // dummy satisfies it while the namespaced codex model resolves via
        // the provider credential store (never sent upstream). See
        // provider-backend-debug SKILL.md step 0.
        if std::env::var("XAI_API_KEY").is_err() {
            std::env::set_var("XAI_API_KEY", "provider-gate-bypass");
        }
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let cwd =
                    std::env::temp_dir().join(format!("grok-live-compact-{}", std::process::id()));
                std::fs::create_dir_all(&cwd).unwrap();
                // NOTE: no ACP `authenticate` call. The harness helper's
                // xai.api_key flow would persist $XAI_API_KEY (a dummy) into
                // the real ~/.grok/auth.json, clobbering the user's xAI
                // credential. The codex model resolves via the provider
                // credential store, which needs no ACP auth handshake.
                let (conn, _init) = connect_noauth(AutoApproveClient, "live-compact-e2e").await;

                let session_id = tokio::time::timeout(
                    acp_harness::RPC_TIMEOUT,
                    conn.new_session(
                        acp::NewSessionRequest::new(cwd.clone())
                            .meta(json!({ "modelId": LIVE_MODEL }).as_object().cloned()),
                    ),
                )
                .await
                .expect("session/new timed out")
                .expect("session/new failed")
                .session_id;

                // Turn 1: seed a large conversation.
                tokio::time::timeout(TURN_TIMEOUT, async {
                    prompt_turn(&conn, &session_id, &big_prompt()).await;
                })
                .await
                .expect("seed turn timed out");

                // Turn 2: manual compaction — ServerFirst must reach the real
                // backend, collect the streamed compaction item, and commit
                // the retained-prefix + blob checkpoint.
                tokio::time::timeout(TURN_TIMEOUT, async {
                    prompt_turn(&conn, &session_id, "/compact").await;
                })
                .await
                .expect("/compact timed out");

                // Hard evidence: sidecar on disk with the v2 shape. The
                // sidecar file is named by checkpoint id.
                let checkpoints_dir =
                    session_compaction_dir(&cwd, &session_id).join("compaction_checkpoints");
                let sidecar = std::fs::read_dir(&checkpoints_dir)
                    .unwrap_or_else(|e| panic!("compaction_checkpoints dir missing: {e}"))
                    .filter_map(|entry| entry.ok().map(|e| e.path()))
                    .find(|path| path.extension().is_some_and(|ext| ext == "json"))
                    .unwrap_or_else(|| panic!("no checkpoint sidecar under {checkpoints_dir:?}"));
                let raw = std::fs::read_to_string(&sidecar).unwrap_or_else(|e| {
                    panic!("checkpoint sidecar unreadable at {sidecar:?}: {e}")
                });
                let file: serde_json::Value = serde_json::from_str(&raw).unwrap();
                assert_eq!(file["kind"], "responses_server", "sidecar kind");
                let wrapper = &file["wrapper"];
                assert!(
                    wrapper["compaction_item"]["encrypted_content"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty()),
                    "wrapper must carry the opaque compaction blob"
                );
                assert!(
                    wrapper["retained_prefix"].as_array().is_some(),
                    "wrapper must carry the retained typed prefix"
                );
                assert!(
                    wrapper.get("output").is_none(),
                    "old unary output shape must be gone"
                );

                // Turn 3: follow-up over the compacted context — the backend
                // accepting the replayed checkpoint IS the replay verification.
                tokio::time::timeout(TURN_TIMEOUT, async {
                    prompt_turn(
                        &conn,
                        &session_id,
                        "What identifier did I ask you to remember? One short sentence.",
                    )
                    .await;
                })
                .await
                .expect("follow-up turn timed out");

                println!("live e2e ok: session {}", session_id.0);
            })
            .await;
    });
}

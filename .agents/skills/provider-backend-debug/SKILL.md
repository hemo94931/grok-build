---
name: provider-backend-debug
description: Debug and fix wire compatibility between grok-build's provider routing and third-party model backends (OpenAI Codex/ChatGPT, Anthropic, GitHub Copilot, OpenRouter, Kimi Coding, Radius, custom endpoints). Use whenever a provider-backed model returns HTTP 400/404, hangs, retries or resamples infinitely, when OAuth login (codex, device-code) needs headless verification, when adding or upgrading a provider endpoint (responses, responses/compact, messages, completions), or when the user asks to test a provider integration end-to-end. Also use when a request works via curl but fails through grok, or when you need to see the exact bytes grok sends. Covers traffic capture via reverse proxy, iterative curl replay probing of backend-rejected parameters, triage of real bugs vs designed behavior, and locking findings into contract tests.
---

# Provider Backend Debug

A repeatable workflow for finding and fixing incompatibilities between grok-build's provider wire layer and third-party model backends. The method is evidence-first: capture the exact bytes, replay them against the real backend, and let the backend itself enumerate what it rejects. Do not guess from error messages surfaced through grok — they are often truncated or wrapped (e.g. `"API error (status 400 Bad Request)"` hides the body's `detail` field).

## The core loop

```
capture → replay → probe → triage → fix at the right layer → lock with tests + hard-evidence E2E
```

Each step below explains *why* it matters, not just what to do.

## Step 0 — Headless auth setup (repo-specific)

Provider logins live in `~/.grok/providers.json` (mode 600). Never print its contents; extract fields programmatically and use them immediately.

- **OAuth without a browser**: `grok login --provider openai-codex --device-auth` prints a device code + URL; the user completes it in a browser (15 min validity). Run it backgrounded with output to a log and poll the log.
- **Headless runs need a dummy xAI key**: `XAI_API_KEY=provider-gate-bypass grok -p "..." -m openai-codex/<model> ...`. The headless ACP auth gate only recognizes xAI-side credentials (provider models are built with `auth_provider: None`, so they don't count as `has_own_credentials`). The dummy key satisfies the gate; namespaced provider models route to the provider credential store and the dummy is never sent upstream. This is a known upstream design gap — do not "fix" it ad hoc mid-debug; note it and move on.
- The background task that refreshes the xAI model catalog will 400 with the dummy key and retry with backoff. That noise is harmless; filter it out of logs (`Incorrect API key provided`).

## Step 1 — Capture the exact wire traffic

Backend errors surfaced through grok lose the response body. Capture it yourself.

1. Start the bundled proxy: `python3 scripts/capture_proxy.py [port] [upstream]` (defaults `8899`, `https://chatgpt.com`). It logs full request/response to `/tmp/capture.log`.
2. Point the model at it via `~/.grok/config.toml` (singular `[model.<id>]`, NOT `[[models]]` — the latter fails config parse with "invalid type: map, expected a string"):
   ```toml
   [model."openai-codex/gpt-5.6-luna"]
   model = "openai-codex/gpt-5.6-luna"
   base_url = "http://127.0.0.1:8899/backend-api"
   ```
3. Run the grok scenario, then read `/tmp/capture.log` for the exact request body and the backend's full error JSON.
4. **Always remove the override and kill the proxy when done.** A stale override silently routes production traffic through a dead proxy.

The proxy is single-threaded. If grok logs `error sending request` transport errors while the capture shows successful 200s, suspect proxy serialization under concurrent connections — it is a capture artifact, not a grok bug (re-run, or accept partial capture).

## Step 2 — Replay and probe

Backends typically reject one parameter per response (`{"detail":"Unsupported parameter: temperature"}`), so a single failed request tells you almost nothing. Enumerate systematically:

1. Replay the captured body verbatim with curl against the **real** endpoint (get token + account id from `~/.grok/providers.json` without printing them). Fix the first reported problem, replay again, repeat until HTTP 200. This yields the minimal transformation set.
2. Then probe candidates one at a time with `scripts/probe_params.py` to build the full accept/reject table — fix only what you can prove is rejected, and document what is accepted so you don't over-strip (over-stripping silently drops features like `prompt_cache_key`).
3. Probe the *right* endpoint: path prefixes matter. On the ChatGPT Codex backend, `responses` and `responses/compact` both live under `codex/`; the unprefixed `responses/compact` answers 404 with an HTML error page (easy to misread as an auth failure).

See `references/codex-backend-quirks.md` for the live-verified accept/reject tables and response-shape quirks for the Codex backend. When debugging a different provider, build the equivalent table first and add it to that reference.

## Step 3 — Triage: real bug vs designed behavior

Not every failure is a wire bug. Check before patching:

- **Retry/resample loops**: grok retries "empty" responses. The Codex backend sends `output: []` in the terminal `response.completed` frame and delivers all content via streamed `output_item.done` events — a client that only reads `response.output` sees a legitimately complete answer as empty and resamples forever (~30 s backoff cycle in logs). Check the capture for repeated byte-identical requests.
- **Deliberate fallbacks**: server-side compaction falls back to local summary when the shrink check fails (`compressed output + prompt envelope < conversation tokens`). On small sessions this is mathematically guaranteed and is correct behavior for every backend, not a compatibility bug. Verify with a session whose content clearly exceeds the envelope (~9.5k tokens with 25 tools + system prompt).
- **Capability gates**: `ProviderCapabilities` fail-closes first-party features (remote compaction, checkpoints) per provider. A missing feature may be a descriptor flag, not a wire failure.
- **Cached negative capability**: unsupported compaction capability is cached (1 h TTL), so a fixed endpoint may still be skipped in a long-lived process. Re-test in a fresh process.

## Step 4 — Fix at the right layer

Pick the narrowest layer; keep provider isolation absolute. First-party xAI traffic (`grok-*` models on xAI URLs) must never enter provider sanitize paths — `ProviderWireRoute::from_config` returns `None` for it, and there is a test locking that.

| Symptom | Layer |
|---|---|
| Request-body fields rejected on the normal path | `xai-grok-sampler/src/provider_wire.rs` `sanitize_body`, inside the matching `ProviderKind` branch only |
| Sealed/frozen bodies (standalone compaction) bypass `sanitize_body` | a dedicated route method (e.g. `sanitize_compact_body`) applied at serialization; remember the `model` id rewrite — sealed bodies carry the namespaced catalog id, which backends reject |
| Wrong URL path for a provider | `endpoint_path` match arm |
| Feature disabled per provider | `ProviderCapabilities` in `xai-grok-shell/src/auth/providers/route.rs` |
| Content lost between stream events and final response | `xai-grok-sampler/src/stream/responses.rs` (stream-layer accumulation/backfill) |

Never let a fix for one provider leak into shared branches: shared post-processing (compaction-item stripping, `x_search` tool filtering) stays byte-identical for other providers.

## Step 5 — Lock it in

- Add/extend `*_matches_contract` tests with the exact accept/reject behavior you verified live, and cite "verified against the live endpoint" in a comment — these tests are the contract documentation.
- Run the crate's unit tests (`cargo test -p xai-grok-sampler --lib`). Before attributing any failure to your change, `git stash` and re-run — this repo has pre-existing failures (2 doom-loop integration tests in `test_actor.rs`).
- Verify E2E with hard evidence, not stdout alone: checkpoint files on disk (`kind: responses_server`, `compaction_summary` items), capture-log request/response interleaving, usage-token accounting, and a follow-up turn proving the backend accepted replayed context.

## Pitfalls

- `GROK_LOG_SAMPLING` must be `true`/`false` (clap parses it as a bool flag value), not `1`.
- `--log-sampling` writes `~/.grok/logs/sampling.jsonl` (request metadata, not full bodies); the debug log (`--debug --debug-file <path>`) carries spans and errors. Use both, but trust the capture proxy over either for wire truth.
- Debug logs can contain full bearer tokens inside serialized configs. Redact before sharing logs or captures.
- `grok models` prints "You are not authenticated" when only the default xAI provider lacks credentials — provider models remain usable; this is not a blocker.

## Files

- `scripts/capture_proxy.py` — single-threaded reverse proxy capturing full request/response to `/tmp/capture.log`
- `scripts/probe_params.py` — per-parameter accept/reject prober against a live endpoint
- `references/codex-backend-quirks.md` — live-verified Codex backend contract: endpoint map, rejected/accepted parameters, response-shape quirks

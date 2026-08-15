# ChatGPT Codex backend — live-verified wire contract

Base URL: `https://chatgpt.com/backend-api`. Auth: `authorization: Bearer <oauth access token>` + `chatgpt-account-id: <accountId>` + `originator: pi` + `openai-beta: responses=experimental` (all four required; missing `chatgpt-account-id` fails).

Everything below was verified against the live endpoint with curl (see `scripts/probe_params.py` to re-verify or extend). Treat this as the reference contract when touching `provider_wire.rs`.

## Endpoint map

| Logical path | Required path on this backend | Notes |
|---|---|---|
| `responses` | `codex/responses` | SSE only: `stream:true` is mandatory (`{"detail":"Stream must be set to true"}`) |
| `responses/compact` | ~~`codex/responses/compact`~~ | **DEPRECATED upstream** — remote compaction now rides the ordinary `codex/responses` request; see the v2 contract section below |

## Remote compaction v2 contract — `compaction_trigger` over `codex/responses`

**Live-verified 2026-08-14** against the production ChatGPT backend (probe scripts:
trigger baseline + header variants + replay/recompact shapes):

- Compaction = ordinary streaming `POST codex/responses` whose `input` ends with a bare
  control item `{"type":"compaction_trigger"}` (request-only, never persisted).
- Verified response: `response.created` → `response.in_progress` →
  `response.output_item.added` → `response.output_item.done` carrying exactly one
  `{"type":"compaction", id:"cmp_…", encrypted_content}` item (probe blobs 1.7–2.5 KB),
  then `response.completed` with **`output: []` (empty, per quirk 1)** and full `usage`
  (`output_tokens`/`total_tokens` present — did-not-shrink inputs available).
- **Header necessity — all verified NOT required**: the trigger alone drives compaction.
  Probed and still HTTP 200 with a valid blob when dropping, independently and combined:
  `x-codex-beta-features: remote_compaction_v2`, `x-codex-turn-metadata`,
  `openai-beta: responses=experimental`. grok keeps sending the first two on compaction
  requests for upstream parity / future enforcement (upstream always advertises them).
- **Replay shape verified**: `[retained user msg…, compaction blob, new user msg]` →
  HTTP 200, follow-up answer correctly used the pre-compaction context.
- **Recompact shape verified**: `[retained…, old blob, new turns…, compaction_trigger]` →
  HTTP 200 with a fresh compaction item (chain semantics work).
- Client-side history rebuild: retained prefix = user/developer/system messages
  (non-final agent messages ≤ 10k tokens), truncated newest-first to a 64k-token
  budget, then the opaque blob as the last element.
- grok classification: HTTP 400/404/405/422/501 or a completed stream without exactly
  one compaction item (after 2 attempts) = endpoint unsupported → negative capability
  cache 1 h → builtin fallback; transport/timeout/cancel fall back without caching.
- **Replay integrity — live-verified failure mode**: a replayed `input` containing a
  `function_call` without its matching `function_call_output` is rejected with 400
  `No tool output found for function call call_…`. Since ToolResult items are never
  retained in the prefix (they live inside the blob), grok's retained-prefix filter
  drops assistant items that carry tool calls entirely — upstream retains *messages*
  only, so this matches reference behavior.
- grok live e2e (`tests/responses_compaction_live_e2e.rs`, gated on
  `GROK_LIVE_CODEX_E2E=1`): seed turn → `/compact` → on-disk sidecar asserted
  (`kind: responses_server`, wrapper = retained prefix + blob, real `cmp_…` id) →
  follow-up turn over the replayed checkpoint accepted.

Legacy unary contract kept below for reference only.

## Request parameter contract — `codex/responses`

**Rejected with HTTP 400** (`{"detail":"Unsupported parameter: <name>"}` unless noted):

- Any `input` item with `role:"system"` → `"System messages are not allowed"`. System prompts must travel in the top-level `instructions` string.
- `includeSystemPrompt` (grok-internal flag; rejected outright)
- `max_output_tokens`, `max_tool_calls`
- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`
- `stream_options`, `truncation`, `metadata`, `safety_identifier`, `background`
- `service_tier` → `"Unsupported service_tier: auto"`
- `model` carrying the namespaced catalog id (`openai-codex/gpt-5.6-luna`) → `"The '<id>' model is not supported when using Codex with a ChatGPT account."` — always send the bare upstream model id.

**Accepted (HTTP 200)**: `tool_choice`, `parallel_tool_calls`, `prompt_cache_key`, `text`, `include: ["reasoning.encrypted_content"]`, `tools` (function schemas must themselves be valid — `additionalProperties: false` required in `parameters`).

## Request parameter contract — `codex/responses/compact` (LEGACY, deprecated upstream)

Same base, but a *different* rejection set — probe each endpoint separately:

- Rejected: `service_tier`, `prompt_cache_retention`, `prompt_cache_options`
- Accepted: `prompt_cache_key`, `reasoning`, `text`, `tools`, `model`, `input`, `instructions`, `parallel_tool_calls`

Response shape: `object: "response.compaction"`, `output` = retained message items + one `compaction_summary` item carrying `encrypted_content`, plus `usage`. grok's validator accepts item types `compaction` | `compaction_summary` and rejects `compaction_trigger` | `summary` | `context` | `context_summary`.

## Response-shape quirks (client must tolerate)

1. **`output: []` in terminal frames.** All content (text, function calls, reasoning) arrives exclusively via streamed `response.output_item.done` events; the final `response.completed` frame carries an empty `output` array. A client that builds its final response only from `response.output` will treat every turn as empty and retry/resample forever. Reconstruct `output` from the streamed item events.
2. **No `data: [DONE]` sentinel.** The SSE stream simply ends after `response.completed`. Readers that require a `[DONE]` marker misclassify clean EOF as truncation.
3. **Extra fields everywhere** — `tool_usage`, `frequency_penalty`, `safety_identifier`, `phase:"final_answer"` on message items, etc. Deserializers must ignore unknown fields.
4. **Checkpoints replay as `compaction_summary` items in `input`** and are accepted on subsequent `codex/responses` calls (verified: follow-up turns answered from compacted context).

## Auth/account notes

- Credentials are OAuth (device-code or browser flow via `grok login --provider openai-codex`), stored in `~/.grok/providers.json` with `access`, `refresh`, `expires`, `accountId`. Never print the file; extract fields programmatically.
- The access token is a JWT; `chatgpt-account-id` must match the token's account claim.
- grok refreshes the token via the provider store; long debug sessions are fine.

## Session-side gotchas observed

- grok's prompt envelope (system prompt + 25 tool schemas) is ~9.5k tokens. Server-side compaction's shrink check (`compressed output + envelope < conversation tokens`) fails by construction on small sessions and falls back to local summary — designed behavior, every backend. Test remote compaction with ≥ ~20k-token sessions (e.g. `--prompt-file` with ~80 KB of text).
- Unsupported compaction capability is cached in-process for 1 h; re-test fixes in a fresh process.

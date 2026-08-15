# ChatGPT Codex backend — live-verified wire contract

Base URL: `https://chatgpt.com/backend-api`. Auth: `authorization: Bearer <oauth access token>` + `chatgpt-account-id: <accountId>` + `originator: pi` + `openai-beta: responses=experimental` (all four required; missing `chatgpt-account-id` fails).

Everything below was verified against the live endpoint with curl (see `scripts/probe_params.py` to re-verify or extend). Treat this as the reference contract when touching `provider_wire.rs`.

## Endpoint map

| Logical path | Required path on this backend | Notes |
|---|---|---|
| `responses` | `codex/responses` | SSE only: `stream:true` is mandatory (`{"detail":"Stream must be set to true"}`) |
| `responses/compact` | ~~`codex/responses/compact`~~ | **DEPRECATED upstream** — remote compaction now rides the ordinary `codex/responses` request; see the v2 contract section below |

## Remote compaction v2 contract — `compaction_trigger` over `codex/responses`

Derived from upstream codex source (`compact_remote_v2.rs` @ `1bb6384`) and the
pi-codex-compact plugin; **not yet live-probed by grok** — re-verify with
`scripts/probe_params.py` before relying on header minimality:

- Compaction = ordinary streaming `POST codex/responses` whose `input` ends with a bare
  control item `{"type":"compaction_trigger"}` (request-only, never persisted).
- Request headers: ordinary turn headers plus `x-codex-beta-features: remote_compaction_v2`
  and `x-codex-turn-metadata: {"request_kind":"compaction","compaction":{"implementation":"responses_compaction_v2","strategy":"memento"}}`
  (upstream always sends both on compaction requests; whether they are strictly
  required is a Phase-0 probe item).
- Response: the compaction blob is expected via `response.output_item.done` frames —
  per quirk 1 below the terminal `response.completed.response.output` is empty on this
  backend. Require exactly one distinct `{"type":"compaction"|"compaction_summary", id?,
  encrypted_content}` item; tolerate other items.
- Client-side history rebuild: retained prefix = user/developer/system messages
  (non-final agent messages ≤ 10k tokens), truncated newest-first to a 64k-token
  budget, then the opaque blob as the last element. Recompact = expand prior
  checkpoint (incl. old blob) and append the trigger again.
- grok classification: HTTP 400/404/405/422/501 or a completed stream without exactly
  one compaction item (after 2 attempts) = endpoint unsupported → negative capability
  cache 1 h → builtin fallback; transport/timeout/cancel fall back without caching.

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

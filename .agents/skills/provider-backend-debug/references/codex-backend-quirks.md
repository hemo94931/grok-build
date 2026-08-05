# ChatGPT Codex backend — live-verified wire contract

Base URL: `https://chatgpt.com/backend-api`. Auth: `authorization: Bearer <oauth access token>` + `chatgpt-account-id: <accountId>` + `originator: pi` + `openai-beta: responses=experimental` (all four required; missing `chatgpt-account-id` fails).

Everything below was verified against the live endpoint with curl (see `scripts/probe_params.py` to re-verify or extend). Treat this as the reference contract when touching `provider_wire.rs`.

## Endpoint map

| Logical path | Required path on this backend | Notes |
|---|---|---|
| `responses` | `codex/responses` | SSE only: `stream:true` is mandatory (`{"detail":"Stream must be set to true"}`) |
| `responses/compact` | `codex/responses/compact` | Unprefixed `responses/compact` → **404 with an HTML error page** (not JSON) |

The `codex/` prefix applies per-endpoint; grok's `endpoint_path` must rewrite both.

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

## Request parameter contract — `codex/responses/compact`

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

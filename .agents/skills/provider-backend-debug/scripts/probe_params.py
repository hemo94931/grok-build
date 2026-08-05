#!/usr/bin/env python3
"""Probe which request parameters a model backend accepts or rejects.

Sends a minimal valid request plus ONE candidate parameter at a time and
reports HTTP status + error detail per parameter. Backends typically reject
a single parameter per response ("Unsupported parameter: X"), so fixing
one error at a time by hand is slow; this enumerates the full table in one
run. Probe every distinct endpoint you care about (e.g. `responses` and
`responses/compact` have different rejection sets).

Usage:
    export PROBE_TOKEN=<bearer token>          # required
    export PROBE_ACCOUNT_ID=<account id>       # optional, codex only
    python3 probe_params.py <endpoint_url> [base_body.json] [candidates.json]

    endpoint_url     full URL to POST to
    base_body.json   minimal known-good body (default: codex responses shape)
    candidates.json  {"param": value, ...} (default: built-in list)

Read the token WITHOUT printing it, e.g.:
    export PROBE_TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.grok/providers.json'))['openai-codex']['access'])")
    export PROBE_ACCOUNT_ID=$(python3 -c "import json;print(json.load(open('$HOME/.grok/providers.json'))['openai-codex']['accountId'])")
"""
import json
import os
import subprocess
import sys

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else None
if not ENDPOINT:
    sys.exit(__doc__)

TOKEN = os.environ.get("PROBE_TOKEN", "")
ACCT = os.environ.get("PROBE_ACCOUNT_ID", "")
if not TOKEN:
    sys.exit("PROBE_TOKEN is required")

BASE_BODY = {
    "model": "gpt-5.6-luna",
    "input": [{"type": "message", "role": "user",
               "content": [{"type": "input_text", "text": "hi"}]}],
    "stream": True,
    "store": False,
}
DEFAULT_CANDIDATES = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 4096,
    "max_tool_calls": 5,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
    "stream_options": {"include_usage": True},
    "truncation": "disabled",
    "metadata": {"k": "v"},
    "safety_identifier": "x",
    "service_tier": "auto",
    "background": False,
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "prompt_cache_key": "k",
    "prompt_cache_retention": "24h",
    "prompt_cache_options": {"mode": "default"},
    "text": {"format": {"type": "text"}, "verbosity": "medium"},
    "include": ["reasoning.encrypted_content"],
}

body_path = sys.argv[2] if len(sys.argv) > 2 else None
cand_path = sys.argv[3] if len(sys.argv) > 3 else None
base = json.load(open(body_path)) if body_path else BASE_BODY
candidates = json.load(open(cand_path)) if cand_path else DEFAULT_CANDIDATES

headers = [
    f"authorization: Bearer {TOKEN}",
    "originator: pi",
    "openai-beta: responses=experimental",
    "content-type: application/json",
]
if ACCT:
    headers.append(f"chatgpt-account-id: {ACCT}")

for key, value in candidates.items():
    body = dict(base)
    body[key] = value
    cmd = ["curl", "-sS", "-o", "/tmp/probe-resp.txt", "-w", "%{http_code}", ENDPOINT]
    for h in headers:
        cmd += ["-H", h]
    cmd += ["-d", json.dumps(body), "--max-time", "90"]
    status = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    detail = ""
    if status != "200":
        try:
            detail = json.load(open("/tmp/probe-resp.txt")).get("detail", "")[:90]
        except Exception:
            detail = open("/tmp/probe-resp.txt").read()[:90]
    print(f"{key:24s} -> {status} {detail}")

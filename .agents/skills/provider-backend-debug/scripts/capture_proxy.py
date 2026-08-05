#!/usr/bin/env python3
"""Reverse proxy that captures full HTTP request/response traffic.

Usage:
    python3 capture_proxy.py [port] [upstream]

    port      listen port on 127.0.0.1 (default 8899)
    upstream  scheme://host to forward to (default https://chatgpt.com)

Writes an append-only log to /tmp/capture.log with headers (authorization
truncated) and complete bodies. Point grok at the proxy via a model
base_url override in ~/.grok/config.toml:

    [model."openai-codex/gpt-5.6-luna"]
    model = "openai-codex/gpt-5.6-luna"
    base_url = "http://127.0.0.1:8899/backend-api"

NOTE: single-threaded. Concurrent client connections can surface as
transport errors on the grok side — a capture artifact, not a grok bug.
Always remove the config override when done.
"""
import http.server
import ssl
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
UPSTREAM = sys.argv[2] if len(sys.argv) > 2 else "https://chatgpt.com"
LOG = "/tmp/capture.log"


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        with open(LOG, "ab") as f:
            f.write(b"=== REQUEST " + self.path.encode() + b"\n")
            for k, v in self.headers.items():
                if k.lower() == "authorization":
                    v = v[:20] + "...<redacted>"
                f.write(f"{k}: {v}\n".encode())
            f.write(b"--- body ---\n" + body + b"\n\n")

        req = urllib.request.Request(UPSTREAM + self.path, data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection", "accept-encoding"):
                req.add_header(k, v)
        try:
            resp = urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context())
            status, resp_body = resp.status, resp.read()
            resp_headers = dict(resp.headers)
        except urllib.error.HTTPError as e:
            status, resp_body = e.code, e.read()
            resp_headers = dict(e.headers)

        with open(LOG, "ab") as f:
            f.write(f"=== RESPONSE {status}\n".encode() + resp_body + b"\n\n")

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() in ("transfer-encoding", "connection", "content-encoding"):
                continue
            self.send_header(k, v)
        self.send_header("content-length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"capture proxy: 127.0.0.1:{PORT} -> {UPSTREAM}, log: {LOG}")
    http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

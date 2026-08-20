#!/usr/bin/env python3
"""Loopback streaming proxy that makes /metrics unavailable to a benchmark."""

from __future__ import annotations

import argparse
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_host = "127.0.0.1"
    upstream_port = 30000

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def _metrics_unavailable(self) -> None:
        body = json.dumps({"detail": "metrics intentionally hidden"}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _proxy(self) -> None:
        if self.path.split("?", 1)[0] == "/metrics":
            self._metrics_unavailable()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.casefold() not in HOP_BY_HOP and key.casefold() != "host"
        }
        upstream = http.client.HTTPConnection(
            self.upstream_host, self.upstream_port, timeout=600
        )
        try:
            upstream.request(self.command, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.casefold() not in HOP_BY_HOP and key.casefold() != "content-length":
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            upstream.close()
            self.close_connection = True

    do_GET = _proxy
    do_POST = _proxy


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, default=30001)
    parser.add_argument("--upstream-port", type=int, default=30000)
    args = parser.parse_args()
    ProxyHandler.upstream_port = args.upstream_port
    ProxyServer(("127.0.0.1", args.listen_port), ProxyHandler).serve_forever()


if __name__ == "__main__":
    main()

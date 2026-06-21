"""Static file server with cross-origin-isolation headers.

A plain `python -m http.server --directory site` is enough to run the exported app
in a normal browser (the Shinylive service worker supplies isolation). This helper
adds COOP/COEP headers explicitly, which some headless/CI browsers need.

    python scripts/serve_coi.py site 8000
"""
import functools
import http.server
import socketserver
import sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "site"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Service-Worker-Allowed", "/")
        super().end_headers()


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=DIR)) as httpd:
    print(f"serving {DIR} on http://127.0.0.1:{PORT}")
    httpd.serve_forever()

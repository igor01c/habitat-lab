#!/usr/bin/env python3
"""Serve the benchmark inspector from a results directory.

Copies inspector.html into the results directory (so relative paths work),
then starts a local HTTP server and opens the browser.

Usage:
    python -m habitat_object_benchmark.scripts.serve_inspector \
        --results-dir data/datasets/amara-spatial-10k/results \
        [--port 8765]
"""

import argparse
import http.server
import os
import shutil
import socketserver
import threading
import webbrowser
from pathlib import Path

INSPECTOR_HTML = Path(__file__).parent.parent / "inspector.html"


def main():
    parser = argparse.ArgumentParser(description="Serve the benchmark inspector")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    json_path = results_dir / "results.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"results.json not found in {results_dir}\n"
            f"Run build_results_json.py first."
        )

    # Copy the latest inspector.html into results_dir
    dst = results_dir / "inspector.html"
    shutil.copy2(INSPECTOR_HTML, dst)
    print(f"Copied inspector.html → {dst}")

    url = f"http://localhost:{args.port}/inspector.html"
    print(f"Serving {results_dir}")
    print(f"Open: {url}")

    os.chdir(results_dir)

    # Open browser after a short delay so the server is ready
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # silence request logs
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()

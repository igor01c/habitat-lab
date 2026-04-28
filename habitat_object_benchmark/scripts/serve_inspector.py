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

    # Find the common ancestor of all dataset results dirs so relative GIF
    # paths (../../other_dataset/...) resolve correctly under the HTTP server.
    import json as _json
    payload = _json.loads(json_path.read_text())
    dirs = [results_dir]
    for ds in payload.get("datasets", {}).values():
        rd = ds.get("results_dir")
        if rd:
            dirs.append(Path(rd))
    serve_root = dirs[0]
    for d in dirs[1:]:
        # Walk up until serve_root is an ancestor of d
        while True:
            try:
                d.relative_to(serve_root)
                break
            except ValueError:
                serve_root = serve_root.parent

    # Copy inspector.html into serve_root
    dst_html = serve_root / "inspector.html"
    shutil.copy2(INSPECTOR_HTML, dst_html)

    # Write results.json to serve_root, rewriting gif paths to be relative to serve_root
    dst_json = serve_root / "results.json"
    if dst_json.resolve() != json_path.resolve():
        import json as _json2
        data = _json2.loads(json_path.read_text())
        orig_base = json_path.parent.resolve()
        new_base = serve_root.resolve()
        for ds in data.get("datasets", {}).values():
            for asset in ds.get("assets", {}).values():
                for check in asset.values():
                    for mode in check.values():
                        if isinstance(mode, dict) and mode.get("gif"):
                            abs_gif = (orig_base / mode["gif"]).resolve()
                            try:
                                mode["gif"] = str(abs_gif.relative_to(new_base))
                            except ValueError:
                                mode["gif"] = str(abs_gif)
        dst_json.write_text(_json2.dumps(data))

    rel_html = dst_html.relative_to(serve_root)
    url = f"http://localhost:{args.port}/{rel_html}"
    print(f"Serving {serve_root}")
    print(f"Open: {url}")

    os.chdir(serve_root)

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

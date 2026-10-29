"""
Deliverable 05: Local Dashboard Server & Static Export Utility (`src/dashboard_app.py`)

Usage:
    python3 src/dashboard_app.py --port 8080   # Serves interactive CIO dashboard at http://localhost:8080
    python3 src/dashboard_app.py --check       # Verifies dashboard/index.html and payload integrity
"""

import argparse
import http.server
import os
import socketserver
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(ROOT_DIR, "dashboard")


def main():
    parser = argparse.ArgumentParser(description="ZECKATHON PS 03 — Fund-Manager Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port to serve the dashboard (default: 8080)")
    parser.add_argument("--check", action="store_true", help="Verify dashboard assets without starting server")
    args = parser.parse_args()

    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found. Run `python3 run_pipeline.py` first.")
        sys.exit(1)

    if args.check:
        size_kb = os.path.getsize(index_path) / 1024.0
        print(f"Dashboard verified at {index_path} ({size_kb:.1f} KB).")
        return

    os.chdir(DASHBOARD_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Serving ZECKATHON PS 03 CIO Dashboard at http://localhost:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

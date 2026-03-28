#!/usr/bin/env python3
"""
Wonderstand Ingest Server — Receives voice memos via HTTP, processes them.

Designed to run on an always-on Mac behind a Cloudflare tunnel.
Pairs with an iOS Shortcut that POSTs recordings after Voice Memos closes.

Usage:
    python ingest_server.py                    # Start on port 8787
    python ingest_server.py --port 9000        # Custom port
    python ingest_server.py --process-now      # Process all pending files in inbox

Endpoints:
    POST /upload          — Upload an audio file (multipart form or raw body)
    GET  /status          — Check server status and pending/processed counts
    GET  /health          — Simple health check
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BASE_DIR = Path(__file__).parent
INBOX_DIR = BASE_DIR / "inbox"
PROCESSED_FILE = BASE_DIR / ".processed_recordings"
OUTPUT_DIR = BASE_DIR / "output"

INBOX_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class IngestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/upload":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "No data")
            return

        # Get filename from header or generate one
        filename = self.headers.get("X-Filename", "")
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memo_{ts}.m4a"

        # Sanitize filename
        filename = Path(filename).name
        filepath = INBOX_DIR / filename

        # Save the file
        with open(filepath, "wb") as f:
            remaining = content_length
            while remaining > 0:
                chunk = self.rfile.read(min(8192, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"[Ingest] Received: {filename} ({size_mb:.1f} MB)")

        # Trigger processing in background
        threading.Thread(target=process_file, args=(filepath,), daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "received",
            "filename": filename,
            "size_mb": round(size_mb, 1),
        }).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if self.path == "/status":
            inbox_count = len(list(INBOX_DIR.glob("*.m4a"))) + len(list(INBOX_DIR.glob("*.qta")))
            processed = set()
            if PROCESSED_FILE.exists():
                processed = set(PROCESSED_FILE.read_text().strip().split("\n"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "running",
                "inbox_pending": inbox_count,
                "total_processed": len(processed),
            }).encode())
            return

        self.send_error(404)

    def log_message(self, format, *args):
        # Quieter logging
        pass


def process_file(filepath):
    """Run Wonderstand pipeline on a single file."""
    filename = filepath.name
    basename = filepath.stem

    # Check if already processed
    processed = set()
    if PROCESSED_FILE.exists():
        processed = set(PROCESSED_FILE.read_text().strip().split("\n"))
    if filename in processed:
        print(f"[Ingest] Already processed: {filename}")
        return

    output_json = OUTPUT_DIR / f"{basename}.json"
    output_notes = OUTPUT_DIR / f"{basename}_notes.md"

    print(f"[Ingest] Processing: {filename}...")
    venv_python = BASE_DIR / "venv" / "bin" / "python"
    cmd = [
        str(venv_python),
        str(BASE_DIR / "multimodal_analyzer.py"),
        str(filepath),
        "--audio-only",
        "--meeting-notes",
        "--diarize",
        "--output", str(output_json),
        "--notes-output", str(output_notes),
    ]

    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            # Mark as processed
            with open(PROCESSED_FILE, "a") as f:
                f.write(filename + "\n")
            print(f"[Ingest] Done: {filename} -> {output_notes.name}")
        else:
            print(f"[Ingest] Failed: {filename}")
            print(result.stderr[-500:] if result.stderr else "no stderr")
    except subprocess.TimeoutExpired:
        print(f"[Ingest] Timeout: {filename} (>30 min)")


def process_inbox():
    """Process all pending files in inbox."""
    audio_files = sorted(INBOX_DIR.glob("*.m4a")) + sorted(INBOX_DIR.glob("*.qta"))
    if not audio_files:
        print("[Ingest] Inbox empty.")
        return
    print(f"[Ingest] Found {len(audio_files)} files in inbox.")
    for f in audio_files:
        process_file(f)


def main():
    parser = argparse.ArgumentParser(description="Wonderstand Ingest Server")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--process-now", action="store_true", help="Process inbox and exit")
    args = parser.parse_args()

    if args.process_now:
        process_inbox()
        return

    server = HTTPServer(("0.0.0.0", args.port), IngestHandler)
    print(f"[Ingest] Wonderstand ingest server listening on port {args.port}")
    print(f"[Ingest] Upload: POST http://localhost:{args.port}/upload")
    print(f"[Ingest] Status: GET  http://localhost:{args.port}/status")
    print(f"[Ingest] Inbox:  {INBOX_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Ingest] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()

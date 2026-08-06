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
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*\.(m4a|wav|mp3|caf|qta|opus|aiff|flac)$")
RESERVED_NAMES = {".seen", ".processed_recordings", ".uploaded_memos", ".DS_Store"}


def _safe_filename(raw: str) -> str:
    """Reject anything that could clobber internal state files. Returns a safe
    name or a server-generated fallback if the input is malformed/reserved."""
    name = Path(raw or "").name
    if not name or name.startswith(".") or name in RESERVED_NAMES or not SAFE_FILENAME_RE.match(name):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"memo_{ts}.m4a"
    return name

BASE_DIR = Path(__file__).parent
INBOX_DIR = BASE_DIR / "inbox"
PROCESSED_FILE = BASE_DIR / ".processed_recordings"
OUTPUT_DIR = BASE_DIR / "output"
SPINE_PATH = BASE_DIR / "events.ndjson"
CONTACTS_PATH = BASE_DIR / "contacts.yaml"
VOICE_PROFILES_DIR = BASE_DIR / "voice_profiles"

INBOX_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Always-on capture layer extensions (additive — /upload still works unchanged).
from spine import EventSpine
import policy_resolver

SPINE = EventSpine(SPINE_PATH)
try:
    CONTACTS = policy_resolver.load_contacts(CONTACTS_PATH)
except Exception as e:
    print(f"[Ingest] contacts.yaml load failed ({e}); using empty contacts")
    CONTACTS = {"defaults": {"unknown_two_party": "B", "unknown_one_party": "A"}}

# Guard against accidental API billing — claude work must subprocess `claude` CLI.
if os.environ.get("ANTHROPIC_API_KEY"):
    print("[Ingest] WARNING: ANTHROPIC_API_KEY is set; the LLM path must NOT use this.")


class IngestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/upload":
            self._handle_upload(sidecar=None)
            return
        if self.path == "/ingest":
            sidecar = self._read_sidecar()
            self._handle_upload(sidecar=sidecar)
            return
        self.send_error(404)

    def _read_sidecar(self) -> dict:
        """Sidecar JSON describes the recording context: counterparties, outbound flag,
        calendar event snapshot frozen at record-start. Sent base64-encoded in X-Sidecar
        header to keep the multipart story simple for the iOS Shortcut."""
        raw = self.headers.get("X-Sidecar", "")
        if not raw:
            return {}
        try:
            return json.loads(base64.b64decode(raw))
        except Exception as e:
            print(f"[Ingest] sidecar parse failed: {e}")
            return {}

    def _handle_upload(self, sidecar):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "No data")
            return

        filename = _safe_filename(self.headers.get("X-Filename", ""))
        filepath = INBOX_DIR / filename

        # Resolve policy BEFORE persisting the body to disk. For consent=none we
        # never want the audio touching durable storage. Drain the request body
        # into a bounded memory buffer so we can still 403 cleanly.
        decision = None
        if sidecar:
            try:
                decision = policy_resolver.resolve(
                    counterparties=sidecar.get("counterparties", []),
                    outbound=bool(sidecar.get("outbound", False)),
                    contacts=CONTACTS,
                    chase_state=sidecar.get("chase_state", "FL"),
                )
            except policy_resolver.RecordingRefused as e:
                SPINE.append("capture.refused", {
                    "filename": filename,
                    "sidecar": sidecar,
                    "reason": str(e),
                })
                # Drain the body off the socket without persisting it.
                remaining = content_length
                while remaining > 0:
                    chunk = self.rfile.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "refused", "reason": str(e)}).encode())
                return

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

        event_id = SPINE.append("capture.started", {
            "filename": filename,
            "size_mb": round(size_mb, 1),
            "sidecar": sidecar or {},
            "policy": decision,
        })

        threading.Thread(target=process_file, args=(filepath, event_id, decision), daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "received",
            "filename": filename,
            "size_mb": round(size_mb, 1),
            "event_id": event_id,
            "policy": decision,
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


def process_file(filepath, capture_event_id=None, policy_decision=None):
    """Run Wonderstand pipeline on a single file. Emits spine events around the work."""
    filename = filepath.name
    basename = filepath.stem

    processed = set()
    if PROCESSED_FILE.exists():
        processed = set(PROCESSED_FILE.read_text().strip().split("\n"))
    if filename in processed:
        print(f"[Ingest] Already processed: {filename}")
        return

    output_json = OUTPUT_DIR / f"{basename}.json"
    output_notes = OUTPUT_DIR / f"{basename}_notes.md"

    SPINE.append("transcript.started", {
        "filename": filename,
        "capture_event_id": capture_event_id,
    })

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
            import fcntl as _fcntl
            with open(PROCESSED_FILE, "a") as f:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
                try:
                    f.write(filename + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
            print(f"[Ingest] Done: {filename} -> {output_notes.name}")
            SPINE.append("transcript.completed", {
                "filename": filename,
                "capture_event_id": capture_event_id,
                "output_notes": str(output_notes.name),
            })
        else:
            print(f"[Ingest] Failed: {filename}")
            stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
            print(stderr_tail)
            SPINE.append("error", {
                "where": "transcript",
                "filename": filename,
                "capture_event_id": capture_event_id,
                "stderr_tail": stderr_tail,
            })
    except subprocess.TimeoutExpired:
        print(f"[Ingest] Timeout: {filename} (>30 min)")
        SPINE.append("error", {
            "where": "transcript_timeout",
            "filename": filename,
            "capture_event_id": capture_event_id,
        })
    finally:
        # Tier B is a consent guarantee — audio MUST be shredded whether or not
        # transcription succeeded. If transcription failed, we lose the data, which
        # is the correct behavior given the consent constraint.
        if policy_decision and policy_decision.get("tier") == "B":
            try:
                if filepath.exists():
                    filepath.unlink()
                SPINE.append("audio.shredded", {
                    "filename": filename,
                    "capture_event_id": capture_event_id,
                    "reason": "tier_B_transcript_only",
                })
                print(f"[Ingest] Tier B: shredded audio for {filename}")
            except OSError as e:
                SPINE.append("error", {"where": "tier_b_shred", "filename": filename, "err": str(e)})


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

"""Append-only NDJSON event log. Single source of truth.

Thread-safe and cross-process safe via fcntl.flock. Every append is fsynced
to disk before returning, so a crash can lose at most an in-flight append.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


SCHEMA_VERSION = 1


class EventSpine:
    """Append-only NDJSON event log at <path>. Thread-safe (file-lock)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file so readers don't have to special-case missing file.
        if not self.path.exists():
            self.path.touch()
        # In-process lock — fcntl handles cross-process but is not reentrant
        # on the same fd across threads in CPython for all platforms.
        self._tlock = threading.Lock()

    # ----------------------- write path -----------------------

    def append(self, event_type: str, payload: dict) -> str:
        """Append an event. Returns event_id (uuid4 hex). fsynced before return."""
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        event_id = uuid.uuid4().hex
        row = {
            "event_id": event_id,
            "type": event_type,
            "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "payload": payload,
            "v": SCHEMA_VERSION,
        }
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        data = line.encode("utf-8")

        with self._tlock:
            # Open in append+binary so we get atomic O_APPEND writes on POSIX.
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                # Exclusive advisory lock for cross-process safety. flock is
                # released on fd close, but we release explicitly anyway.
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    # Single write() of <PIPE_BUF (4096) is atomic on POSIX,
                    # and O_APPEND guarantees the seek+write is atomic
                    # regardless. We loop for partial writes defensively.
                    view = memoryview(data)
                    while view:
                        n = os.write(fd, view)
                        if n <= 0:
                            raise OSError("write returned non-positive count")
                        view = view[n:]
                    os.fsync(fd)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return event_id

    # ----------------------- read path -----------------------

    def _iter_rows(self) -> Iterator[dict]:
        # Share-lock the file while we snapshot it, so we don't read a
        # partial line that another process is mid-writing.
        fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            try:
                with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as f:
                    for lineno, raw in enumerate(f, 1):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            yield json.loads(raw)
                        except json.JSONDecodeError:
                            # Fail-soft for I/O / corruption: skip bad row.
                            # (Programming errors would have raised at append.)
                            continue
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def read_all(self) -> Iterator[dict]:
        """Yield every event row in append order."""
        yield from self._iter_rows()

    def read_since(self, event_id: Optional[str]) -> Iterator[dict]:
        """Yield events after `event_id` (exclusive). None => from start."""
        if event_id is None:
            yield from self._iter_rows()
            return
        seen = False
        for row in self._iter_rows():
            if seen:
                yield row
            elif row.get("event_id") == event_id:
                seen = True
        # If event_id was never seen, we yield nothing — caller can detect
        # that by checking find(event_id) first if they care.

    def find(self, event_id: str) -> Optional[dict]:
        """Return the row with this event_id, or None."""
        for row in self._iter_rows():
            if row.get("event_id") == event_id:
                return row
        return None


def idempotency_key(event_hash: str, prompt_version: str, model_id: str) -> str:
    """Returns sha256 hex of the tuple. Used to cache claude -p results."""
    h = hashlib.sha256()
    # Length-prefix each field so ("a", "bc", "d") != ("ab", "c", "d").
    for part in (event_hash, prompt_version, model_id):
        b = part.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


# ----------------------- smoke test -----------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        spine_path = Path(td) / "spine.ndjson"
        spine = EventSpine(spine_path)

        # Append 3 events.
        id1 = spine.append("capture_started", {"session_id": "s1", "tier": "A"})
        id2 = spine.append("transcript_ready", {"session_id": "s1", "words": 42})
        id3 = spine.append("capture_ended", {"session_id": "s1", "duration_s": 137.5})

        assert id1 != id2 != id3, "event_ids must be unique"

        # Read back.
        rows = list(spine.read_all())
        assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
        assert [r["event_id"] for r in rows] == [id1, id2, id3], "order preserved"
        assert rows[0]["type"] == "capture_started"
        assert rows[1]["payload"]["words"] == 42
        assert all(r["v"] == 1 for r in rows), "schema version stamped"
        assert all(r["ts_utc"].endswith("Z") for r in rows), "iso8601 utc"

        # find()
        assert spine.find(id2) == rows[1]
        assert spine.find("deadbeef" * 4) is None

        # read_since()
        since = list(spine.read_since(id1))
        assert [r["event_id"] for r in since] == [id2, id3], "read_since exclusive"
        assert list(spine.read_since(None)) == rows, "None => from start"

        # idempotency_key determinism + sensitivity.
        k1 = idempotency_key("hash-abc", "v3", "claude-opus-4-7")
        k2 = idempotency_key("hash-abc", "v3", "claude-opus-4-7")
        k3 = idempotency_key("hash-abc", "v3", "claude-opus-4-6")
        k4 = idempotency_key("hash-abd", "v3", "claude-opus-4-7")
        assert k1 == k2, "deterministic on identical inputs"
        assert k1 != k3, "sensitive to model_id"
        assert k1 != k4, "sensitive to event_hash"
        assert len(k1) == 64 and all(c in "0123456789abcdef" for c in k1), "sha256 hex"

        # Length-prefixing guard: ("a","bc",...) vs ("ab","c",...) must differ.
        ka = idempotency_key("a", "bc", "x")
        kb = idempotency_key("ab", "c", "x")
        assert ka != kb, "length-prefixed: no boundary collision"

        print("OK")

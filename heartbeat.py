"""Heartbeat — daily digest of pipeline health. Not a pager."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


# --- Time helpers -----------------------------------------------------------

def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse an ISO8601 ts_utc into an aware datetime; return None on garbage."""
    if not ts:
        return None
    try:
        # Accept trailing Z
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


# --- Spine reading (duck-typed; we do not import spine.py) -----------------

def _iter_spine(spine_path: Path):
    """Yield event dicts from an NDJSON spine. Silently skip malformed rows.
    Takes a shared file lock during read so concurrent appenders don't tear
    a line mid-write (which would silently under-count the digest)."""
    if not spine_path.exists():
        return
    import fcntl as _fcntl
    with spine_path.open("r", encoding="utf-8") as fh:
        try:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_SH)
        except OSError:
            pass
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):
                    yield ev
        finally:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
            except OSError:
                pass


# --- Disk audio -------------------------------------------------------------

def _dir_audio_mb(d: Path) -> float:
    """Total size in MB of audio-ish files under d (recursive). 0 if missing."""
    if not d or not d.exists():
        return 0.0
    exts = {".m4a", ".wav", ".mp3", ".caf", ".opus", ".ogg", ".flac", ".aiff"}
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            if Path(f).suffix.lower() in exts:
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
    return round(total / (1024 * 1024), 3)


# --- Permission probes ------------------------------------------------------

_PERM_CACHE: dict = {}
_PERM_CACHE_TTL_SEC = 300  # 5 min — perms don't change often, and we don't want every digest read shelling out.


def _cached_perm(key: str, probe):
    now = time.time()
    rec = _PERM_CACHE.get(key)
    if rec and (now - rec[0]) < _PERM_CACHE_TTL_SEC:
        return rec[1]
    val = probe()
    _PERM_CACHE[key] = (now, val)
    return val


def check_screen_recording_permission() -> bool:
    """Return True if Mac has Screen Recording granted to this process.
    Result cached for 5 minutes to keep heartbeat digest reads cheap and to
    avoid races on shared probe paths."""
    def _probe() -> bool:
        if not shutil.which("screencapture"):
            return False
        # Per-call unique tempfile — no shared-path race between concurrent probes.
        with tempfile.NamedTemporaryFile(prefix="sctest_", suffix=".png", delete=False) as tf:
            tmp = Path(tf.name)
        try:
            tmp.unlink(missing_ok=True)  # screencapture wants to create the file
            proc = subprocess.run(
                ["screencapture", "-x", "-t", "png", str(tmp)],
                capture_output=True,
                timeout=10,
            )
            return bool(proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1024)
        except (subprocess.TimeoutExpired, OSError):
            return False
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return _cached_perm("screen", _probe)


def check_microphone_permission() -> Optional[bool]:
    """Return True/False if microphone access can be probed; None if no probe tool.
    Result cached for 5 minutes."""
    def _probe() -> Optional[bool]:
        if shutil.which("sox"):
            try:
                proc = subprocess.run(
                    ["sox", "-t", "coreaudio", "default", "-n", "trim", "0", "0.1"],
                    capture_output=True,
                    timeout=5,
                )
                return proc.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                return False
        if shutil.which("ffmpeg"):
            try:
                with tempfile.NamedTemporaryFile(prefix="mictest_", suffix=".wav", delete=False) as tf:
                    tmp = Path(tf.name)
                tmp.unlink(missing_ok=True)
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-f", "avfoundation", "-i", ":0",
                        "-t", "0.2", str(tmp),
                    ],
                    capture_output=True,
                    timeout=8,
                )
                ok = proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                if ok:
                    return True
                err = (proc.stderr or b"").decode("utf-8", errors="ignore").lower()
                if "permission" in err or "not authorized" in err or "denied" in err:
                    return False
                return False
            except (subprocess.TimeoutExpired, OSError):
                return False
        return None
    return _cached_perm("mic", _probe)


def _is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# --- Gap detection ----------------------------------------------------------

def _gaps_during_calendar_events(events: list[dict]) -> list[str]:
    """Calendar event titles for which no capture.started occurred within ±5min."""
    cal = []
    caps = []
    for ev in events:
        t = ev.get("type", "")
        ts = _parse_iso(ev.get("ts_utc", ""))
        if ts is None:
            continue
        if t == "calendar.event_observed":
            payload = ev.get("payload") or {}
            title = payload.get("title")
            if title:
                # Use event ts as anchor; payload.start optional.
                anchor = _parse_iso(payload.get("start", "")) or ts
                cal.append((title, anchor))
        elif t.startswith("capture") and "started" in t:
            caps.append(ts)
    gaps: list[str] = []
    window = timedelta(minutes=5)
    for title, anchor in cal:
        if not any(abs(c - anchor) <= window for c in caps):
            gaps.append(title)
    return gaps


# --- Pending profile count --------------------------------------------------

def _count_pending_profiles(voice_profiles_dir: Optional[Path]) -> int:
    if not voice_profiles_dir:
        return 0
    pending = voice_profiles_dir / "pending.json"
    if not pending.exists():
        return 0
    try:
        data = json.loads(pending.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return len(data)
        if isinstance(data, list):
            return len(data)
    except (OSError, json.JSONDecodeError):
        return 0
    return 0


# --- Public API -------------------------------------------------------------

def daily_digest(
    spine_path: Path,
    since_hours: int = 24,
    output: Optional[Path] = None,
    *,
    inbox_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    voice_profiles_dir: Optional[Path] = None,
) -> dict:
    """Walk spine over last since_hours and return a digest dict."""
    spine_path = Path(spine_path)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=since_hours)

    # Default disk dirs relative to spine if not provided
    spine_parent = spine_path.parent
    if voice_profiles_dir is None:
        # heuristic: sibling voice_profiles dir if present
        candidate = spine_parent / "voice_profiles"
        voice_profiles_dir = candidate if candidate.exists() else None

    base_for_inbox = (voice_profiles_dir.parent if voice_profiles_dir else spine_parent)
    if inbox_dir is None:
        inbox_dir = base_for_inbox / "inbox"
    if output_dir is None:
        output_dir = base_for_inbox / "output"

    in_window: list[dict] = []
    for ev in _iter_spine(spine_path):
        ts = _parse_iso(ev.get("ts_utc", ""))
        if ts is None:
            continue
        if ts >= cutoff:
            in_window.append(ev)

    captures = transcripts = claude_calls = errors = captures_refused = audio_shredded = 0
    for ev in in_window:
        t = (ev.get("type") or "").lower()
        if t == "capture.started":
            captures += 1
        elif t == "capture.refused":
            captures_refused += 1
        elif t.startswith("transcript"):
            transcripts += 1
        elif t.startswith("claude"):
            claude_calls += 1
        elif t == "audio.shredded":
            audio_shredded += 1
        elif t.startswith("error"):
            errors += 1

    gaps = _gaps_during_calendar_events(in_window)

    disk_mb = round(_dir_audio_mb(inbox_dir) + _dir_audio_mb(output_dir), 3)
    pending = _count_pending_profiles(voice_profiles_dir)

    mic_perm = check_microphone_permission()
    screen_perm = check_screen_recording_permission()
    ffmpeg_ok = _is_ffmpeg_available()

    digest = {
        "window_hours": since_hours,
        "captures": captures,
        "captures_refused": captures_refused,
        "audio_shredded": audio_shredded,
        "transcripts": transcripts,
        "claude_calls": claude_calls,
        "errors": errors,
        "gaps_during_calendar_events": gaps,
        "disk_audio_mb": disk_mb,
        "pending_profiles": pending,
        "permissions": {
            "screen_recording": bool(screen_perm),
            "microphone": bool(mic_perm) if mic_perm is not None else False,
            "ffmpeg": bool(ffmpeg_ok),
        },
        "_microphone_probe": (
            "unknown (no sox/ffmpeg avfoundation)" if mic_perm is None else
            ("granted" if mic_perm else "denied-or-failed")
        ),
        "generated_at": now.isoformat(),
    }

    if output is not None:
        _write_markdown(Path(output), digest)

    return digest


def _write_markdown(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    perms = d.get("permissions", {})
    gaps = d.get("gaps_during_calendar_events", [])
    lines = [
        f"# Wonderstand Heartbeat — {d.get('generated_at','')}",
        "",
        f"Window: last **{d.get('window_hours', 0)}h**",
        "",
        "## Counts",
        f"- captures: {d.get('captures', 0)}",
        f"- transcripts: {d.get('transcripts', 0)}",
        f"- claude_calls: {d.get('claude_calls', 0)}",
        f"- errors: {d.get('errors', 0)}",
        "",
        "## Disk",
        f"- audio_mb: {d.get('disk_audio_mb', 0)}",
        f"- pending_profiles: {d.get('pending_profiles', 0)}",
        "",
        "## Permissions",
        f"- screen_recording: {perms.get('screen_recording')}",
        f"- microphone: {perms.get('microphone')} (probe: {d.get('_microphone_probe','')})",
        f"- ffmpeg: {perms.get('ffmpeg')}",
        "",
        "## Calendar gaps (events with no capture within ±5min)",
    ]
    if gaps:
        for g in gaps:
            lines.append(f"- {g}")
    else:
        lines.append("- (none)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Self-test --------------------------------------------------------------

def _selftest() -> bool:
    tmpdir = Path(tempfile.mkdtemp(prefix="hb_selftest_"))
    spine = tmpdir / "spine.ndjson"
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(minutes=10)).isoformat()
    cal_anchor = (now - timedelta(minutes=30)).isoformat()
    old = (now - timedelta(hours=48)).isoformat()

    events = [
        # In-window: one of each tracked category
        {"event_id": "1", "type": "capture.started", "ts_utc": recent,
         "payload": {}, "v": 1},
        {"event_id": "2", "type": "transcript.completed", "ts_utc": recent,
         "payload": {}, "v": 1},
        {"event_id": "3", "type": "claude.invoked", "ts_utc": recent,
         "payload": {}, "v": 1},
        {"event_id": "4", "type": "error.something", "ts_utc": recent,
         "payload": {}, "v": 1},
        # Calendar event with NO matching capture nearby → should be a gap
        {"event_id": "5", "type": "calendar.event_observed", "ts_utc": cal_anchor,
         "payload": {"title": "1:1 with Elie", "start": cal_anchor}, "v": 1},
        # Out-of-window event — must NOT be counted
        {"event_id": "6", "type": "capture.started", "ts_utc": old,
         "payload": {}, "v": 1},
    ]
    with spine.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    inbox = tmpdir / "inbox"
    out = tmpdir / "output"
    inbox.mkdir()
    out.mkdir()
    # Drop a fake audio file to verify disk accounting
    (inbox / "fake.m4a").write_bytes(b"\x00" * (1024 * 1024))  # 1 MB

    md_path = tmpdir / "digest.md"
    digest = daily_digest(
        spine,
        since_hours=24,
        output=md_path,
        inbox_dir=inbox,
        output_dir=out,
        voice_profiles_dir=tmpdir / "voice_profiles",
    )

    ok = True
    checks = [
        ("captures", digest["captures"], 1),
        ("transcripts", digest["transcripts"], 1),
        ("claude_calls", digest["claude_calls"], 1),
        ("errors", digest["errors"], 1),
        ("window_hours", digest["window_hours"], 24),
    ]
    for name, got, want in checks:
        if got != want:
            print(f"FAIL: {name} got {got} want {want}")
            ok = False

    if "1:1 with Elie" not in digest["gaps_during_calendar_events"]:
        print(f"FAIL: gap missing, got {digest['gaps_during_calendar_events']}")
        ok = False

    if digest["disk_audio_mb"] < 0.9 or digest["disk_audio_mb"] > 1.2:
        print(f"FAIL: disk_audio_mb unexpected: {digest['disk_audio_mb']}")
        ok = False

    if not md_path.exists():
        print("FAIL: markdown not written")
        ok = False

    print(f"permissions: {digest['permissions']}")
    print(f"mic probe: {digest['_microphone_probe']}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if _selftest():
        print("OK")
    else:
        raise SystemExit(1)

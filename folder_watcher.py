"""folder_watcher.py — Atomic file watch + opus re-encode.

Poll-based watcher (stdlib only). Calls on_new_file(path) exactly once per
atomically-arrived file. Tracks seen files in inbox_dir/.seen so restarts
don't re-fire. Includes opus_reencode() and is_ffmpeg_available() helpers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Seen-set persistence
# ---------------------------------------------------------------------------

def _seen_path(inbox_dir: Path) -> Path:
    return inbox_dir / ".seen"


def _load_seen(inbox_dir: Path) -> set[str]:
    p = _seen_path(inbox_dir)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return set(str(x) for x in data)
    except (json.JSONDecodeError, OSError):
        pass
    return set()


def _save_seen(inbox_dir: Path, seen: set[str]) -> None:
    p = _seen_path(inbox_dir)
    tmp = p.with_suffix(".seen.tmp")
    try:
        tmp.write_text(json.dumps(sorted(seen)))
        os.replace(tmp, p)
    except OSError:
        # best-effort; don't crash the watcher on a persistence hiccup
        pass


# ---------------------------------------------------------------------------
# watch()
# ---------------------------------------------------------------------------

def watch(
    inbox_dir: Path,
    on_new_file: Callable[[Path], None],
    extensions: tuple[str, ...] = (".m4a", ".wav", ".mp3", ".caf"),
    poll_interval: float = 2.0,
    stable_for: float = 3.0,
    stop_event=None,
) -> None:
    """Blocking poll-based watcher; fires on_new_file exactly once per atomic arrival.

    If stop_event (threading.Event) is provided, the loop checks it each tick
    and returns cleanly when set — lets a thread shut the watcher down without
    killing the process.
    """
    inbox_dir = Path(inbox_dir)
    inbox_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = _load_seen(inbox_dir)
    candidates: dict[str, tuple[int, float, float]] = {}
    ext_lower = tuple(e.lower() for e in extensions)

    while True:
        if stop_event is not None and stop_event.is_set():
            return
        now = time.time()
        try:
            entries = list(os.scandir(inbox_dir))
        except OSError:
            entries = []

        present_names: set[str] = set()
        for entry in entries:
            try:
                if not entry.is_file():
                    continue
            except OSError:
                continue
            name = entry.name
            if name.startswith("."):
                continue
            if not name.lower().endswith(ext_lower):
                continue
            if name in seen:
                continue
            present_names.add(name)
            try:
                st = entry.stat()
            except OSError:
                continue
            size = st.st_size
            mtime = st.st_mtime

            prev = candidates.get(name)
            if prev is None or prev[0] != size or prev[1] != mtime:
                # changed (or new) — reset stability clock
                candidates[name] = (size, mtime, now)
            else:
                # unchanged since last sample
                stable_since = prev[2]
                if (now - stable_since) >= stable_for:
                    # fire
                    path = Path(entry.path)
                    try:
                        on_new_file(path)
                    except Exception:
                        # caller's problem; record as seen anyway so we don't loop
                        pass
                    seen.add(name)
                    _save_seen(inbox_dir, seen)
                    candidates.pop(name, None)

        # forget candidates whose files vanished
        for gone in list(candidates.keys()):
            if gone not in present_names:
                candidates.pop(gone, None)

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def is_ffmpeg_available() -> bool:
    """Return True if `ffmpeg` is on PATH and runs."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def opus_reencode(src: Path, dst: Path, bitrate_kbps: int = 24) -> Path:
    """Re-encode src → dst as mono 16 kHz voice-tuned opus via ffmpeg. Returns dst."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"src not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # encode to a temp path then move into place so partial files never appear.
    # Keep the original suffix so ffmpeg picks the right muxer; just prefix.
    tmp_dst = dst.with_name(f".{dst.name}.part{dst.suffix}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "libopus",
        "-b:a", f"{bitrate_kbps}k",
        "-application", "voip",
        str(tmp_dst),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH")
    except subprocess.TimeoutExpired as e:
        try:
            tmp_dst.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg timed out after {e.timeout}s")

    if proc.returncode != 0:
        try:
            tmp_dst.unlink(missing_ok=True)
        except OSError:
            pass
        stderr_tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-200:]
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr_tail}")

    # success — move into place
    os.replace(tmp_dst, dst)

    # "Replaces src on success only if dst != src" — if dst is at a different
    # path than src, the contract just expects dst to exist; we don't delete src
    # here because callers may want to archive it. If dst == src we already
    # replaced via the temp file move.
    return dst


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import signal
    import sys
    import threading

    test_dir = Path("/tmp/wstest_watch")
    # clean slate
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    fired: list[Path] = []
    fired_lock = threading.Lock()

    def on_new(path: Path) -> None:
        with fired_lock:
            fired.append(path)

    # Background dropper: after 1s, write file A; small gap, write file B.
    def dropper():
        time.sleep(1.0)
        # use subprocess as specified in the contract
        subprocess.run(
            ["sh", "-c", "printf 'aaaaaaaaaaaaaaaaaaaa' > /tmp/wstest_watch/a.m4a"],
            check=False,
        )
        time.sleep(0.3)
        subprocess.run(
            ["sh", "-c", "printf 'bbbbbbbbbbbbbbbbbbbb' > /tmp/wstest_watch/b.wav"],
            check=False,
        )

    t = threading.Thread(target=dropper, daemon=True)
    t.start()

    # Run watch() for ~8s with a signal-based timeout.
    class _Stop(Exception):
        pass

    def _handler(signum, frame):
        raise _Stop()

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(8)

    errors: list[str] = []
    try:
        # short stable_for so the 8s budget is enough
        watch(
            test_dir,
            on_new,
            extensions=(".m4a", ".wav"),
            poll_interval=0.5,
            stable_for=1.5,
        )
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001
        errors.append(f"watch raised: {e!r}")
    finally:
        signal.alarm(0)

    # also test idempotency across restart: run watch a second time briefly,
    # should NOT re-fire the already-seen files.
    refire: list[Path] = []

    def on_new_2(path: Path) -> None:
        refire.append(path)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(3)
    try:
        watch(
            test_dir,
            on_new_2,
            extensions=(".m4a", ".wav"),
            poll_interval=0.5,
            stable_for=1.0,
        )
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001
        errors.append(f"watch(restart) raised: {e!r}")
    finally:
        signal.alarm(0)

    names = sorted(p.name for p in fired)
    expected = ["a.m4a", "b.wav"]

    print(f"fired: {names}")
    print(f"refire after restart: {[p.name for p in refire]}")

    ok = True
    if names != expected:
        print(f"FAIL: expected {expected}, got {names}")
        ok = False
    else:
        print("PASS: both files fired exactly once")

    if refire:
        print(f"FAIL: restart re-fired {refire}")
        ok = False
    else:
        print("PASS: restart did not re-fire seen files")

    ffmpeg_ok = is_ffmpeg_available()
    print(f"ffmpeg available: {ffmpeg_ok}")
    # is_ffmpeg_available must be callable + cached
    if is_ffmpeg_available() != ffmpeg_ok:
        print("FAIL: ffmpeg cache inconsistent")
        ok = False
    else:
        print("PASS: is_ffmpeg_available cached")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        ok = False

    print("OK" if ok else "NOT OK")
    sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""
Wonderstand UI — browse recordings, group voice samples by person, search
summaries & transcripts. Localhost-only (security baseline).

Run:
    cd /Users/cf/Projects/wonderstand
    venv/bin/python ui/server.py            # -> http://127.0.0.1:8770

Data sources (read-only unless you edit in the UI):
    output/**/*.json          transcript segments (speaker, timestamp, text)
    output/**/*_notes.md      auto meeting notes (optional)
    voice_profiles/profiles.json   enrolled voice profiles
    ui/meta.json              editable metadata this UI owns (location,
                              summary, title, per-session speaker renames)
"""
import json
import os
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory, abort

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
PROFILES_FILE = ROOT / "voice_profiles" / "profiles.json"
META_FILE = ROOT / "ui" / "meta.json"
CACHE_DIR = ROOT / "ui" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Where source audio might live, matched by basename when a session references it.
VOICE_MEMOS_DIR = Path.home() / "Library" / "Group Containers" / \
    "group.com.apple.VoiceMemos.shared" / "Recordings"
AUDIO_SEARCH_DIRS = [VOICE_MEMOS_DIR, ROOT / "inbox", OUTPUT_DIR]

app = Flask(__name__, static_folder=None)
_meta_lock = threading.Lock()

# ---------------------------------------------------------------- meta store
def load_meta() -> dict:
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"sessions": {}, "people": {}}


def save_meta(meta: dict) -> None:
    with _meta_lock:
        tmp = META_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(META_FILE)


def load_profiles() -> dict:
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_profiles(profiles: dict) -> None:
    tmp = PROFILES_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profiles, indent=2))
    tmp.replace(PROFILES_FILE)


# ---------------------------------------------------------------- indexing
DATE_RE = re.compile(r"(20\d{2})[-_ ]?(\d{2})[-_ ]?(\d{2})[ _-]?(\d{2})?(\d{2})?(\d{2})?")


def parse_date_from_name(name: str, fallback_mtime: float) -> str:
    m = DATE_RE.search(name)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        hh = m.group(4) or "00"
        mm = m.group(5) or "00"
        ss = m.group(6) or "00"
        try:
            return datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss)).isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_mtime).isoformat()


def read_segments(data) -> list:
    segs = data.get("segments") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    out = []
    for s in segs or []:
        if not isinstance(s, dict):
            continue
        if s.get("type") and s.get("type") != "audio":
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = s.get("timestamp", s.get("start", 0.0))
        dur = s.get("duration")
        end = s.get("end", (start + dur) if dur is not None else start)
        out.append({
            "start": float(start or 0.0),
            "end": float(end or start or 0.0),
            "speaker": s.get("speaker") or s.get("speaker_label") or "UNKNOWN",
            "text": text,
        })
    return out


def session_id_for(path: Path) -> str:
    return str(path.relative_to(OUTPUT_DIR)).replace(os.sep, "~")


def path_for_session_id(sid: str) -> Path:
    rel = sid.replace("~", os.sep)
    p = (OUTPUT_DIR / rel).resolve()
    if OUTPUT_DIR.resolve() not in p.parents and p != OUTPUT_DIR.resolve():
        abort(400)
    return p


def find_audio(basename: str):
    if not basename:
        return None
    stem = Path(basename).stem
    for d in AUDIO_SEARCH_DIRS:
        if not d.exists():
            continue
        for ext in (".qta", ".m4a", ".wav", ".mp3", ".mov", ".mp4"):
            cand = d / (stem + ext)
            if cand.exists():
                return str(cand)
    return None


def build_index() -> dict:
    meta = load_meta()
    profiles = load_profiles()
    known_names = set(profiles.keys())
    sessions = []

    for jf in sorted(OUTPUT_DIR.rglob("*.json")):
        if jf.name.endswith(".tmp"):
            continue
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        segs = read_segments(data)
        if not segs:
            continue

        sid = session_id_for(jf)
        smeta = meta.get("sessions", {}).get(sid, {})
        rename = smeta.get("speaker_names", {})

        # distinct speakers, renamed via per-session override
        raw_speakers = []
        for s in segs:
            if s["speaker"] not in raw_speakers:
                raw_speakers.append(s["speaker"])
        participants = []
        for rs in raw_speakers:
            disp = rename.get(rs, rs)
            participants.append({
                "raw": rs,
                "name": disp,
                "known": disp in known_names,
                "seconds": round(sum(s["end"] - s["start"] for s in segs if s["speaker"] == rs), 1),
            })

        duration = round(max((s["end"] for s in segs), default=0.0), 1)
        full_text = " ".join(s["text"] for s in segs)

        notes_path = jf.with_name(jf.stem + "_notes.md")
        audio = smeta.get("audio") or find_audio(smeta.get("source") or jf.name)

        default_title = smeta.get("title") or jf.stem.replace("_diar", "").replace("_", " ")
        date = smeta.get("date") or parse_date_from_name(jf.name, jf.stat().st_mtime)

        sessions.append({
            "id": sid,
            "title": default_title,
            "date": date,
            "duration": duration,
            "participants": participants,
            "location": smeta.get("location", ""),
            "location_confidence": smeta.get("location_confidence", ""),
            "summary": smeta.get("summary", ""),
            "tags": smeta.get("tags", []),
            "has_notes": notes_path.exists(),
            "has_audio": bool(audio),
            "text_len": len(full_text),
            "_search": (default_title + " " + smeta.get("summary", "") + " " +
                        smeta.get("location", "") + " " +
                        " ".join(p["name"] for p in participants) + " " + full_text).lower(),
        })

    sessions.sort(key=lambda s: s["date"], reverse=True)

    # People: enrolled profiles + appearance counts across sessions
    appearances = {}
    for s in sessions:
        for p in s["participants"]:
            if p["known"]:
                appearances.setdefault(p["name"], {"sessions": 0, "seconds": 0.0})
                appearances[p["name"]]["sessions"] += 1
                appearances[p["name"]]["seconds"] += p["seconds"]

    people = []
    for name, data in profiles.items():
        ap = appearances.get(name, {"sessions": 0, "seconds": 0.0})
        people.append({
            "name": name,
            "enrolled_seconds": round(data.get("total_seconds", 0), 0),
            "source_files": data.get("source_files", []),
            "session_count": ap["sessions"],
            "heard_seconds": round(ap["seconds"], 0),
        })
    people.sort(key=lambda p: (-p["session_count"], p["name"]))

    return {"people": people, "sessions": sessions}


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(ROOT / "ui", "index.html")


@app.get("/api/state")
def api_state():
    return jsonify(build_index())


@app.get("/api/session/<sid>")
def api_session(sid):
    p = path_for_session_id(sid)
    if not p.exists():
        abort(404)
    data = json.loads(p.read_text())
    segs = read_segments(data)
    meta = load_meta()
    smeta = meta.get("sessions", {}).get(sid, {})
    rename = smeta.get("speaker_names", {})
    for s in segs:
        s["display_speaker"] = rename.get(s["speaker"], s["speaker"])
    notes_path = p.with_name(p.stem + "_notes.md")
    notes = notes_path.read_text() if notes_path.exists() else ""
    return jsonify({"id": sid, "segments": segs, "notes": notes, "meta": smeta})


@app.post("/api/session/<sid>")
def api_session_update(sid):
    body = request.get_json(force=True) or {}
    meta = load_meta()
    meta.setdefault("sessions", {}).setdefault(sid, {})
    for k in ("title", "location", "location_confidence", "summary", "tags",
              "speaker_names", "audio", "source", "date"):
        if k in body:
            meta["sessions"][sid][k] = body[k]
    save_meta(meta)
    return jsonify({"ok": True, "session": meta["sessions"][sid]})


@app.post("/api/person/rename")
def api_person_rename():
    body = request.get_json(force=True) or {}
    old, new = body.get("old"), (body.get("new") or "").strip()
    if not old or not new:
        abort(400)
    profiles = load_profiles()
    if old in profiles and new != old:
        profiles[new] = profiles.pop(old)
        save_profiles(profiles)
    # cascade rename into per-session speaker overrides
    meta = load_meta()
    for sid, sm in meta.get("sessions", {}).items():
        sn = sm.get("speaker_names", {})
        changed = False
        for raw, nm in list(sn.items()):
            if nm == old:
                sn[raw] = new
                changed = True
        if changed:
            sm["speaker_names"] = sn
    save_meta(meta)
    return jsonify({"ok": True})


@app.get("/api/audio/<sid>")
def api_audio(sid):
    """Serve session audio as mp3 (transcoded + cached), with range support."""
    meta = load_meta()
    smeta = meta.get("sessions", {}).get(sid, {})
    p = path_for_session_id(sid)
    src = smeta.get("audio") or find_audio(smeta.get("source") or (p.name if p.exists() else ""))
    if not src or not os.path.exists(src):
        abort(404)
    cache = CACHE_DIR / (sid + ".mp3")
    if not cache.exists() or os.path.getmtime(cache) < os.path.getmtime(src):
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-b:a", "96k", str(cache)],
            check=True, capture_output=True,
        )
    return send_file(cache, mimetype="audio/mpeg", conditional=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")  # localhost only
    a = ap.parse_args()
    print(f"Wonderstand UI -> http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)

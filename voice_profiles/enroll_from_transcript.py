#!/usr/bin/env python3
"""
Enroll a named speaker using the diarization ALREADY baked into a transcript
JSON, so we never re-run pyannote. Pulls the target speaker's segments from the
transcript, slices those exact time ranges out of the source audio, computes a
wespeaker embedding, and merges it into voice_profiles/profiles.json.

Usage:
    python enroll_from_transcript.py \
        --transcript ../output/today-0721/1437_5min_diar.json \
        --audio "/path/to/recording.qta" \
        --speaker SPEAKER_00 --name Shaylinn

    # inspect distinct speaker labels + how much audio each has
    python enroll_from_transcript.py --transcript <json> --stats
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enroll  # get_embedding_model, compute_embedding, load/save_profiles, extract_audio

SAMPLE_RATE = 16000


def load_segments(transcript_path):
    with open(transcript_path) as f:
        data = json.load(f)
    segs = data.get("segments", data if isinstance(data, list) else [])
    out = []
    for s in segs:
        spk = s.get("speaker") or s.get("speaker_label") or "UNKNOWN"
        start = s.get("timestamp", s.get("start", 0.0))
        dur = s.get("duration")
        end = s.get("end", (start + dur) if dur is not None else start)
        out.append((float(start), float(end), spk, s.get("text", "")))
    return out


def stats(transcript_path):
    segs = load_segments(transcript_path)
    by = {}
    for start, end, spk, _ in segs:
        by.setdefault(spk, [0.0, 0])
        by[spk][0] += max(0.0, end - start)
        by[spk][1] += 1
    print(f"Speakers in {os.path.basename(transcript_path)}:")
    for spk, (secs, n) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"  {spk:12s}  {secs:6.1f}s  {n} segments")


def enroll(transcript_path, audio_path, raw_speaker, name, max_secs=60.0):
    segs = load_segments(transcript_path)
    target = [(s, e) for s, e, spk, _ in segs if spk == raw_speaker]
    if not target:
        labels = sorted({spk for _, _, spk, _ in segs})
        raise SystemExit(f"No segments for '{raw_speaker}'. Present: {labels}")

    print(f"[Enroll] Extracting full audio from {audio_path} ...")
    import importlib
    e = importlib.import_module("enroll")
    audio = e.extract_audio(audio_path)

    chunks, total = [], 0.0
    for start, end in target:
        if total >= max_secs:
            break
        s = int(start * SAMPLE_RATE)
        en = int(end * SAMPLE_RATE)
        if 0 <= s < en <= len(audio):
            chunks.append(audio[s:en])
            total += (end - start)
    if not chunks:
        raise SystemExit("No usable audio slices (timestamps out of range).")

    combined = np.concatenate(chunks)
    print(f"[Enroll] {name}: {total:.1f}s across {len(chunks)} segments -> embedding ...")
    inference = e.get_embedding_model()
    emb = np.array(e.compute_embedding(inference, combined))

    profiles = e.load_profiles()
    src = os.path.basename(audio_path)
    if name in profiles:
        old = np.array(profiles[name]["embedding"])
        avg = (old + emb) / 2
        avg = avg / np.linalg.norm(avg)
        profiles[name]["embedding"] = avg.tolist()
        profiles[name].setdefault("source_files", []).append(src)
        profiles[name]["total_seconds"] = profiles[name].get("total_seconds", 0) + total
        print(f"[Enroll] Merged into existing '{name}' profile.")
    else:
        emb = emb / np.linalg.norm(emb)
        profiles[name] = {
            "embedding": emb.tolist(),
            "source_files": [src],
            "total_seconds": total,
        }
        print(f"[Enroll] Created new profile '{name}'.")
    e.save_profiles(profiles)
    print(f"[Enroll] Saved. '{name}' now {profiles[name]['total_seconds']:.0f}s "
          f"from {len(profiles[name]['source_files'])} file(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--speaker")
    ap.add_argument("--name")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--max-secs", type=float, default=60.0)
    a = ap.parse_args()
    if a.stats:
        stats(a.transcript)
    else:
        if not (a.audio and a.speaker and a.name):
            raise SystemExit("Need --audio, --speaker, --name (or --stats)")
        enroll(a.transcript, a.audio, a.speaker, a.name, a.max_secs)

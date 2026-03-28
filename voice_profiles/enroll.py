#!/usr/bin/env python3
"""
Voice Profile Enrollment — Extract speaker embeddings and save as profiles.

Usage:
    # Enroll from a recording with known speakers (uses diarization)
    python enroll.py recording.m4a --speakers "SPEAKER_00=Chase,SPEAKER_01=Mom"

    # Enroll from a solo recording (single speaker)
    python enroll.py solo_memo.m4a --name Chase

    # List saved profiles
    python enroll.py --list
"""
import argparse
import json
import os
import sys
import tempfile
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROFILES_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(PROFILES_DIR, "profiles.json")
SAMPLE_RATE = 16000


def extract_audio(path):
    """Extract audio as float32 numpy array at 16kHz mono."""
    import av
    container = av.open(path)
    audio_stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

    samples = []
    for frame in container.decode(audio_stream):
        for r in resampler.resample(frame):
            data = r.to_ndarray().flatten().astype(np.float32) / 32768.0
            samples.append(data)
    container.close()
    return np.concatenate(samples) if samples else np.array([], dtype=np.float32)


def get_embedding_model():
    """Load the speaker embedding model."""
    from pyannote.audio import Model, Inference
    hf_token = os.environ.get("HF_TOKEN")
    model = Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=hf_token
    )
    inference = Inference(model, window="whole")
    return inference


def compute_embedding(inference, audio_samples):
    """Compute speaker embedding from audio samples."""
    # Write to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_samples * 32768).astype(np.int16).tobytes())
        embedding = inference(tmp.name)
        os.unlink(tmp.name)
    return embedding


def load_profiles():
    """Load saved profiles."""
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {}


def save_profiles(profiles):
    """Save profiles to disk."""
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


def enroll_solo(audio_path, name):
    """Enroll a single speaker from a solo recording."""
    print(f"[Enroll] Extracting audio from {audio_path}...")
    audio = extract_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"[Enroll] Got {duration:.1f}s of audio")

    # Use first 60s max for embedding (more than enough)
    max_samples = 60 * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    print("[Enroll] Computing speaker embedding...")
    inference = get_embedding_model()
    embedding = compute_embedding(inference, audio)

    profiles = load_profiles()
    profiles[name] = {
        "embedding": embedding.tolist(),
        "source_files": [os.path.basename(audio_path)],
        "total_seconds": min(duration, 60),
    }
    save_profiles(profiles)
    print(f"[Enroll] Saved profile for '{name}' ({min(duration, 60):.0f}s of audio)")


def enroll_diarized(audio_path, speaker_map):
    """Enroll speakers from a multi-speaker recording using diarization."""
    from diarization import SpeakerDiarizer

    print(f"[Enroll] Running diarization on {audio_path}...")
    diarizer = SpeakerDiarizer(num_speakers=len(speaker_map))
    diarizer.process(audio_path)

    audio = diarizer._full_audio
    inference = get_embedding_model()
    profiles = load_profiles()

    for raw_label, name in speaker_map.items():
        # Collect audio segments for this speaker
        speaker_audio = []
        total_secs = 0
        for start, end, spk in diarizer.segments:
            if spk == raw_label and total_secs < 60:
                s = int(start * SAMPLE_RATE)
                e = int(end * SAMPLE_RATE)
                chunk = audio[s:e]
                speaker_audio.append(chunk)
                total_secs += (end - start)

        if not speaker_audio:
            print(f"[Enroll] No audio found for {raw_label} ({name}), skipping")
            continue

        combined = np.concatenate(speaker_audio)
        print(f"[Enroll] Computing embedding for '{name}' ({total_secs:.1f}s)...")
        embedding = compute_embedding(inference, combined)

        if name in profiles:
            # Average with existing embedding
            old = np.array(profiles[name]["embedding"])
            new = np.array(embedding)
            averaged = (old + new) / 2
            averaged = averaged / np.linalg.norm(averaged)
            profiles[name]["embedding"] = averaged.tolist()
            profiles[name]["source_files"].append(os.path.basename(audio_path))
            profiles[name]["total_seconds"] += total_secs
        else:
            profiles[name] = {
                "embedding": embedding.tolist(),
                "source_files": [os.path.basename(audio_path)],
                "total_seconds": total_secs,
            }

        print(f"[Enroll] Saved profile for '{name}'")

    save_profiles(profiles)


def identify_speaker(embedding, threshold=0.5):
    """Match an embedding against saved profiles. Returns (name, score) or None."""
    profiles = load_profiles()
    if not profiles:
        return None

    best_name = None
    best_score = -1

    emb = np.array(embedding)
    emb = emb / np.linalg.norm(emb)

    for name, data in profiles.items():
        stored = np.array(data["embedding"])
        stored = stored / np.linalg.norm(stored)
        score = float(np.dot(emb, stored))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return None


def list_profiles():
    """Print saved profiles."""
    profiles = load_profiles()
    if not profiles:
        print("No voice profiles saved.")
        return
    print(f"\nSaved voice profiles ({len(profiles)}):")
    print("-" * 40)
    for name, data in profiles.items():
        sources = ", ".join(data["source_files"])
        print(f"  {name}: {data['total_seconds']:.0f}s from {sources}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice profile enrollment")
    parser.add_argument("source", nargs="?", help="Audio/video file")
    parser.add_argument("--name", help="Speaker name (for solo recordings)")
    parser.add_argument("--speakers", help="Speaker map: 'SPEAKER_00=Chase,SPEAKER_01=Mom'")
    parser.add_argument("--list", action="store_true", help="List saved profiles")
    args = parser.parse_args()

    if args.list:
        list_profiles()
    elif args.source and args.name:
        enroll_solo(args.source, args.name)
    elif args.source and args.speakers:
        speaker_map = {}
        for pair in args.speakers.split(","):
            k, v = pair.strip().split("=", 1)
            speaker_map[k.strip()] = v.strip()
        enroll_diarized(args.source, speaker_map)
    else:
        parser.print_help()

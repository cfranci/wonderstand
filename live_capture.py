#!/usr/bin/env python3
"""
Live Meeting Capture — Records audio in rolling chunks, transcribes with
Whisper, extracts meeting intelligence, and creates a Notion page with notes.

Usage:
    python live_capture.py --duration 60
    python live_capture.py --audio-device 4 --meeting-notes
    python live_capture.py --duration 300 --meeting-notes

Press Ctrl+C to stop recording and generate final notes.

NOTE: Screen capture requires Screen Recording permission for Terminal.
      Run from Terminal.app and grant permission in System Settings >
      Privacy & Security > Screen Recording. Without it, audio-only mode
      is used automatically.
"""

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime


# Chunk config
CHUNK_DURATION = 30   # seconds per audio chunk sent to Whisper
CHUNK_OVERLAP = 2     # overlap between chunks


class LiveCapture:
    """Records audio live, transcribes in real-time, extracts meeting notes."""

    def __init__(self, audio_device="0", chunk_duration=CHUNK_DURATION,
                 output_dir=None, meeting_notes=False, notes_path=None,
                 speakers=None, max_duration=None, notion_sync=False):
        self.audio_device = audio_device
        self.chunk_duration = chunk_duration
        self.output_dir = output_dir or "output/live_session"
        self.meeting_notes = meeting_notes
        self.notes_path = notes_path or "output/live_meeting_notes.md"
        self.speakers = speakers or {}
        self.max_duration = max_duration
        self.notion_sync = notion_sync

        self._stop = threading.Event()
        self._recording_proc = None
        self._chunk_index = 0

        # Pipeline components (loaded lazily)
        self._whisper_model = None
        self._llm_model = None
        self._llm_tokenizer = None
        self._meeting_intel = None
        self._notion = None

        # Accumulated results
        self._all_segments = []
        self._prev_text = ""

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_models(self):
        """Load Whisper and LLM models."""
        from lightning_whisper_mlx import LightningWhisperMLX
        from config import WHISPER_MODEL, SUMMARY_LLM_MODEL

        print("[Live] Loading Whisper model...")
        self._whisper_model = LightningWhisperMLX(
            model=WHISPER_MODEL, batch_size=12, quant=None
        )
        print("[Live] Whisper loaded.")

        if self.meeting_notes:
            from mlx_lm import load as load_lm
            from meeting_intelligence import MeetingIntelligence

            print(f"[Live] Loading summary LLM: {SUMMARY_LLM_MODEL}")
            self._llm_model, self._llm_tokenizer = load_lm(SUMMARY_LLM_MODEL)
            print("[Live] LLM loaded.")

            self._meeting_intel = MeetingIntelligence(
                llm_model=self._llm_model,
                llm_tokenizer=self._llm_tokenizer,
                output_path=self.notes_path,
            )

    def _record_audio_chunk(self):
        """Record one audio chunk using ffmpeg, return path to WAV file."""
        chunk_path = os.path.join(
            self.output_dir, f"chunk_{self._chunk_index:04d}.wav"
        )
        duration = self.chunk_duration + CHUNK_OVERLAP

        cmd = [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-i", f":{self.audio_device}",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-t", str(duration),
            chunk_path,
        ]

        try:
            self._recording_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._recording_proc.wait()
            self._recording_proc = None

            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000:
                self._chunk_index += 1
                return chunk_path
        except Exception as e:
            print(f"[Live] Recording error: {e}")
        return None

    def _transcribe(self, wav_path):
        """Transcribe a WAV file with Whisper."""
        result = self._whisper_model.transcribe(wav_path)
        text = result.get("text", "").strip()
        return self._deduplicate(text)

    def _deduplicate(self, text):
        """Remove overlapping words between consecutive chunks."""
        if not self._prev_text or not text:
            self._prev_text = text
            return text

        prev_words = self._prev_text.split()
        curr_words = text.split()
        max_check = min(len(prev_words), len(curr_words), max(3, len(curr_words) // 2))

        best = 0
        for n in range(1, max_check + 1):
            if ([w.lower().strip(".,!?;:") for w in prev_words[-n:]] ==
                    [w.lower().strip(".,!?;:") for w in curr_words[:n]]):
                best = n

        self._prev_text = text
        return " ".join(curr_words[best:]) if best else text

    def _process_chunk(self, wav_path, chunk_num, elapsed):
        """Transcribe and feed into meeting intelligence."""
        text = self._transcribe(wav_path)
        if not text or not text.strip():
            print(f"  [Chunk {chunk_num}] (silence)")
            return

        timestamp = chunk_num * self.chunk_duration
        segment = {
            "timestamp": float(timestamp),
            "end_timestamp": float(timestamp + self.chunk_duration),
            "audio": {
                "segments": [{"time": float(timestamp), "text": text, "speaker": ""}],
                "full_text": text,
            },
            "video": {"segments": []},
            "summary": text[:200],
        }
        self._all_segments.append(segment)

        # Print live
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        print(f"  [{mins}:{secs:02d}] {text[:120]}")

        # Feed meeting intelligence
        if self._meeting_intel:
            self._meeting_intel.process_segment(segment, realtime=True)

            # Update Notion sync file when we have new data
            if self._notion and self._meeting_intel._segment_count > 0:
                data = self._meeting_intel.get_summary_data()
                if data.get("action_items") or data.get("decisions") or data.get("topics"):
                    self._notion.update_notes(data)

    def run(self):
        """Main capture loop."""
        now = datetime.now()
        session_name = now.strftime("Live Meeting — %b %d, %Y %I:%M %p")

        print(f"\n{'='*60}")
        print(f"  🎙️  LIVE MEETING CAPTURE")
        print(f"{'='*60}")
        print(f"  Session:       {session_name}")
        print(f"  Audio device:  {self.audio_device}")
        print(f"  Chunk size:    {self.chunk_duration}s")
        print(f"  Meeting notes: {'ON → ' + self.notes_path if self.meeting_notes else 'Off'}")
        print(f"  Notion sync:   {'ON' if self.notion_sync else 'Off'}")
        if self.max_duration:
            print(f"  Max duration:  {self.max_duration}s")
        print(f"  Output:        {self.output_dir}")
        print(f"{'='*60}")
        print(f"  Press Ctrl+C to stop.\n")

        # Initialize Notion sync if enabled
        if self.notion_sync:
            from notion_sync import NotionSync
            self._notion = NotionSync(session_name=session_name)
            print(f"[Live] Notion sync file: {self._notion.sync_file}")

        # Load models
        self._load_models()
        print("\n[Live] Recording started. Speak now...\n")

        start_time = time.time()
        chunk_num = 0

        # Signal handler
        def handle_stop(signum, frame):
            print("\n\n[Live] Stopping capture...")
            self._stop.set()
            if self._recording_proc:
                self._recording_proc.terminate()

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        while not self._stop.is_set():
            elapsed = time.time() - start_time
            if self.max_duration and elapsed >= self.max_duration:
                print(f"\n[Live] Duration limit reached ({self.max_duration}s).")
                break

            wav = self._record_audio_chunk()
            if wav and not self._stop.is_set():
                self._process_chunk(wav, chunk_num, elapsed)
                # Clean up wav to save disk
                try:
                    os.remove(wav)
                except OSError:
                    pass

            chunk_num += 1

        # Finalize
        total = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  CAPTURE COMPLETE — {int(total)}s recorded")
        print(f"{'='*60}")

        # Flush meeting notes
        if self._meeting_intel:
            self._meeting_intel.flush()
            print(f"  Meeting notes: {self.notes_path}")

            # Save JSON too
            data = self._meeting_intel.get_summary_data()
            json_path = os.path.join(self.output_dir, "meeting_data.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  JSON data:     {json_path}")

            # Finalize Notion sync
            if self._notion:
                self._notion.finalize(data, total)
                print(f"  Notion sync:   {self._notion.sync_file} (completed)")

        # Save full transcript
        if self._all_segments:
            transcript_path = os.path.join(self.output_dir, "transcript.json")
            with open(transcript_path, "w") as f:
                json.dump({
                    "session": session_name,
                    "duration": int(total),
                    "segments": self._all_segments,
                    "total_segments": len(self._all_segments),
                }, f, indent=2, ensure_ascii=False)
            print(f"  Transcript:    {transcript_path}")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Live meeting capture with real-time transcription and notes",
    )
    parser.add_argument(
        "--audio-device", default="0",
        help="AVFoundation audio device index (default: 0 = MacBook mic). "
             "Run 'ffmpeg -f avfoundation -list_devices true -i \"\"' to see devices.",
    )
    parser.add_argument(
        "--chunk-duration", type=int, default=CHUNK_DURATION,
        help=f"Seconds per recording chunk (default: {CHUNK_DURATION})",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Max recording duration in seconds (default: unlimited, Ctrl+C to stop)",
    )
    parser.add_argument(
        "--meeting-notes", action="store_true",
        help="Extract structured meeting notes (action items, decisions, etc.)",
    )
    parser.add_argument(
        "--notes-output", default=None,
        help="Path for meeting notes markdown (default: output/live_meeting_notes.md)",
    )
    parser.add_argument(
        "--speakers", default=None,
        help="Map speaker IDs to names: 'SPEAKER_00=Colin,SPEAKER_01=Ryan'",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for session files (default: output/live_session)",
    )
    parser.add_argument(
        "--notion", action="store_true",
        help="Auto-create a Notion page and sync meeting notes in real-time",
    )

    args = parser.parse_args()

    speaker_map = {}
    if args.speakers:
        for pair in args.speakers.split(","):
            if "=" in pair:
                k, v = pair.strip().split("=", 1)
                speaker_map[k.strip()] = v.strip()

    # --notion implies --meeting-notes
    if args.notion:
        args.meeting_notes = True

    capture = LiveCapture(
        audio_device=args.audio_device,
        chunk_duration=args.chunk_duration,
        output_dir=args.output_dir,
        meeting_notes=args.meeting_notes,
        notes_path=args.notes_output,
        speakers=speaker_map,
        max_duration=args.duration,
        notion_sync=args.notion,
    )
    capture.run()


if __name__ == "__main__":
    main()

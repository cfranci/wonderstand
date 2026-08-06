#!/usr/bin/env python3
"""
Live Meeting Capture — Records audio in rolling chunks, transcribes with
Whisper, extracts meeting intelligence, and streams JSON output.

Usage:
    python live_capture.py --duration 60
    python live_capture.py --stream --diarize
    python live_capture.py --duration 300 --stream --meeting-notes
    python live_capture.py --stream --diarize --meeting-notes

Press Ctrl+C to stop recording and generate final notes.

Stream mode (--stream) outputs JSON lines to stdout:
    {"type": "transcript", "speaker": "Colin", "text": "...", "start": 1.0, "end": 5.2}
    {"type": "action", "text": "Colin to send the proposal", "speaker": "Colin"}
    {"type": "decision", "text": "Going with Option B for pricing"}
    {"type": "question", "text": "What's the timeline?", "speaker": "Maria"}

All status/log output goes to stderr when streaming, keeping stdout clean
for downstream consumers (VNOCH, terminal UI, etc.).

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
import sys
import threading
import time
from datetime import datetime


# Chunk config
CHUNK_DURATION = 30   # seconds per audio chunk sent to Whisper
CHUNK_OVERLAP = 2     # overlap between chunks

# Streaming mode uses smaller batches for faster intelligence extraction
STREAM_EXTRACTION_BATCH = 3

# lightning_whisper_mlx segment format: [start_frames, end_frames, text]
# Convert to seconds: frames * HOP_LENGTH / SAMPLE_RATE = frames * 0.01
_WHISPER_FRAME_TO_SEC = 160 / 16000  # HOP_LENGTH / SAMPLE_RATE


class LiveCapture:
    """Records audio live, transcribes in real-time, extracts meeting notes."""

    def __init__(self, audio_device="0", chunk_duration=CHUNK_DURATION,
                 output_dir=None, meeting_notes=False, notes_path=None,
                 speakers=None, max_duration=None, notion_sync=False,
                 stream=False, diarize=False):
        self.audio_device = audio_device
        self.chunk_duration = chunk_duration
        self.output_dir = output_dir or "output/live_session"
        self.meeting_notes = meeting_notes
        self.notes_path = notes_path or "output/live_meeting_notes.md"
        self.speakers = speakers or {}
        self.max_duration = max_duration
        self.notion_sync = notion_sync
        self.stream = stream
        self.diarize = diarize

        self._stop = threading.Event()
        self._recording_proc = None
        self._chunk_index = 0

        # Pipeline components (loaded lazily)
        self._whisper_model = None
        self._llm_model = None
        self._llm_tokenizer = None
        self._meeting_intel = None
        self._notion = None
        self._diarizer = None

        # Accumulated results
        self._all_segments = []
        self._prev_text = ""

        os.makedirs(self.output_dir, exist_ok=True)

    # ── I/O helpers ──────────────────────────────────────────────

    def _emit(self, data):
        """Write a JSON line to stdout for stream consumers."""
        sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _log(self, msg):
        """Log a status message. Stderr when streaming, stdout otherwise."""
        if self.stream:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        else:
            print(msg)

    # ── Model loading ────────────────────────────────────────────

    def _load_models(self):
        """Load Whisper, optional LLM, and optional diarization models."""
        from lightning_whisper_mlx import LightningWhisperMLX
        from config import WHISPER_MODEL, SUMMARY_LLM_MODEL

        self._log("[Live] Loading Whisper model...")
        self._whisper_model = LightningWhisperMLX(
            model=WHISPER_MODEL, batch_size=12, quant=None
        )
        self._log("[Live] Whisper loaded.")

        if self.meeting_notes:
            from mlx_lm import load as load_lm
            from meeting_intelligence import MeetingIntelligence

            self._log(f"[Live] Loading summary LLM: {SUMMARY_LLM_MODEL}")
            self._llm_model, self._llm_tokenizer = load_lm(SUMMARY_LLM_MODEL)
            self._log("[Live] LLM loaded.")

            batch_size = STREAM_EXTRACTION_BATCH if self.stream else None
            self._meeting_intel = MeetingIntelligence(
                llm_model=self._llm_model,
                llm_tokenizer=self._llm_tokenizer,
                output_path=self.notes_path,
                extraction_batch_size=batch_size,
            )

        if self.diarize:
            from diarization import SpeakerDiarizer
            self._diarizer = SpeakerDiarizer(speaker_map=self.speakers)
            self._log("[Live] Loading diarization model...")
            self._diarizer.load_model()
            self._log("[Live] Diarization model loaded.")

    # ── Recording ────────────────────────────────────────────────

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
            self._log(f"[Live] Recording error: {e}")
        return None

    # ── Transcription ────────────────────────────────────────────

    def _transcribe_raw(self, wav_path):
        """Transcribe a WAV file with Whisper. Returns full result dict."""
        return self._whisper_model.transcribe(wav_path)

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

    # ── Data collection ─────────────────────────────────────────

    def _build_transcript_segments(self, whisper_segments, deduped_text,
                                   time_offset, diar_segments):
        """Build transcript segment dicts from a processed chunk.

        Returns list of {type, speaker, text, start, end} ready for
        both stdout streaming and Notion sync.

        lightning_whisper_mlx segments are [start_frames, end_frames, text].
        Convert frames to seconds with _WHISPER_FRAME_TO_SEC.
        """
        segments = []

        if whisper_segments:
            for wseg in whisper_segments:
                # Handle [start_frames, end_frames, text] list format
                if isinstance(wseg, (list, tuple)) and len(wseg) >= 3:
                    seg_start = wseg[0] * _WHISPER_FRAME_TO_SEC + time_offset
                    seg_end = wseg[1] * _WHISPER_FRAME_TO_SEC + time_offset
                    seg_text = wseg[2].strip() if isinstance(wseg[2], str) else ""
                # Handle dict format (standard Whisper)
                elif isinstance(wseg, dict):
                    seg_start = wseg.get("start", 0) + time_offset
                    seg_end = wseg.get("end", 0) + time_offset
                    seg_text = wseg.get("text", "").strip()
                else:
                    continue

                if not seg_text:
                    continue

                speaker = ""
                if self._diarizer and diar_segments:
                    speaker = self._diarizer.get_speaker(seg_start, seg_end)
                    if speaker == "UNKNOWN":
                        speaker = ""

                segments.append({
                    "type": "transcript",
                    "speaker": speaker,
                    "text": seg_text,
                    "start": round(seg_start, 1),
                    "end": round(seg_end, 1),
                })
        else:
            speaker = self._dominant_speaker(diar_segments)
            segments.append({
                "type": "transcript",
                "speaker": speaker,
                "text": deduped_text,
                "start": round(time_offset, 1),
                "end": round(time_offset + self.chunk_duration, 1),
            })

        return segments

    def _collect_new_intel(self, prev_counts):
        """Collect meeting intelligence items added since prev_counts.

        Returns dict with lists: {actions: [...], decisions: [...], questions: [...]}.
        Each item has {type, text, speaker, time}.
        """
        result = {"actions": [], "decisions": [], "questions": []}
        if not self._meeting_intel:
            return result

        mi = self._meeting_intel

        for item in mi.action_items[prev_counts["actions"]:]:
            result["actions"].append({
                "type": "action",
                "text": item["task"],
                "speaker": item.get("owner", ""),
                "time": item.get("time", 0),
            })

        for item in mi.decisions[prev_counts["decisions"]:]:
            result["decisions"].append({
                "type": "decision",
                "text": item["decision"],
                "speaker": item.get("speaker", ""),
                "time": item.get("time", 0),
            })

        for item in mi.questions[prev_counts["questions"]:]:
            result["questions"].append({
                "type": "question",
                "text": item["question"],
                "speaker": item.get("asked_by", ""),
                "time": item.get("time", 0),
            })

        return result

    def _dispatch_intel(self, new_intel):
        """Send new intelligence items to stream and/or Notion."""
        all_items = new_intel["actions"] + new_intel["decisions"] + new_intel["questions"]

        if self.stream:
            for item in all_items:
                self._emit(item)

        if self._notion:
            if new_intel["actions"]:
                self._notion.append_actions(new_intel["actions"])
            if new_intel["decisions"]:
                self._notion.append_decisions(new_intel["decisions"])
            if new_intel["questions"]:
                self._notion.append_questions(new_intel["questions"])

    @staticmethod
    def _dominant_speaker(diar_segments):
        """Find the speaker with the most talk time in a set of segments."""
        if not diar_segments:
            return ""
        durations = {}
        for s, e, spk in diar_segments:
            durations[spk] = durations.get(spk, 0) + (e - s)
        return max(durations, key=durations.get) if durations else ""

    def _intel_counts(self):
        """Snapshot current meeting intelligence item counts."""
        if not self._meeting_intel:
            return {"actions": 0, "decisions": 0, "questions": 0}
        return {
            "actions": len(self._meeting_intel.action_items),
            "decisions": len(self._meeting_intel.decisions),
            "questions": len(self._meeting_intel.questions),
        }

    # ── Chunk processing ─────────────────────────────────────────

    def _process_chunk(self, wav_path, chunk_num, elapsed):
        """Transcribe, optionally diarize, and emit results for one chunk."""
        # Transcribe with Whisper
        result = self._transcribe_raw(wav_path)
        full_text = result.get("text", "").strip()
        whisper_segments = result.get("segments", [])

        # Deduplicate overlapping text at chunk boundaries
        deduped_text = self._deduplicate(full_text)

        if not deduped_text or not deduped_text.strip():
            self._log(f"  [Chunk {chunk_num}] (silence)")
            return

        # Wall-clock time offset for accurate streaming timestamps
        time_offset = elapsed

        # Run diarization on this chunk
        diar_segments = []
        if self._diarizer:
            try:
                diar_segments = self._diarizer.process_wav(
                    wav_path, time_offset=time_offset
                )
            except Exception as e:
                self._log(f"  [Diarization] Error on chunk {chunk_num}: {e}")

        # Build transcript segments (used by stream + Notion)
        transcript_segs = self._build_transcript_segments(
            whisper_segments, deduped_text, time_offset, diar_segments
        )

        # Stream JSON lines
        if self.stream:
            for seg in transcript_segs:
                self._emit(seg)

        # Push transcript to Notion
        if self._notion:
            self._notion.append_transcript(transcript_segs)

        # Human-readable log line
        speaker_tag = ""
        if diar_segments:
            dom = self._dominant_speaker(diar_segments)
            if dom:
                speaker_tag = f"{dom}: "

        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self._log(f"  [{mins}:{secs:02d}] {speaker_tag}{deduped_text[:120]}")

        # Build segment for meeting intelligence
        timestamp = chunk_num * self.chunk_duration
        segment = {
            "timestamp": float(timestamp),
            "end_timestamp": float(timestamp + self.chunk_duration),
            "audio": {
                "segments": [{
                    "time": float(timestamp),
                    "text": deduped_text,
                    "speaker": speaker_tag.rstrip(": "),
                }],
                "full_text": deduped_text,
            },
            "video": {"segments": []},
            "summary": deduped_text[:200],
        }
        self._all_segments.append(segment)

        # Feed meeting intelligence, dispatch new items to stream + Notion
        if self._meeting_intel:
            prev = self._intel_counts()
            self._meeting_intel.process_segment(segment, realtime=True)
            new_intel = self._collect_new_intel(prev)
            self._dispatch_intel(new_intel)

    # ── Main loop ────────────────────────────────────────────────

    def run(self):
        """Main capture loop."""
        now = datetime.now()
        session_name = now.strftime("Live Meeting — %b %d, %Y %I:%M %p")

        # Session start event
        if self.stream:
            self._emit({
                "type": "session_start",
                "session": session_name,
                "timestamp": now.isoformat(),
                "config": {
                    "chunk_duration": self.chunk_duration,
                    "diarize": self.diarize,
                    "meeting_notes": self.meeting_notes,
                },
            })

        self._log(f"\n{'='*60}")
        self._log(f"  LIVE MEETING CAPTURE")
        self._log(f"{'='*60}")
        self._log(f"  Session:       {session_name}")
        self._log(f"  Audio device:  {self.audio_device}")
        self._log(f"  Chunk size:    {self.chunk_duration}s")
        self._log(f"  Stream mode:   {'ON' if self.stream else 'Off'}")
        self._log(f"  Diarization:   {'ON' if self.diarize else 'Off'}")
        self._log(f"  Meeting notes: {'ON -> ' + self.notes_path if self.meeting_notes else 'Off'}")
        self._log(f"  Notion sync:   {'ON' if self.notion_sync else 'Off'}")
        if self.max_duration:
            self._log(f"  Max duration:  {self.max_duration}s")
        self._log(f"  Output:        {self.output_dir}")
        self._log(f"{'='*60}")
        self._log(f"  Press Ctrl+C to stop.\n")

        # Initialize Notion sync — creates page immediately
        if self.notion_sync:
            from notion_sync import NotionSync
            self._notion = NotionSync(session_name=session_name)
            page_id, page_url = self._notion.create_page()
            self._log(f"[Live] Notion page: {page_url}")

        # Load models
        self._load_models()
        self._log("\n[Live] Recording started. Speak now...\n")

        start_time = time.time()
        chunk_num = 0

        # Signal handler
        def handle_stop(signum, frame):
            self._log("\n\n[Live] Stopping capture...")
            self._stop.set()
            if self._recording_proc:
                self._recording_proc.terminate()

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        while not self._stop.is_set():
            elapsed = time.time() - start_time
            if self.max_duration and elapsed >= self.max_duration:
                self._log(f"\n[Live] Duration limit reached ({self.max_duration}s).")
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

        # ── Finalize ─────────────────────────────────────────────
        total = time.time() - start_time
        self._log(f"\n{'='*60}")
        self._log(f"  CAPTURE COMPLETE — {int(total)}s recorded")
        self._log(f"{'='*60}")

        # Flush meeting notes (may extract one last batch)
        if self._meeting_intel:
            prev = self._intel_counts()
            self._meeting_intel.flush()
            new_intel = self._collect_new_intel(prev)
            self._dispatch_intel(new_intel)

            self._log(f"  Meeting notes: {self.notes_path}")

            # Save JSON
            data = self._meeting_intel.get_summary_data()
            json_path = os.path.join(self.output_dir, "meeting_data.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._log(f"  JSON data:     {json_path}")

        # Finalize Notion page
        if self._notion:
            self._notion.finalize(total)
            self._log(f"  Notion page:   {self._notion.page_url} (complete)")

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
            self._log(f"  Transcript:    {transcript_path}")

        # Session end event
        if self.stream:
            self._emit({
                "type": "session_end",
                "duration": round(total, 1),
                "chunks": chunk_num,
                "segments": len(self._all_segments),
            })

        self._log(f"{'='*60}\n")


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
    parser.add_argument(
        "--stream", action="store_true",
        help="Output JSON lines to stdout (transcript, actions, decisions). "
             "Status logs go to stderr. Designed for piping to VNOCH or other consumers.",
    )
    parser.add_argument(
        "--diarize", action="store_true",
        help="Enable real-time speaker diarization on each chunk. "
             "Requires HF_TOKEN env var for pyannote.",
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
        stream=args.stream,
        diarize=args.diarize,
    )
    capture.run()


if __name__ == "__main__":
    main()

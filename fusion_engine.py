import json
import queue
import threading
from collections import defaultdict

from config import FUSION_WINDOW, TIMESTAMP_TOLERANCE, SUMMARY_LLM_MODEL, SUMMARY_MAX_TOKENS


class FusionEngine:
    def __init__(self, audio_queue, video_queue, output_path=None,
                 use_llm_summary=True, gpu_lock=None):
        self.audio_queue = audio_queue
        self.video_queue = video_queue
        self.output_path = output_path
        self.use_llm_summary = use_llm_summary
        self.gpu_lock = gpu_lock
        self._stop = threading.Event()

        self.audio_segments = []
        self.video_segments = []
        self.fused_results = []

        self.audio_done = False
        self.video_done = False

        # LLM for summaries
        self._llm_model = None
        self._llm_tokenizer = None

    def load_summary_model(self):
        """Load small LLM for summary generation."""
        if not self.use_llm_summary:
            return
        from mlx_lm import load
        print(f"[Fusion] Loading summary LLM: {SUMMARY_LLM_MODEL}")
        self._llm_model, self._llm_tokenizer = load(SUMMARY_LLM_MODEL)
        print("[Fusion] Summary model loaded.")

    def stop(self):
        self._stop.set()

    def _drain_queues(self):
        """Pull all available items from both queues."""
        while True:
            try:
                item = self.audio_queue.get_nowait()
                if item.get("done"):
                    self.audio_done = True
                else:
                    self.audio_segments.append(item)
            except queue.Empty:
                break

        while True:
            try:
                item = self.video_queue.get_nowait()
                if item.get("done"):
                    self.video_done = True
                else:
                    self.video_segments.append(item)
            except queue.Empty:
                break

    def _find_matching_video(self, audio_ts):
        """Find video segments that overlap with an audio timestamp."""
        matches = []
        for vs in self.video_segments:
            vs_start = vs["timestamp"]
            vs_end = vs.get("end_timestamp", vs_start)
            if (
                vs_start - TIMESTAMP_TOLERANCE <= audio_ts <= vs_end + TIMESTAMP_TOLERANCE
            ):
                matches.append(vs)
        return matches

    def _find_matching_audio(self, video_start, video_end):
        """Find audio segments within a video time range."""
        matches = []
        for aseg in self.audio_segments:
            a_ts = aseg["timestamp"]
            a_end = a_ts + aseg.get("duration", 0)
            if a_ts <= video_end + TIMESTAMP_TOLERANCE and a_end >= video_start - TIMESTAMP_TOLERANCE:
                matches.append(aseg)
        return matches

    def fuse(self):
        """Fuse all collected audio and video segments."""
        # Group by time windows
        all_timestamps = set()
        for a in self.audio_segments:
            all_timestamps.add(a["timestamp"])
        for v in self.video_segments:
            all_timestamps.add(v["timestamp"])

        if not all_timestamps:
            return

        sorted_ts = sorted(all_timestamps)

        # Build windows
        windows = []
        window_start = sorted_ts[0]
        while window_start <= sorted_ts[-1]:
            window_end = window_start + FUSION_WINDOW
            windows.append((window_start, window_end))
            window_start = window_end

        # Assign each segment to exactly ONE window (closest match)
        # This prevents audio/video bleeding across window boundaries
        assigned_audio = set()
        assigned_video = set()

        for w_start, w_end in windows:
            audio_in_window = []
            for i, a in enumerate(self.audio_segments):
                if i in assigned_audio:
                    continue
                if w_start - TIMESTAMP_TOLERANCE <= a["timestamp"] < w_end:
                    audio_in_window.append(a)
                    assigned_audio.add(i)

            video_in_window = []
            for i, v in enumerate(self.video_segments):
                if i in assigned_video:
                    continue
                if w_start - TIMESTAMP_TOLERANCE <= v["timestamp"] < w_end:
                    video_in_window.append(v)
                    assigned_video.add(i)

            if not audio_in_window and not video_in_window:
                continue

            # Merge consecutive same-speaker segments into conversation turns
            merged_audio = self._merge_speaker_turns(audio_in_window)

            fused_entry = {
                "timestamp": round(w_start, 2),
                "end_timestamp": round(w_end, 2),
                "audio": {
                    "segments": [
                        {
                            "time": a["timestamp"],
                            "text": a["text"],
                            "speaker": a.get("speaker", ""),
                        }
                        for a in merged_audio
                    ],
                    "full_text": " ".join(a["text"] for a in merged_audio),
                },
                "video": {
                    "segments": [
                        {
                            "time": v["timestamp"],
                            "end_time": v.get("end_timestamp", v["timestamp"]),
                            "description": v["description"],
                            "scene_change": v.get("scene_change", False),
                        }
                        for v in video_in_window
                    ],
                },
                "summary": self._generate_summary(merged_audio, video_in_window),
            }
            self.fused_results.append(fused_entry)

    @staticmethod
    def _merge_speaker_turns(audio_segments):
        """Merge consecutive segments from the same speaker into single turns.

        Turns [SPEAKER_01: "hello", SPEAKER_01: "how are you", SPEAKER_00: "good"]
        into  [SPEAKER_01: "hello how are you", SPEAKER_00: "good"]
        """
        if not audio_segments:
            return audio_segments

        merged = []
        current = None

        for seg in audio_segments:
            speaker = seg.get("speaker", "")
            if current and current.get("speaker", "") == speaker:
                # Same speaker — merge text
                current["text"] = current["text"] + " " + seg["text"]
                current["duration"] = current.get("duration", 0) + seg.get("duration", 0)
            else:
                # New speaker — start new turn
                if current:
                    merged.append(current)
                current = dict(seg)  # copy

        if current:
            merged.append(current)

        return merged

    def _generate_summary(self, audio_segments, video_segments):
        """Generate a fused summary using LLM or fallback to simple concat."""
        if not self.use_llm_summary or self._llm_model is None:
            return self._generate_summary_simple(audio_segments, video_segments)

        # Build prompt for the LLM
        prompt_parts = []

        if audio_segments:
            speech_lines = []
            for a in audio_segments:
                speaker = a.get("speaker", "")
                prefix = f"[{speaker}] " if speaker else ""
                speech_lines.append(f"{prefix}{a['text']}")
            prompt_parts.append("SPEECH TRANSCRIPT:\n" + "\n".join(speech_lines))

        if video_segments:
            visual_lines = []
            for v in video_segments:
                desc = v.get("description", "")
                if isinstance(desc, str) and desc:
                    visual_lines.append(desc[:500])
            if visual_lines:
                prompt_parts.append("VISUAL OBSERVATIONS:\n" + "\n".join(visual_lines))

        if not prompt_parts:
            return "No data"

        system_prompt = (
            "You are a video analysis assistant. Given speech transcripts and "
            "visual observations from the same time window of a video, write "
            "a concise 2-3 sentence summary that fuses both modalities into "
            "a coherent description. Focus on what is being discussed, what "
            "is shown on screen, and how they relate. If speakers are labeled, "
            "reference them. Be factual and specific."
        )

        user_content = "\n\n".join(prompt_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        chat_prompt = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            from mlx_lm import generate
            if self.gpu_lock:
                with self.gpu_lock:
                    summary = generate(
                        self._llm_model, self._llm_tokenizer,
                        prompt=chat_prompt,
                        max_tokens=SUMMARY_MAX_TOKENS,
                        verbose=False,
                    )
            else:
                summary = generate(
                    self._llm_model, self._llm_tokenizer,
                    prompt=chat_prompt,
                    max_tokens=SUMMARY_MAX_TOKENS,
                    verbose=False,
                )
            return summary.strip() if isinstance(summary, str) else str(summary).strip()
        except Exception as e:
            print(f"[Fusion] LLM summary failed: {e}, falling back to simple")
            return self._generate_summary_simple(audio_segments, video_segments)

    def _generate_summary_simple(self, audio_segments, video_segments):
        """Original concatenation-based summary as fallback."""
        parts = []
        if audio_segments:
            text = " ".join(a["text"] for a in audio_segments)
            parts.append(f"Speech: {text}")
        if video_segments:
            desc = " | ".join(
                v["description"][:150] for v in video_segments
                if isinstance(v.get("description"), str)
            )
            if desc:
                parts.append(f"Visual: {desc}")
        return " // ".join(parts) if parts else "No data"

    def process(self, realtime_callback=None):
        """Main fusion loop. Runs until both pipelines signal done."""
        print("[Fusion] Starting fusion engine.")
        last_fuse_count = 0

        while not self._stop.is_set():
            self._drain_queues()

            # In realtime mode, fuse periodically as data arrives
            total = len(self.audio_segments) + len(self.video_segments)
            if realtime_callback and total > last_fuse_count + 3:
                self.fused_results.clear()
                self.fuse()
                if self.fused_results:
                    realtime_callback(self.fused_results[-1])
                last_fuse_count = total

            if self.audio_done and self.video_done:
                break

            threading.Event().wait(0.5)

        # Final fuse pass
        self._drain_queues()
        self.fused_results.clear()
        self.fuse()

        if self.output_path:
            self.save(self.output_path)

        print(f"[Fusion] Complete. {len(self.fused_results)} fused segments.")
        return self.fused_results

    def save(self, path):
        """Save fused results to JSON."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def _default(obj):
            """Fallback serializer for non-standard types."""
            if hasattr(obj, "text"):
                return obj.text
            return str(obj)

        with open(path, "w") as f:
            json.dump(
                {"segments": self.fused_results, "total_segments": len(self.fused_results)},
                f,
                indent=2,
                ensure_ascii=False,
                default=_default,
            )
        print(f"[Fusion] Saved to {path}")

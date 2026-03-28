import queue
import threading
import sys

from audio_pipeline import AudioPipeline
from video_pipeline import VideoPipeline
from fusion_engine import FusionEngine
from config import AUDIO_QUEUE_SIZE, VIDEO_QUEUE_SIZE, DEFAULT_OUTPUT


# Global lock to serialize MLX/Metal GPU operations across threads.
# Metal command buffers crash when two threads submit concurrently.
gpu_lock = threading.Lock()


class StreamProcessor:
    def __init__(
        self,
        video_path,
        audio_model=None,
        video_fps=None,
        output_path=None,
        realtime=False,
        diarize=False,
        use_llm_summary=True,
        speaker_map=None,
        meeting_notes=False,
        meeting_notes_path=None,
    ):
        self.video_path = video_path
        self.realtime = realtime
        self.output_path = output_path or DEFAULT_OUTPUT
        self.diarize = diarize
        self.speaker_map = speaker_map or {}
        self.meeting_notes = meeting_notes

        self.audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
        self.video_queue = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)

        self.audio_pipeline = AudioPipeline(
            model_size=audio_model, result_queue=self.audio_queue
        )
        self.video_pipeline = VideoPipeline(
            result_queue=self.video_queue, fps=video_fps
        )
        self.fusion_engine = FusionEngine(
            self.audio_queue, self.video_queue,
            output_path=self.output_path,
            use_llm_summary=use_llm_summary,
            gpu_lock=gpu_lock,
        )

        # Meeting intelligence (optional, shares LLM with fusion engine)
        self.meeting_intel = None
        if meeting_notes:
            from meeting_intelligence import MeetingIntelligence
            notes_path = meeting_notes_path or "output/meeting_notes.md"
            self.meeting_intel = MeetingIntelligence(
                gpu_lock=gpu_lock,
                output_path=notes_path,
            )

        # Speaker diarization (optional, runs on CPU)
        self.diarizer = None
        if diarize:
            from diarization import SpeakerDiarizer
            self.diarizer = SpeakerDiarizer(speaker_map=self.speaker_map)

        self._threads = []
        self._shutdown = threading.Event()

    def _audio_worker(self):
        try:
            self.audio_pipeline.process(
                self.video_path, gpu_lock=gpu_lock, diarizer=self.diarizer
            )
        except Exception as e:
            print(f"[Audio] Error: {e}")
            import traceback; traceback.print_exc()
            self.audio_queue.put({"type": "audio", "done": True})

    def _video_worker(self):
        try:
            self.video_pipeline.process(self.video_path, gpu_lock=gpu_lock)
        except Exception as e:
            print(f"[Video] Error: {e}")
            import traceback; traceback.print_exc()
            self.video_queue.put({"type": "video", "done": True})

    def _realtime_print(self, segment):
        """Print a fused segment to terminal in realtime."""
        ts = segment["timestamp"]
        end = segment["end_timestamp"]
        print(f"\n{'='*60}")
        print(f"  [{ts:.1f}s - {end:.1f}s]")
        print(f"{'='*60}")
        if segment["audio"]["full_text"]:
            # Show speaker-labeled text if available
            for aseg in segment["audio"]["segments"]:
                speaker = aseg.get("speaker", "")
                speaker_tag = f"[{speaker}] " if speaker else ""
                print(f"  \U0001f3a4 {speaker_tag}{aseg['text']}")
        for vs in segment["video"]["segments"]:
            desc = vs["description"]
            if isinstance(desc, str):
                change = " (SCENE CHANGE)" if vs.get("scene_change") else ""
                print(f"  \U0001f441  {desc[:300]}{change}")
        if segment.get("summary"):
            print(f"  \U0001f4dd {segment['summary'][:300]}")
        print()

        # Feed meeting intelligence in real-time
        if self.meeting_intel:
            self.meeting_intel.process_segment(segment, realtime=True)

    def run(self):
        """Start all processing threads and wait for completion."""
        print(f"\nMulti-Modal Analyzer v2")
        print(f"{'=' * 50}")
        print(f"  Source:       {self.video_path}")
        print(f"  Audio model:  {self.audio_pipeline.model_size}")
        print(f"  Video FPS:    {self.video_pipeline.fps}")
        print(f"  Output:       {self.output_path}")
        print(f"  Realtime:     {self.realtime}")
        print(f"  Diarization:  {self.diarize}")
        print(f"  LLM Summary:  {self.fusion_engine.use_llm_summary}")
        print(f"  Meeting Notes:{' ' + self.meeting_intel.output_path if self.meeting_intel else ' Off'}")
        print(f"{'=' * 50}\n")

        # Run diarization FIRST (CPU-only, before GPU models load)
        if self.diarize and self.diarizer:
            self.diarizer.process(self.video_path)

        # Load models sequentially (they share GPU memory)
        print("Loading models...")
        self.audio_pipeline.load_model()
        self.video_pipeline.load_model()
        self.fusion_engine.load_summary_model()

        # Share LLM with meeting intelligence (no need to load twice)
        if self.meeting_intel and self.fusion_engine._llm_model:
            self.meeting_intel.set_llm(
                self.fusion_engine._llm_model,
                self.fusion_engine._llm_tokenizer,
            )
            print("[Meeting] Intelligence module ready (sharing summary LLM).")

        print("Models loaded. Starting analysis...\n")

        # Start processing threads
        audio_thread = threading.Thread(
            target=self._audio_worker, name="AudioWorker", daemon=True
        )
        video_thread = threading.Thread(
            target=self._video_worker, name="VideoWorker", daemon=True
        )

        audio_thread.start()
        video_thread.start()
        self._threads = [audio_thread, video_thread]

        # Run fusion in main thread (handles Ctrl+C)
        callback = self._realtime_print if self.realtime else None
        try:
            results = self.fusion_engine.process(realtime_callback=callback)
        except KeyboardInterrupt:
            print("\n[Main] Interrupted. Shutting down...")
            self.shutdown()
            return []

        # Generate final meeting notes from ALL fused results
        # (realtime callbacks only see partial data as it streams in)
        if self.meeting_intel:
            print(f"[Meeting] Generating meeting notes from {len(results)} segments...")
            self.meeting_intel.reset()
            for seg in results:
                self.meeting_intel.process_segment(seg, realtime=False)
            self.meeting_intel.flush()
            print(f"[Meeting] Notes saved to: {self.meeting_intel.output_path}")

        # Wait for workers to finish
        for t in self._threads:
            t.join(timeout=5)

        print(f"\nAnalysis complete. {len(results)} segments fused.")
        if self.output_path:
            print(f"Results saved to: {self.output_path}")

        return results

    def shutdown(self):
        """Gracefully stop all pipelines."""
        self._shutdown.set()
        self.audio_pipeline.stop()
        self.video_pipeline.stop()
        self.fusion_engine.stop()
        for t in self._threads:
            t.join(timeout=3)

import queue
import tempfile
import threading
import wave
import numpy as np
import av

from config import (
    WHISPER_MODEL,
    AUDIO_CHUNK_DURATION,
    AUDIO_OVERLAP,
    AUDIO_SAMPLE_RATE,
)


class AudioPipeline:
    def __init__(self, model_size=None, result_queue=None):
        self.model_size = model_size or WHISPER_MODEL
        self.result_queue = result_queue or queue.Queue()
        self.model = None
        self._stop = threading.Event()
        self._prev_text = ""  # Track previous chunk text for deduplication

    def load_model(self):
        from lightning_whisper_mlx import LightningWhisperMLX

        print(f"[Audio] Loading Whisper model: {self.model_size}")
        self.model = LightningWhisperMLX(
            model=self.model_size, batch_size=12, quant=None
        )
        print("[Audio] Model loaded.")

    def stop(self):
        self._stop.set()

    def _deduplicate_text(self, current_text):
        """Remove overlapping words between consecutive chunks.

        With overlap between chunks, Whisper often re-transcribes the last
        few words from the previous chunk. This finds and removes that
        prefix overlap from the current text.
        """
        if not self._prev_text or not current_text:
            self._prev_text = current_text
            return current_text

        prev_words = self._prev_text.split()
        curr_words = current_text.split()

        # Look for the longest suffix of prev_words that matches
        # a prefix of curr_words (check up to ~half of current text)
        max_check = min(len(prev_words), len(curr_words),
                        max(3, len(curr_words) // 2))

        best_overlap = 0
        for overlap_len in range(1, max_check + 1):
            prev_suffix = prev_words[-overlap_len:]
            curr_prefix = curr_words[:overlap_len]
            # Case-insensitive, punctuation-stripped comparison
            if ([w.lower().strip(".,!?;:") for w in prev_suffix] ==
                    [w.lower().strip(".,!?;:") for w in curr_prefix]):
                best_overlap = overlap_len

        if best_overlap > 0:
            deduplicated = " ".join(curr_words[best_overlap:])
        else:
            deduplicated = current_text

        self._prev_text = current_text  # Store ORIGINAL for next comparison
        return deduplicated

    def extract_audio_chunks(self, video_path):
        """Extract audio from video in chunks using PyAV."""
        container = av.open(video_path)
        audio_stream = container.streams.audio[0]

        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE
        )

        samples = np.array([], dtype=np.float32)
        chunk_samples = int(AUDIO_CHUNK_DURATION * AUDIO_SAMPLE_RATE)
        overlap_samples = int(AUDIO_OVERLAP * AUDIO_SAMPLE_RATE)
        step_samples = chunk_samples - overlap_samples
        chunk_index = 0

        for frame in container.decode(audio_stream):
            if self._stop.is_set():
                break

            resampled = resampler.resample(frame)
            for r in resampled:
                audio_data = r.to_ndarray().flatten().astype(np.float32) / 32768.0
                samples = np.concatenate([samples, audio_data])

            while len(samples) >= chunk_samples:
                chunk = samples[:chunk_samples]
                timestamp = chunk_index * (step_samples / AUDIO_SAMPLE_RATE)
                yield timestamp, chunk
                samples = samples[step_samples:]
                chunk_index += 1

        # Yield remaining audio if substantial (>100ms)
        if len(samples) > AUDIO_SAMPLE_RATE * 0.1:
            timestamp = chunk_index * (step_samples / AUDIO_SAMPLE_RATE)
            # Pad to at least 2 seconds for Whisper context
            min_samples = int(AUDIO_SAMPLE_RATE * 2.0)
            if len(samples) < min_samples:
                samples = np.pad(samples, (0, min_samples - len(samples)))
            yield timestamp, samples

        container.close()

    def transcribe_chunk(self, audio_chunk, gpu_lock=None):
        """Transcribe a single audio chunk."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes((audio_chunk * 32768).astype(np.int16).tobytes())

            # Acquire GPU lock for the MLX inference call
            if gpu_lock:
                with gpu_lock:
                    result = self.model.transcribe(f.name)
            else:
                result = self.model.transcribe(f.name)

        return result.get("text", "").strip()

    def process(self, video_path, gpu_lock=None, diarizer=None):
        """Process all audio from a video file, pushing results to queue.

        If a diarizer with speaker_chunks() is available, uses diarization-aligned
        chunking (each chunk = one speaker) for much better speaker accuracy.
        Falls back to fixed-size chunking otherwise.
        """
        if self.model is None:
            self.load_model()

        print(f"[Audio] Processing: {video_path}")

        # Use diarization-driven chunking when available
        if diarizer and hasattr(diarizer, 'speaker_chunks') and diarizer._full_audio is not None:
            self._process_diarized(diarizer, gpu_lock)
        else:
            self._process_fixed_chunks(video_path, gpu_lock, diarizer)

        self.result_queue.put({"type": "audio", "done": True})
        print("[Audio] Processing complete.")

    def _process_diarized(self, diarizer, gpu_lock=None):
        """Transcribe audio split at diarization boundaries — one speaker per chunk."""
        print("[Audio] Using diarization-aligned chunking for better speaker accuracy")
        for start, end, speaker, samples in diarizer.speaker_chunks():
            if self._stop.is_set():
                break
            text = self.transcribe_chunk(samples, gpu_lock=gpu_lock)
            if text and text.strip():
                duration = round(end - start, 2)
                entry = {
                    "type": "audio",
                    "timestamp": round(start, 2),
                    "duration": duration,
                    "text": text.strip(),
                    "speaker": speaker,
                }
                self.result_queue.put(entry)
                print(f"  [Audio @ {entry['timestamp']:.1f}s] [{speaker}] {text[:80]}")

    def _process_fixed_chunks(self, video_path, gpu_lock=None, diarizer=None):
        """Original fixed-size chunk processing (fallback)."""
        for timestamp, chunk in self.extract_audio_chunks(video_path):
            if self._stop.is_set():
                break
            text = self.transcribe_chunk(chunk, gpu_lock=gpu_lock)
            if text:
                text = self._deduplicate_text(text)
                if text.strip():  # May be empty after dedup
                    duration = round(len(chunk) / AUDIO_SAMPLE_RATE, 2)
                    entry = {
                        "type": "audio",
                        "timestamp": round(timestamp, 2),
                        "duration": duration,
                        "text": text,
                    }
                    # Add speaker label if diarizer is available
                    if diarizer:
                        entry["speaker"] = diarizer.get_speaker(
                            timestamp, timestamp + duration
                        )
                    self.result_queue.put(entry)
                    speaker_tag = f" [{entry.get('speaker', '')}]" if diarizer else ""
                    print(f"  [Audio @ {entry['timestamp']:.1f}s{speaker_tag}] {text[:80]}")

import os
import tempfile
import wave
import numpy as np
import av

from config import (
    DIARIZATION_MODEL,
    HF_TOKEN_ENV_VAR,
    AUDIO_SAMPLE_RATE,
    DIARIZATION_NUM_SPEAKERS,
    DIARIZATION_MIN_SEGMENT,
    DIARIZATION_MAX_SEGMENT,
)


class SpeakerDiarizer:
    """Runs pyannote speaker diarization on full audio, provides speaker lookup."""

    def __init__(self, speaker_map=None, num_speakers=None):
        self.segments = []  # List of (start, end, speaker_label)
        self._pipeline = None
        self.speaker_map = speaker_map or {}  # e.g. {"SPEAKER_00": "Colin"}
        self.num_speakers = num_speakers or DIARIZATION_NUM_SPEAKERS
        self._full_audio = None  # Cached full audio samples

    def load_model(self):
        """Load pyannote pipeline. Requires HF_TOKEN env var."""
        from pyannote.audio import Pipeline
        import torch

        hf_token = os.environ.get(HF_TOKEN_ENV_VAR)
        if not hf_token:
            raise ValueError(
                f"Speaker diarization requires a HuggingFace token. "
                f"Set the {HF_TOKEN_ENV_VAR} environment variable.\n"
                f"Get a free token at https://huggingface.co/settings/tokens\n"
                f"after accepting the model license at "
                f"https://huggingface.co/{DIARIZATION_MODEL}"
            )

        print(f"[Diarization] Loading model: {DIARIZATION_MODEL}")
        self._pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, token=hf_token
        )
        # Force CPU -- MPS support is incomplete for pyannote
        self._pipeline.to(torch.device("cpu"))
        print("[Diarization] Model loaded (running on CPU).")

    def process(self, video_path):
        """Run diarization on the full audio track.

        Extracts audio from video, runs pyannote, stores speaker segments.
        This should be called BEFORE the main pipeline starts.
        """
        if self._pipeline is None:
            self.load_model()

        print(f"[Diarization] Extracting full audio from {video_path}...")
        self._full_audio = self._extract_full_audio(video_path)

        # Write to temp WAV file (pyannote needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(
                    (self._full_audio * 32768).astype(np.int16).tobytes()
                )

            duration = len(self._full_audio) / AUDIO_SAMPLE_RATE
            print(f"[Diarization] Running speaker diarization on {duration:.1f}s of audio...")

            # Pass num_speakers hint if available — greatly improves accuracy
            kwargs = {}
            if self.num_speakers is not None:
                kwargs["num_speakers"] = self.num_speakers
                print(f"[Diarization] Hinting num_speakers={self.num_speakers}")

            result = self._pipeline(tmp.name, **kwargs)

        # pyannote v4 returns DiarizeOutput; annotation is in .speaker_diarization
        diarization = getattr(result, "speaker_diarization", result)

        # Convert pyannote output to simple list of (start, end, speaker)
        raw_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append((turn.start, turn.end, speaker))

        # Merge adjacent same-speaker segments that are close together
        self.segments = self._merge_segments(raw_segments)

        n_speakers = len(set(s[2] for s in self.segments))
        print(f"[Diarization] Found {n_speakers} speaker(s), "
              f"{len(raw_segments)} raw -> {len(self.segments)} merged segment(s).")

        # Auto-identify speakers using saved voice profiles
        if not self.speaker_map:
            self._identify_from_profiles()

        return self.segments

    def _merge_segments(self, segments):
        """Merge consecutive same-speaker segments with small gaps between them."""
        if not segments:
            return segments

        merged = [segments[0]]
        for start, end, speaker in segments[1:]:
            prev_start, prev_end, prev_speaker = merged[-1]
            # Merge if same speaker and gap is smaller than threshold
            if speaker == prev_speaker and (start - prev_end) < DIARIZATION_MIN_SEGMENT:
                merged[-1] = (prev_start, end, speaker)
            else:
                merged.append((start, end, speaker))
        return merged

    def _identify_from_profiles(self):
        """Match detected speakers against saved voice profiles."""
        import os as _os
        profiles_file = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "voice_profiles", "profiles.json"
        )
        if not _os.path.exists(profiles_file):
            return

        import json
        with open(profiles_file) as f:
            profiles = json.load(f)
        if not profiles:
            return

        try:
            import tempfile
            import wave
            from pyannote.audio import Model, Inference

            hf_token = os.environ.get(HF_TOKEN_ENV_VAR)
            model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=hf_token
            )
            inference = Inference(model, window="whole")
        except Exception as e:
            print(f"[Diarization] Could not load embedding model: {e}")
            return

        # Get unique speaker labels
        unique_speakers = set(s[2] for s in self.segments)

        for raw_label in unique_speakers:
            # Collect up to 60s of audio for this speaker
            speaker_audio = []
            total_secs = 0
            for start, end, spk in self.segments:
                if spk == raw_label and total_secs < 60:
                    s = int(start * AUDIO_SAMPLE_RATE)
                    e = int(end * AUDIO_SAMPLE_RATE)
                    if self._full_audio is not None and e <= len(self._full_audio):
                        speaker_audio.append(self._full_audio[s:e])
                        total_secs += (end - start)

            if not speaker_audio or total_secs < 3:
                continue

            combined = np.concatenate(speaker_audio)

            # Compute embedding
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes((combined * 32768).astype(np.int16).tobytes())
                embedding = inference(tmp.name)
                _os.unlink(tmp.name)

            # Match against profiles
            emb = np.array(embedding)
            emb = emb / np.linalg.norm(emb)

            best_name = None
            best_score = -1
            for name, data in profiles.items():
                stored = np.array(data["embedding"])
                stored = stored / np.linalg.norm(stored)
                score = float(np.dot(emb, stored))
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= 0.5 and best_name:
                self.speaker_map[raw_label] = best_name
                print(f"[Diarization] {raw_label} identified as {best_name} "
                      f"(confidence: {best_score:.2f})")
            else:
                print(f"[Diarization] {raw_label} not matched "
                      f"(best: {best_name} @ {best_score:.2f})")

    def _extract_full_audio(self, video_path):
        """Extract complete audio track as float32 numpy array."""
        container = av.open(video_path)
        audio_stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE
        )

        all_samples = []
        for frame in container.decode(audio_stream):
            resampled = resampler.resample(frame)
            for r in resampled:
                data = r.to_ndarray().flatten().astype(np.float32) / 32768.0
                all_samples.append(data)

        container.close()
        return np.concatenate(all_samples) if all_samples else np.array([], dtype=np.float32)

    def process_wav(self, wav_path, time_offset=0.0):
        """Run diarization on a single WAV file for incremental/streaming use.

        Unlike process() which takes a full video, this works on individual
        WAV chunks recorded by live_capture. Segments are accumulated so
        get_speaker() works across all processed chunks.

        Args:
            wav_path: Path to WAV file (16kHz, mono, pcm_s16le).
            time_offset: Added to all timestamps for absolute timing.

        Returns:
            List of (start, end, speaker) tuples for this chunk.
        """
        if self._pipeline is None:
            self.load_model()

        kwargs = {}
        if self.num_speakers is not None:
            kwargs["num_speakers"] = self.num_speakers

        result = self._pipeline(wav_path, **kwargs)
        diarization = getattr(result, "speaker_diarization", result)

        raw_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append((
                turn.start + time_offset,
                turn.end + time_offset,
                speaker,
            ))

        merged = self._merge_segments(raw_segments)

        # Apply speaker name mapping
        mapped = [
            (s, e, self.speaker_map.get(spk, spk))
            for s, e, spk in merged
        ]

        # Accumulate for get_speaker() lookups
        self.segments.extend(mapped)

        return mapped

    def get_speaker(self, start_time, end_time):
        """Find the dominant speaker for a given time range.

        For each Whisper transcript segment, find which pyannote speaker
        has the most temporal overlap with it.

        Returns: speaker label string (e.g., "SPEAKER_00") or "UNKNOWN"
        """
        if not self.segments:
            return "UNKNOWN"

        speaker_overlap = {}  # speaker -> total overlap seconds
        for seg_start, seg_end, speaker in self.segments:
            overlap_start = max(start_time, seg_start)
            overlap_end = min(end_time, seg_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        if not speaker_overlap:
            return "UNKNOWN"

        raw_speaker = max(speaker_overlap, key=speaker_overlap.get)
        # Apply name mapping if provided
        return self.speaker_map.get(raw_speaker, raw_speaker)

    def speaker_chunks(self):
        """Yield (start, end, speaker, audio_samples) aligned to diarization boundaries.

        Instead of fixed-size chunks, this splits audio at speaker boundaries
        so each chunk belongs to exactly one speaker. Long segments are split
        at DIARIZATION_MAX_SEGMENT to keep Whisper accurate.

        Requires process() to have been called first (caches full audio).
        """
        if self._full_audio is None:
            raise RuntimeError("Call process() before speaker_chunks()")

        for start, end, speaker in self.segments:
            duration = end - start
            if duration < 0.3:
                continue  # Skip tiny fragments

            # Split long segments for Whisper accuracy
            seg_start = start
            while seg_start < end:
                seg_end = min(seg_start + DIARIZATION_MAX_SEGMENT, end)
                # Don't create tiny trailing segments
                if end - seg_end < 1.0:
                    seg_end = end

                sample_start = int(seg_start * AUDIO_SAMPLE_RATE)
                sample_end = int(seg_end * AUDIO_SAMPLE_RATE)
                samples = self._full_audio[sample_start:sample_end]

                # Pad short segments to 2s minimum for Whisper context
                min_samples = int(AUDIO_SAMPLE_RATE * 2.0)
                if len(samples) < min_samples:
                    samples = np.pad(samples, (0, min_samples - len(samples)))

                mapped_speaker = self.speaker_map.get(speaker, speaker)
                yield seg_start, seg_end, mapped_speaker, samples

                seg_start = seg_end

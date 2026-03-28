import queue
import threading
import numpy as np
import cv2
from PIL import Image
import av

from config import (
    VLM_MODEL,
    VIDEO_FPS,
    VLM_PROMPT_OCR,
    SCENE_CHANGE_THRESHOLD,
    SCENE_MIN_INTERVAL,
    SCENE_MAX_INTERVAL,
)


class VideoPipeline:
    def __init__(self, model_name=None, result_queue=None, fps=None):
        self.model_name = model_name or VLM_MODEL
        self.result_queue = result_queue or queue.Queue()
        self.fps = fps or VIDEO_FPS
        self.model = None
        self.processor = None
        self._stop = threading.Event()
        # Scene change detection state
        self._prev_histogram = None
        self._last_analyzed_ts = -999.0  # Force first frame to be analyzed

    def load_model(self):
        from mlx_vlm import load

        print(f"[Video] Loading VLM: {self.model_name}")
        self.model, self.processor = load(self.model_name)
        print("[Video] Model loaded.")

    def stop(self):
        self._stop.set()

    def _detect_scene_change(self, pil_image):
        """Compare frame to previous using histogram correlation.

        Returns True if the frame is visually different enough to warrant
        analysis. Uses HSV color histogram — robust to minor brightness
        changes but sensitive to actual content changes.
        """
        # Convert PIL to OpenCV format
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # Calculate histogram (Hue and Saturation channels)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if self._prev_histogram is None:
            self._prev_histogram = hist
            return True  # First frame is always a "scene change"

        # Compare with previous frame's histogram
        # correlation: 1.0 = identical, 0.0 = completely different
        correlation = cv2.compareHist(
            self._prev_histogram, hist, cv2.HISTCMP_CORREL
        )

        self._prev_histogram = hist

        # Trigger when correlation drops below threshold
        return correlation < (1.0 - SCENE_CHANGE_THRESHOLD)

    def extract_frames(self, video_path):
        """Extract frames at target FPS from video using PyAV."""
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        source_fps = float(video_stream.average_rate)
        frame_interval = max(1, int(source_fps / self.fps))

        frame_count = 0
        for frame in container.decode(video_stream):
            if self._stop.is_set():
                break

            if frame_count % frame_interval == 0:
                timestamp = float(frame.pts * video_stream.time_base)
                pil_image = frame.to_image()
                # Resize to reduce memory and processing time
                pil_image.thumbnail((512, 512), Image.LANCZOS)
                yield timestamp, pil_image

            frame_count += 1

        container.close()

    @staticmethod
    def _clean_repetition(text):
        """Detect and truncate repeated phrases in VLM output.

        The small 2.2B VLM often loops, producing output like:
        'Number of guests:\nNumber of guests:\nNumber of guests:...'
        This detects repeated lines/phrases and cuts them off.
        """
        if not text:
            return text

        lines = text.split("\n")
        cleaned = []
        repeat_count = 0
        last_line = None

        for line in lines:
            stripped = line.strip()
            if stripped == last_line and stripped:
                repeat_count += 1
                if repeat_count >= 2:
                    # Already saw this line 3+ times, skip
                    continue
            else:
                repeat_count = 0
            last_line = stripped
            cleaned.append(line)

        result = "\n".join(cleaned)

        # Also catch phrase-level repetition within a single line
        # e.g. "The website has a lot of text. The website has a lot of text."
        sentences = result.split(". ")
        if len(sentences) > 3:
            seen = set()
            deduped = []
            for s in sentences:
                key = s.strip().lower()
                if key and key in seen:
                    continue
                seen.add(key)
                deduped.append(s)
            result = ". ".join(deduped)

        return result.strip()

    def analyze_frame(self, image, prompt, gpu_lock=None):
        """Analyze a single frame with the VLM."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(
            self.processor, self.model.config, prompt, num_images=1
        )

        try:
            if gpu_lock:
                with gpu_lock:
                    result = generate(
                        self.model,
                        self.processor,
                        formatted,
                        image=[image],
                        max_tokens=400,  # Increased from 200 for OCR output
                        verbose=False,
                    )
            else:
                result = generate(
                    self.model,
                    self.processor,
                    formatted,
                    image=[image],
                    max_tokens=400,
                    verbose=False,
                )
            # generate() may return a GenerationResult object or a string
            if hasattr(result, "text"):
                return result.text.strip()
            return str(result).strip()
        except Exception as e:
            print(f"  [Video] Analysis error: {e}")
            return str(e)

    def process(self, video_path, gpu_lock=None):
        """Process video frames with scene change detection."""
        if self.model is None:
            self.load_model()

        print(f"[Video] Processing: {video_path} @ {self.fps} FPS (scene change detection)")

        prompt = VLM_PROMPT_OCR  # OCR-focused prompt

        for timestamp, image in self.extract_frames(video_path):
            if self._stop.is_set():
                break

            time_since_last = timestamp - self._last_analyzed_ts

            # Always respect minimum interval to avoid GPU overload
            if time_since_last < SCENE_MIN_INTERVAL:
                continue

            # Check for visual change
            is_scene_change = self._detect_scene_change(image)

            if not is_scene_change and time_since_last < SCENE_MAX_INTERVAL:
                # No scene change and analyzed recently — skip
                continue

            # Either scene changed OR it's been too long (periodic refresh)
            self._last_analyzed_ts = timestamp

            change_tag = "CHANGE" if is_scene_change else "periodic"
            print(f"  [Video @ {timestamp:.1f}s] Analyzing frame ({change_tag})...")

            description = self.analyze_frame(image, prompt, gpu_lock=gpu_lock)
            description = self._clean_repetition(description)

            entry = {
                "type": "video",
                "timestamp": round(timestamp, 2),
                "end_timestamp": round(timestamp, 2),
                "description": description,
                "scene_change": is_scene_change,
            }
            self.result_queue.put(entry)
            desc_preview = description[:80] if isinstance(description, str) else ""
            print(f"  [Video @ {timestamp:.1f}s] {desc_preview}")

        self.result_queue.put({"type": "video", "done": True})
        print("[Video] Processing complete.")

# Model defaults
WHISPER_MODEL = "distil-large-v3"
VLM_MODEL = "mlx-community/SmolVLM2-2.2B-Instruct-mlx"

# Audio processing
AUDIO_CHUNK_DURATION = 8.0  # seconds per chunk (was 2.0; 8s gives Whisper full sentences)
AUDIO_OVERLAP = 1.5  # seconds overlap between chunks (was 0.5)
AUDIO_SAMPLE_RATE = 16000  # Whisper expects 16kHz

# Deduplication
DEDUP_OVERLAP_RATIO = 0.6  # minimum word overlap ratio to consider duplicated

# Video processing
VIDEO_FPS = 1.0  # frames per second to extract (scene detection decides which to analyze)
CONTEXT_FRAMES = 4  # kept for backwards compat

# Scene change detection
SCENE_CHANGE_THRESHOLD = 0.35  # histogram correlation delta to trigger analysis (0-1)
SCENE_MIN_INTERVAL = 2.0  # minimum seconds between analyzed frames
SCENE_MAX_INTERVAL = 10.0  # force analysis after this many seconds even without change

# Fusion
FUSION_WINDOW = 10.0  # seconds to batch before fusing (was 5.0)
TIMESTAMP_TOLERANCE = 1.0  # seconds tolerance for alignment (was 0.5)

# Queues
AUDIO_QUEUE_SIZE = 200
VIDEO_QUEUE_SIZE = 50

# Output
DEFAULT_OUTPUT = "output/analysis.json"

# Speaker diarization
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
DIARIZATION_NUM_SPEAKERS = None  # Set to 2 if you know exactly how many speakers
DIARIZATION_MIN_SEGMENT = 1.0   # Merge same-speaker segments shorter than this (seconds)
DIARIZATION_MAX_SEGMENT = 15.0  # Max segment length before splitting for Whisper

# LLM summarization
SUMMARY_LLM_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SUMMARY_MAX_TOKENS = 200

# OCR-focused VLM prompt
VLM_PROMPT_OCR = (
    "You are an OCR and scene analysis assistant. For this screenshot:\n"
    "1. READ and transcribe ALL visible text exactly as written (menus, labels, "
    "buttons, URLs, headers, body text, error messages, chat messages).\n"
    "2. Describe the UI layout, application, and any people visible.\n"
    "3. Note any interactive elements (buttons, forms, dropdowns) and their states.\n"
    "Format: Start with 'TEXT ON SCREEN:' listing all readable text, then "
    "'SCENE:' with the visual description."
)

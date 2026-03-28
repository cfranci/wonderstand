#!/usr/bin/env python3
"""
Real-time Multi-Modal Video+Audio Analyzer for Apple Silicon.

Transcribes audio (Lightning Whisper MLX) and analyzes video frames
(SmolVLM2 via mlx-vlm) simultaneously, then fuses both streams into
combined multi-modal insights with LLM-generated summaries.

Usage:
    python multimodal_analyzer.py video.mp4
    python multimodal_analyzer.py video.mp4 --audio-model large-v3 --realtime
    python multimodal_analyzer.py video.mp4 --video-fps 0.5 --output results.json
    python multimodal_analyzer.py video.mp4 --diarize --realtime
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Real-time multi-modal video+audio analysis (Apple Silicon)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                              Basic analysis
  %(prog)s video.mp4 --realtime                    Stream results live
  %(prog)s video.mp4 --audio-model large-v3        High-quality audio
  %(prog)s video.mp4 --video-fps 0.5               Slower but thorough
  %(prog)s video.mp4 --audio-only                  Transcription only
  %(prog)s video.mp4 --video-only                  Visual analysis only
  %(prog)s video.mp4 --diarize --realtime          With speaker identification
  %(prog)s video.mp4 --no-llm-summary              Skip LLM summaries
  %(prog)s video.mp4 --meeting-notes --diarize      Extract meeting notes
  %(prog)s video.mp4 --meeting-notes --realtime     Live meeting notes
        """,
    )
    parser.add_argument("source", help="Path to video file")
    parser.add_argument(
        "--audio-model",
        default=None,
        help="Whisper model size (tiny, base, small, distil-large-v3, large-v3). "
        "Default: distil-large-v3",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Frames per second to extract (default: 1.0). Scene change "
        "detection decides which frames to actually analyze.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON path (default: output/analysis.json)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Stream fused results to terminal in real-time",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only run audio transcription (skip video analysis)",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Only run video analysis (skip audio transcription)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires HF_TOKEN env var and pyannote.audio)",
    )
    parser.add_argument(
        "--no-llm-summary",
        action="store_true",
        help="Disable LLM-powered summaries (use simple concatenation instead)",
    )
    parser.add_argument(
        "--speakers",
        default=None,
        help="Map speaker IDs to names. Format: 'SPEAKER_00=Colin,SPEAKER_01=Ryan'",
    )
    parser.add_argument(
        "--meeting-notes",
        action="store_true",
        help="Extract structured meeting notes (topics, action items, decisions) to Markdown",
    )
    parser.add_argument(
        "--notes-output",
        default=None,
        help="Path for meeting notes Markdown file (default: output/meeting_notes.md)",
    )

    args = parser.parse_args()

    # Parse speaker name mapping
    args.speaker_map = {}
    if args.speakers:
        for pair in args.speakers.split(","):
            if "=" in pair:
                k, v = pair.strip().split("=", 1)
                args.speaker_map[k.strip()] = v.strip()

    if not os.path.isfile(args.source):
        print(f"Error: File not found: {args.source}")
        sys.exit(1)

    if args.audio_only and args.video_only:
        print("Error: Cannot use both --audio-only and --video-only")
        sys.exit(1)

    # Single-modality shortcuts
    if args.audio_only:
        _run_audio_only(args)
        return

    if args.video_only:
        _run_video_only(args)
        return

    # Full multi-modal analysis
    from stream_processor import StreamProcessor

    processor = StreamProcessor(
        video_path=args.source,
        audio_model=args.audio_model,
        video_fps=args.video_fps,
        output_path=args.output,
        realtime=args.realtime,
        diarize=args.diarize,
        use_llm_summary=not args.no_llm_summary,
        speaker_map=args.speaker_map,
        meeting_notes=args.meeting_notes,
        meeting_notes_path=args.notes_output,
    )
    processor.run()


def _run_audio_only(args):
    """Run just audio transcription, with optional meeting notes."""
    import json
    import threading
    from audio_pipeline import AudioPipeline

    pipeline = AudioPipeline(model_size=args.audio_model)
    pipeline.load_model()

    # Optional diarization for audio-only mode
    diarizer = None
    if args.diarize:
        from diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer()
        diarizer.process(args.source)

    pipeline.process(args.source, diarizer=diarizer)

    results = []
    while not pipeline.result_queue.empty():
        item = pipeline.result_queue.get()
        if not item.get("done"):
            results.append(item)

    output_path = args.output or "output/transcription.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"segments": results}, f, indent=2, ensure_ascii=False)
    print(f"\nTranscription saved to: {output_path}")

    # Meeting notes extraction from transcription results
    if args.meeting_notes and results:
        from meeting_intelligence import MeetingIntelligence
        from config import SUMMARY_LLM_MODEL

        print("[Meeting] Loading LLM for meeting notes...")
        from mlx_lm import load as load_lm
        gpu_lock = threading.Lock()
        llm_model, llm_tokenizer = load_lm(SUMMARY_LLM_MODEL)

        notes_path = args.notes_output or "output/meeting_notes.md"
        meeting_intel = MeetingIntelligence(
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            gpu_lock=gpu_lock,
            output_path=notes_path,
            speakers=set(args.speaker_map.values()) if args.speaker_map else set(),
        )

        # Convert transcription results to fused-segment format
        for seg in results:
            fused = {
                "timestamp": seg.get("timestamp", 0),
                "end_timestamp": seg.get("timestamp", 0) + seg.get("duration", 0),
                "audio": {
                    "segments": [{
                        "text": seg.get("text", ""),
                        "speaker": seg.get("speaker", ""),
                    }]
                },
                "video": {"segments": []},
            }
            meeting_intel.process_segment(fused)

        meeting_intel.flush()
        print(f"Meeting notes saved to: {notes_path}")


def _run_video_only(args):
    """Run just video analysis."""
    import json
    from video_pipeline import VideoPipeline

    pipeline = VideoPipeline(fps=args.video_fps)
    pipeline.load_model()
    pipeline.process(args.source)

    results = []
    while not pipeline.result_queue.empty():
        item = pipeline.result_queue.get()
        if not item.get("done"):
            results.append(item)

    output_path = args.output or "output/video_analysis.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"segments": results}, f, indent=2, ensure_ascii=False)
    print(f"\nVideo analysis saved to: {output_path}")


if __name__ == "__main__":
    main()

"""
Meeting Intelligence — Real-time extraction of structured notes from fused segments.

Processes fused audio+video segments and uses the summary LLM to extract:
- Discussion topics / agenda items
- Key points and decisions
- Action items (with assigned speaker)
- Product features / plans mentioned
- Questions raised (with who asked)

Outputs a continuously-updated Markdown file and optional real-time terminal display.
"""

import json
import os
import re
import threading
from datetime import timedelta


# ---------- config constants ----------
MEETING_NOTES_MAX_TOKENS = 150
# How many fused segments to batch before running extraction
EXTRACTION_BATCH_SIZE = 20
# Rolling context window: keep last N extraction results for continuity
CONTEXT_WINDOW = 2


def _ts_fmt(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    td = timedelta(seconds=int(seconds))
    total_secs = int(td.total_seconds())
    hours, remainder = divmod(total_secs, 3600)
    mins, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


class MeetingIntelligence:
    """Extracts structured meeting notes from fused segments in real time."""

    def __init__(self, llm_model=None, llm_tokenizer=None, gpu_lock=None,
                 output_path=None, speakers=None):
        self._llm_model = llm_model
        self._llm_tokenizer = llm_tokenizer
        self.gpu_lock = gpu_lock
        self.output_path = output_path or "output/meeting_notes.md"
        self.speakers = speakers or set()

        # Accumulated structured data
        self.topics = []          # {time, title, points: [], speaker}
        self.action_items = []    # {time, task, owner, context}
        self.decisions = []       # {time, decision, speaker}
        self.questions = []       # {time, question, asked_by, answered}
        self.product_features = []  # {time, feature, detail, speaker}
        self.key_quotes = []      # {time, quote, speaker}

        # Processing state
        self._pending_segments = []
        self._recent_extractions = []  # rolling context
        self._current_topic = None
        self._segment_count = 0
        self._lock = threading.Lock()

        # Deduplication: track normalized text of items to avoid repeats
        self._seen_actions = set()
        self._seen_decisions = set()
        self._seen_questions = set()
        self._seen_features = set()

    def set_llm(self, model, tokenizer):
        """Set LLM after loading (shared with FusionEngine)."""
        self._llm_model = model
        self._llm_tokenizer = tokenizer

    def reset(self):
        """Clear all accumulated data for a fresh run."""
        self.topics.clear()
        self.action_items.clear()
        self.decisions.clear()
        self.questions.clear()
        self.product_features.clear()
        self.key_quotes.clear()
        self._pending_segments.clear()
        self._recent_extractions.clear()
        self._current_topic = None
        self._segment_count = 0
        self.speakers.clear()
        self._seen_actions.clear()
        self._seen_decisions.clear()
        self._seen_questions.clear()
        self._seen_features.clear()

    def process_segment(self, fused_segment, realtime=False):
        """Process a single fused segment. Called from fusion engine callback.

        Batches segments and runs extraction every EXTRACTION_BATCH_SIZE segments.
        """
        with self._lock:
            self._pending_segments.append(fused_segment)
            self._segment_count += 1

            # Discover speakers from data
            for aseg in fused_segment.get("audio", {}).get("segments", []):
                spk = aseg.get("speaker", "")
                if spk and spk != "UNKNOWN":
                    self.speakers.add(spk)

        if len(self._pending_segments) >= EXTRACTION_BATCH_SIZE:
            self._extract_batch(realtime=realtime)

    def flush(self):
        """Process any remaining pending segments and write final output."""
        if self._pending_segments:
            self._extract_batch(realtime=False)
        self._write_markdown()

    def _extract_batch(self, realtime=False):
        """Run LLM extraction on batched segments."""
        with self._lock:
            batch = list(self._pending_segments)
            self._pending_segments.clear()

        if not batch:
            return

        # Build transcript for the batch
        transcript_lines = []
        time_start = batch[0].get("timestamp", 0)
        time_end = batch[-1].get("end_timestamp", 0)

        for seg in batch:
            ts = seg.get("timestamp", 0)
            for aseg in seg.get("audio", {}).get("segments", []):
                speaker = aseg.get("speaker", "")
                text = aseg.get("text", "").strip()
                if text:
                    prefix = f"[{speaker}]" if speaker else "[?]"
                    transcript_lines.append(f"{_ts_fmt(ts)} {prefix} {text}")

            # Include screen context (brief)
            for vseg in seg.get("video", {}).get("segments", []):
                desc = vseg.get("description", "")
                if isinstance(desc, str) and desc:
                    # Just take the first line / 100 chars for context
                    brief = desc.split("\n")[0][:100]
                    transcript_lines.append(f"{_ts_fmt(ts)} [SCREEN] {brief}")

        if not transcript_lines:
            return

        transcript = "\n".join(transcript_lines)

        # Skip rolling context for small LLMs — it causes repetition
        # Just pass the fresh transcript
        extraction = self._run_extraction(transcript, "", time_start, time_end)

        if extraction:
            self._parse_extraction(extraction, time_start, time_end)
            # Add to rolling context
            summary = extraction[:200] if len(extraction) > 200 else extraction
            self._recent_extractions.append(summary)

            if realtime:
                self._print_realtime(extraction, time_start, time_end)
                self._write_markdown()

    def _run_extraction(self, transcript, context, time_start, time_end):
        """Use LLM to extract structured meeting intelligence."""
        if self._llm_model is None:
            return None

        system_prompt = (
            "Summarize ONLY what was explicitly said. Do NOT invent or infer.\n"
            "Output max 3-5 bullets total. Use this format:\n\n"
            "TOPIC: <what they discussed in 5 words>\n"
            "KEY POINTS:\n- <fact stated> (Speaker)\n"
            "ACTION ITEMS:\n- <task someone agreed to do> → Owner: <name>\n\n"
            "Rules:\n"
            "- ONLY list action items someone explicitly committed to\n"
            "- If no action items, skip that section entirely\n"
            "- Max 5 bullets total across all sections\n"
            "- Use speaker names from brackets exactly as shown"
        )

        user_content = (
            f"{context}"
            f"TRANSCRIPT ({_ts_fmt(time_start)} - {_ts_fmt(time_end)}):\n"
            f"{transcript}"
        )

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
                    result = generate(
                        self._llm_model, self._llm_tokenizer,
                        prompt=chat_prompt,
                        max_tokens=MEETING_NOTES_MAX_TOKENS,
                        verbose=False,
                    )
            else:
                result = generate(
                    self._llm_model, self._llm_tokenizer,
                    prompt=chat_prompt,
                    max_tokens=MEETING_NOTES_MAX_TOKENS,
                    verbose=False,
                )
            return result.strip() if isinstance(result, str) else str(result).strip()
        except Exception as e:
            print(f"[Meeting] Extraction failed: {e}")
            return None

    def _parse_extraction(self, text, time_start, time_end):
        """Parse LLM output into structured data."""
        time_ref = round(time_start, 1)
        current_section = None

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            upper = line.upper()
            if upper.startswith("TOPIC:"):
                title = line[6:].strip().strip("*").strip()
                if title and title.lower() not in ("none", "n/a", ""):
                    # Check if this is a genuinely new topic
                    if not self._current_topic or self._current_topic["title"].lower() != title.lower():
                        self._current_topic = {
                            "time": time_ref,
                            "title": title,
                            "points": [],
                            "time_range": f"{_ts_fmt(time_start)} - {_ts_fmt(time_end)}"
                        }
                        self.topics.append(self._current_topic)
                current_section = "topic"
                continue
            elif upper.startswith("KEY POINT"):
                current_section = "key_points"
                continue
            elif upper.startswith("ACTION ITEM"):
                current_section = "action_items"
                continue
            elif upper.startswith("DECISION"):
                current_section = "decisions"
                continue
            elif upper.startswith("QUESTION"):
                current_section = "questions"
                continue
            elif upper.startswith("PRODUCT") or upper.startswith("FEATURE"):
                current_section = "product"
                continue

            # Parse bullet items
            if line.startswith("- ") or line.startswith("• "):
                item_text = line[2:].strip()
                speaker = self._extract_speaker(item_text)

                # Normalize for dedup — strip speaker tags, lowercase, trim
                dedup_key = re.sub(r'\([^)]*\)\s*$', '', item_text).strip().lower()

                if current_section == "key_points" and self._current_topic:
                    self._current_topic["points"].append(item_text)
                elif current_section == "action_items":
                    owner = ""
                    # Parse "→ Owner: Name" pattern
                    if "→" in item_text:
                        parts = item_text.split("→", 1)
                        task = parts[0].strip()
                        owner_part = parts[1].strip()
                        if ":" in owner_part:
                            owner = owner_part.split(":", 1)[1].strip()
                        else:
                            owner = owner_part
                    elif "owner:" in item_text.lower():
                        idx = item_text.lower().index("owner:")
                        task = item_text[:idx].strip().rstrip("-–—")
                        owner = item_text[idx + 6:].strip()
                    else:
                        task = item_text
                        owner = speaker
                    task_key = re.sub(r'\([^)]*\)\s*$', '', task).strip().lower()
                    if task_key not in self._seen_actions:
                        self._seen_actions.add(task_key)
                        self.action_items.append({
                            "time": time_ref,
                            "task": task,
                            "owner": owner or "Unassigned",
                            "time_str": _ts_fmt(time_start),
                        })
                elif current_section == "decisions":
                    if dedup_key not in self._seen_decisions:
                        self._seen_decisions.add(dedup_key)
                        self.decisions.append({
                            "time": time_ref,
                            "decision": item_text,
                            "speaker": speaker,
                            "time_str": _ts_fmt(time_start),
                        })
                elif current_section == "questions":
                    asked_by = ""
                    if "asked by:" in item_text.lower():
                        idx = item_text.lower().index("asked by:")
                        question = item_text[:idx].strip().rstrip("-–—")
                        asked_by = item_text[idx + 9:].strip()
                    else:
                        question = item_text
                        asked_by = speaker
                    q_key = re.sub(r'\([^)]*\)\s*$', '', question).strip().lower()
                    if q_key not in self._seen_questions:
                        self._seen_questions.add(q_key)
                        self.questions.append({
                            "time": time_ref,
                            "question": question,
                            "asked_by": asked_by,
                            "time_str": _ts_fmt(time_start),
                        })
                elif current_section == "product":
                    if dedup_key not in self._seen_features:
                        self._seen_features.add(dedup_key)
                        self.product_features.append({
                            "time": time_ref,
                            "feature": item_text,
                            "speaker": speaker,
                            "time_str": _ts_fmt(time_start),
                        })

    @staticmethod
    def _extract_speaker(text):
        """Pull speaker name from parentheses at end of line, e.g. '... (Host)'."""
        match = re.search(r'\(([^)]+)\)\s*$', text)
        return match.group(1) if match else ""

    def _print_realtime(self, extraction, time_start, time_end):
        """Print extraction to terminal in real-time."""
        print(f"\n{'─'*60}")
        print(f"  📋 MEETING NOTES  [{_ts_fmt(time_start)} - {_ts_fmt(time_end)}]")
        print(f"{'─'*60}")
        for line in extraction.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Colorize section headers
            upper = line.upper()
            if any(upper.startswith(h) for h in ("TOPIC:", "KEY POINT", "ACTION ITEM",
                                                  "DECISION", "QUESTION", "PRODUCT", "FEATURE")):
                print(f"  \033[1m{line}\033[0m")
            elif line.startswith("- ") or line.startswith("• "):
                print(f"    {line}")
            else:
                print(f"  {line}")
        print()

    def _write_markdown(self):
        """Write accumulated meeting notes to Markdown file."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        lines = []
        lines.append("# Meeting Notes\n")
        lines.append(f"_Auto-generated from video analysis_\n")

        # Speakers
        if self.speakers:
            lines.append(f"**Participants:** {', '.join(sorted(self.speakers))}\n")

        # Topics / Agenda
        if self.topics:
            lines.append("---\n")
            lines.append("## 📌 Discussion Topics\n")
            for i, topic in enumerate(self.topics, 1):
                lines.append(f"### {i}. {topic['title']}")
                lines.append(f"*{topic.get('time_range', _ts_fmt(topic['time']))}*\n")
                if topic["points"]:
                    for pt in topic["points"]:
                        lines.append(f"- {pt}")
                    lines.append("")

        # Action Items
        if self.action_items:
            lines.append("---\n")
            lines.append("## ✅ Action Items\n")
            lines.append("| Time | Task | Owner |")
            lines.append("|------|------|-------|")
            for ai in self.action_items:
                task = ai["task"].replace("|", "\\|")
                lines.append(f"| {ai['time_str']} | {task} | {ai['owner']} |")
            lines.append("")

        # Decisions
        if self.decisions:
            lines.append("---\n")
            lines.append("## 🔨 Decisions Made\n")
            for d in self.decisions:
                speaker_tag = f" — {d['speaker']}" if d["speaker"] else ""
                lines.append(f"- **[{d['time_str']}]** {d['decision']}{speaker_tag}")
            lines.append("")

        # Product Features / Plans
        if self.product_features:
            lines.append("---\n")
            lines.append("## 🚀 Product Features & Plans\n")
            for pf in self.product_features:
                speaker_tag = f" ({pf['speaker']})" if pf["speaker"] else ""
                lines.append(f"- **[{pf['time_str']}]** {pf['feature']}{speaker_tag}")
            lines.append("")

        # Questions
        if self.questions:
            lines.append("---\n")
            lines.append("## ❓ Questions Raised\n")
            for q in self.questions:
                asked = f" — *{q['asked_by']}*" if q["asked_by"] else ""
                lines.append(f"- **[{q['time_str']}]** {q['question']}{asked}")
            lines.append("")

        # Stats
        lines.append("---\n")
        lines.append(f"*{self._segment_count} segments processed · "
                      f"{len(self.topics)} topics · "
                      f"{len(self.action_items)} action items · "
                      f"{len(self.decisions)} decisions*\n")

        with open(self.output_path, "w") as f:
            f.write("\n".join(lines))

    def get_summary_data(self):
        """Return all extracted data as a dict (for JSON export)."""
        return {
            "speakers": sorted(self.speakers),
            "topics": self.topics,
            "action_items": self.action_items,
            "decisions": self.decisions,
            "questions": self.questions,
            "product_features": self.product_features,
            "segments_processed": self._segment_count,
        }

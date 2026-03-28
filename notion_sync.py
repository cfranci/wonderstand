"""
Notion Sync — Creates and updates a Notion page with live meeting notes.

Used by live_capture.py to push meeting intelligence to Notion in real-time.
Requires the Notion MCP integration (runs via Claude Code).

Since we can't call MCP tools directly from Python, this module writes
a JSON "sync file" that the orchestrating Claude session reads and
pushes to Notion.
"""

import json
import os
import time
from datetime import datetime


SYNC_DIR = "output/notion_sync"


class NotionSync:
    """Writes meeting notes to a sync file for Notion updates."""

    def __init__(self, session_name=None):
        self.session_name = session_name or datetime.now().strftime(
            "Live Meeting — %b %d, %Y %I:%M %p"
        )
        self.sync_dir = SYNC_DIR
        os.makedirs(self.sync_dir, exist_ok=True)

        self.sync_file = os.path.join(self.sync_dir, "current_session.json")
        self.page_id = None  # Set externally after Notion page is created

        # Write initial sync state
        self._write_sync({
            "status": "started",
            "session_name": self.session_name,
            "started_at": datetime.now().isoformat(),
            "page_id": None,
            "notes": None,
            "update_count": 0,
        })

    def _write_sync(self, data):
        """Write sync state to JSON file."""
        with open(self.sync_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def update_notes(self, meeting_data):
        """Update sync file with latest meeting notes."""
        self._write_sync({
            "status": "recording",
            "session_name": self.session_name,
            "started_at": getattr(self, '_started_at', datetime.now().isoformat()),
            "page_id": self.page_id,
            "notes": meeting_data,
            "updated_at": datetime.now().isoformat(),
            "update_count": getattr(self, '_update_count', 0) + 1,
        })
        self._update_count = getattr(self, '_update_count', 0) + 1

    def finalize(self, meeting_data, duration_seconds):
        """Write final sync state."""
        self._write_sync({
            "status": "completed",
            "session_name": self.session_name,
            "page_id": self.page_id,
            "notes": meeting_data,
            "duration": duration_seconds,
            "completed_at": datetime.now().isoformat(),
            "update_count": getattr(self, '_update_count', 0) + 1,
        })

    @staticmethod
    def format_notion_content(meeting_data, session_name=None, duration=None):
        """Format meeting intelligence data as Notion-flavored markdown content."""
        lines = []

        if duration:
            mins = int(duration) // 60
            secs = int(duration) % 60
            lines.append(f"**Duration:** {mins}m {secs}s")
            lines.append("")

        speakers = meeting_data.get("speakers", [])
        if speakers:
            lines.append(f"**Participants:** {', '.join(speakers)}")
            lines.append("")

        segments = meeting_data.get("segments_processed", 0)
        lines.append(f"**Segments processed:** {segments}")
        lines.append("")

        # Action Items
        action_items = meeting_data.get("action_items", [])
        if action_items:
            lines.append("## Action Items")
            lines.append("")
            for ai in action_items:
                task = ai.get("task", "")
                owner = ai.get("owner", "Unassigned")
                time_str = ai.get("time_str", "")
                lines.append(f"- [ ] **[{time_str}]** {task} — *{owner}*")
            lines.append("")

        # Decisions
        decisions = meeting_data.get("decisions", [])
        if decisions:
            lines.append("## Decisions Made")
            lines.append("")
            for d in decisions:
                decision = d.get("decision", "")
                speaker = d.get("speaker", "")
                time_str = d.get("time_str", "")
                speaker_tag = f" — {speaker}" if speaker else ""
                lines.append(f"- **[{time_str}]** {decision}{speaker_tag}")
            lines.append("")

        # Product Features
        features = meeting_data.get("product_features", [])
        if features:
            lines.append("## Product Features & Plans")
            lines.append("")
            for pf in features:
                feature = pf.get("feature", "")
                speaker = pf.get("speaker", "")
                time_str = pf.get("time_str", "")
                speaker_tag = f" ({speaker})" if speaker else ""
                lines.append(f"- **[{time_str}]** {feature}{speaker_tag}")
            lines.append("")

        # Questions
        questions = meeting_data.get("questions", [])
        if questions:
            lines.append("## Questions Raised")
            lines.append("")
            for q in questions:
                question = q.get("question", "")
                asked_by = q.get("asked_by", "")
                time_str = q.get("time_str", "")
                asked_tag = f" — *{asked_by}*" if asked_by else ""
                lines.append(f"- **[{time_str}]** {question}{asked_tag}")
            lines.append("")

        # Topics
        topics = meeting_data.get("topics", [])
        if topics:
            lines.append("## Discussion Topics")
            lines.append("")
            for i, topic in enumerate(topics, 1):
                title = topic.get("title", "")
                time_range = topic.get("time_range", "")
                lines.append(f"### {i}. {title}")
                if time_range:
                    lines.append(f"*{time_range}*")
                points = topic.get("points", [])
                for pt in points:
                    lines.append(f"- {pt}")
                lines.append("")

        # Stats footer
        stats_parts = []
        if segments:
            stats_parts.append(f"{segments} segments")
        if topics:
            stats_parts.append(f"{len(topics)} topics")
        if action_items:
            stats_parts.append(f"{len(action_items)} action items")
        if decisions:
            stats_parts.append(f"{len(decisions)} decisions")
        if stats_parts:
            lines.append("---")
            lines.append(f"*{' | '.join(stats_parts)}*")

        return "\n".join(lines)

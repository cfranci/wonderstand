"""
Notion Sync — Live-syncs meeting notes to a Notion page via direct API.

Creates a page at session start with toggleable section headings (Transcript,
Action Items, Decisions, Questions). Appends blocks to each section in
real-time as data arrives — no polling, no sync files, no replacing content.

Requires:
    NOTION_TOKEN          — Internal integration token
    NOTION_PARENT_PAGE_ID — Page ID to create meeting notes under

Rate limits: Notion allows 3 req/sec. We batch block appends per chunk,
so typical usage is 1-4 requests per ~30s chunk — well under the limit.
"""

import os
import time
from datetime import datetime

try:
    from notion_client import Client
    from notion_client.errors import APIResponseError
except ImportError:
    Client = None
    APIResponseError = None


class NotionSync:
    """Appends meeting notes to a Notion page in real-time."""

    def __init__(self, session_name=None, token=None, parent_page_id=None):
        if Client is None:
            raise ImportError(
                "notion-client is required for Notion sync. "
                "Install it: python -m pip install notion-client"
            )

        self.token = token or os.environ.get("NOTION_TOKEN")
        self.parent_page_id = parent_page_id or os.environ.get("NOTION_PARENT_PAGE_ID")
        self.session_name = session_name or datetime.now().strftime(
            "Live Meeting — %b %d, %Y %I:%M %p"
        )

        if not self.token:
            raise ValueError(
                "Notion sync requires NOTION_TOKEN env var. "
                "Create an integration at https://www.notion.so/my-integrations"
            )
        if not self.parent_page_id:
            raise ValueError(
                "Notion sync requires NOTION_PARENT_PAGE_ID env var. "
                "Set it to the page ID where meeting notes should be created."
            )

        self.client = Client(auth=self.token)
        self.page_id = None
        self.page_url = None

        # Block IDs of the toggleable section headings
        self._transcript_id = None
        self._actions_id = None
        self._decisions_id = None
        self._questions_id = None

        self._last_request = 0.0

    # ── Throttle / retry ─────────────────────────────────────────

    def _throttle(self):
        """Stay under 3 req/sec."""
        gap = time.time() - self._last_request
        if gap < 0.35:
            time.sleep(0.35 - gap)
        self._last_request = time.time()

    def _call(self, fn, *args, **kwargs):
        """Notion API call with throttle + retry on 429."""
        for attempt in range(3):
            self._throttle()
            try:
                return fn(*args, **kwargs)
            except APIResponseError as e:
                if e.status == 429:
                    time.sleep(min(2 ** attempt, 5))
                    continue
                raise
        raise RuntimeError("Notion API rate-limited after 3 retries")

    # ── Page lifecycle ───────────────────────────────────────────

    def create_page(self):
        """Create meeting notes page with section headings. Returns (page_id, url)."""
        now_str = datetime.now().strftime("%B %d, %Y %I:%M %p")

        resp = self._call(
            self.client.pages.create,
            parent={"page_id": self.parent_page_id},
            properties={
                "title": {"title": [{"text": {"content": self.session_name}}]},
            },
            icon={"type": "emoji", "emoji": "🎙️"},
            children=[
                _callout(f"📅 {now_str}  ·  Recording in progress…", "gray_background"),
                _divider(),
                _toggle_heading("Transcript"),
                _toggle_heading("Action Items"),
                _toggle_heading("Decisions"),
                _toggle_heading("Questions"),
            ],
        )

        self.page_id = resp["id"]
        self.page_url = resp.get("url", "")

        # Fetch section block IDs so we can append to each
        children = self._call(
            self.client.blocks.children.list, block_id=self.page_id,
        )
        headings = [b for b in children["results"] if b["type"] == "heading_2"]

        if len(headings) >= 4:
            self._transcript_id = headings[0]["id"]
            self._actions_id = headings[1]["id"]
            self._decisions_id = headings[2]["id"]
            self._questions_id = headings[3]["id"]

        return self.page_id, self.page_url

    def finalize(self, duration):
        """Mark the page header as complete."""
        if not self.page_id:
            return

        mins = int(duration) // 60
        secs = int(duration) % 60
        now_str = datetime.now().strftime("%B %d, %Y %I:%M %p")

        # Find the callout block and update it
        children = self._call(
            self.client.blocks.children.list, block_id=self.page_id,
        )
        for block in children["results"]:
            if block["type"] == "callout":
                self._call(
                    self.client.blocks.update,
                    block_id=block["id"],
                    callout={
                        "rich_text": [{
                            "text": {
                                "content": (
                                    f"📅 {now_str}  ·  "
                                    f"Duration: {mins}m {secs}s  ·  ✅ Complete"
                                ),
                            },
                        }],
                        "color": "green_background",
                    },
                )
                break

    # ── Append methods ───────────────────────────────────────────

    def append_transcript(self, segments):
        """Append transcript lines to the Transcript section.

        Args:
            segments: list of {speaker, text, start, end}
        """
        if not segments or not self._transcript_id:
            return

        blocks = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker", "")
            start = seg.get("start", 0)
            ts = f"{int(start) // 60}:{int(start) % 60:02d}"

            rich = []
            if speaker:
                rich.append(_bold(f"[{ts}] {speaker}: "))
            else:
                rich.append(_bold(f"[{ts}] "))
            rich.append(_text(text))

            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": rich},
            })

        if blocks:
            self._call(
                self.client.blocks.children.append,
                block_id=self._transcript_id,
                children=blocks,
            )

    def append_actions(self, items):
        """Append to-do items to the Action Items section.

        Args:
            items: list of {text, speaker}
        """
        if not items or not self._actions_id:
            return

        blocks = []
        for item in items:
            label = item.get("text", "")
            owner = item.get("speaker", "")
            if owner:
                label = f"{label} — {owner}"
            blocks.append({
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [_text(label)],
                    "checked": False,
                },
            })

        if blocks:
            self._call(
                self.client.blocks.children.append,
                block_id=self._actions_id,
                children=blocks,
            )

    def append_decisions(self, items):
        """Append to the Decisions section.

        Args:
            items: list of {text, speaker}
        """
        if not items or not self._decisions_id:
            return

        blocks = []
        for item in items:
            label = item.get("text", "")
            speaker = item.get("speaker", "")
            if speaker:
                label = f"{label} — {speaker}"
            blocks.append(_bullet(label))

        if blocks:
            self._call(
                self.client.blocks.children.append,
                block_id=self._decisions_id,
                children=blocks,
            )

    def append_questions(self, items):
        """Append to the Questions section.

        Args:
            items: list of {text, speaker}
        """
        if not items or not self._questions_id:
            return

        blocks = []
        for item in items:
            label = item.get("text", "")
            asker = item.get("speaker", "")
            if asker:
                label = f"{label} — {asker}"
            blocks.append(_bullet(label))

        if blocks:
            self._call(
                self.client.blocks.children.append,
                block_id=self._questions_id,
                children=blocks,
            )


# ── Block helpers ────────────────────────────────────────────────

def _text(content):
    return {"type": "text", "text": {"content": content}}


def _bold(content):
    return {"type": "text", "text": {"content": content}, "annotations": {"bold": True}}


def _callout(text, color="gray_background"):
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [_text(text)],
            "color": color,
            "icon": {"type": "emoji", "emoji": "📋"},
        },
    }


def _divider():
    return {"object": "block", "type": "divider", "divider": {}}


def _toggle_heading(title):
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [_text(title)],
            "is_toggleable": True,
        },
    }


def _bullet(text):
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": [_text(text)]},
    }

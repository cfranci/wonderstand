# Wonderstand Always-On Extension — Install Guide

This guide covers the manual install steps for the always-on capture extension.
The Python pieces are already in place and tested:
- `spine.py` — append-only event log
- `policy_resolver.py` + `contacts.yaml` — consent tier decisions
- `voice_profiles_provisional.py` — pending pool + 3-match promotion + rollback
- `folder_watcher.py` — atomic file watch + opus re-encode
- `heartbeat.py` — daily digest + permission probes
- `ingest_server.py` — patched with `/ingest` endpoint and spine hooks

This document covers the parts that require your hands.

---

## 1. Edit `contacts.yaml`

Open `contacts.yaml` and replace the sample entries with real numbers/emails of
people you talk to. Consent values:

- `full` — record everything, always tier A (90-day audio retention)
- `transcript_only` — tier B, transcript kept, audio shredded after transcribing
- `none` — refuse to record; the server returns 403 and deletes the upload
- `default` — fall through to the defaults block (regime-based)

```yaml
"+15551234567":
  name: "Real Person"
  consent: full
```

Sane defaults are already set: unknown two-party calls → tier B (transcript only),
unknown one-party → tier A. Edit the `TWO_PARTY_STATES` list if you want to be
more or less conservative.

---

## 2. Install the iOS Shortcut (manual, in the Shortcuts app)

iOS Shortcuts cannot be installed via the CLI from a Mac. You build the shortcut
once in the Shortcuts app on your iPhone. Steps:

1. Open Shortcuts on iPhone → tap "+" to create a new shortcut.
2. Add these actions in order:

   **A. "Record Audio"** (Media)
   - Quality: Very High
   - Start: On Tap
   - Stop: On Tap

   **B. "Get Current Date"** → variable `record_started_at` (ISO 8601)

   **C. "Find Calendar Events Where"** (Calendar)
   - Filter: Start Date is within the next 15 minutes OR ended within the last 15 minutes
   - Limit: 1
   - Sort: Start Date

   **D. "Dictionary"** (Scripting) — build the sidecar payload
   ```
   {
     "counterparties": [],
     "outbound": true,
     "chase_state": "FL",
     "calendar_snapshot": {
       "title": <Title from "Calendar Event">,
       "start_iso": <Start Date from "Calendar Event">,
       "attendees": []
     },
     "record_started_at": <record_started_at variable>
   }
   ```
   (Leave `counterparties` empty if it was an in-person meeting or an ad-hoc
   recording; populate from a separate prompt if it was a phone call.)

   **E. "Base64 Encode"** (Scripting) — input: the dictionary from step D, JSON-stringified

   **F. "Get Contents of URL"** (Web)
   - URL: `https://<your-tunnel-or-tailscale>/ingest`
   - Method: POST
   - Headers:
     - `X-Filename`: `Recording_<record_started_at>.m4a`
     - `X-Sidecar`: the base64 string from step E
   - Request Body: File → the recording from step A

3. Name the shortcut "Wonderstand Capture".
4. Triggers (Personal Automation):
   - **"In Meeting" Focus turns on** → run shortcut
   - **Calendar event with title containing "Call:" starts** → run shortcut
   - (Manual is also fine — Hey Siri "Wonderstand capture")

---

## 3. Caffeinate Mac during calendar events (LaunchAgent)

Already included as a sample LaunchAgent below. Drop into
`~/Library/LaunchAgents/com.cf.wonderstand-caffeinate.plist` and `launchctl load`.
The agent runs `caffeinate -i` whenever a Calendar event tagged "Meeting" is in
progress (uses `icalBuddy` to check).

See `com.cf.wonderstand-caffeinate.plist` in this directory.

---

## 4. ScreenCaptureKit daemon for Mac calls (Swift, you compile)

The Python pipeline handles audio that lands in `inbox/`. To capture Zoom / Meet /
FaceTime audio on the Mac, you need a Swift helper using ScreenCaptureKit. Source
is at `mac_call_capture.swift` in this directory.

Build:
```bash
cd /Users/cf/Projects/wonderstand
swiftc -O mac_call_capture.swift -o mac_call_capture
```

First run will prompt for Screen Recording permission. Grant it in System
Settings → Privacy & Security → Screen Recording.

Configure the daemon's launchd plist (`com.cf.wonderstand-screencap.plist`) to
watch for `zoom.us`, `Google Chrome` with `meet.google.com`, `FaceTime` running,
and start the capture for the duration of that process. Output goes to `inbox/`.

---

## 5. Heartbeat as a daily cron

Add to your crontab:
```cron
0 8 * * * cd /Users/cf/Projects/wonderstand && venv/bin/python -c "import heartbeat; from pathlib import Path; heartbeat.daily_digest(Path('events.ndjson'), since_hours=24, output=Path('output/digest_$(date +\%Y\%m\%d).md'))"
```

This writes a markdown digest to `output/` every morning. No pager, no Pushover.
Read it when you sit down at the Mac.

---

## 6. Verify the pipeline

```bash
cd /Users/cf/Projects/wonderstand
venv/bin/python test_integration.py
```

All 10 tests should pass. If they don't, something is wrong before you trust the
system with real recordings.

---

## What's NOT built (and why)

- **iPhone cellular call recording.** Stock iOS does not expose the call audio
  stream to apps. Room-mic on speakerphone is the only honest path.
- **Twilio VoIP routing.** Deferred until you decide whether to route business
  calls through a controlled number.
- **Apple Developer Program / TestFlight iOS app.** Personal-team signing doesn't
  grant CallKit/VoIP entitlements, so the iOS app would just duplicate the
  Shortcut for more cost. Skipped.
- **Cross-device dedup beyond start-time overlap.** Will be added once you see
  duplicate transcripts in practice.

---

## Emergency: rollback a poisoned voice profile

```bash
cd /Users/cf/Projects/wonderstand
venv/bin/python -c "
from voice_profiles_provisional import ProvisionalProfiles
from pathlib import Path
pp = ProvisionalProfiles(Path('voice_profiles'))
n = pp.rollback('NAME_TO_REMOVE')
print(f'removed {n} entries')
"
```

This removes the named profile from `profiles.json` AND any pending entries
bound to that name. Logged in `voice_profiles/binding_log.jsonl`.

---

## Where to look when something feels off

| Symptom | Check |
|---|---|
| "No recordings showing up" | `events.ndjson` tail — any `capture.started`? |
| "Got 403 on upload" | Check `contacts.yaml` for `consent: none`. |
| "Two transcripts for one call" | Look for capture.started pairs in spine with overlapping windows; dual-device dedup is deferred. |
| "Wrong speaker name in transcript" | Check `voice_profiles/pending.json` for cross-bound entry; rollback that profile. |
| "ScreenCaptureKit silently failing" | Heartbeat will flag permissions: false. Re-grant Screen Recording. |

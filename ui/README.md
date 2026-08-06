# Wonderstand UI

A simple localhost dashboard for the recordings wonderstand transcribes:
browse sessions, group voice samples by person, search summaries + transcripts,
play audio, and label speakers.

## Run

```bash
cd /Users/cf/Projects/wonderstand
venv/bin/python ui/server.py            # http://127.0.0.1:8770
```

Bound to `127.0.0.1` only (security baseline). Add `--port N` to change the port.

## What it shows

- **Voices** (left): every enrolled profile from `voice_profiles/profiles.json`,
  with how many recordings they appear in. Click a voice to filter to their
  recordings. Click **edit** to rename a profile (cascades everywhere).
- **Recordings** (main): one card per transcript in `output/`. Green chips are
  enrolled voices; orange chips are speakers not yet identified (`SPEAKER_00`…).
  Search box filters by title, summary, location, person, or transcript text.
- **Detail** (click a card): editable location + confidence + summary, the audio
  player (`.qta` is transcoded to mp3 on demand), and the speaker-labeled
  transcript. Click a timestamp to jump the audio there. Use **label speakers**
  to rename `SPEAKER_00` → a real name for that recording.

## Data it owns

`ui/meta.json` — the editable layer (location, summary, custom title, per-session
speaker names, audio path). Everything else is read from wonderstand's existing
`output/` transcripts and `voice_profiles/`. Nothing here modifies your source
recordings.

## Enrolling a voice from an already-transcribed session

No need to re-run diarization — pull the speaker straight from the transcript:

```bash
cd voice_profiles
python enroll_from_transcript.py --transcript ../output/<file>.json --stats   # see speakers
python enroll_from_transcript.py --transcript ../output/<file>.json \
    --audio "/path/to/source.qta" --speaker SPEAKER_00 --name Larry
```

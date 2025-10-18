# Stream + Deepgram Transcription System

Real-time speech-to-text with NLP consolidation and WebSocket streaming.

## Prerequisites
- Python 3.11 (managed automatically by `uv`)
- A Stream app with API key/secret
- A Deepgram API key
- A hosted call UI to join (e.g. the sample UI linked from the quickstart)

## Setup
1. Install dependencies and create the virtual environment:
   ```bash
   uv sync
   ```
2. Create a `.env` file with the credentials required by the tutorial:
   ```bash
   STREAM_API_KEY=your-stream-api-key
   STREAM_API_SECRET=your-stream-api-secret
   STREAM_BASE_URL=https://video.stream-io-api.com/
   DEEPGRAM_API_KEY=your-deepgram-api-key
   EXAMPLE_BASE_URL=https://your-call-ui-base-url
   ```
   `EXAMPLE_BASE_URL` should point to the base join URL of the frontend you plan to use. The hosted quickstart UI linked in the docs provides a value you can copy verbatim.

## Usage

### 1. Start Live Transcription

```bash
# Default room
uv run python main.py

# Specify room/call ID
uv run python main.py --call-id topic-a-ai
```

Saves transcripts to `transcripts/YYYY-MM-DD-<call_id>.jsonl`

---

### 2. Run Consolidation (NLP Processing)

```bash
# Process existing transcript from start
uv run python consolidate.py --file transcripts/2025-09-30-abc123.jsonl --from-start

# Live mode: watch for new data (use with main.py)
uv run python consolidate.py --follow --call-id my-meeting-room
```

Outputs consolidated paragraphs via:
- **Console**: `user: <speaker>\ncontent: <paragraph>`
- **WebSocket**: `ws://127.0.0.1:8799` (JSON)
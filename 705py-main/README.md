# The new function is to deal with the fragment text

It update the main.py by generate a jsonl document to record the text and then through consolidate.py(nature language process) to get the complete paragraph



# Stream Python AI + Deepgram Quickstart

This project follows Stream's Python AI quickstart but swaps the OpenAI realtime integration for the Deepgram speech-to-text plugin.

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

## Run the example
Launch the Deepgram-enabled quickstart bot:
```bash
uv run python main.py
```

The script will:
- Create a Stream call and a temporary user plus Deepgram bot identity
- Open (or print) a join URL for your browser
- Join the call as the bot and stream audio into Deepgram
- Log partial and final transcripts from remote participants

Press `Ctrl+C` to end the session. The temporary users are cleaned up automatically.

Set `LOG_LEVEL=DEBUG` in your environment to see partial transcript logs.

```
uv run python consolidate.py --file transcripts/2025-09-30-abc123.jsonl --from-start --min-duration 5 --min-chars 30
```